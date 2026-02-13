"""
ai_rename_extension.py — Nautilus (GNOME 46+) AI-powered file renaming extension.

DEVELOPMENT WORKFLOW
--------------------
This file lives in the project repository and is **not** edited in-place inside
the Nautilus extensions directory.  install.sh creates a symlink so Nautilus
loads it directly from the repository:

    ~/.local/share/nautilus-python/extensions/ai_rename_extension.py
        -> /path/to/repo/src/ai_rename_extension.py   (absolute symlink)

After any change to this file, restart Nautilus to pick up the changes:

    nautilus -q

Nautilus will relaunch automatically the next time Files is opened.
"""

# Force unbuffered output so prints from background threads appear immediately
import functools
print = functools.partial(print, flush=True)

import base64
import io
import json
import mimetypes
import os
import re
import subprocess
import threading
import time
import concurrent.futures
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import gi  # type: ignore[import-untyped]
gi.require_version("Nautilus", "4.0")
gi.require_version("GLib", "2.0")
gi.require_version("GObject", "2.0")
gi.require_version("Gio", "2.0")
from gi.repository import GLib, GObject, Gio, Nautilus  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Optional C-extension imports — loaded on the main thread at startup to
# avoid silent crashes when importing from background threads in the Nautilus
# Python process.
# ---------------------------------------------------------------------------

try:
    from PIL import Image  # type: ignore[import-untyped]
    HAS_PIL = True
except ImportError:
    Image = None  # type: ignore[assignment]
    HAS_PIL = False
    print("[AI Rename] Pillow not installed; image files will use filename only")

try:
    import fitz  # type: ignore[import-untyped]
    HAS_FITZ = True
except ImportError:
    fitz = None  # type: ignore[assignment]
    HAS_FITZ = False

try:
    import pypdf  # type: ignore[import-untyped]
    HAS_PYPDF = True
except ImportError:
    pypdf = None  # type: ignore[assignment]
    HAS_PYPDF = False

try:
    import tinytag  # type: ignore[import-untyped]
    HAS_TINYTAG = True
except ImportError:
    tinytag = None  # type: ignore[assignment]
    HAS_TINYTAG = False


class LLMClient:
    """Client for the Anthropic Claude API.

    Usage::

        client = LLMClient(api_key="...")
        name = client.generate_from_text(content, "report.txt")
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"
    TIMEOUT = 60

    _API_BASE = "https://api.anthropic.com/v1/messages"
    _ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model if model is not None else self.DEFAULT_MODEL

    # ------------------------------------------------------------------
    # Public generation methods
    # ------------------------------------------------------------------

    def generate_from_text(
        self,
        content: str,
        original_name: str,
    ) -> str | None:
        """Return a suggested filename derived from *content*.

        Args:
            content: Text excerpt from the file (will be truncated to 2 000 chars).
            original_name: Current filename, including extension.

        Returns:
            Suggested filename string, or None on failure.
        """
        excerpt = content.strip()[:2000] if content and content.strip() else "(no readable content)"

        prompt = (
            "You are a file-naming assistant. Your only job is to output a short, "
            "descriptive filename stem.\n\n"
            f"Original filename: {original_name}\n"
            f"File content preview:\n{excerpt}\n\n"
            "Rules — read carefully:\n"
            "- Output ONLY the filename stem. No extension, no explanation, no extra words.\n"
            "- Use 3 to 6 lowercase words separated by hyphens "
            "(example: quarterly-budget-summary).\n"
            "- Only letters, numbers, and hyphens — no spaces, underscores, or punctuation.\n"
            "- Be specific and descriptive based on the actual content above.\n\n"
            "Filename stem:"
        )
        return self._call_claude([{"type": "text", "text": prompt}], max_tokens=50)

    def generate_from_image(
        self,
        filepath: str,
        original_name: str,
    ) -> str | None:
        """Return a suggested filename derived from image content.

        Args:
            filepath: Absolute path to the image file.
            original_name: Current filename, including extension.

        Returns:
            Suggested filename string, or None on failure.
        """
        print(f"[AI Rename] generate_from_image: {original_name!r}, path={filepath!r}")

        # Detect MIME type from extension; Claude accepts jpeg/png/webp/gif.
        mime_type, _ = mimetypes.guess_type(filepath)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
        print(f"[AI Rename] generate_from_image: detected mime={mime_type!r}")

        _MAX_DIRECT_BYTES = 10 * 1024 * 1024  # 10 MB

        try:
            file_size = os.path.getsize(filepath)
            print(f"[AI Rename] generate_from_image: file_size={file_size} bytes")

            if file_size > _MAX_DIRECT_BYTES and HAS_PIL:
                # Large file: resize with PIL to stay within API limits.
                print(f"[AI Rename] generate_from_image: large file, resizing with PIL")
                img = Image.open(filepath)  # type: ignore[union-attr]
                print(f"[AI Rename] generate_from_image: PIL opened, mode={img.mode!r}, size={img.size}")
                if img.mode in ('RGBA', 'LA', 'PA'):
                    bg = Image.new('RGB', img.size, (255, 255, 255))  # type: ignore[union-attr]
                    bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                    img = bg
                elif img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                img.thumbnail((1024, 1024), Image.Resampling.BILINEAR)  # type: ignore[union-attr]
                print(f"[AI Rename] generate_from_image: thumbnailed to {img.size}")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85, optimize=False)
                image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                mime_type = "image/jpeg"
            else:
                # Fast path: raw bytes, no PIL involved.
                with open(filepath, "rb") as fh:
                    raw = fh.read()
                image_b64 = base64.b64encode(raw).decode("ascii")

            print(f"[AI Rename] generate_from_image: b64 length={len(image_b64)}, mime={mime_type!r}")
        except Exception as exc:
            print(f"[AI Rename] generate_from_image: failed to read {filepath!r}: {exc}")
            return None

        prompt = (
            "You are a file-naming assistant. Your only job is to output a short, "
            "descriptive filename stem for this image.\n\n"
            f"Original filename: {original_name}\n\n"
            "Rules — read carefully:\n"
            "- Output ONLY the filename stem. No extension, no explanation, no extra words.\n"
            "- Use 3 to 6 lowercase words separated by hyphens "
            "(example: golden-gate-bridge-sunset).\n"
            "- Only letters, numbers, and hyphens — no spaces, underscores, or punctuation.\n"
            "- Be specific and descriptive based on what you see in the image.\n\n"
            "Filename stem:"
        )
        parts: List[Dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_b64}},
            {"type": "text", "text": prompt},
        ]
        result = self._call_claude(parts, max_tokens=50)
        print(f"[AI Rename] generate_from_image: result={result!r}")
        return result

    def generate_from_audio(
        self,
        filepath: str,
        original_name: str,
    ) -> str | None:
        """Return a suggested filename derived from audio content.

        Args:
            filepath: Absolute path to the audio file (must be ≤20 MB).
            original_name: Current filename, including extension.

        Returns:
            Suggested filename string, or None on failure.
        """
        print(f"[AI Rename] generate_from_audio: {original_name!r}, path={filepath!r}")

        mime_type, _ = mimetypes.guess_type(filepath)
        if not mime_type or not mime_type.startswith("audio/"):
            mime_type = "audio/mpeg"
        print(f"[AI Rename] generate_from_audio: detected mime={mime_type!r}")

        try:
            file_size = os.path.getsize(filepath)
            print(f"[AI Rename] generate_from_audio: file_size={file_size} bytes")
            with open(filepath, "rb") as fh:
                raw = fh.read()
            audio_b64 = base64.b64encode(raw).decode("ascii")
            print(f"[AI Rename] generate_from_audio: b64 length={len(audio_b64)}, mime={mime_type!r}")
        except Exception as exc:
            print(f"[AI Rename] generate_from_audio: failed to read {filepath!r}: {exc}")
            return None

        prompt = (
            "You are a file-naming assistant. Your only job is to output a short, "
            "descriptive filename stem for this audio file.\n\n"
            f"Original filename: {original_name}\n\n"
            "Rules — read carefully:\n"
            "- Output ONLY the filename stem. No extension, no explanation, no extra words.\n"
            "- Use 3 to 6 lowercase words separated by hyphens "
            "(example: jazz-piano-improvisation-session).\n"
            "- Only letters, numbers, and hyphens — no spaces, underscores, or punctuation.\n"
            "- Be specific and descriptive based on what you hear in the audio.\n\n"
            "Filename stem:"
        )
        parts: List[Dict[str, Any]] = [
            {"type": "audio", "source": {"type": "base64", "media_type": mime_type, "data": audio_b64}},
            {"type": "text", "text": prompt},
        ]
        result = self._call_claude(parts, max_tokens=50)
        print(f"[AI Rename] generate_from_audio: result={result!r}")
        return result

    def generate_batch(
        self,
        files: List[tuple[str, str]],
    ) -> Optional[Dict[str, str]]:
        """Generate filename stems for multiple text files in one API request.

        Args:
            files: List of (original_name, content_excerpt) pairs.
                   Submit at most 10 items per call to keep the prompt concise.

        Returns:
            Dict mapping each original_name to a raw (unsanitised) filename stem,
            or None if the response could not be parsed as JSON (caller should fall
            back to individual requests).
        """
        # Build the file-listing block — short per-file excerpts keep the
        # prompt within the context window.
        file_lines: List[str] = []
        for i, (name, content) in enumerate(files, 1):
            excerpt = (
                content.strip()[:500]
                if content and content.strip()
                else "(no readable content)"
            )
            file_lines.append(f"{i}. {name}\n{excerpt}")
        files_block = "\n\n".join(file_lines)

        # Use the first two filenames in the example so the model sees the format.
        ex_names = [name for name, _ in files[:2]]
        if len(ex_names) >= 2:
            example = (
                f'{{"{ex_names[0]}": "descriptive-stem-here", '
                f'"{ex_names[1]}": "another-descriptive-stem"}}'
            )
        else:
            example = '{"example.txt": "descriptive-stem-here"}'

        prompt = (
            "You are a file-naming assistant. For each file listed below, suggest a "
            "descriptive filename stem.\n\n"
            "Output ONLY a JSON object mapping each original filename to a suggested stem.\n"
            f"Example format: {example}\n\n"
            "Rules — read carefully:\n"
            "- Each value must be 3 to 6 lowercase words separated by hyphens "
            "(example: quarterly-budget-summary).\n"
            "- Only letters, numbers, and hyphens — no spaces, underscores, or punctuation.\n"
            "- Be specific and descriptive based on the actual file content.\n"
            "- Output ONLY the JSON object — no explanation, no preamble.\n\n"
            f"Files:\n\n{files_block}\n\n"
            "JSON:"
        )

        # Allow ~30 tokens per entry plus overhead for the JSON structure.
        max_tokens = max(100, len(files) * 30)
        raw = self._call_claude([{"type": "text", "text": prompt}], max_tokens=max_tokens)
        if not raw:
            return None

        # Attempt 1: the whole response is valid JSON.
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract the first { … } block from the response (handles
        # models that emit a brief preamble before the JSON object).
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        print(f"[AI Rename] Batch JSON parse failed. Raw: {raw[:200]!r}")
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _MAX_RETRIES = 3

    def _call_claude(
        self,
        parts: List[Dict[str, Any]],
        max_tokens: int = 50,
    ) -> str | None:
        """POST a content request to the Claude API and return the reply text.

        Retries up to 3 times on HTTP 429 (rate limit) with 2-5 s back-off.
        """
        print(f"[AI Rename] _call_claude: POST to Claude API (model={self.model}), {len(parts)} part(s), max_tokens={max_tokens}")
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "messages": [{"role": "user", "content": parts}],
        }
        request_body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._API_BASE,
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self._ANTHROPIC_VERSION,
            },
            method="POST",
        )
        for attempt in range(self._MAX_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                    raw_body = resp.read().decode("utf-8")
                print(f"[AI Rename] _call_claude: HTTP 200, response={raw_body[:300]!r}")
                data = json.loads(raw_body)
                text = data["content"][0]["text"].strip()
                print(f"[AI Rename] _call_claude: extracted text={text!r}")
                return text
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < self._MAX_RETRIES - 1:
                    exc.read()  # consume body to release connection
                    wait = 2 + attempt * 1.5  # 2 s, 3.5 s, 5 s
                    print(f"[AI Rename] Rate limited (429), retrying in {wait:.1f}s (attempt {attempt + 1}/{self._MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                error_body = exc.read().decode("utf-8", errors="replace")[:300]
                print(f"[AI Rename] Claude API error {exc.code}: {error_body}")
                return None
            except Exception as exc:
                print(f"[AI Rename] Claude call failed (model={self.model}): {exc}")
                return None
        return None


class FileContentExtractor:
    """Extract a text or image summary from any file for LLM analysis.

    Usage::

        extractor = FileContentExtractor()
        result = extractor.extract("/home/user/report.pdf")
        # {"type": "text", "content": "...", "mime": "application/pdf"}
    """

    MAX_TEXT_CHARS = 3000
    MAX_IMAGE_DIM = 256  # Optimized for PNG files - vision models work great with 256px

    # application/* MIME types whose content is human-readable text.
    _TEXT_APP_MIMES = frozenset({
        "application/json",
        "application/xml",
        "application/javascript",
        "application/x-sh",
        "application/x-python",
        "application/x-perl",
        "application/x-ruby",
        "application/x-yaml",
        "application/toml",
    })

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(self, filepath: str) -> dict:
        """Return a content dict suitable for passing to LLMClient.

        Returns:
            {
                "type":    "text" | "image" | "audio",
                "content": str   (text excerpt, or filepath for image/audio),
                "mime":    str   (detected MIME type),
            }
        """
        mime = self._detect_mime(filepath)
        category = self._categorize(mime)
        print(f"[AI Rename] extract: {filepath!r} → mime={mime!r} category={category!r}")

        if category == "image":
            # Return the filepath directly — generate_from_image() reads raw bytes
            # and sends them to Claude without any PIL preprocessing (fast path).
            print(f"[AI Rename] extract: image detected, returning filepath for direct API upload")
            return {"type": "image", "content": filepath, "mime": mime}

        elif category == "pdf":
            content = self._extract_pdf(filepath)

        elif category == "docx":
            content = self._extract_docx(filepath)

        elif category == "av":
            metadata = self._extract_av(filepath)
            if metadata:
                # Good metadata (Title/Artist/etc.) → use as text (fast path).
                content = metadata
            else:
                file_size = os.path.getsize(filepath)
                _MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20 MB
                if mime.startswith("audio/") and file_size <= _MAX_AUDIO_BYTES:
                    # No metadata + small audio → send raw bytes to Claude.
                    print(f"[AI Rename] extract: audio, no metadata, returning filepath for direct API upload")
                    return {"type": "audio", "content": filepath, "mime": mime}
                else:
                    # Video, or large audio → fallback description.
                    content = f"Audio/video file ({mime}), size: {self._human_size(filepath)}"

        elif category == "text":
            content = self._extract_text(filepath)

        else:  # unknown binary
            content = f"Binary file ({mime}), size: {self._human_size(filepath)}"

        return {"type": "text", "content": content, "mime": mime}

    # ------------------------------------------------------------------
    # MIME detection & categorisation
    # ------------------------------------------------------------------

    def _detect_mime(self, filepath: str) -> str:
        """Guess MIME type using Gio (magic bytes + filename) then stdlib fallback."""
        try:
            # Read a small header so Gio can use magic-byte sniffing.
            try:
                with open(filepath, "rb") as fh:
                    header = fh.read(512)
            except OSError:
                header = None

            mime, _ = Gio.content_type_guess(filepath, header)
            if mime:
                return mime
        except Exception:
            pass

        mime, _ = mimetypes.guess_type(filepath)
        return mime or "application/octet-stream"

    def _categorize(self, mime: str) -> str:
        if mime == "application/pdf":
            return "pdf"
        if mime in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ):
            return "docx"
        if mime.startswith("image/"):
            return "image"
        if mime.startswith(("audio/", "video/")):
            return "av"
        if mime.startswith("text/") or mime in self._TEXT_APP_MIMES:
            return "text"
        return "binary"

    # ------------------------------------------------------------------
    # Per-type extraction methods
    # ------------------------------------------------------------------

    def _extract_text(self, filepath: str) -> str:
        try:
            with open(filepath, encoding="utf-8") as fh:
                return fh.read(self.MAX_TEXT_CHARS)
        except UnicodeDecodeError:
            pass
        except Exception as exc:
            print(f"[AI Rename] Text read failed for {filepath!r}: {exc}")
            return ""
        try:
            with open(filepath, encoding="latin-1") as fh:
                return fh.read(self.MAX_TEXT_CHARS)
        except Exception as exc:
            print(f"[AI Rename] Latin-1 fallback failed for {filepath!r}: {exc}")
            return ""

    def _extract_pdf(self, filepath: str) -> str:
        # Attempt 1: PyMuPDF (fitz)
        if HAS_FITZ:
            try:
                doc = fitz.open(filepath)  # type: ignore[union-attr]
                text = doc[0].get_text() if doc.page_count > 0 else ""
                doc.close()
                if text.strip():
                    return text[: self.MAX_TEXT_CHARS]
            except Exception as exc:
                print(f"[AI Rename] PyMuPDF failed for {filepath!r}: {exc}")

        # Attempt 2: pypdf
        if HAS_PYPDF:
            try:
                reader = pypdf.PdfReader(filepath)  # type: ignore[union-attr]
                if reader.pages:
                    text = reader.pages[0].extract_text() or ""
                    if text.strip():
                        return text[: self.MAX_TEXT_CHARS]
            except Exception as exc:
                print(f"[AI Rename] pypdf failed for {filepath!r}: {exc}")

        return f"PDF file, size: {self._human_size(filepath)} (no extractable text)"

    def _extract_docx(self, filepath: str) -> str:
        WNS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                with zf.open("word/document.xml") as fh:
                    root = ET.parse(fh).getroot()
            paragraphs = []
            for para in root.iter(f"{WNS}p"):
                text = "".join(node.text or "" for node in para.iter(f"{WNS}t"))
                if text.strip():
                    paragraphs.append(text)
            return "\n".join(paragraphs)[: self.MAX_TEXT_CHARS]
        except Exception as exc:
            print(f"[AI Rename] DOCX extraction failed for {filepath!r}: {exc}")
            return f"DOCX file, size: {self._human_size(filepath)}"

    def _extract_av(self, filepath: str) -> str | None:
        """Return metadata string if useful tags exist, or None if no metadata found."""
        if HAS_TINYTAG:
            try:
                tag = tinytag.TinyTag.get(filepath)  # type: ignore[union-attr]
                fields = {
                    "Title": tag.title,
                    "Artist": tag.artist,
                    "Album": tag.album,
                    "Genre": tag.genre,
                    "Year": tag.year,
                }
                lines = [f"{k}: {v}" for k, v in fields.items() if v]
                if lines:
                    return "\n".join(lines)
            except Exception as exc:
                print(f"[AI Rename] Audio/video metadata failed for {filepath!r}: {exc}")

        return None  # No useful metadata found

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _human_size(filepath: str) -> str:
        try:
            n = os.path.getsize(filepath)
            for unit in ("B", "KB", "MB", "GB"):
                if n < 1024 or unit == "GB":
                    return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
                n /= 1024
        except OSError:
            pass
        return "unknown size"


# ---------------------------------------------------------------------------
# Filename sanitisation
# ---------------------------------------------------------------------------

MAX_FILENAME_CHARS = 60


def sanitize_filename(raw: str, original_ext: str, max_chars: int = MAX_FILENAME_CHARS) -> str:
    """Clean LLM output into a safe, filesystem-friendly filename.

    Args:
        raw: Raw string returned by the LLM (may contain quotes, extensions, …).
        original_ext: Extension of the original file, with leading dot (e.g. ".pdf").
                      Pass an empty string for extension-less files.

    Returns:
        A sanitised filename with *original_ext* appended.
    """
    # 1. Strip surrounding whitespace and any quote / backtick characters.
    name = raw.strip().strip('"`\'')

    # 2. Handle "Label: value" preamble that small models sometimes emit
    #    (e.g. "Here is a filename: my-report" → "my-report").
    #    Only trigger when ": " (colon + space) is present, so legitimate
    #    filenames containing a bare colon are unaffected.
    if ": " in name:
        name = name.rsplit(": ", 1)[-1].strip()

    # 3. Remove any extension the LLM may have appended (we re-add the
    #    original one at the end so the extension is always authoritative).
    name = os.path.splitext(name)[0]

    # 4. Replace spaces, underscores, slashes, and other word-separator
    #    characters with hyphens.
    name = re.sub(r"[ _/\\|]+", "-", name)

    # 5. Remove every character that is not alphanumeric, a hyphen, or a dot.
    name = re.sub(r"[^a-zA-Z0-9\-.]", "", name)

    # 6. Collapse runs of consecutive hyphens into one.
    name = re.sub(r"-{2,}", "-", name)

    # 7. Enforce lowercase.
    name = name.lower()

    # 8. Truncate to max_chars, then strip any trailing hyphens/dots
    #    that may land at the cut point.
    name = name[:max_chars].rstrip("-.")

    # 9. Default if the result is empty after all the above.
    if not name:
        name = "renamed-file"

    # 10. Append the original extension (normalise: ensure it starts with '.').
    if original_ext and not original_ext.startswith("."):
        original_ext = "." + original_ext

    return name + original_ext


# ---------------------------------------------------------------------------
# Undo manager
# ---------------------------------------------------------------------------

class UndoManager:
    """Persist rename history and support undoing the most recent renames.

    History is stored as a JSON array in
    ``~/.config/nautilus-ai-rename/undo_history.json``.  Each entry is a
    plain object ``{"old": <original path>, "new": <renamed path>}``.

    Usage::

        um = UndoManager()
        um.record("/home/user/IMG_001.jpg", "/home/user/sunset-beach.jpg")
        um.undo_last_batch()   # renames sunset-beach.jpg back to IMG_001.jpg
    """

    _CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "nautilus-ai-rename")
    _HISTORY_FILE = "undo_history.json"
    _MAX_HISTORY = 100

    def __init__(self) -> None:
        self._history_path = os.path.join(self._CONFIG_DIR, self._HISTORY_FILE)
        os.makedirs(self._CONFIG_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self._pending: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, old_path: str, new_path: str) -> None:
        """Queue a rename event (thread-safe). Call flush() to persist."""
        with self._lock:
            self._pending.append({"old": old_path, "new": new_path})

    def flush(self) -> None:
        """Persist all queued records to disk in one write."""
        with self._lock:
            if not self._pending:
                return
            history = self._load()
            history.extend(self._pending)
            self._pending.clear()
            self._save(history[-self._MAX_HISTORY:])

    def undo_last_batch(self, count: int = 1) -> int:
        """Undo the last *count* renames.

        Processes entries newest-first so that repeated renames of the same
        file are unwound in the correct order.

        Returns:
            Number of files successfully renamed back.
        """
        history = self._load()
        batch = history[-count:]                     # the N most recent entries
        prefix = history[: len(history) - len(batch)]  # everything before the batch

        success_count = 0
        failed: list[dict] = []

        for entry in reversed(batch):  # newest first
            try:
                gfile = Gio.File.new_for_path(entry["new"])
                gfile.set_display_name(os.path.basename(entry["old"]), None)
                success_count += 1
            except Exception as exc:
                print(f"[AI Rename] Undo failed for {entry['new']!r}: {exc}")
                failed.append(entry)

        # Keep only the entries that could not be undone.
        self._save(prefix + failed)
        return success_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> list:
        try:
            with open(self._history_path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            print("[AI Rename] Undo history has unexpected format; resetting.")
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            print(f"[AI Rename] Corrupt undo history, resetting: {exc}")
        return []

    def _save(self, history: list) -> None:
        try:
            with open(self._history_path, "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2)
        except OSError as exc:
            print(f"[AI Rename] Failed to save undo history: {exc}")


# ---------------------------------------------------------------------------
# Nautilus extension entry point
# ---------------------------------------------------------------------------

class AIRenameExtension(GObject.GObject, Nautilus.MenuProvider):
    """Nautilus MenuProvider that adds AI-powered renaming to the context menu.

    Menu structure (right-click on one or more files):

        Rename 'photo.jpg' with AI          ← single file; activates directly
        Undo last AI rename                 ← only when history exists

        Rename 3 files with AI              ← multi-file selection
        Undo last AI rename
    """

    # When at least this many text files are selected, use batch mode.
    _BATCH_THRESHOLD = 5
    # Maximum number of text files to include in a single batch LLM request.
    _BATCH_SIZE = 10

    # ------------------------------------------------------------------
    # Nautilus MenuProvider interface
    # ------------------------------------------------------------------

    def get_file_items(self, files: List) -> List:
        """Build the context-menu items for the selected *files*.

        Called by Nautilus on every right-click; must return instantly.
        Heavy work (LLM calls, file I/O) is offloaded to daemon threads.
        """
        local_files = [
            f for f in files
            if f.get_uri_scheme() == "file" and not f.is_directory()
        ]
        if not local_files:
            return []

        if len(local_files) == 1:
            label = f"Rename \u2018{local_files[0].get_name()}\u2019 with AI"
        else:
            label = f"Rename {len(local_files)} files with AI"

        rename_item = Nautilus.MenuItem(
            name="AIRename::rename",
            label=label,
            tip="Suggest a new filename using Claude AI",
        )
        rename_item.connect("activate", self._on_rename, local_files)

        items = [rename_item]

        # Undo entry — only shown when there is recorded history.
        # Use a fast existence check rather than parsing the JSON file.
        _undo_path = os.path.join(
            UndoManager._CONFIG_DIR, UndoManager._HISTORY_FILE
        )
        if os.path.isfile(_undo_path):
            undo_item = Nautilus.MenuItem(
                name="AIRename::undo",
                label="Undo last AI rename",
                tip="Reverse the most recent AI-driven rename",
            )
            undo_item.connect("activate", self._on_undo)
            items.append(undo_item)

        return items

    def get_background_items(self, _current_folder: Any) -> List:
        return []

    # ------------------------------------------------------------------
    # Menu-item callbacks (main thread — must not block)
    # ------------------------------------------------------------------

    def _on_rename(self, _: Any, files: List) -> None:
        threading.Thread(
            target=self._rename_worker,
            args=(files,),
            daemon=True,
        ).start()

    def _on_undo(self, _: Any) -> None:
        threading.Thread(
            target=self._undo_worker,
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    def _rename_worker(self, files: List) -> None:
        """Run in a daemon thread: extract → LLM → sanitize → rename.

        Text files are renamed in batch when 5+ are selected (up to 10 per
        request).  Images always go through the vision API individually.
        """
        cfg = self._load_settings()
        api_key = cfg["api-key"]

        if not api_key:
            self._notify(
                "AI Rename — API key missing",
                "Run: dconf write /com/github/ai-rename/api-key \"'your-key-here'\"",
            )
            return

        print(f"[AI Rename] Starting rename of {len(files)} file(s) via Claude")

        client = LLMClient(api_key, cfg["model"])
        extractor = FileContentExtractor()
        undo_mgr = UndoManager()
        renamed = 0
        failed = 0

        # ---- Phase 1: extract content from every file ----
        text_items: List[Dict[str, Any]] = []
        image_items: List[Dict[str, Any]] = []
        audio_items: List[Dict[str, Any]] = []

        for nfile in files:
            filepath = unquote(nfile.get_uri()[7:])   # strip "file://"
            original_name = os.path.basename(filepath)
            original_ext = os.path.splitext(original_name)[1]
            parent_dir = os.path.dirname(filepath)

            try:
                result = extractor.extract(filepath)
                item: Dict[str, Any] = {
                    "nfile": nfile,
                    "filepath": filepath,
                    "original_name": original_name,
                    "original_ext": original_ext,
                    "parent_dir": parent_dir,
                    "content": result["content"],
                }
                if result["type"] == "image":
                    image_items.append(item)
                elif result["type"] == "audio":
                    audio_items.append(item)
                else:
                    text_items.append(item)
            except Exception as exc:
                print(f"[AI Rename] Extraction failed for {original_name!r}: {exc}")
                failed += 1

        # ---- Phase 2: process individual files in parallel ----
        print(f"[AI Rename] Phase 2: {len(image_items)} image(s), {len(audio_items)} audio file(s), {len(text_items)} text file(s)")

        # Collect individual-API items: images, audio, and small text sets.
        # Text files above the batch threshold go through batch mode instead.
        futures: list[concurrent.futures.Future] = []  # type: ignore[type-arg]

        use_batch = len(text_items) >= self._BATCH_THRESHOLD

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for item in image_items:
                futures.append(executor.submit(
                    self._apply_rename, item, client, undo_mgr, cfg, is_image=True,
                ))
            for item in audio_items:
                futures.append(executor.submit(
                    self._apply_rename, item, client, undo_mgr, cfg, is_audio=True,
                ))
            if not use_batch:
                for item in text_items:
                    futures.append(executor.submit(
                        self._apply_rename, item, client, undo_mgr, cfg,
                    ))

            for future in concurrent.futures.as_completed(futures):
                r, f = future.result()
                renamed += r
                failed += f

        # ---- Phase 3: batch text files (if above threshold) ----
        if use_batch:
            chunks = [
                text_items[i : i + self._BATCH_SIZE]
                for i in range(0, len(text_items), self._BATCH_SIZE)
            ]
            for chunk in chunks:
                r, f = self._process_text_batch(chunk, client, undo_mgr, cfg)
                renamed += r
                failed += f

        # Flush all queued undo records to disk in one write.
        undo_mgr.flush()

        if failed == 0:
            self._notify(
                "AI Rename complete",
                f"Renamed {renamed} file{'s' if renamed != 1 else ''}.",
            )
        else:
            self._notify(
                "AI Rename — partial failure",
                f"Renamed {renamed}, failed to rename {failed}.",
            )

    def _process_text_batch(
        self,
        chunk: List[Dict[str, Any]],
        client: "LLMClient",
        undo_mgr: "UndoManager",
        cfg: Dict[str, Any],
    ) -> tuple[int, int]:
        """Send one batch LLM request for *chunk*; fall back to individual on failure.

        Returns:
            (renamed, failed) counts.
        """
        batch_input = [(item["original_name"], item["content"]) for item in chunk]
        mapping = client.generate_batch(batch_input)

        if mapping is None:
            # JSON parse failed or LLM returned nothing — retry each file individually.
            print(
                f"[AI Rename] Batch of {len(chunk)} failed; "
                "falling back to individual requests."
            )
            renamed, failed = 0, 0
            for item in chunk:
                r, f = self._apply_rename(item, client, undo_mgr, cfg, is_image=False)
                renamed += r
                failed += f
            return renamed, failed

        renamed, failed = 0, 0
        for item in chunk:
            raw = mapping.get(item["original_name"])
            if not raw:
                # LLM omitted this file — fall back to an individual request.
                print(
                    f"[AI Rename] Batch omitted {item['original_name']!r}; "
                    "retrying individually."
                )
                r, f = self._apply_rename(item, client, undo_mgr, cfg, is_image=False)
                renamed += r
                failed += f
                continue

            try:
                new_name = sanitize_filename(
                    raw, item["original_ext"], max_chars=cfg["max-filename-length"]
                )
                new_name = self._resolve_collision(item["parent_dir"], new_name)
                gfile = Gio.File.new_for_path(item["filepath"])
                gfile.set_display_name(new_name, None)
                undo_mgr.record(
                    item["filepath"], os.path.join(item["parent_dir"], new_name)
                )
                renamed += 1
                GLib.idle_add(item["nfile"].invalidate_extension_info)
            except Exception as exc:
                print(f"[AI Rename] Failed to rename {item['original_name']!r}: {exc}")
                failed += 1

        return renamed, failed

    def _apply_rename(
        self,
        item: Dict[str, Any],
        client: "LLMClient",
        undo_mgr: "UndoManager",
        cfg: Dict[str, Any],
        *,
        is_image: bool = False,
        is_audio: bool = False,
    ) -> tuple[int, int]:
        """Call the LLM for a single *item* and rename the file.

        Returns:
            (renamed, failed) — always sums to 1.
        """
        try:
            if is_image:
                print(f"[AI Rename] _apply_rename: image path for {item['original_name']!r}, filepath={item['content']!r}")
                raw = client.generate_from_image(item["content"], item["original_name"])
            elif is_audio:
                print(f"[AI Rename] _apply_rename: audio path for {item['original_name']!r}, filepath={item['content']!r}")
                raw = client.generate_from_audio(item["content"], item["original_name"])
            else:
                raw = client.generate_from_text(item["content"], item["original_name"])

            print(f"[AI Rename] _apply_rename: raw suggestion={raw!r}")
            if not raw:
                print(f"[AI Rename] No suggestion returned for {item['original_name']!r}")
                return 0, 1

            new_name = sanitize_filename(
                raw, item["original_ext"], max_chars=cfg["max-filename-length"]
            )
            new_name = self._resolve_collision(item["parent_dir"], new_name)
            gfile = Gio.File.new_for_path(item["filepath"])
            gfile.set_display_name(new_name, None)
            undo_mgr.record(
                item["filepath"], os.path.join(item["parent_dir"], new_name)
            )
            GLib.idle_add(item["nfile"].invalidate_extension_info)
            return 1, 0
        except Exception as exc:
            print(f"[AI Rename] Failed to rename {item['original_name']!r}: {exc}")
            return 0, 1

    def _undo_worker(self) -> None:
        """Run in a daemon thread: undo the most recent rename batch."""
        undo_mgr = UndoManager()
        count = undo_mgr.undo_last_batch(1)
        if count:
            self._notify(
                "AI Rename — undone",
                f"Reversed {count} rename{'s' if count != 1 else ''}.",
            )
        else:
            self._notify(
                "AI Rename — nothing to undo",
                "No recent rename could be reversed.",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_collision(parent_dir: str, filename: str) -> str:
        """Return *filename*, appending -1, -2 … if the path already exists."""
        candidate = filename
        stem, ext = os.path.splitext(filename)
        n = 1
        while os.path.exists(os.path.join(parent_dir, candidate)):
            candidate = f"{stem}-{n}{ext}"
            n += 1
        return candidate

    @staticmethod
    def _load_settings() -> Dict[str, Any]:
        """Read extension settings from GSettings (dconf backend)."""
        defaults: Dict[str, Any] = {
            "model": LLMClient.DEFAULT_MODEL,
            "api-key": "",  # Must be set via dconf or gsettings
            "max-filename-length": MAX_FILENAME_CHARS,
        }
        try:
            s = Gio.Settings.new("com.github.ai-rename")
            available = s.list_keys()

            return {
                "model": s.get_string("model") or defaults["model"],
                "api-key": s.get_string("api-key") if "api-key" in available else "",
                "max-filename-length": s.get_int("max-filename-length") or defaults["max-filename-length"],
            }
        except Exception as exc:
            print(f"[AI Rename] Failed to load settings: {exc}")
            return defaults

    @staticmethod
    def _notify(title: str, body: str) -> None:
        try:
            subprocess.Popen(["notify-send", title, body])
        except Exception as exc:
            print(f"[AI Rename] notify-send failed: {exc}")