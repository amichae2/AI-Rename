# AI Rename — Nautilus Extension

A GNOME Files (Nautilus) extension that uses the **Anthropic Claude API** to generate
intelligent, content-aware filenames. Right-click any file or selection of files and
choose **Rename '…' with AI** to get a descriptive name based on the actual content.

## Features

- **Content-aware renaming** — analyses text, images, PDFs, DOCX, audio, and video
  to generate meaningful filenames
- **Multi-format support** — text files, images (JPEG/PNG/WebP/GIF), PDFs, DOCX,
  audio (MP3/FLAC/OGG/WAV), and video (metadata-based)
- **Batch processing** — automatically batches 5+ text files into a single API call
  for efficiency
- **Image vision** — resizes large images before sending to Claude's vision API for
  analysis
- **Audio analysis** — extracts metadata (title/artist/album) when available, or sends
  raw audio (<20 MB) directly to Claude for content-based naming
- **Undo support** — every rename is logged; use the **Undo last AI rename** context
  menu item to reverse it
- **Filename sanitisation** — outputs lowercase, hyphen-separated names (3–6 words,
  max 60 characters by default, configurable)
- **Collision handling** — appends `-1`, `-2`, etc. when a filename already exists
- **Desktop notifications** — sends a notification when renaming completes

## How It Works

1. Detects the file type using Gio magic-byte sniffing and MIME detection.
2. Extracts content — a text excerpt, image bytes, audio data, PDF text, or DOCX
   paragraphs.
3. Sends a prompt to the **Anthropic Claude API** asking for a concise, descriptive
   filename stem.
4. Sanitises the response (lowercase, hyphens only, configurable max length).
5. Renames the file via Gio so Nautilus refreshes without a manual reload.
6. Logs the rename to `~/.config/nautilus-ai-rename/undo_history.json`.
7. Sends a desktop notification with the result.

## Prerequisites

### Anthropic API Key

This extension requires an [Anthropic](https://www.anthropic.com/) API key.

1. Create an account at [console.anthropic.com](https://console.anthropic.com/).
2. Navigate to **API Keys** and generate a new key.
3. Set the environment variable so Nautilus can access it:

```bash
# Add to ~/.bashrc or ~/.profile so it persists across sessions:
export ANTHROPIC_API_KEY="sk-ant-..."
```

> **Note:** You can also store the key via GSettings (see [Configuration](#configuration)),
> but the environment variable is recommended for security.

### System Packages

```bash
sudo apt install python3-nautilus python3-pil
```

| Package | Purpose |
|---|---|
| `python3-nautilus` | Nautilus Python extension bindings (GObject introspection) |
| `python3-pil` | Image resizing before sending to the vision API (Pillow) |

### Optional Dependencies

**Better PDF text extraction:**

```bash
# Option A — PyMuPDF (faster, extracts text + structure)
pip install pymupdf

# Option B — pypdf (pure Python, no native dependencies)
pip install pypdf
```

Without either, PDF files fall back to filename-only context.

**Audio/video metadata extraction:**

```bash
pip install tinytag
```

With tinytag, the extension extracts ID3/Vorbis/FLAC tags (title, artist, album, genre,
year) and uses them as context. Without it, small audio files (<20 MB) are sent directly
to Claude for audio analysis; larger files and video use filename-only context.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ai-rename.git
cd ai-rename
./install.sh
```

The script will:

- Symlink `src/ai_rename_extension.py` into
  `~/.local/share/nautilus-python/extensions/` (absolute path via `realpath`).
- Copy the GSettings schema to `~/.local/share/glib-2.0/schemas/` and compile it.
- Create `~/.config/nautilus-ai-rename/` for undo history.
- Restart Nautilus (`nautilus -q`).

The script is idempotent — safe to run again after pulling updates.

## Uninstallation

```bash
./uninstall.sh
```

Removes the symlink and schema, recompiles schemas, and restarts Nautilus.
Your config directory is preserved:

```bash
rm -rf ~/.config/nautilus-ai-rename   # remove manually if desired
```

## Configuration

Settings live in GSettings under the schema `com.github.ai-rename`.

```bash
# View current settings
gsettings list-recursively com.github.ai-rename

# Change the model
gsettings set com.github.ai-rename model 'claude-sonnet-4-5-20250929'

# Store the API key in GSettings (instead of the environment variable)
gsettings set com.github.ai-rename api-key 'sk-ant-...'

# Shorten generated filenames
gsettings set com.github.ai-rename max-filename-length 40
```

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | string | `claude-haiku-4-5-20251001` | Claude model identifier |
| `api-key` | string | *(empty)* | Anthropic API key (falls back to `ANTHROPIC_API_KEY` env var) |
| `max-filename-length` | integer | `60` | Max characters in the filename stem (extension not counted) |

### Available Models

| Model | ID | Notes |
|---|---|---|
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | Fast and cost-effective (default) |
| Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | Balanced speed and quality |
| Claude Opus 4.5 | `claude-opus-4-5-20250520` | Highest quality |

The default (Haiku 4.5) is recommended — it's fast, inexpensive, and more than
capable of generating good filenames.

## Troubleshooting

**Menu item does not appear:**

```bash
# Run Nautilus in the foreground to see Python errors
nautilus --no-desktop 2>&1 | grep -i 'ai.rename\|python\|error'

# Verify the extension symlink exists
ls -la ~/.local/share/nautilus-python/extensions/ai_rename_extension.py
```

**Rename fails silently:**

```bash
# Watch for [AI Rename] log lines
journalctl --user -f | grep 'AI Rename'
```

**API key not found:**

- Ensure `ANTHROPIC_API_KEY` is exported in the shell environment that Nautilus
  inherits. If you launch Nautilus from the GNOME desktop, add the export to
  `~/.profile` (not just `~/.bashrc`) and log out/in.
- Alternatively, store the key via GSettings:
  ```bash
  gsettings set com.github.ai-rename api-key 'sk-ant-...'
  ```

**Rate limiting (HTTP 429):**

The extension automatically retries up to 3 times with increasing back-off
(2 s, 3.5 s, 5 s). If you hit persistent rate limits, consider upgrading your
Anthropic plan or reducing batch sizes.

**Missing Python dependencies:**

```bash
# Check that nautilus-python bindings are installed
python3 -c "import gi; gi.require_version('Nautilus', '4.0')"

# Check for Pillow
python3 -c "from PIL import Image; print('OK')"
```

## File Structure

```
ai-rename/
├── src/
│   └── ai_rename_extension.py   # Main extension code
├── schemas/
│   └── com.github.ai-rename.gschema.xml   # GSettings schema
├── install.sh       # Installation script (creates symlink + compiles schema)
├── uninstall.sh     # Uninstallation script
├── LICENSE          # MIT License
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
