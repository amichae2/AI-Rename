#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXT_SRC="$(realpath "$SCRIPT_DIR/src/ai_rename_extension.py")"
EXT_DIR="$HOME/.local/share/nautilus-python/extensions"
EXT_LINK="$EXT_DIR/ai_rename_extension.py"

SCHEMA_SRC="$SCRIPT_DIR/schemas/com.github.ai-rename.gschema.xml"
SCHEMA_DIR="$HOME/.local/share/glib-2.0/schemas"
SCHEMA_DEST="$SCHEMA_DIR/com.github.ai-rename.gschema.xml"

CONFIG_DIR="$HOME/.config/nautilus-ai-rename"

echo "Installing AI Rename Nautilus extension..."

# 1. Create required directories
mkdir -p "$EXT_DIR"
mkdir -p "$SCHEMA_DIR"
mkdir -p "$CONFIG_DIR"

# 2. Symlink the extension (absolute path; -f replaces any existing file/symlink)
echo "  Symlinking: $EXT_LINK -> $EXT_SRC"
ln -sf "$EXT_SRC" "$EXT_LINK"

# 3. Copy the GSettings schema
echo "  Copying schema: $SCHEMA_DEST"
cp "$SCHEMA_SRC" "$SCHEMA_DEST"

# 4. Compile schemas
echo "  Compiling GSettings schemas..."
glib-compile-schemas "$SCHEMA_DIR"

# 5. Restart Nautilus so it picks up the new extension
echo "  Restarting Nautilus..."
nautilus -q || true   # -q sends SIGTERM to running instances; ignore error if none running

echo ""
echo "Installation complete."
echo "  Extension: $EXT_LINK"
echo "  Schema:    $SCHEMA_DEST"
echo "  Config:    $CONFIG_DIR"
echo ""
echo "Right-click any file in Nautilus and choose 'AI Rename...' to get started."
