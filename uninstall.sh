#!/usr/bin/env bash
set -euo pipefail

EXT_LINK="$HOME/.local/share/nautilus-python/extensions/ai_rename_extension.py"
SCHEMA_DEST="$HOME/.local/share/glib-2.0/schemas/com.github.ai-rename.gschema.xml"
SCHEMA_DIR="$HOME/.local/share/glib-2.0/schemas"

echo "Uninstalling AI-Rename Nautilus extension..."

# 1. Remove the extension symlink
if [ -L "$EXT_LINK" ]; then
    echo "  Removing symlink: $EXT_LINK"
    rm "$EXT_LINK"
elif [ -f "$EXT_LINK" ]; then
    echo "  Removing file: $EXT_LINK"
    rm "$EXT_LINK"
else
    echo "  Extension not found, skipping: $EXT_LINK"
fi

# 2. Remove the GSettings schema
if [ -f "$SCHEMA_DEST" ]; then
    echo "  Removing schema: $SCHEMA_DEST"
    rm "$SCHEMA_DEST"
else
    echo "  Schema not found, skipping: $SCHEMA_DEST"
fi

# 3. Recompile schemas so the removed schema is no longer registered
echo "  Recompiling GSettings schemas..."
glib-compile-schemas "$SCHEMA_DIR"

# 4. Restart Nautilus
echo "  Restarting Nautilus..."
nautilus -q || true

echo ""
echo "Uninstallation complete."
echo "  Config directory preserved at: $HOME/.config/nautilus-ai-rename"
echo "  Remove it manually if you no longer need it:"
echo "    rm -rf $HOME/.config/nautilus-ai-rename"
