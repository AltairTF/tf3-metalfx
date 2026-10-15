#!/bin/sh
# Installs the MetalFX mod for Transport Fever 3 (macOS) into ~/.local/share/tf3-metalfx.
#
#   macos/install.sh              install (from a release folder, or builds from source first)
#   macos/install.sh --uninstall  remove it again (your tf3_dlss.cfg is kept as a backup)
#
# Nothing in the game folder is changed. The mod is switched on by the Steam launch option printed at
# the end, and off again by clearing that launch option.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/.local/share/tf3-metalfx"

if [ "$1" = "--uninstall" ]; then
  [ -f "$DEST/tf3_dlss.cfg" ] && cp "$DEST/tf3_dlss.cfg" "$HOME/tf3_dlss.cfg.backup" && echo "Settings saved to ~/tf3_dlss.cfg.backup"
  rm -rf "$DEST"
  echo "Removed $DEST. Also clear the launch option of Transport Fever 3 in Steam."
  exit 0
fi

# Release folder: bin/ next to this script. Source tree: build-macos/bin (built if missing).
if [ -f "$HERE/bin/VkLayer_tf3_dlss.dylib" ]; then
  SRC_BIN="$HERE/bin"; SRC_CFG="$HERE/tf3_dlss.cfg"
else
  ROOT="$(cd "$HERE/.." && pwd)"
  [ -f "$ROOT/build-macos/bin/VkLayer_tf3_dlss.dylib" ] || "$HERE/build.sh"
  SRC_BIN="$ROOT/build-macos/bin"; SRC_CFG="$HERE/tf3_dlss.cfg"
fi

mkdir -p "$DEST/bin"
cp "$SRC_BIN/VkLayer_tf3_dlss.dylib" "$SRC_BIN/VkLayer_tf3_dlss.json" "$DEST/bin/"
cp "$HERE/launch.sh" "$DEST/launch.sh"
chmod +x "$DEST/launch.sh"
if [ -f "$DEST/tf3_dlss.cfg" ]; then
  echo "Keeping your existing settings ($DEST/tf3_dlss.cfg)."
else
  cp "$SRC_CFG" "$DEST/tf3_dlss.cfg"
fi
# Files downloaded from the internet are quarantined; the game couldn't load the library then.
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true

echo
echo "Installed in $DEST"
echo
echo "Last step, in Steam: right-click Transport Fever 3 > Properties > General > Launch options, paste:"
echo
echo "    $DEST/launch.sh %command%"
echo
echo "Then in the game's graphics options: Resolution scale 50-67%, MSAA off."
