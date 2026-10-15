#!/bin/sh
# Steam launch wrapper for macOS. Steam launch options: <path to this file> %command%
# Steam on macOS doesn't accept VAR=value in launch options (OS Error 260), so the variables are set here,
# for the game process only.
#
# Layouts:
#   installed (macos/install.sh): <dir>/launch.sh, <dir>/bin/VkLayer_tf3_dlss.dylib, <dir>/tf3_dlss.cfg
#   development (source tree):    macos/launch.sh, build-macos/bin/..., build-macos/tf3_dlss.cfg
HERE="$(dirname "$(realpath "$0")")"  # realpath: the script may be started through a symlink
if [ -n "$TF3_METALFX_LAYER_DIR" ]; then
  LAYER_DIR="$TF3_METALFX_LAYER_DIR"
  ENV_FILE="$LAYER_DIR/../launch.env"
elif [ -f "$HERE/bin/VkLayer_tf3_dlss.dylib" ]; then
  LAYER_DIR="$HERE/bin"
  ENV_FILE="$HERE/launch.env"
else
  LAYER_DIR="$HERE/../build-macos/bin"
  ENV_FILE="$HERE/../build-macos/launch.env"
fi
export VK_ADD_LAYER_PATH="$(cd "$LAYER_DIR" && pwd)"
export VK_INSTANCE_LAYERS=VK_LAYER_TF3_dlss
# Optional extra variables for development runs (TF3_DLSS_TEST_UPSCALER, TF3_DLSS_CAPTURE, ...),
# one VAR=value per line; delete the file to go back to a normal run.
if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
fi
echo "$(date '+%F %T') launch: $*" >> "$LAYER_DIR/../launch.log" 2>/dev/null
exec "$@"
