#!/bin/sh
# Makes a release zip for other players: dist-macos/TF3-MetalFX-macOS-<version>.zip
# (layer + manifest, settings, launch wrapper, installer, README and licences). No game files.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Always rebuild, so the zip never ships a stale layer.
if [ -f build-macos/build.ninja ]; then cmake --build build-macos; else macos/build.sh; fi
VER="$(grep -m1 'define TF3_DLSS_VERSION' layer/src/layer.cpp | sed "s/.*\"\(.*\)\".*/\1/")"
OUT="dist-macos/TF3-MetalFX"
rm -rf dist-macos && mkdir -p "$OUT/bin" "$OUT/licenses"
cp build-macos/bin/VkLayer_tf3_dlss.dylib build-macos/bin/VkLayer_tf3_dlss.json "$OUT/bin/"
cp macos/tf3_dlss.cfg macos/launch.sh macos/install.sh "$OUT/"
cp docs/TUTORIAL_MACOS.md "$OUT/LEIA-ME.md"
cp LICENSE "$OUT/licenses/TF3-DLSS-LICENSE.txt"
cp package/licenses/*.txt "$OUT/licenses/"  # incl. NVIDIA SDK licence: NGX helper headers are compiled in
codesign --force --sign - "$OUT/bin/VkLayer_tf3_dlss.dylib"  # ad-hoc signature (required on Apple Silicon)
(cd dist-macos && zip -qr "TF3-MetalFX-macOS-$VER.zip" TF3-MetalFX)
echo "dist-macos/TF3-MetalFX-macOS-$VER.zip"
