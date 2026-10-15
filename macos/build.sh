#!/bin/sh
# Builds the macOS layer (Apple Silicon): build-macos/bin/VkLayer_tf3_dlss.dylib + manifest.
#
#   macos/build.sh            build (downloads what is missing the first time)
#   macos/build.sh --harness  also build the test harness (fake_tf3) in build-harness/
#
# Needs: Xcode Command Line Tools (clang, the Metal/MetalFX SDK) and Homebrew. Installs with Homebrew,
# if missing: cmake, ninja, glslang, vulkan-headers. The NVIDIA NGX headers (only used to stub DLSS out
# on macOS) are fetched from github.com/NVIDIA/DLSS into third_party/nvsdk (not committed).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

need() { command -v "$1" >/dev/null 2>&1; }
if ! need brew; then echo "Homebrew is needed: https://brew.sh"; exit 1; fi
for pkg in cmake ninja glslang vulkan-headers; do
  brew list "$pkg" >/dev/null 2>&1 || brew install "$pkg"
done

NV="$ROOT/third_party/nvsdk"
if [ ! -f "$NV/include/nvsdk_ngx_vk.h" ]; then
  echo "Fetching the NVIDIA NGX headers (build-time only)..."
  rm -rf "$NV"
  git clone --depth 1 --filter=blob:none --sparse https://github.com/NVIDIA/DLSS.git "$NV"
  git -C "$NV" sparse-checkout set include
fi

cmake -S macos -B build-macos -G Ninja -DCMAKE_BUILD_TYPE=Release -DDLSS_SDK="$NV"
cmake --build build-macos

if [ "$1" = "--harness" ]; then
  GAME="${TF3_GAME_DIR:-$HOME/Library/Application Support/Steam/steamapps/common/Transport Fever 3}"
  if [ ! -f "$GAME/libvulkan.1.dylib" ]; then
    echo "Set TF3_GAME_DIR to the game folder (it provides the Vulkan loader and MoltenVK)."; exit 1
  fi
  cmake -S harness -B build-harness -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DVulkan_INCLUDE_DIR="$(brew --prefix)/include" -DVulkan_LIBRARY="$GAME/libvulkan.1.dylib" -DDLSS_SDK="$NV"
  cmake --build build-harness
  cp macos/VkLayer_tf3_dlss.json build-harness/layer/
fi
echo "Built: $ROOT/build-macos/bin/VkLayer_tf3_dlss.dylib"
