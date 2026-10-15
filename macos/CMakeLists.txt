# macOS (Apple Silicon) build of the layer: arm64 dylib, NGX stubbed (no DLSS on the Mac),
# FSR 3.1 runtime not loaded. The upscaler on macOS is MetalFX (docs/MACOS_PORT.md, step 3).
#
#   cmake -S macos -B build-macos -G Ninja -DDLSS_SDK=<folder with include/nvsdk_ngx_vk.h>
#   cmake --build build-macos           -> build-macos/bin: VkLayer_tf3_dlss.dylib + manifest
cmake_minimum_required(VERSION 3.20)
project(tf3_metalfx_layer CXX OBJCXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0)

find_program(GLSLANG glslangValidator REQUIRED)
find_path(VULKAN_INCLUDE vulkan/vulkan.h HINTS /opt/homebrew/include "$ENV{VULKAN_SDK}/include" REQUIRED)
set(DLSS_SDK "" CACHE PATH "Folder with include/nvsdk_ngx_vk.h (headers only, for the NGX stub)")
if(NOT EXISTS "${DLSS_SDK}/include/nvsdk_ngx_vk.h")
  message(FATAL_ERROR "Set -DDLSS_SDK to a folder containing include/nvsdk_ngx_vk.h")
endif()

set(LAYER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../layer")
set(GEN_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GEN_DIR}")
set(SHADER_INCS "")
foreach(SRC camera_mv.comp mv_debug.comp sharpen.comp overlay.comp test_upscale.comp object_mv.vert object_mv.frag)
  get_filename_component(NAME "${SRC}" NAME_WE)
  get_filename_component(EXT "${SRC}" LAST_EXT)
  if(EXT STREQUAL ".comp")
    set(OUT "${GEN_DIR}/${NAME}.spv.inc")
  else()
    string(SUBSTRING "${EXT}" 1 -1 STAGE)
    set(OUT "${GEN_DIR}/${NAME}_${STAGE}.spv.inc")
  endif()
  add_custom_command(OUTPUT "${OUT}"
    COMMAND "${GLSLANG}" --target-env vulkan1.1 -V -x -o "${OUT}" "${LAYER_DIR}/shaders/${SRC}"
    DEPENDS "${LAYER_DIR}/shaders/${SRC}" COMMENT "Compiling ${SRC}")
  list(APPEND SHADER_INCS "${OUT}")
endforeach()

# camera_mv.comp again with DEPTH_MS: the multisampled-depth variant (MSAA on)
add_custom_command(OUTPUT "${GEN_DIR}/camera_mv_ms.spv.inc"
  COMMAND "${GLSLANG}" --target-env vulkan1.1 -V -x -DDEPTH_MS -o "${GEN_DIR}/camera_mv_ms.spv.inc" "${LAYER_DIR}/shaders/camera_mv.comp"
  DEPENDS "${LAYER_DIR}/shaders/camera_mv.comp" COMMENT "Compiling camera_mv.comp (DEPTH_MS)")
list(APPEND SHADER_INCS "${GEN_DIR}/camera_mv_ms.spv.inc")

add_library(VkLayer_tf3_dlss SHARED "${LAYER_DIR}/src/layer.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/../harness/ngx_stub.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/metalfx_bridge.mm" ${SHADER_INCS})
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/metalfx_bridge.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
target_link_libraries(VkLayer_tf3_dlss PRIVATE "-framework Metal" "-framework MetalFX" "-framework Foundation"
                      "-framework AppKit" "-framework CoreGraphics")
target_include_directories(VkLayer_tf3_dlss PRIVATE "${VULKAN_INCLUDE}" "${DLSS_SDK}/include" "${GEN_DIR}"
                           "${CMAKE_CURRENT_SOURCE_DIR}"
                           "${LAYER_DIR}/../third_party/ffx-api/include")
target_compile_definitions(VkLayer_tf3_dlss PRIVATE VK_USE_PLATFORM_METAL_EXT VK_ENABLE_BETA_EXTENSIONS)
# the FidelityFX header marks its (unused here) exports with __declspec
target_compile_options(VkLayer_tf3_dlss PRIVATE "-D__declspec(x)=" -fvisibility=hidden -Wno-unused-parameter)
set_target_properties(VkLayer_tf3_dlss PROPERTIES PREFIX "" SUFFIX ".dylib"
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/VkLayer_tf3_dlss.json" "${CMAKE_BINARY_DIR}/bin/VkLayer_tf3_dlss.json" COPYONLY)
