# Test harness: fake_tf3, a small Vulkan program that renders frames shaped like Transport Fever 3's
# (scene -> tonemap -> FSR 1 EASU -> RCAS -> blit -> UI), so the layer can be tested without the game.
#
# Windows: builds fake_tf3 only; scripts\run_harness.ps1 runs it with the layer from build\bin.
# Elsewhere (Linux, CI): also builds the layer itself as a shared library, with NGX stubbed out (no
# DLSS there); harness/run.sh runs it with the test upscaler on a CPU Vulkan driver (lavapipe).
cmake_minimum_required(VERSION 3.20)
project(tf3_dlss_harness CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_program(GLSLANG glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" REQUIRED)

set(GEN_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GEN_DIR}")

# GLSL -> SPIR-V words in a .spv.inc file: NAME.comp -> NAME.spv.inc, NAME.vert -> NAME_vert.spv.inc
function(compile_shaders OUT_VAR DIR)
  set(outs "")
  foreach(SRC ${ARGN})
    get_filename_component(NAME "${SRC}" NAME_WE)
    get_filename_component(EXT "${SRC}" LAST_EXT)
    if(EXT STREQUAL ".comp")
      set(OUT "${GEN_DIR}/${NAME}.spv.inc")
    else()
      string(SUBSTRING "${EXT}" 1 -1 STAGE)
      set(OUT "${GEN_DIR}/${NAME}_${STAGE}.spv.inc")
    endif()
    add_custom_command(OUTPUT "${OUT}"
      COMMAND "${GLSLANG}" --target-env vulkan1.1 -V -x -o "${OUT}" "${DIR}/${SRC}"
      DEPENDS "${DIR}/${SRC}" COMMENT "Compiling ${SRC}")
    list(APPEND outs "${OUT}")
  endforeach()
  set(${OUT_VAR} ${outs} PARENT_SCOPE)
endfunction()

compile_shaders(FAKE_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  scene.vert scene.frag fullscreen.vert tonemap.frag easu.frag rcas.frag blit.frag ui.frag smoke.frag)
add_executable(fake_tf3 fake_tf3.cpp ${FAKE_SHADERS})
target_include_directories(fake_tf3 PRIVATE "${GEN_DIR}")
find_package(Threads REQUIRED)
target_link_libraries(fake_tf3 PRIVATE Vulkan::Vulkan Threads::Threads)
if(WIN32)
  target_compile_definitions(fake_tf3 PRIVATE VK_USE_PLATFORM_WIN32_KHR NOMINMAX WIN32_LEAN_AND_MEAN
                                              _CRT_SECURE_NO_WARNINGS)
endif()

if(NOT WIN32)
  set(DLSS_SDK "$ENV{HOME}/DLSS" CACHE PATH "NVIDIA DLSS SDK folder (only its headers are used here)")
  set(LAYER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../layer")
  compile_shaders(LAYER_SHADERS "${LAYER_DIR}/shaders"
    camera_mv.comp mv_debug.comp sharpen.comp overlay.comp test_upscale.comp object_mv.vert object_mv.frag reactive_mask.comp
    reactive_const.frag)
  add_custom_command(OUTPUT "${GEN_DIR}/camera_mv_ms.spv.inc"
    COMMAND "${GLSLANG}" --target-env vulkan1.1 -V -x -DDEPTH_MS -o "${GEN_DIR}/camera_mv_ms.spv.inc" "${LAYER_DIR}/shaders/camera_mv.comp"
    DEPENDS "${LAYER_DIR}/shaders/camera_mv.comp" COMMENT "Compiling camera_mv.comp (DEPTH_MS)")
  list(APPEND LAYER_SHADERS "${GEN_DIR}/camera_mv_ms.spv.inc")
  set(LAYER_EXTRA "")
  if(APPLE)  # MetalFX bridge (macOS build of the layer)
    enable_language(OBJCXX)
    set(LAYER_EXTRA "${LAYER_DIR}/../macos/metalfx_bridge.mm")
    set_source_files_properties("${LAYER_EXTRA}" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  endif()
  add_library(VkLayer_tf3_dlss SHARED "${LAYER_DIR}/src/layer.cpp" ngx_stub.cpp ${LAYER_EXTRA} ${LAYER_SHADERS})
  if(APPLE)
    target_include_directories(VkLayer_tf3_dlss PRIVATE "${LAYER_DIR}/../macos")
    target_compile_definitions(VkLayer_tf3_dlss PRIVATE VK_USE_PLATFORM_METAL_EXT)
    target_link_libraries(VkLayer_tf3_dlss PRIVATE "-framework Metal" "-framework MetalFX" "-framework Foundation"
                          "-framework AppKit" "-framework CoreGraphics" "-framework IOKit")
  endif()
  target_include_directories(VkLayer_tf3_dlss PRIVATE "${DLSS_SDK}/include" "${GEN_DIR}"
                             "${LAYER_DIR}/../third_party/ffx-api/include")
  target_link_libraries(VkLayer_tf3_dlss PRIVATE Vulkan::Headers)
  # the FidelityFX header marks its (unused here) exports with __declspec
  target_compile_options(VkLayer_tf3_dlss PRIVATE "-D__declspec(x)=")
  if(NOT APPLE)
    target_link_options(VkLayer_tf3_dlss PRIVATE -Wl,--no-undefined)
  endif()
  set_target_properties(VkLayer_tf3_dlss PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/layer")
  if(APPLE)  # the manifest names the Linux library; on macOS the layer is a .dylib
    set(LAYER_JSON_SRC "${CMAKE_CURRENT_SOURCE_DIR}/VkLayer_tf3_dlss_linux.json")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LAYER_JSON_SRC}")
    file(READ "${LAYER_JSON_SRC}" LAYER_JSON)
    string(REPLACE "VkLayer_tf3_dlss.so" "VkLayer_tf3_dlss.dylib" LAYER_JSON "${LAYER_JSON}")
    file(WRITE "${CMAKE_BINARY_DIR}/layer/VkLayer_tf3_dlss.json" "${LAYER_JSON}")
  else()
    configure_file(VkLayer_tf3_dlss_linux.json "${CMAKE_BINARY_DIR}/layer/VkLayer_tf3_dlss.json" COPYONLY)
  endif()
endif()
