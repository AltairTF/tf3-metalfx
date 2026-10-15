cmake_minimum_required(VERSION 3.20)
project(tf3_dlss_layer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT so the DLL has no runtime dependencies of its own (matches nvsdk_ngx_s.lib)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(DEFINED ENV{VULKAN_SDK})
  set(VULKAN_SDK "$ENV{VULKAN_SDK}")
else()
  set(VULKAN_SDK "C:/VulkanSDK/1.4.363.0")
endif()
set(DLSS_SDK "$ENV{USERPROFILE}/Projects/DLSS" CACHE PATH "NVIDIA DLSS SDK folder")
# FSR 3.1: AMD FidelityFX API headers are in the repo; the signed runtime DLL is not (download it
# with scripts/get_fsr.ps1, from the FidelityFX SDK v1.1.4 PrebuiltSignedDLL folder).
set(FFX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ffx-api")
set(FFX_DLL "${FFX_DIR}/bin/amd_fidelityfx_vk.dll" CACHE FILEPATH "amd_fidelityfx_vk.dll (FidelityFX SDK v1.1.4)")
message(STATUS "Vulkan SDK: ${VULKAN_SDK}")
message(STATUS "DLSS SDK:   ${DLSS_SDK}")
if(NOT EXISTS "${DLSS_SDK}/include/nvsdk_ngx_vk.h")
  message(FATAL_ERROR "DLSS SDK not found at ${DLSS_SDK}")
endif()

# Compile our shaders to SPIR-V words that layer.cpp #includes.
# camera_mv.comp -> camera_mv.spv.inc, object_mv.vert -> object_mv_vert.spv.inc, ...
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
  add_custom_command(
    OUTPUT "${OUT}"
    COMMAND "${VULKAN_SDK}/Bin/glslangValidator.exe" --target-env vulkan1.1 -V -x
            -o "${OUT}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SRC}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SRC}"
    COMMENT "Compiling ${SRC}")
  list(APPEND SHADER_INCS "${OUT}")
endforeach()

# camera_mv.comp again with DEPTH_MS: the multisampled-depth variant (MSAA on)
add_custom_command(
  OUTPUT "${GEN_DIR}/camera_mv_ms.spv.inc"
  COMMAND "${VULKAN_SDK}/Bin/glslangValidator.exe" --target-env vulkan1.1 -V -x -DDEPTH_MS
          -o "${GEN_DIR}/camera_mv_ms.spv.inc" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/camera_mv.comp"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/camera_mv.comp"
  COMMENT "Compiling camera_mv.comp (DEPTH_MS)")
list(APPEND SHADER_INCS "${GEN_DIR}/camera_mv_ms.spv.inc")

add_library(VkLayer_tf3_dlss SHARED src/layer.cpp ${SHADER_INCS})
target_include_directories(VkLayer_tf3_dlss PRIVATE "${VULKAN_SDK}/Include" "${DLSS_SDK}/include" "${GEN_DIR}"
                           "${FFX_DIR}/include")
target_compile_definitions(VkLayer_tf3_dlss PRIVATE
  VK_USE_PLATFORM_WIN32_KHR NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
target_link_libraries(VkLayer_tf3_dlss PRIVATE
  "${DLSS_SDK}/lib/Windows_x86_64/x64/nvsdk_ngx_s$<$<CONFIG:Debug>:_dbg>.lib"
  advapi32 version)
if(MSVC)
  target_compile_options(VkLayer_tf3_dlss PRIVATE /W3 /permissive-)
endif()

set_target_properties(VkLayer_tf3_dlss PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# The layer manifest and the DLSS / FSR runtimes sit next to the layer DLL
configure_file(VkLayer_tf3_dlss.json "${CMAKE_BINARY_DIR}/bin/VkLayer_tf3_dlss.json" COPYONLY)
if(EXISTS "${FFX_DLL}")
  add_custom_command(TARGET VkLayer_tf3_dlss POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FFX_DLL}" "${CMAKE_BINARY_DIR}/bin/amd_fidelityfx_vk.dll")
else()
  message(WARNING "amd_fidelityfx_vk.dll not found at ${FFX_DLL}: FSR won't be available (run scripts/get_fsr.ps1)")
endif()
add_custom_command(TARGET VkLayer_tf3_dlss POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${DLSS_SDK}/lib/Windows_x86_64/rel/nvngx_dlss.dll" "${CMAKE_BINARY_DIR}/bin/nvngx_dlss.dll"
  # developer build (on-screen debug overlays and hotkeys), used when dlss_dev=1
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin/dev"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${DLSS_SDK}/lib/Windows_x86_64/dev/nvngx_dlss.dll" "${CMAKE_BINARY_DIR}/bin/dev/nvngx_dlss.dll")

# version.dll stand-in that goes next to TransportFever3.exe and enables the layer (drop-in install).
# Built into build\proxy, not build\bin, so it never sits next to the layer during development.
add_library(tf3_dlss_proxy SHARED proxy/version_proxy.cpp proxy/version.def)
target_compile_definitions(tf3_dlss_proxy PRIVATE UNICODE _UNICODE WIN32_LEAN_AND_MEAN)
set_target_properties(tf3_dlss_proxy PROPERTIES
  OUTPUT_NAME version
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/proxy"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/proxy"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/proxy")
