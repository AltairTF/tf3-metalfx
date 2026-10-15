# macOS port: MetalFX Temporal for Transport Fever 3

Fork of tf3-dlss (Windows: DLSS / FSR 3.1) for macOS on Apple Silicon. DLSS and AMD's signed FSR DLL
only exist for Windows, so the upscaler on the Mac is Apple's **MetalFX Temporal**. MetalFX takes
the same inputs DLSS takes (color, depth, motion vectors, jitter), and this layer already produces
them.

## Target setup (tested)
- Transport Fever 3 (Steam 3493540), native arm64 Mach-O. The binary is ad-hoc signed with no
  hardened runtime.
- The game uses Vulkan through its bundled loader (`libvulkan.1.4.335.dylib`) on MoltenVK 1.4.2
  (`libMoltenVK.dylib`, MoltenVK_icd.json). KosmicKrisp is also shipped but the game doesn't select it.
- Apple M4, macOS 27, 5120x2880 display, game fullscreen at 2560x1440.

## How the layer gets loaded on macOS
On Windows a `version.dll` stand-in enables the layer. macOS needs no proxy, because the Vulkan loader
reads environment variables, and Steam can set them per game:

```
VK_ADD_LAYER_PATH="<game>/tf3-metalfx/build-macos/bin" VK_INSTANCE_LAYERS=VK_LAYER_TF3_dlss %command%
```
(Steam > Transport Fever 3 > Properties > Launch options.) Clearing the field turns the mod off.

## Steps

### Step 1: layer on macOS, no upscaler
Goal: the layer builds as an arm64 dylib, the game loads it, and the game looks and plays exactly as
before.
- Build with CMake + Ninja + Apple clang (`macos/CMakeLists.txt`). NGX is stubbed like the Linux
  harness build. FSR is not loaded.
- Port the Windows-only parts: paths, file-change polling, `__rdtsc`, `.dll` loading, `%TEMP%`.
- Enable the layer through Steam launch options. Settings: `enabled=0` (tracking only).
- Done when: `tf3_dlss.log` shows the layer started, the game's tonemap / EASU / RCAS / blit passes were
  found by SPIR-V debug name, frames are counted, and the game runs with no crash and no visual change.

### Step 2: jitter and motion vectors
Goal: the inputs MetalFX needs, produced in the real game.
- Turn on Halton jitter and camera + object motion vectors with the test upscaler or `mv_debug`
  (the layer's own debug views), still on MoltenVK.
- Check the MoltenVK specifics: compute pipelines, depth image reads (D32/D24S8 layouts), storage image
  formats, and that the game's MSAA depth resolve gives a usable depth.
- Done when: the motion-vector debug view moves with the camera and with vehicles. With jitter on and
  the test upscaler, the image is stable (no swimming) when the camera is still.

### Step 3: MetalFX Temporal
Goal: replace the upscaler call with `MTLFXTemporalScaler`.
- Use `VK_EXT_metal_objects` (MoltenVK) to get the `MTLDevice`, the `MTLCommandQueue` and an
  `MTLTexture` for each input/output `VkImage` (color, depth, motion, output).
- Sync: the Vulkan work up to RCAS goes on the queue first, then MetalFX encodes on a Metal command
  buffer, and the Vulkan blit reads the result. Start with the simplest correct option (submit split +
  `MTLSharedEvent` exported from a VkSemaphore/VkEvent), then optimize.
- Settings in `tf3_dlss.cfg`: `upscaler=metalfx`, sharpness through the layer's existing CAS pass.
- Done when: at 67% / 75% the image is visibly sharper and steadier than FSR 1 and NIS (Steam F12
  screenshots, same spot), with no ghosting on still scenery and acceptable trails on vehicles, at
  the same or better frame rate.

## Verification method (all steps)
- Steam overlay screenshot (F12) captures the real swapchain. The game's own F9 screenshot renders a
  separate 5K frame without the upscaler path, so it is useless for A/B.
- Logs: `tf3_dlss.log` (layer) and
  `~/Library/Application Support/Steam/userdata/<id>/3493540/local/crash_dump/stdout.txt` (game).
- Undo: clear the Steam launch options.

## Log
- 2026-10-14 step 1: `macos/CMakeLists.txt` builds an arm64 dylib (Apple clang, no warnings). Mac changes
  in layer.cpp are `#ifdef __APPLE__` only: mach_absolute_time instead of __rdtsc, dladdr module dir,
  `/` path separator, getprogname() == "TransportFever3" process check, macOS info in the log header.
  NGX is stubbed with harness/ngx_stub.cpp (build-time NVIDIA headers only, passed with -DDLSS_SDK, not
  committed).
- A standalone test (game loader libvulkan.1.dylib + MoltenVK_icd.json + VK_ADD_LAYER_PATH) creates
  an instance and device with the layer on Apple M4, and the layer logs and reads its cfg.
- Starting ./TransportFever3 from a terminal does NOT work: Steam relaunches the game (the parent
  becomes steam_osx) and the VK_* variables are lost. The Steam launch options are the way in. The
  Steam-launched game runs with VK_ICD_FILENAMES=MoltenVK_icd.json, DYLD_LIBRARY_PATH=.: and the Steam
  overlay in DYLD_INSERT_LIBRARIES.
- Launch: Steam on macOS rejects `VAR=value %command%` with "OS Error 260". Use the wrapper
  `macos/launch.sh` (it sets the VK_* variables, then `exec "$@"`) through a symlink with no spaces:
  launch options `~/.local/share/tf3-metalfx/launch.sh %command%`. Steam runs `<game>/run.sh` as %command%.
- 2026-10-14 23:31 STEP 1 DONE in game (enabled=0, upscaler=off): 2713 frames, normal shutdown, no game
  errors. The layer identified TONEMAP, FSR_EASU, FSR_RCAS (even with the NIS shaders installed) and the
  final blit (samples the RCAS output, set 1 binding 0). Camera matrices are read correctly ("LOOKS
  RIGHT", fov 1.047, near 4). Scene viewport 1920x1080 at 75%, swapchain 2560x1440 B8G8R8A8.
  Depth: fmt 130 = D32_SFLOAT_S8_UINT, 1920x1080.
  Open point for step 2: `[obj] ... position unclear, no object motion (attrPosition at location -1)`.
  The probe finds the per-instance model matrix but not the vertex position input, so vehicle motion
  vectors are off. Check whether Windows logs the same, or whether this is a MoltenVK/shader-name
  difference.
- 2026-10-14 step 2, run 1 (test upscaler, jitter + MV on, 8 two-frame captures): 6100 frames, no errors.
  The object pass works (77k moved instances). The camera MV field looks right in shape (smooth
  zoom/pan pattern), BUT consecutive captured frames show far more real motion than the MVs (e.g. MV
  1.4 px while the image moved ~20 px), and some frames get huge MVs (380 px). Suspect: the camera
  UBO read at vkQueueSubmit is stale or out of step on macOS. Added TF3_DLSS_TRACE=start,count
  (per-frame [trace] + [trace-present]: is the camera rewritten after submit?).
  Capture note: depth.bin from D32_SFLOAT_S8 via MoltenVK shows row stripes; it is only a recording
  artefact, the MV pass samples the depth texture directly.
  Tool: tools/mvcap_check.py (pure stdlib; R11G11B10 colour, warps frame pairs, writes PNGs).
- Run 2 (trace): the camera UBO is NOT rewritten between submit and present, and it moves smoothly
  every frame. Rules out "read too early".
- Run 3 (trace-reuse): the game keeps a ring of camera buffers. The last 2 frames' buffers never
  change and buffers 3+ frames old are reused. Frame time on the Mac varies a lot (7-85 ms, median
  30 ms), so the camera step changes every frame. The earlier pair analysis was too noisy (flat
  desert, HDR, uncompensated jitter); tools/mvcap_check.py now uses gamma luma, textured pixels only and
  jitter compensation, and it compares the frame's own MV against A's MV (one-frame-lag hypothesis) and
  0.5x/2x.
- Runs 4-5: captures were triggered relative to the process start (long menus wasted them) ->
  TF3_DLSS_CAPTURE_REL=1 counts from the first FSR frame. JSON check: B.prev_view_proj == A.cur_view_proj
  exactly, so the layer's matrix bookkeeping is consistent. But the pair metric was invalid: writing
  ~10 MB of files on the game thread after frame A stalled the game, so frame B's camera jumped far
  (time-based camera) and warping can't explain such jumps. Fix: snapshot the readback buffers into RAM
  after QueueWaitIdle and write the files on a detached thread.
- Run 6 (final in-game run): captures without the stall. CPU recomputation of the camera MVs from the
  recorded depth + JSON matrices (same convention as camera_mv.comp) explains the images (warp error
  0.04-0.05, at the noise floor), but the GPU-computed MVs don't. Fitting shows the GPU MVs match a
  near-constant depth (~0.989): the compute shader read the wrong depth.
  ROOT CAUSE: with TF3's MSAA on (user: numMSAASamples=2) the game depth image is multisampled. The
  layer sampled it through a plain sampler2D, which is invalid. NVIDIA tolerates it, but MoltenVK returns
  unrelated values. (vkCmdCopyImageToBuffer of the MS image, used by the recordings, still gave sample 0,
  which is why the recorded depth looked right.)
  FIX (layer, all platforms):
    * ImageInfo records the sample count ("samples=N" in [image] logs).
    * camera_mv.comp is built twice; DEPTH_MS uses sampler2DMS and reads sample 0. RecordUpscale
      picks the variant from the game depth's sample count and logs "[mv] game depth ... has N sample(s)".
    * The object pass compares against the camera pass's R32F depth copy when the game depth is MS.
    * Recordings copy that R32F depth (valid for MS too); the JSON says "depth_is_r32f": 1.
    * TF3_DLSS_MS_DEPTH=0 forces the old path (for A/B only).
  VERIFIED offline on MoltenVK (game's own loader + libMoltenVK) with the harness (fake_tf3, now with
  --msaa N, portability enumeration and portability_subset): no MSAA, MSAA 2x and MSAA 4x all pass
  (camera and object MVs within 0.002 px). The old path with MSAA 2x fails (flat depth: cube not found),
  which reproduces the in-game symptom.
- Harness on macOS: cmake -S harness -B build-harness -G Ninja -DVulkan_INCLUDE_DIR=/opt/homebrew/include
  -DVulkan_LIBRARY=<game>/libvulkan.1.dylib -DDLSS_SDK=<headers>; copy macos/VkLayer_tf3_dlss.json into
  build-harness/layer; run with DYLD_LIBRARY_PATH=<game> VK_DRIVER_FILES=<game>/MoltenVK_icd.json
  VK_ADD_LAYER_PATH=build-harness/layer VK_INSTANCE_LAYERS=VK_LAYER_TF3_dlss TF3_DLSS_FORCE=1
  TF3_DLSS_TEST_UPSCALER=1 TF3_DLSS_CAPTURE=60,140,220 build-harness/fake_tf3 [--msaa 2].
- STEP 2 status: camera + object MVs correct on MoltenVK in the harness, including MSAA. The fix has not
  been confirmed in the game yet; the user ended in-game testing for now. Confirm with one capture run
  (tools/mvcap_check.py should report "mv" best and GOOD on moving pairs) at the start of step 3.
  The game is left in a safe state: layer loaded passively (enabled=0, upscaler=off, no launch.env).

## Step 3: MetalFX Temporal (implemented 2026-10-15, verified in the harness)
- macos/metalfx_bridge.{h,mm}: a C interface over MetalFX (own MTLCommandQueue, scaler, shared-event
  helpers, macOS hotkeys).
- Hand-off inside the game's command buffer, after the MV/object passes: copy the colour input into our
  own image, memory barrier, vkCmdSetEvent(in[slot]), vkCmdWaitEvents(out[slot]). MoltenVK encodes these
  as encodeSignalEvent/encodeWaitForEvent on the events' MTLSharedEvents with "signaledValue + 1",
  computed on the CPU when it encodes, which happens at vkQueueSubmit (MoltenVK source MVKSync.mm,
  MVKEventNative). At our vkQueueSubmit hook: wait until the slot's previous Metal command buffer is done,
  bump both events to an even value (= "not set"), call the driver, then commit wait(in, v+1) ->
  MTLFXTemporalScaler -> signal(out, v+1) on our queue. A failed submit or encode still signals "out" from
  the CPU. If MVK_CONFIG_PREFILL_METAL_COMMAND_BUFFERS is set, MetalFX is disabled (values can't line up).
- Inputs: colour = copy of the EASU input (B10G11R11 float, MetalFX accepts RG11B10Float), depth = the
  camera pass's R32F copy (MetalFX accepts R32Float), motion = R16G16 float image (MetalFX convention =
  ours: pixels, pointing to the previous position, top-left origin, scale 1). Output = our RGBA8 image (now
  also COLOR_ATTACHMENT usage = MetalFX's render-target requirement). depthReversed = false.
- Jitter sign measured in the harness (still camera, frame-to-frame output change): +1,+1 = 0.10-0.22,
  -1,-1 = 0.9-2.6 -> default mfx_jitter_sign_x/y = 1 (same as DLSS).
- VK_EXT_metal_objects is added at device creation when available. upscaler=metalfx (or auto on macOS).
- MetalFX GPU cost on Apple M4 (standalone bench, tools in scratch): 1280x720->1440p ~8-10 ms,
  1707x960->1440p ~12-14 ms, 1920x1080->1440p ~14-15 ms. Cost follows the input pixel count, so render
  scale 50-58% is the sweet spot.
- Harness regression (MoltenVK of the game): test upscaler with/without MSAA, MetalFX with/without MSAA,
  overlay + hotkeys, auto backend selection: all 12/12 checks pass, no hangs.
- In-game test config: macos/tf3_dlss.step3.cfg (metalfx, overlay, debug_keys) + launch.env with 3
  relative capture points to re-check the MSAA fix with tools/mvcap_check.py.
- 2026-10-15 IN-GAME RESULT (MSAA off, 65% / 60% / 50%): MetalFX runs with no errors or hangs (5388 frames;
  Ctrl+Shift+D toggles and Ctrl+Shift+Up/Down sharpness work on macOS). User: the image changes "a LOT"
  between on and off. The 1:1 crops confirm it: MetalFX gives clean, continuous road markings and edges
  where NIS/FSR 1 shows staircase aliasing and softness.
  FPS shown with MetalFX: 65% -> 25 fps (39.9 ms), 50% -> 38-45 fps (22-26 ms).
  MSAA fix confirmed in game: on a moving capture (mean MV 166 px) the GPU MVs match the CPU
  recomputation within 0.03 px (before the fix: 10-97 px). The other capture pairs had a still camera.
  Overlay title now says METALFX TEMPORAL (was hard-coded to DLSS). Capture env removed after the test.
