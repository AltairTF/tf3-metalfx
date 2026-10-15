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

## Current state (2026-10-15)
Good enough for a smooth image (user, after the in-game reactive-mask test). What runs by default on macOS:
MetalFX Temporal with the layer's jitter, camera + vehicle motion vectors (MSAA-depth fix), RCAS with
motion-adaptive sharpening, the reactive mask for particles / people and animals / water, mip bias, and the
optional `upscale_to` cap. Frame generation (`frame_gen=1`) stays experimental and off.
- Tested settings: game window 1920x1080 on the 4K LG UltraFine (macOS "looks like 1080p"), MSAA off, resolution
  scale 60-75% (1152x648 and 1440x810 -> 1920x1080). 37-47 fps; the GPU is the limit (99% busy). MetalFX costs
  2.4-3.3 ms, the reactive mask 0.6-0.8 ms, sharpening ~0.5 ms, motion vectors ~0.1 ms.
- Open: the game presents with VK_PRESENT_MODE_IMMEDIATE (no vsync) and its frame time moves between 21 and
  27 ms, so on a 60 Hz screen without VRR frames stay up for 1 or 2 refreshes unevenly (judder) and can tear.
  `fps_limit=30` (every frame exactly 2 refreshes) addresses it; see "Frame rate lock" at the end.

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

## Roadmap after step 3 (ideas from OptiScaler)
OptiScaler (github.com/optiscaler/OptiScaler) swaps one upscaler for another in games that already call
DLSS 2+ / FSR 2+ / XeSS, on Windows. TF3 only has FSR 1, and OptiScaler is GPL-3.0 while this layer is MIT,
so it is used for ideas only: no code is copied. Where an idea needs an algorithm, it is written from an
MIT source (AMD FidelityFX, Dear ImGui).

1. MetalFX GPU time: the MetalFX command buffer's GPUStartTime..GPUEndTime in the info box, the log and
   the perf=1 report. Also measure skip_game_fsr=1 (the game's 1:1 EASU + RCAS draws are wasted while
   MetalFX runs). Done: ~3.4 ms at 50% in game; the game's FSR 1 passes cost ~0.1 ms, so skip stays off.
2. RCAS sharpening (AMD FidelityFX FSR 1) plus motion-adaptive sharpening (OptiScaler's "MAS": more
   sharpness where the motion vectors are long), replacing the CAS-style pass. Done (runs in game).
3. Frame generation with MTLFXFrameInterpolator (macOS 26+, supported on the M4). TF3 draws its UI after
   the blit, so the MetalFX output is a free UI-less image, which is what OptiScaler's HUDfix has to search
   for. Phases: harness prototype, in game with the UI interpolated, then a separate UI texture.
   Prototype done (frame_gen=1, off by default). Works in the harness; in game it is smooth but the map
   labels shake, there is tearing, and it costs ~6.3 ms (see "Roadmap step 3" below). Stays experimental.
4. Reactive mask (MetalFX reactiveMaskTexture, macOS 14.4+) for smoke, particles, water and glass.
   Done (reactive=1, default on); verified in the harness and in game (see "Roadmap step 4" below).
5. Optional: an in-game menu (Dear ImGui), FSR 3.1 built from source on MoltenVK as a second backend.

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
  scale 50-58% is the sweet spot. (Superseded: measured in game, 1280x720->1440p is ~3.4 ms; see the
  in-game result under "Roadmap steps 1 and 2".)
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

## Roadmap steps 1 and 2 (implemented 2026-10-15, verified in the harness and in game)
- MetalFX GPU time: metalfx_bridge.mm adds a completion handler to each MetalFX command buffer and keeps
  GPUEndTime - GPUStartTime (the median of the last 63 for the info box; median, mean and slowest per perf=1
  window). Measured with a probe on the M4:
  a command buffer that starts with encodeWaitForEvent gets its GPUStartTime only after the event is
  signalled (0.25 ms after, with the signal 100 ms after commit), so the wait for the game's inputs is not
  in the number. Shown as "GPU x.x MS" after the sizes in the info box, "[mfx] frame N: MetalFX GPU ..." every
  log_every frames, and "[perf]   MetalFX GPU: ..." in the perf=1 report (the overlay's perf line says MFX).
- Harness (1024x576 -> 2048x1152, M4): MetalFX 2.3-2.9 ms per frame. The 8-10 ms at 720p -> 1440p in the
  tutorial came from a standalone bench and still has to be re-measured in game.
- Sharpening (layer/shaders/sharpen.comp): sharpen_type=rcas (default) is RCAS from AMD FidelityFX FSR 1,
  with its noise detection and with lobe x sharpness (1 = full RCAS, 0.5 = one stop less, 0 = off, so the
  motion boost has no jump at 0). sharpen_type=cas keeps the old pass. Motion-adaptive sharpening adds
  motion_sharpness x clamp((|mv| - motion_threshold) / (motion_limit - motion_threshold)), |mv| in output
  pixels (defaults 0.4, 0, 10, OptiScaler's defaults). sharpen_debug=1 tints the boost red. The pass also
  runs with sharpness=0 when motion_sharpness > 0. New macOS hotkeys: Ctrl+Shift+S (RCAS <-> CAS),
  Ctrl+Shift+A (motion boost on/off).
- RCAS is weaker than the old CAS at the same value (harness: Laplacian energy 4.69 at RCAS 0.3, 6.73 at
  CAS 0.3, 7.44 at RCAS 1.0), partly because RCAS does nothing next to pure black or white (the
  checkerboard). Default sharpness raised from 0.3 to 0.5.
- Harness checks (game's MoltenVK, Khronos validation): test upscaler with sharpening + motion boost, with
  skip_game_fsr=1 and with MSAA 2x: all checks pass, 0 validation errors. MetalFX with MSAA 2x and
  skip_game_fsr=1: all checks pass; the same 41 validation messages as the previous commit (VkEvent on a
  portability device, metal-objects export without the instance flag, objects left at device destroy).
  Motion boost: during the pan (frame 149) 96% of the pixels are tinted (all but the UI panel); with a still
  camera (frame 299) only the moving cube (1.7%). Frame time with the new pass equals the previous commit's
  (5.52 vs 5.54 ms, same run; run-to-run noise is larger).
- 2026-10-15 IN-GAME RESULT (50%, 1280x720 -> 2560x1440, MSAA off, perf=1, about 3.5 minutes after the
  save loaded; build-macos/tf3_dlss.log): no errors, clean exit after 12043 frames, about 8700 of them
  upscaled.
  - MetalFX GPU over 8607 frames in 42 perf windows: 3.3-3.5 ms per frame in 28 windows (5865 frames,
    weighted mean 3.45 ms). The other 14 windows had single command buffers of 19-85 ms, which pushed their
    means to 4.0-5.7 ms and the info box's old exponential average to 6.6 ms. Those spikes came in the
    busiest scenes (37-42 fps): the GPU interleaves the game's work with ours, so a command buffer's
    start-to-end time sometimes covers some of the game's work. Since then the info box shows the median
    of the last 63 command buffers, and the perf=1 report gives median, mean and slowest.
  - The bench's 8-10 ms at 720p -> 1440p was about 2.5x too high. Frame time 17-33 ms (30-59 fps)
    depending on the scene, so MetalFX is 10-20% of the frame.
  - The layer's Vulkan passes, medians of the 42 windows (range in brackets): motion vectors 0.30 ms
    (0.09-0.71), sharpening 0.69 ms (0.64-1.41; the compute pass plus the copy back into the output
    image), the "upscaler" part 3.6 ms (3.4-6.9; the wait for MetalFX). The game's own FSR 1 passes cost
    0.10 ms (0.08-0.78), so skip_game_fsr=1 saves about 0.1 ms: the default stays 0, and the cfg says so.
  - Sharpening: the user switched RCAS <-> CAS (Ctrl+Shift+S) and the motion boost (Ctrl+Shift+A) a few
    times; both hotkeys work on macOS. In the harness the old CAS pass cost about the same (0.49-0.74 ms vs
    0.52-0.78 ms at 2048x1152), so most of the 0.69 ms is the pass itself and its copy, not the RCAS math.
  - Not measured yet: 58% and 67% render scale.
- Possible saving: the sharpening pass writes a temporary image and copies it back into the output
  (2560x1440 RGBA8, about 30 MB of traffic per frame). With MetalFX, the scaler could write into the
  temporary image and the sharpening pass into the output, which drops the copy. Worth doing with the
  backend split before step 3, since frame generation also needs the output path reworked.

## Roadmap step 3: frame generation (prototype 2026-10-15, verified in the harness)
- Probe (scratchpad fgprobe.mm, M4, macOS 27): MTLFXFrameInterpolator is supported. Depth and motion can stay
  at the render size while colour and output are at the output size. Motion vectors use the same convention
  as the temporal scaler (render pixels, pointing to the previous position, scale 1): a square moving
  80,40 px per frame came out exactly at the midpoint, and only when the frames are fed in order (fed the
  same pair again and again, the output was the current frame). The UI test passed: a white bar composited
  into the "ui" texture (uiTextureComposited) came out intact while the square moved.
- Cost at 1280x720 -> 2560x1440, back-to-back command buffers: 7.7-9.8 ms GPU and about 3 ms CPU to encode
  per generated frame; the temporal scaler measured the same way: 2.3 ms (3.4 ms in game). Attaching the
  temporal scaler to the interpolator (descriptor.scaler) saved nothing. In the harness at 2048x1152 the
  interpolation is 3.9 ms (median). So on a base M4 at 1440p, expect roughly 6-8 ms per generated frame.
- Design (layer.cpp, "Frame generation"; metalfx_bridge.mm mfx_fg_*): with frame_gen=1 at swapchain creation
  the swapchain gets transfer usage, 3 images and FIFO. At each present of a frame the MetalFX upscaler ran on,
  the layer acquires a second swapchain image and submits one command buffer of its own on the game's queue
  that waits for the present's semaphores, copies the finished frame (UI included) into uiImage, hands over
  to MetalFX through a VkEvent pair exported as MTLSharedEvents (the same hand-off as the upscaler), copies the
  generated frame into the second image and copies this frame's upscaled image into prevImage. It signals
  two semaphores: one for presenting the generated image, one for the game's image, which are presented in
  that order. MetalFX gets colour = the upscaled image (no UI), prevColor = the last one, depth (R32F),
  motion (RG16F, render size), ui = the finished frame with uiTextureComposited, jitter, near / far,
  vertical field of view and aspect from the projection matrix, and deltaTime from the present times. History
  resets after a skipped frame, a camera cut or a size change. The game's next frame is submitted after the
  hand-off on the same queue, so the shared inputs (depth, motion, the output image) can't be overwritten
  while MetalFX reads them. The presents run outside the layer's lock, since FIFO can block them.
- With FIFO the two images are shown at successive refreshes: the game runs at most at half the refresh rate
  (30 -> 60 on the LG UltraFine 5K at 60 Hz, no VRR). MTLFXFrameInterpolator always makes the midpoint, so
  other ratios would need presentation timing (VK_GOOGLE_display_timing), not done.
- Harness (fake_tf3 --speed K multiplies the camera pan and the cube speed): all checks pass with
  frame_gen=1, with and without MSAA 2x; Khronos validation shows only the 4 kinds of messages seen before.
  tools fg_check.py / fg_cube.py on TF3_FG_DUMP frames: at 20 px per frame of pan the generated frame is
  shifted by exactly 10 px (match error 0.16 vs 0.33 between the real frames, so it is not a blend); a cube
  moving 30 px per frame is drawn halfway (edges 987 -> 1002 -> 1017); the UI panel is identical to the
  current frame's. The cube's trailing edge smears over about 15 px where the wall is revealed (expected for
  interpolation; small at normal speeds). Game frame time in the harness: 3.9 ms off, 9.7 ms on.
- Known limits: the info box is drawn into the upscaled image, so MetalFX moves it with the scene (it shakes
  in generated frames); the UI separation needs the game's blit to copy the upscaled image unchanged (true in
  the harness, to be checked in game with Ctrl+Shift+G); frame_gen=1 needs a restart; macOS 26+.
- 2026-10-15 IN-GAME RESULT (50%, 1280x720 -> 2560x1440, LG UltraFine 5K at 60 Hz; build-macos/tf3_dlss.run8-fg.log):
  runs without errors (about 1000 generated frames, Ctrl+Shift+F / G work). The game already asked for FIFO
  with 3 images. Interpolation GPU 6.2-8.2 ms per generated frame (median 6.3). Game frame rate with
  frame generation 23-29 fps (34-42 ms), without it 38-47 fps in the same session, so near cities it falls
  below 30 and the 60 Hz screen gets an uneven mix of generated and repeated frames. The user: smooth, but
  "bastante tearing", the UI "balançando" (shaking), and bad near a city.
  The two Ctrl+Shift+G dumps had a still camera (only the info box text changed), so they show no motion,
  but they confirm that the game's blit copies the upscaled image unchanged (99.5% of the pixels identical
  outside the UI) and that the generated frame carries the current frame's UI. The UI shaking follows from
  that: labels tied to the map (stations, vehicle icons, town names) are UI, so MetalFX puts them where they
  are in the current frame over a scene half a frame behind; they jump back and forth on every shown frame.
  The info box shakes for the opposite reason (it is in the image MetalFX moves). The cause of the tearing is
  not known yet (with FIFO the screen itself should not tear: it may be seams in the interpolated image or the
  uneven pacing below 30 fps).
  Conclusion for now: frame generation stays experimental and off by default. On a base M4 with a 60 Hz
  screen it can at best turn 30 fps into 60, and costs about 6 ms of the 33 ms it has.

## Roadmap step 4: reactive mask (2026-10-15, verified in the harness and in game)
- Why TF3 needs it: the game has no motion vectors; the layer makes them for the camera and for moving vehicles
  only. Particles (smoke, steam: misc/particle.fs), rain (atmosphere/rain.fs), skinned meshes (people, animals:
  mat/vs/skinning) and water (mat/fs/water*.fs) move on their own, so MetalFX reprojects their history with the
  camera's motion and smears them.
- How (layer.cpp "Reactive mask"): at vkCreateGraphicsPipelines the layer classifies each 3D colour pipeline
  (depth test on, no tessellation): particle/rain and water by the fragment shader's SPIR-V names (noise0Tex,
  tex + noise, waterDepthTex), animated by the skinned vertex shader, transparent = blended without depth writes.
  For each kind with a strength > 0 it creates a copy of the pipeline for its own render pass (RGBA8 + D32 depth,
  1 sample, depth test LEQUAL, no depth writes, alpha blend: dst.a += src.a x blend constant). Particles and
  transparent keep the game's fragment shader (alpha = coverage); animated and water use reactive_const.frag
  (alpha 1). New hook vkCmdPushConstants keeps the game's push constants. During the frame, draws of these
  pipelines in the main 3D pass are remembered with their state (sets with dynamic offsets, vertex/index buffers,
  push constants). In RecordMetalFx, after the motion-vector pass, the layer copies the R32F depth into its D32
  buffer (through a buffer), replays the draws with this frame's jitter, and reactive_mask.comp copies the alpha
  into an R32F mask. The scaler is created with reactiveMaskTextureEnabled (R32Float; a probe on the M4 accepted
  R8, R16F, R32F and RGBA8). With no draws in a frame the mask is cleared. Pipelines get no copy when reactive=0
  or their kind's strength is 0 at creation (restart to change). Ctrl+Shift+R cycles on -> on + red tint -> off.
- Harness: fake_tf3 --smoke draws a blended, drifting, flickering puff (push constants, no depth writes) after the
  opaque scene (classified "transparent"; the harness has no particle/skinned shaders). With
  reactive_transparent=0.9: 1 pipeline copy, 1 draw replayed per frame, the recorded mask (mvcap *.reactive.bin)
  covers 12-13% of the pixels with the puff's alpha x 0.9. All motion-vector checks pass with and without MSAA 2x;
  Khronos validation shows only the 41 messages of the previous commits (VkEvent on portability, metal-objects
  export, leaks at device destroy). Forcing the mask to 1 everywhere raises the output's frame-to-frame change from
  4.80 to 5.77 (red channel, mean), so MetalFX does use the mask. reactive_debug=1 tints 12% of the final frame.
  GPU cost in the harness is within run-to-run noise (one draw).
- To check in game: the [reactive] log lines (pipeline copies by kind, draws replayed per frame by kind, draws of
  copied pipelines outside the main 3D pass), reactive_debug=1 / Ctrl+Shift+R near smoke, a city (people) and
  water, and the GPU time with perf=1 (the pass is inside the "upscaler" part).
- 2026-10-15 IN-GAME RESULT (1152x648 and 1440x810 -> 1920x1080, MSAA off, perf=1; build-macos/tf3_dlss.log):
  no errors, clean exit after 13778 frames. 100+ pipeline copies (42 particle, 58 animated). Per frame 126-715
  particle draws and 1 water draw were replayed; no animated draws were seen in that session (no people on
  screen, or they are drawn some other way: to check with the red tint in a busy street), and no draws of copied
  pipelines happened outside the main 3D pass. Cost 0.6-0.8 ms (the "upscaler" part 3.5-3.8 ms with the mask,
  2.9 ms after Ctrl+Shift+R turned it off). User: good enough for smoothing the image.

## upscale_to and the perf=1 system line (2026-10-15)
- `upscale_to=<lines>`: MetalFX writes an image at most that tall; the game's blit (std.fs, `texture(tex, uv)`)
  stretches it with the layer's linear sampler, and the UI is drawn after it at the screen's resolution. The
  output is also capped at MetalFX's maximum scale (mfx_max_scale, 3x on the M4). The mip bias follows the
  upscaled size. Harness: passes at 3840x2160, scale 0.275. In game so far only with a 1920x1080 swapchain, where
  upscale_to=1080 changes nothing; a 4K swapchain with scale 25-30% is still to be tried.
- perf=1 adds a "system:" line: GPU busy % (IOAccelerator "Device Utilization %"), the process's and the busiest
  threads' CPU, and a verdict (GPU, one game thread, or the 60 Hz display).
- Harness manifest fix: the layer library is .dylib on macOS.

## Frame rate lock: fps_limit (2026-10-15, verified outside the game and in game)
At 37-47 fps on a 60 Hz screen without VRR, frames are shown for 1 or 2 refreshes in an uneven pattern, which
looks like jumps; the game presents with IMMEDIATE, so it can also tear. TF3 has no frame limit or vsync option.
`fps_limit=30` (layer.cpp "Frame rate lock", off by default, needs a restart): the swapchain is created with FIFO,
and at each present (PacedPresent, when frame generation is off) the layer sleeps until half a refresh before the
refresh n after the previous frame's (n = refresh rate / fps_limit = 2), then hands the frame over; FIFO shows it
at that refresh. The refresh grid comes from a CVDisplayLink on the main display (mfx_vsync in the bridge). The
sleep also holds the game's thread, so it can't run ahead. A late frame goes out at once and takes the first
refresh it can still make.
- Tried first and dropped: VK_GOOGLE_display_timing (MoltenVK has it). In a test window presentAtTime wasn't
  honoured (frames reached the screen ~24 ms before the asked time; the pacing then drifted to 29.2 fps), and the
  "actual" present times are the hand-over times (~1 ms after the present), not refreshes.
- Test (scratchpad fpstest: a 480x270 Cocoa window on the game's loader + MoltenVK with the layer, CPU work 21-27
  ms per frame like TF3, IMMEDIATE asked): 30.0 fps; 599 of 599 frames 2 refreshes after the previous one, 0
  late; the game thread waits ~7 ms per frame. With 30-36 ms frames (below 30 fps) it falls to a 2/3-refresh mix
  (24.4 fps, 45% late): the lock needs the game to hold 30 fps.
- Log: `[fps]` every 300 frames (frames, late, refreshes between frames, time the game thread was held).
- 2026-10-15 IN-GAME RESULT (fps_limit=30, 60 Hz LG UltraFine): 6299 frames locked, 6146 (97.6%) exactly 2
  refreshes after the previous one, 121 after 3 and 31 after 4 or more (loading and busy moments), 134 late. The
  user accepted it ("ok, done then"). The info box is off by default again (overlay=0 in every cfg).
