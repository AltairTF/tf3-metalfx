# TF3-DLSS

NVIDIA DLSS and AMD FSR 3.1 for **Transport Fever 3**. The mod replaces the game's built-in FSR 1
upscaler for a sharper, steadier image at the same frame rate, or more frames for the same image.
Unofficial and experimental; not affiliated with Urban Games, NVIDIA or AMD.

**[⬇ Download the latest release](https://github.com/lucasmokwa/tf3-dlss/releases/latest)**

> **macOS (Apple Silicon):** this fork adds a macOS build that uses Apple's **MetalFX Temporal** through
> MoltenVK, plus a fix for motion vectors with MSAA on. Install and usage (Portuguese):
> [docs/TUTORIAL_MACOS.md](docs/TUTORIAL_MACOS.md). Technical log: [docs/MACOS_PORT.md](docs/MACOS_PORT.md).
> Quick start from source: `macos/build.sh && macos/install.sh`, then put the printed launch option in Steam.

## Install

You need Windows 10 or 11 and Transport Fever 3 on Steam. DLSS needs an NVIDIA RTX card (20 series or
newer); FSR 3.1 works on any GPU, and the mod picks the right one for you.

1. Open the game folder: in Steam, right-click Transport Fever 3 → *Manage* → *Browse local files*.
2. Copy everything from the zip into that folder, next to `TransportFever3.exe`: `version.dll` and
   the `TF3-DLSS` folder. No game files are replaced.
3. Start the game from Steam and pick a **Resolution scale** in its graphics options:
   67% (Quality) or 50% (Performance) for more frames, or 100% for native-resolution anti-aliasing
   (DLAA).

Settings are in `tf3_dlss.cfg` in the mod folder; every line is explained there, and changes apply
as soon as the file is saved, also while playing. To compare with the game's own upscaling in game,
set `debug_keys=1`: then **Ctrl+Shift+D** turns the upscaler on and off and **Ctrl+Shift+U**
switches between DLSS and FSR.

**Update:** close the game and copy the new zip's contents over the old ones.
**Uninstall:** delete `version.dll` and the `TF3-DLSS` folder from the game folder.
**Not working?** See `TF3-DLSS\README.txt`. If the mod couldn't start, it leaves a
`TF3-DLSS-DID-NOT-START.txt` in the `TF3-DLSS` folder that says why. Bug reports: attach `tf3_dlss.log` from the same
folder, with your GPU and resolution scale.

### What is version.dll?

The mod itself is a Vulkan layer (`TF3-DLSS\bin\VkLayer_tf3_dlss.dll`), and Vulkan only loads layers
it has been told about. `version.dll` is a small stand-in for the Windows file of the same name:
the game loads it from its own folder at startup, it tells Vulkan where the layer is for this one
game process, and passes everything else on to the real `version.dll` in Windows. Nothing is
written to the registry or outside the game folder, and no other program sees the mod.

## Known limitations

- People and animals (skinned models) get camera motion only.
- Smoke, clouds and other effects without depth get no motion vectors and can trail.
- Vulkan renderer only (TF3's only renderer).

---

The rest of this page is for developers.

## How it works

| Step | What the layer does |
|---|---|
| Find the passes | Recognises the game's tonemap, FSR EASU/RCAS and final blit by their SPIR-V debug names |
| Jitter | Shifts the 3D scene's viewport by a sub-pixel Halton offset each frame (overlays and UI aren't touched) |
| Camera motion | Compute pass: rebuilds each pixel's previous screen position from depth and the camera matrices |
| Object motion | Finds moving models (vehicles) and redraws them into the motion-vector image with last frame's model matrix |
| Upscale | After RCAS, records DLSS or FSR 3.1 into the game's command buffer, then swaps the blit's input for the upscaled image |
| Extras | Texture LOD bias, sharpening (RCAS with a motion-adaptive boost for DLSS and MetalFX, FSR's own RCAS for FSR), hover-highlight depth fix |

A release enables the layer with `version.dll` in the game folder (`layer/proxy/`): when the game
loads it, it sets `VK_ADD_LAYER_PATH` to `TF3-DLSS\bin` and adds `VK_LAYER_TF3_dlss` to
`VK_INSTANCE_LAYERS` / `VK_LOADER_LAYERS_ENABLE` for that process only (the same variables
`run_tf3.ps1` sets), then forwards every `version.dll` call to `System32\version.dll`.

## Repository layout

| Folder | Contents |
|---|---|
| `layer/` | Layer source (`src/layer.cpp`), shaders (`shaders/`), `version.dll` stand-in (`proxy/`), CMake project, layer manifest |
| `scripts/` | `build.ps1`, `run_tf3.ps1` (dev launch), `run_validation.ps1`, `get_fsr.ps1`, `package.ps1` |
| `package/` | Files shipped as-is in a release: `README.txt`, `licenses/` |
| `third_party/ffx-api/` | AMD FidelityFX API headers (MIT); the runtime DLL is downloaded into `bin/`, not committed |
| `tools/` | RenderDoc analysis scripts used during development |
| `tf3_dlss.cfg` | Settings, documented line by line (the release copy) |

## Building

Needs Windows with:

- Visual Studio 2022+ Build Tools (C++), CMake and Ninja
- the [Vulkan SDK](https://vulkan.lunarg.com/)
- the [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS) in `%USERPROFILE%\Projects\DLSS` (or set `-DDLSS_SDK=...`)

```powershell
cd scripts
.\get_fsr.ps1        # once: AMD's signed amd_fidelityfx_vk.dll (FidelityFX SDK v1.1.4)
.\build.ps1          # -> build\bin: VkLayer_tf3_dlss.dll, manifest, nvngx_dlss.dll, amd_fidelityfx_vk.dll
                     #    build\proxy: version.dll
.\run_tf3.ps1        # starts TF3 with the layer from build\bin (log: logs\layer.log)
```

`run_tf3.ps1` enables the layer for that launch only and uses the repo's `tf3_dlss.cfg`. A release
installed in the game folder steps aside for it: its `version.dll` sees the layer is already enabled.

`run_validation.ps1` does the same with the Khronos validation layer below ours
(`-Sync` adds synchronisation validation); messages go to `logs\validation.log`.

## Measuring the mod's cost

Set `perf=1` in `tf3_dlss.cfg` (live) and every `perf_seconds` the log gets a `[perf]` report:

- frame time (average and slowest 1%) and which upscaler ran,
- the CPU time of the layer's own code per frame, by kind of call (draws, descriptor binds, render
  passes, recording the upscaler, ...), with calls per frame and how often game threads waited for
  the layer's lock,
- GPU time of each pass the layer adds (motion vectors, DLSS/FSR, sharpening, the rest), from
  timestamp queries, and of the game's own FSR 1 passes that the upscaler makes redundant,
- on macOS, MetalFX's own GPU time (its Metal command buffers run on a separate queue, so the
  Vulkan timestamps only see the hand-off).

With `overlay=1` the info box's last line shows the totals. To compare settings, keep the camera
still in the same scene and change one cfg line at a time (most apply live), or toggle the
upscaler with Ctrl+Shift+D (`debug_keys=1`): with it off the report shows what the layer's tracking
alone costs. `skip_game_fsr=1` leaves out the game's FSR 1 draws while an upscaler runs.

## Testing without the game

`harness/fake_tf3` is a small Vulkan program that renders frames shaped like TF3's (3D scene with
the game's vertex inputs and camera block, tonemap, FSR 1 EASU and RCAS, blit, UI) and checks what the
layer does with them:

- the upscaled image reaches the screen: the FSR 1 stand-ins draw magenta, so any magenta left
  means the blit swap failed, and the UI must stay on top;
- motion vectors, from recordings of a still camera, a sideways pan and an upward pan with an object
  moving in front of a wall, match the exact values to within 0.02 px;
- no Khronos validation errors;
- its CPU time per frame with and without the layer (the scene is recorded on 4 threads, like the game).

```powershell
cd scripts
.\run_harness.ps1                 # with DLSS / FSR 3.1 from build\bin
.\run_harness.ps1 -Validation     # plus the validation layer (Vulkan SDK)
.\run_harness.ps1 -TestUpscaler   # a plain resize in place of DLSS / FSR, as in CI
```

Results go to `logs\harness`. CI runs the same checks on Linux before every build (`harness/run.sh`):
there the layer is built as a shared library with NGX stubbed out (`harness/ngx_stub.cpp`), on Mesa's
CPU Vulkan driver, and `TF3_DLSS_TEST_UPSCALER=1` swaps DLSS and FSR for a bilinear resize. A failing
check stops the build, so nothing is released. Each CI run's summary page shows the checks and what the layer costs
(CPU time per frame with and without it, its own `[perf]` report and lock waits), from `harness/summary.py`,
next to the numbers of the latest build of `main` and the change.

## Releasing

Every push to `main` is released by GitHub Actions (`.github/workflows/release.yml`), as the next
minor version after the latest release (0.2.0, 0.3.0, ...). Pushes that only change Markdown files
are skipped. The workflow stamps the version into its own copy of `layer.cpp`, the manifests and
`package/README.txt`, runs `package.ps1` and publishes `TF3-DLSS-<version>.zip`; nothing is committed
back and binaries are never committed.

For a bigger jump (say 1.0.0), set `TF3_DLSS_VERSION` in `layer/src/layer.cpp` to it: a version newer
than the latest release is used as is. Before pushing, make sure `tf3_dlss.cfg` has the developer
options off, or the package step fails.

The CI build fetches the DLSS SDK from [NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) at the tag in
`DLSS_SDK_REF`, the Vulkan SDK from LunarG and the FSR runtime with `get_fsr.ps1`. Pull requests get the
same build, with the zip under the run's artifacts instead of a release.

After the release, the `nexus` job uploads the same zip to the
[Nexus Mods page](https://www.nexusmods.com/transportfever3/mods/17) as a new version of the main file
(`NEXUS_FILE_ID`), archives the previous version, sets the page's version and adds the release's PR
titles as the changelog. It needs a Nexus API key in the `NEXUSMODS_API_KEY` repository secret and is
skipped without one. If Nexus fails, re-run just that job; the GitHub release is already out.

## Hotkeys

All hotkeys need `debug_keys=1` in the cfg (off by default, so no key press changes the upscaler
for players):

| Keys | Does |
|---|---|
| Ctrl+Shift+D | upscaler on / off |
| Ctrl+Shift+U | switch DLSS <-> FSR 3.1 |
| Ctrl+Shift+M | motion vectors: camera + objects -> camera only -> off |
| Ctrl+Shift+J | jitter on / off |
| Ctrl+Shift+V | debug view: off -> motion vectors -> jitter |
| Ctrl+Shift+T | automated camera test, recorded to `mvcap/` next to the log |
| Ctrl+Shift+Y | automated quick-tilt test, records 8 frames in a row |
| Ctrl+Shift+L | log every draw of one frame (pass, jitter on or off, pipeline state, shader names) |
| Ctrl+Shift+O | on-screen info box on / off (`overlay` in the cfg) |
| Ctrl+Shift+Up / Down | sharpness of the running upscaler +0.1 / -0.1 |
| Ctrl+Shift+K | jitter sign of the running upscaler: x+y+ -> x-y+ -> x+y- -> x-y- |

Changes made with these keys last until the cfg file is saved again.

## License

MIT for the source code in this repository (see `LICENSE`). Release binaries also contain NVIDIA
code that the MIT license doesn't cover: `VkLayer_tf3_dlss.dll` links NVIDIA's NGX SDK library, and
`nvngx_dlss.dll` is NVIDIA's DLSS runtime. Those stay under NVIDIA's terms, which the release passes
on to players. The FSR runtime is AMD's, under MIT. See `package/licenses/THIRD-PARTY.txt`.

This software contains source code provided by NVIDIA Corporation. NVIDIA, RTX and DLSS are
trademarks of NVIDIA Corporation; AMD, FidelityFX and FSR are trademarks of Advanced Micro
Devices, Inc.
