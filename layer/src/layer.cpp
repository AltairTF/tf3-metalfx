// VK_LAYER_TF3_dlss - DLSS / FSR 3.1 for Transport Fever 3, as a Vulkan layer
//
//   Game frame (relevant part):    tonemap -> FSR EASU (1:1) -> FSR RCAS (1:1) -> blit to 4K -> UI
//   With the layer:                tonemap -> EASU -> RCAS -> [DLSS or FSR 3.1: EASU input + depth
//                                  + motion vectors -> 4K image] -> blit samples that image -> UI
//
//   * Installed as an implicit layer, but only active inside TransportFever3.exe (other programs
//     fail the loader negotiation, so the loader skips the layer for them).
//   * Shader identification by SPIR-V debug names (FSR EASU / RCAS / tonemap).
//   * Tracks images, views, framebuffers, render passes, layouts and descriptor sets.
//   * Adds DLSS's Vulkan extensions (and FSR's device features) at creation, initialises NGX and
//     loads amd_fidelityfx_vk.dll.
//   * After RCAS's render pass ends, records the upscaler into the game's command buffer, then
//     restores the game's graphics bindings.
//   * When the final blit draws, temporarily binds a descriptor set pointing at the upscaled image.
//   * Camera motion vectors from depth (compute), object motion by redrawing moving models with
//     last frame's model matrix, sub-pixel jitter through the scene viewport.
//   * Settings: tf3_dlss.cfg in the mod folder (or %TF3_DLSS_CFG%), reloaded live when cfg_live=1.
//     Log: tf3_dlss.log in the mod folder (or %TF3_DLSS_LOG%), falling back to %TEMP%.
// Detailed logging: first 3 frames after the game's FSR pass first runs, then every log_every frames.
//
// Copyright (c) 2026 Lucas. MIT license (see LICENSE).
// This software contains source code provided by NVIDIA Corporation.

#define TF3_DLSS_VERSION "0.1.0"

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_helpers_vk.h"

// AMD FidelityFX API (FSR 3.1). Headers only: amd_fidelityfx_vk.dll is loaded at run time.
#include "ffx_api/ffx_api.h"
#include "ffx_api/ffx_upscale.h"
#include "ffx_api/vk/ffx_api_vk.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>  // __rdtsc (perf=1 timings)
#define TF3_TICKS() __rdtsc()
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#define TF3_TICKS() __rdtsc()
#elif defined(__APPLE__)
#include <mach/mach_time.h>  // Apple Silicon: no TSC; mach_absolute_time is a cheap monotonic counter
#define TF3_TICKS() mach_absolute_time()
#else
#define TF3_TICKS() (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count()
#endif

#ifdef __APPLE__
#include <vulkan/vulkan_metal.h>  // VK_EXT_metal_objects: MoltenVK's Metal objects for MetalFX
#include "metalfx_bridge.h"
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <sys/sysctl.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <windows.h>
#define TF3_EXPORT extern "C" __declspec(dllexport)
#else
#define TF3_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define LAYER_NAME "VK_LAYER_TF3_dlss"

#ifdef _WIN32
#define TF3_SEP L"\\"
#else
#define TF3_SEP L"/"
#endif

namespace {

// =============================================================================================
// Paths, process check, logging
// =============================================================================================
// Directory containing this DLL (nvngx_dlss.dll and amd_fidelityfx_vk.dll sit next to it).
std::wstring ModuleDir() {
#ifdef _WIN32
    HMODULE mod = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&ModuleDir), &mod);
    wchar_t buf[MAX_PATH] = {};
    GetModuleFileNameW(mod, buf, MAX_PATH);
    std::wstring p(buf);
    size_t slash = p.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : p.substr(0, slash);
#elif defined(__APPLE__)
    Dl_info info{};
    if (!dladdr(reinterpret_cast<const void*>(&ModuleDir), &info) || !info.dli_fname) return L".";
    std::string p(info.dli_fname);
    size_t slash = p.find_last_of('/');
    p = slash == std::string::npos ? std::string(".") : p.substr(0, slash);
    return std::wstring(p.begin(), p.end());
#else
    return L".";
#endif
}

// The mod folder: where tf3_dlss.cfg, the log and the README live. The DLL is in its "bin"
// subfolder in a release; in a development build there is no "bin" parent, so it's the DLL folder.
std::wstring RootDir() {
    std::wstring dir = ModuleDir();
    size_t slash = dir.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return dir;
    std::wstring leaf = dir.substr(slash + 1);
    for (auto& c : leaf) c = (wchar_t)std::towlower(c);
    return leaf == L"bin" ? dir.substr(0, slash) : dir;
}

std::wstring TempDir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH] = {};
    DWORD n = GetTempPathW(MAX_PATH, buf);
    if (n == 0 || n >= MAX_PATH) return L".";  // n is the needed size when the buffer is too small
    std::wstring t(buf, n);
    while (!t.empty() && (t.back() == L'\\' || t.back() == L'/')) t.pop_back();
    return t.empty() ? L"." : t;
#else
    return L"/tmp";
#endif
}

std::string Narrow(const std::wstring& w) {
    std::string s;
#ifdef _WIN32
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    s.resize(n > 0 ? n : 0);
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
#else
    for (wchar_t c : w) s += (char)c;
#endif
    return s;
}

FILE* OpenFile(const std::wstring& path, const wchar_t* mode) {
#ifdef _WIN32
    return _wfopen(path.c_str(), mode);
#else
    return std::fopen(Narrow(path).c_str(), Narrow(mode).c_str());
#endif
}

// The layer is installed for every Vulkan program, but only does anything inside the game.
// TF3_DLSS_FORCE=1 makes it active in any process (testing); TF3_DLSS_DISABLE=1 turns it off.
bool IsGameProcess() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    const char* dis = std::getenv("TF3_DLSS_DISABLE");
    const char* force = std::getenv("TF3_DLSS_FORCE");
    if (dis && *dis == '1') cached = 0;
    else if (force && *force == '1') cached = 1;
    else {
#ifdef _WIN32
        wchar_t buf[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, buf, MAX_PATH);
        const wchar_t* name = std::wcsrchr(buf, L'\\');
        name = name ? name + 1 : buf;
        cached = _wcsicmp(name, L"TransportFever3.exe") == 0 ? 1 : 0;
#elif defined(__APPLE__)
        const char* name = getprogname();  // macOS: the game is a bare executable named TransportFever3
        cached = name && std::strcmp(name, "TransportFever3") == 0 ? 1 : 0;
#else
        cached = 1;
#endif
    }
    return cached != 0;
}

std::mutex g_logLock;
FILE* g_log = nullptr;
std::wstring g_logPath;
bool g_logTried = false;

void Log(const char* fmt, ...) {
    std::lock_guard<std::mutex> lk(g_logLock);
    if (!g_log) {
        if (g_logTried) return;
        g_logTried = true;
        const char* env = std::getenv("TF3_DLSS_LOG");
        std::vector<std::wstring> paths;
        if (env && *env) paths.push_back(std::filesystem::path(env).wstring());
        paths.push_back(RootDir() + TF3_SEP L"tf3_dlss.log");
        paths.push_back(TempDir() + TF3_SEP L"tf3_dlss.log");  // mod folder not writable (e.g. Program Files)
        for (const auto& p : paths)
            if ((g_log = OpenFile(p, L"w"))) {
                g_logPath = p;
                break;
            }
        if (!g_log) return;
    }
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_log, fmt, ap);
    va_end(ap);
    std::fputc('\n', g_log);
    std::fflush(g_log);
}

template <typename T>
std::string H(T handle) {
    char b[32];
    std::snprintf(b, sizeof b, "%p", (const void*)handle);
    return b;
}

// ---------------------------------------------------------------------------------------------
// TF3-DLSS-DID-NOT-START.txt: written into the mod folder when the game loads the layer, kept up
// to date with the reason the upscaler isn't running, and deleted once an upscaled frame has been
// shown. If the game is closed (or crashes) before that, the file tells the player what happened.
// ---------------------------------------------------------------------------------------------
std::string g_statusReason;
bool g_statusCleared = false;

std::vector<std::wstring> StatusPaths() {
    return {RootDir() + TF3_SEP L"TF3-DLSS-DID-NOT-START.txt", TempDir() + TF3_SEP L"TF3-DLSS-DID-NOT-START.txt"};
}

void WriteStatus(const std::string& reason) {
    if (g_statusCleared) return;
    g_statusReason = reason;
    for (const auto& p : StatusPaths()) {
        FILE* f = OpenFile(p, L"w");
        if (!f) continue;
        std::fprintf(f,
                     "TF3-DLSS " TF3_DLSS_VERSION " was loaded by Transport Fever 3, but did not upscale a frame.\n\n"
                     "Reason: %s\n\n"
                     "This file is deleted automatically as soon as DLSS or FSR runs.\n"
                     "Details are in the log: %s\n"
                     "See README.txt (\"If something is wrong\") for help.\n",
                     reason.c_str(), g_logPath.empty() ? "(no log could be written)" : Narrow(g_logPath).c_str());
        std::fclose(f);
        return;  // written to the first location that works
    }
}

void ClearStatus() {
    if (g_statusCleared) return;
    g_statusCleared = true;
    for (const auto& p : StatusPaths()) {
        std::error_code ec;
        std::filesystem::remove(std::filesystem::path(p), ec);
    }
}

// =============================================================================================
// Config file (tf3_dlss.cfg)
// =============================================================================================
enum class UpscalerPref { Auto, Dlss, Fsr, MetalFx, Off };
const char* PrefName(UpscalerPref p) {
    switch (p) {
        case UpscalerPref::Dlss: return "dlss";
        case UpscalerPref::Fsr: return "fsr";
        case UpscalerPref::MetalFx: return "metalfx";
        case UpscalerPref::Off: return "off";
        default: return "auto";
    }
}

struct Config {
    int enabled = 1;            // start with the upscaler on
    UpscalerPref upscaler = UpscalerPref::Auto;  // auto = DLSS on NVIDIA RTX, else FSR 3.1
    int preset = 11;            // DLSS render preset (11 = K)
    int fsrColorSrgb = 1;       // FSR: the input colour is gamma-encoded (sRGB-like), not linear
    int fsrDebug = 0;           // FSR: 1 = input checks (messages in the log), 2 = also FSR's debug view
    float fsrJitterSignX = 1.f; // FSR's jitter sign, like jitter_sign_x/y for DLSS
    float fsrJitterSignY = 1.f;
    float mfxJitterSignX = 1.f;  // MetalFX's jitter sign (macOS). +1,+1 measured steadiest in the harness
    float mfxJitterSignY = 1.f;  // (still camera: 0.1-0.2 vs 0.9-2.6 levels of frame-to-frame change for -1,-1)
    int debugKeys = 0;          // developer hotkeys (tests, draw log, motion/jitter toggles, debug view)
    int motionVectors = 1;      // 1 = camera motion vectors, 0 = zero motion (milestone 3 behaviour)
    float mvScaleX = 1.f;       // debug: multiply motion vectors (use -1 to flip an axis)
    float mvScaleY = 1.f;
    float resetDistance = 1000; // camera jump (world units per frame) that resets DLSS history
    int mvDouble = 1;           // camera motion: 1 = one reprojection matrix built in double, 0 = float (old)
    int jitter = 1;             // sub-pixel camera jitter (needed for DLSS to resolve detail)
    int jitterPhases = 0;       // 0 = NVIDIA's recommendation for the current scale
    int jitterScreenPasses = 0; // debug: 1 = also jitter full-screen passes (reflections, AO, bloom), as before 0.2
    int jitterOtherTargets = 1; // jitter 3D draws into every render-size target, not just the main 3D pass
    int objectMotion = 1;       // motion vectors for moving objects (vehicles), not just the camera
    int objectDepthTest = 1;    // debug: 0 = draw object motion even where the object is hidden
    int fixHighlightDepth = 1;  // keep the hover highlight's overlay from writing depth (needs a restart)
    bool mipBiasAuto = true;    // texture LOD bias: log2(render/display), -1 at 50% (needs a restart)
    float mipBias = 0.f;        // used when mipBiasAuto is false (0 = off)
    float sharpness = 0.f;      // sharpening after DLSS, 0..1 (0 = off)
    int sharpenType = 0;        // 0 = RCAS (AMD FidelityFX FSR 1), 1 = the earlier CAS-style pass
    float motionSharpness = 0.4f;   // extra sharpness for moving pixels, reached at motionLimit (0 = off)
    float motionThreshold = 0.f;    // output pixels per frame below which nothing is added
    float motionLimit = 10.f;       // output pixels per frame where the full motionSharpness applies
    int sharpenDebug = 0;       // 1 = tint where motion changes the sharpening (red = more)
    int frameGen = 0;           // macOS: 1 = MetalFX frame interpolation, one generated frame per game frame
    int upscaleTo = 0;          // >0: the upscaler's image is at most this many lines; the game's blit stretches it
    float fsrSharpness = 0.5f;  // FSR's own sharpening (RCAS), 0..1 (0 = off)
    int overlay = 0;            // 1 = on-screen info box (upscaler, sizes, sharpness, jitter)
    float overlayX = 25.f;      // its top-left corner, percent of the screen from the left ...
    float overlayY = 1.f;       // ... and from the top
    float jitterSignX = 1.f;    // flip if a still image looks fuzzy/shaky (NVIDIA guide 8.4)
    float jitterSignY = 1.f;
    int dlssDev = 0;            // 1 = NVIDIA's developer DLSS build (debug overlays, Ctrl+Alt+F12)
    int debugView = 0;          // 1 = draw motion vectors instead of the DLSS image
    float debugScale = 0.05f;   // debug view colour change per pixel of motion
    float debugObjectScale = 0.05f;  // same, for object motion (pixels written by the object pass)
    int cfgLive = 1;            // re-read this file while the game runs
    int logEvery = 600;         // detailed log every N frames (0 = only the first frames)
    int perf = 0;               // 1 = measure what the layer costs (CPU and GPU), in the log and the info box
    float perfSeconds = 5.f;    // perf=1: averaging window, seconds
    int skipGameFsr = 0;        // don't draw the game's FSR 1 passes while DLSS/FSR 3.1 replaces their result
    // MetalFX reactive mask (macOS): how much MetalFX trusts the current frame over its history where the game
    // draws things without motion vectors, per kind of draw (0 = no mask for it). A kind that is 0 when the
    // game starts stays off until a restart (its pipelines get no replay copy).
    int reactive = 1;               // 0 = no reactive mask
    float reactiveParticles = 0.9f; // smoke, steam, dust and other particles, rain
    float reactiveAnimated = 0.6f;  // skinned meshes: people, animals
    float reactiveWater = 0.3f;     // water surfaces (waves and reflections move without motion vectors)
    float reactiveTransparent = 0.f;  // other blended 3D draws (glass, ...)
    int reactiveDebug = 0;          // 1 = tint the image red where the mask is set
    int fpsLimit = 0;               // macOS: lock the frame rate (30 = every frame 2 refreshes at 60 Hz); 0 = off
} g_cfg;

std::filesystem::path g_cfgPath;
std::filesystem::file_time_type g_cfgTime{};

std::string Trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// Returns true if the file was read.
bool LoadConfig() {
    if (g_cfgPath.empty()) {
        const char* env = std::getenv("TF3_DLSS_CFG");
        g_cfgPath = env && *env ? std::filesystem::path(env) : std::filesystem::path(RootDir()) / "tf3_dlss.cfg";
    }
    std::error_code ec;
    auto t = std::filesystem::last_write_time(g_cfgPath, ec);
    if (ec) {
        Log("[cfg] %s not found, using defaults", g_cfgPath.string().c_str());
        return false;
    }
    g_cfgTime = t;
    std::ifstream f(g_cfgPath);
    Config c;
    std::string line;
    while (std::getline(f, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = Trim(line.substr(0, eq)), v = Trim(line.substr(eq + 1));
        if (k.empty() || v.empty()) continue;
        float fv = (float)std::atof(v.c_str());
        int iv = (int)std::lround(fv);
        if (k == "enabled") c.enabled = iv;
        else if (k == "upscaler") {
            std::string lv = v;
            for (auto& ch : lv) ch = (char)std::tolower((unsigned char)ch);
            if (lv == "auto") c.upscaler = UpscalerPref::Auto;
            else if (lv == "dlss") c.upscaler = UpscalerPref::Dlss;
            else if (lv == "fsr" || lv == "fsr3") c.upscaler = UpscalerPref::Fsr;
            else if (lv == "metalfx" || lv == "mfx") c.upscaler = UpscalerPref::MetalFx;
            else if (lv == "off" || lv == "0") c.upscaler = UpscalerPref::Off;
            else Log("[cfg] upscaler=%s not understood (auto, dlss, fsr, metalfx or off); using auto", v.c_str());
        }
        else if (k == "fsr_color") {
            std::string lv = v;
            for (auto& ch : lv) ch = (char)std::tolower((unsigned char)ch);
            c.fsrColorSrgb = lv != "linear";
        }
        else if (k == "fsr_debug") c.fsrDebug = iv;
        else if (k == "fsr_jitter_sign_x") c.fsrJitterSignX = fv;
        else if (k == "fsr_jitter_sign_y") c.fsrJitterSignY = fv;
        else if (k == "debug_keys") c.debugKeys = iv;
        else if (k == "overlay") c.overlay = iv;
        else if (k == "overlay_x") c.overlayX = std::min(std::max(fv, 0.f), 100.f);
        else if (k == "overlay_y") c.overlayY = std::min(std::max(fv, 0.f), 100.f);
        else if (k == "preset") c.preset = iv;
        else if (k == "motion_vectors") c.motionVectors = iv;
        else if (k == "mv_scale_x") c.mvScaleX = fv;
        else if (k == "mv_scale_y") c.mvScaleY = fv;
        else if (k == "reset_distance") c.resetDistance = fv;
        else if (k == "mv_double") c.mvDouble = iv;
        else if (k == "cfg_live") c.cfgLive = iv;
        else if (k == "debug_view") c.debugView = iv;
        else if (k == "jitter") c.jitter = iv;
        else if (k == "jitter_phases") c.jitterPhases = iv;
        else if (k == "jitter_screen_passes") c.jitterScreenPasses = iv;
        else if (k == "jitter_other_targets") c.jitterOtherTargets = iv;
        else if (k == "jitter_mode" || k == "debug_serialize") {}  // retired settings
        else if (k == "object_motion") c.objectMotion = iv;
        else if (k == "object_depth_test") c.objectDepthTest = iv;
        else if (k == "fix_highlight_depth") c.fixHighlightDepth = iv;
        else if (k == "sharpness") c.sharpness = std::min(std::max(fv, 0.f), 1.f);
        else if (k == "sharpen_type") {
            std::string lv = v;
            for (auto& ch : lv) ch = (char)std::tolower((unsigned char)ch);
            if (lv == "rcas") c.sharpenType = 0;
            else if (lv == "cas") c.sharpenType = 1;
            else Log("[cfg] sharpen_type=%s not understood (rcas or cas); using rcas", v.c_str());
        }
        else if (k == "motion_sharpness") c.motionSharpness = std::min(std::max(fv, -1.f), 1.f);
        else if (k == "motion_threshold") c.motionThreshold = std::max(fv, 0.f);
        else if (k == "motion_limit") c.motionLimit = std::max(fv, 0.f);
        else if (k == "sharpen_debug") c.sharpenDebug = iv;
        else if (k == "fsr_sharpness") c.fsrSharpness = std::min(std::max(fv, 0.f), 1.f);
        else if (k == "mip_bias") {
            c.mipBiasAuto = v == "auto";
            c.mipBias = c.mipBiasAuto ? 0.f : fv;
        }
        else if (k == "jitter_sign_x") c.jitterSignX = fv;
        else if (k == "mfx_jitter_sign_x") c.mfxJitterSignX = fv;
        else if (k == "mfx_jitter_sign_y") c.mfxJitterSignY = fv;
        else if (k == "jitter_sign_y") c.jitterSignY = fv;
        else if (k == "dlss_dev") c.dlssDev = iv;
        else if (k == "debug_scale") c.debugScale = fv;
        else if (k == "debug_object_scale") c.debugObjectScale = fv;
        else if (k == "log_every") c.logEvery = iv;
        else if (k == "perf") c.perf = iv;
        else if (k == "perf_seconds") c.perfSeconds = std::min(std::max(fv, 1.f), 60.f);
        else if (k == "skip_game_fsr") c.skipGameFsr = iv;
        else if (k == "frame_gen") c.frameGen = iv;
        else if (k == "upscale_to") c.upscaleTo = std::max(iv, 0);
        else if (k == "reactive") c.reactive = iv;
        else if (k == "reactive_particles") c.reactiveParticles = std::min(std::max(fv, 0.f), 1.f);
        else if (k == "reactive_animated") c.reactiveAnimated = std::min(std::max(fv, 0.f), 1.f);
        else if (k == "reactive_water") c.reactiveWater = std::min(std::max(fv, 0.f), 1.f);
        else if (k == "reactive_transparent") c.reactiveTransparent = std::min(std::max(fv, 0.f), 1.f);
        else if (k == "reactive_debug") c.reactiveDebug = iv;
        else if (k == "fps_limit") c.fpsLimit = std::max(iv, 0);
        else Log("[cfg] unknown setting '%s' ignored", k.c_str());
    }
    g_cfg = c;
    Log("[cfg] loaded %s: enabled=%d upscaler=%s preset=%d motion_vectors=%d object_motion=%d jitter=%d (phases %d, "
        "sign %g,%g, fsr sign %g,%g) sharpness=%g fsr_sharpness=%g mip_bias=%s%g fsr_color=%s fsr_debug=%d overlay=%d debug_keys=%d "
        "dlss_dev=%d mv_scale=(%g,%g) reset_distance=%g mv_double=%d debug_view=%d cfg_live=%d log_every=%d perf=%d "
        "perf_seconds=%g skip_game_fsr=%d sharpen_type=%s motion_sharpness=%g motion_threshold=%g motion_limit=%g "
        "sharpen_debug=%d frame_gen=%d upscale_to=%d reactive=%d (particles %g, animated %g, water %g, "
        "transparent %g, debug %d) fps_limit=%d",
        g_cfgPath.string().c_str(), c.enabled, PrefName(c.upscaler), c.preset, c.motionVectors, c.objectMotion,
        c.jitter, c.jitterPhases, c.jitterSignX, c.jitterSignY, c.fsrJitterSignX, c.fsrJitterSignY, c.sharpness, c.fsrSharpness,
        c.mipBiasAuto ? "auto " : "", c.mipBias, c.fsrColorSrgb ? "srgb" : "linear", c.fsrDebug, c.overlay,
        c.debugKeys, c.dlssDev, c.mvScaleX, c.mvScaleY, c.resetDistance, c.mvDouble, c.debugView, c.cfgLive, c.logEvery,
        c.perf, c.perfSeconds, c.skipGameFsr, c.sharpenType == 1 ? "cas" : "rcas", c.motionSharpness,
        c.motionThreshold, c.motionLimit, c.sharpenDebug, c.frameGen, c.upscaleTo, c.reactive, c.reactiveParticles,
        c.reactiveAnimated, c.reactiveWater, c.reactiveTransparent, c.reactiveDebug, c.fpsLimit);
    if (c.motionLimit <= c.motionThreshold)
        Log("[cfg] motion_limit (%g) should be above motion_threshold (%g)", c.motionLimit, c.motionThreshold);
    return true;
}

// True if the file changed on disk since it was last read.
bool ConfigChanged() {
    std::error_code ec;
    auto t = std::filesystem::last_write_time(g_cfgPath, ec);
    return !ec && t != g_cfgTime;
}

// =============================================================================================
// Dispatch bookkeeping. Dispatchable children (physical devices, queues, command buffers) share
// their parent's loader dispatch pointer, so that pointer is the map key.
// =============================================================================================
template <typename T>
void* Key(T handle) { return *reinterpret_cast<void**>(handle); }

struct InstanceData {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
    PFN_vkDestroyInstance DestroyInstance = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2 = nullptr;
    uint32_t apiVersion = VK_API_VERSION_1_0;  // what the game asked for
};

// Functions we intercept (hooked) ...
#define DEVICE_HOOKS(X)                       \
    X(DestroyDevice)                          \
    X(CreateShaderModule)                     \
    X(DestroyShaderModule)                    \
    X(CreateGraphicsPipelines)                \
    X(DestroyPipeline)                        \
    X(CreateImage)                            \
    X(DestroyImage)                           \
    X(CreateImageView)                        \
    X(CreateFramebuffer)                      \
    X(CreateRenderPass)                       \
    X(CreateRenderPass2)                      \
    X(CreateRenderPass2KHR)                   \
    X(CreateSwapchainKHR)                     \
    X(BindBufferMemory)                       \
    X(BindBufferMemory2)                      \
    X(BindBufferMemory2KHR)                   \
    X(MapMemory)                              \
    X(UnmapMemory)                            \
    X(CreateDescriptorSetLayout)              \
    X(AllocateDescriptorSets)                 \
    X(UpdateDescriptorSets)                   \
    X(CreateDescriptorUpdateTemplate)         \
    X(CreateDescriptorUpdateTemplateKHR)      \
    X(UpdateDescriptorSetWithTemplate)        \
    X(UpdateDescriptorSetWithTemplateKHR)     \
    X(BeginCommandBuffer)                     \
    X(CmdBindPipeline)                        \
    X(CmdBindDescriptorSets)                  \
    X(CmdPipelineBarrier)                     \
    X(CmdBeginRenderPass)                     \
    X(CmdBeginRenderPass2)                    \
    X(CmdBeginRenderPass2KHR)                 \
    X(CmdEndRenderPass)                       \
    X(CmdEndRenderPass2)                      \
    X(CmdEndRenderPass2KHR)                   \
    X(CmdDraw)                                \
    X(CmdDrawIndexed)                         \
    X(CmdBindVertexBuffers)                   \
    X(CmdBindIndexBuffer)                     \
    X(CmdSetViewport)                         \
    X(CmdSetScissor)                          \
    X(CmdPushConstants)                       \
    X(CreateSampler)                          \
    X(QueueSubmit)                            \
    X(QueuePresentKHR)                        \
    X(GetDeviceQueue)                         \
    X(GetDeviceQueue2)                        \
    X(DestroySwapchainKHR)

// ... and functions we only call ourselves.
#define DEVICE_CALLS(X)            \
    X(GetImageMemoryRequirements)  \
    X(AllocateMemory)              \
    X(BindImageMemory)             \
    X(CreateDescriptorPool)        \
    X(CmdClearColorImage)          \
    X(DeviceWaitIdle)              \
    X(CreateBuffer)                \
    X(GetBufferMemoryRequirements) \
    X(CreatePipelineLayout)        \
    X(CreateComputePipelines)      \
    X(CmdDispatch)                 \
    X(CmdCopyBufferToImage)        \
    X(CmdSetBlendConstants)        \
    X(CmdCopyBuffer)               \
    X(CmdCopyImageToBuffer)        \
    X(QueueWaitIdle)               \
    X(DestroyFramebuffer)          \
    X(CmdCopyImage)                \
    X(ResetDescriptorPool)         \
    X(DestroyImageView)            \
    X(CreateQueryPool)             \
    X(DestroyQueryPool)            \
    X(CmdResetQueryPool)           \
    X(CmdWriteTimestamp)           \
    X(GetQueryPoolResults)

struct DeviceData {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    InstanceData* instance = nullptr;
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr = nullptr;
#define X(name) PFN_vk##name name = nullptr;
    DEVICE_HOOKS(X)
    DEVICE_CALLS(X)
#undef X
};

std::mutex g_lock;
std::unordered_map<void*, InstanceData> g_instances;
std::unordered_map<void*, DeviceData> g_devices;
std::unordered_map<VkQueue, uint32_t> g_queueFamily;  // from vkGetDeviceQueue(2), under g_lock

InstanceData* GetInstance(void* key) {
    std::lock_guard<std::mutex> lk(g_lock);
    auto it = g_instances.find(key);
    return it == g_instances.end() ? nullptr : &it->second;
}
// Every hooked call looks its device up, so each thread remembers the last answer; g_deviceGen
// (bumped when a device is created or destroyed) tells it when that answer may be stale.
std::atomic<uint32_t> g_deviceGen{0};
DeviceData* GetDevice(void* key) {
    thread_local void* lastKey = nullptr;
    thread_local DeviceData* last = nullptr;
    thread_local uint32_t lastGen = UINT32_MAX;
    uint32_t gen = g_deviceGen.load(std::memory_order_acquire);
    if (key == lastKey && gen == lastGen) return last;
    std::lock_guard<std::mutex> lk(g_lock);
    auto it = g_devices.find(key);
    if (it == g_devices.end()) return nullptr;
    lastKey = key;
    last = &it->second;
    lastGen = gen;
    return last;
}

// =============================================================================================
// Shader identification
// =============================================================================================
enum class Kind { Other, FsrEasu, FsrRcas, Compose };

const char* KindName(Kind k) {
    switch (k) {
        case Kind::FsrEasu: return "FSR_EASU";
        case Kind::FsrRcas: return "FSR_RCAS";
        case Kind::Compose: return "TONEMAP";
        default: return "other";
    }
}

// The SPIR-V keeps debug names and the text of included headers, but not the main file's source.
// These identifiers are declared only in each main file (checked against the shared headers):
//   fsr_pass0.fs / fsr_pass1.fs : UBO member "fsrSharpening"; only pass1 defines prng()
//   hdr/compose.fs              : legacyToneMapAndGammaAndContrastBump() / member "applyBloom"
Kind Classify(const void* code, size_t size) {
    std::string_view s(static_cast<const char*>(code), size);
    auto has = [&](const char* t) { return s.find(t) != std::string_view::npos; };
    if (has("fsrSharpening")) return has("prng") ? Kind::FsrRcas : Kind::FsrEasu;
    if (has("legacyToneMapAndGammaAndContrastBump") || has("applyBloom")) return Kind::Compose;
    return Kind::Other;
}

// =============================================================================================
// Resource tracking
// =============================================================================================
struct ImageInfo {
    uint32_t w = 0, h = 0;
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;  // MSAA render targets (TF3's MSAA setting)
};

struct DescItem {
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    VkImageView view = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0, range = 0;
};

struct LayoutBinding {
    uint32_t binding;
    VkDescriptorType type;
    uint32_t count;
};

struct TemplateEntry {
    uint32_t binding, arrayElement, count;
    VkDescriptorType type;
    size_t offset, stride;
};

struct BufferInfo {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize memOffset = 0;
};

struct Mapping {
    char* ptr = nullptr;
    VkDeviceSize offset = 0;
    VkDeviceSize size = VK_WHOLE_SIZE;
};

struct BoundSet {
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::vector<uint32_t> dynOffsets;                   // raw dynamic offsets for this set (for re-binding)
    std::vector<std::pair<uint32_t, uint32_t>> dyn;     // binding -> dynamic offset (few; reused per bind)
};

// ---- Object probe (milestone 6, per-object motion vectors: feasibility check) ----------------
// Ordinary models take their world transform from a per-instance vertex attribute
// (mat4 instAttrModel, shaders/mat/vs/std/*.vs); animated ones use bone matrices in a UBO
// (set 3 binding 11, shaders/mat/vs/skinning/util.vs). The probe reads those per-instance
// matrices on two consecutive frames and logs whether objects can be followed between frames.
struct VsInfo {
    int mdlLoc = -1, posLoc = -1;
    bool skinned = false;
    std::string tag;  // its attr*/instAttr* input names, to recognise it (draw log, highlight fix)
    // Inputs whose values reach gl_Position (from the SPIR-V data flow), as {location, name}.
    bool traced = false;
    std::vector<std::pair<int, std::string>> posInputs;
};
struct PipeVtx {
    bool hasMdl = false, instanceRate = false, skinned = false;
    uint32_t mdlBinding = 0, stride = 0, cols = 0, comps = 0;
    uint32_t colOffset[4] = {};
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    uint32_t posBinding = UINT32_MAX;
    VkFormat posFormat = VK_FORMAT_UNDEFINED;
    uint32_t posOffset = 0, posStride = 0;
    bool posInstanceRate = false;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool primitiveRestart = false;
};
// One game draw of a (possibly) moving mesh, replayed by the object motion pass.
struct ObjDraw {
    PipeVtx layout;                 // vertex layout of the game's pipeline
    VkBuffer posBuf = VK_NULL_HANDLE, mdlBuf = VK_NULL_HANDLE, ib = VK_NULL_HANDLE;
    VkDeviceSize posOff = 0, mdlOff = 0, ibOff = 0;  // mdlOff already includes firstInstance
    VkIndexType ibType = VK_INDEX_TYPE_UINT16;
    bool indexed = false;
    uint32_t count = 0, first = 0, instances = 0;
    int32_t vertexOffset = 0;
    uint64_t mesh = 0;
    uint32_t dst = 0;               // first entry in our previous-matrix buffer (assigned when recorded)
};
struct ObjProbe {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceSize base = 0;  // buffer offset of instance 0 (bound offset)
    uint32_t stride = 0, first = 0, count = 0, cols = 0, comps = 0;
    uint32_t colOffset[4] = {};
    uint64_t mesh = 0;  // identifies the mesh: vertex/index buffer and range
    bool skinned = false;
};

constexpr uint8_t kRcNoneValue = 0;  // PipeInfo::react: not a reactive-mask draw (ReactClass, macOS)
struct PipeInfo;
struct CmdState {
    Kind kind = Kind::Other;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const PipeInfo* pipe = nullptr;  // what the layer knows about `pipeline` (null: nothing)
    std::vector<BoundSet> sets;
    VkFramebuffer fb = VK_NULL_HANDLE;
    VkRenderPass rp = VK_NULL_HANDLE;
    bool fbIsScene = false;  // current framebuffer has a full-size, non-square depth buffer (main camera)
    bool jitterSet = false;  // jitter chosen for this frame (when DLSS was recorded)
    bool capture = false;    // this frame's MV/depth/colour are being copied out for analysis
    bool capReactive = false;  // ... and the reactive mask (macOS)
    std::string capPhase;
    float jx = 0.f, jy = 0.f;  // jitter in render pixels
    uint32_t jitterIdx = 0, jitterPhases = 0;
    std::unordered_map<VkImage, VkImageLayout> layouts;  // layouts as recorded in this command buffer
    // camera UBO
    bool hasView = false;
    VkBuffer viewBuf = VK_NULL_HANDLE;
    VkDeviceSize viewOff = 0, viewRange = 0;
    // DLSS inputs gathered while recording this frame
    VkImageView colorView = VK_NULL_HANDLE;  // EASU input
    VkImage colorImage = VK_NULL_HANDLE;
    VkImage depthImage = VK_NULL_HANDLE;     // depth attachment of the tonemap pass = main depth
    VkImage rcasOut = VK_NULL_HANDLE;
    bool rcasInThisPass = false;
    bool dlssDone = false;
    // MetalFX (macOS): this command buffer signals/waits ring slot mfxSlot; the Metal work is committed at submit
    bool mfxPending = false;
    int mfxSlot = 0;
    float mfxJx = 0, mfxJy = 0;
    bool mfxReset = false;
    uint32_t mfxInW = 0, mfxInH = 0;
    // motion vectors: the matrices for this frame are uploaded at submit into ring slot mvSlot
    bool mvPending = false;
    uint32_t mvSlot = 0;
    // object probe (milestone 6 feasibility): bound vertex/index buffers and the scene draws seen
    std::pair<VkBuffer, VkDeviceSize> vb[16] = {};
    VkBuffer ib = VK_NULL_HANDLE;
    VkDeviceSize ibOff = 0;
    std::vector<struct ObjProbe> probes;
    uint32_t fbW = 0, fbH = 0;  // size of the current framebuffer
    // viewport jitter: the game's viewport and the one currently set on the command buffer
    bool hasGameVp = false, hasCurVp = false;
    VkViewport gameVp{}, curVp{};
    bool hasScissor = false;
    VkRect2D scissor{};
    VkIndexType ibType = VK_INDEX_TYPE_UINT16;
    // object motion pass recorded in this command buffer; previous matrices are filled at submit
    std::vector<ObjDraw> objDraws;
    uint32_t objSlot = 0;
    bool objPending = false;
    // the blit search (OnDraw) only runs again when a descriptor set was bound since the last one
    uint32_t setsGen = 0, blitGen = UINT32_MAX;
    std::pair<int, int> blitHit{-1, -1};
    uint32_t viewGen = UINT32_MAX;  // setsGen when the camera buffer was last looked for (CheckForView)
    // the scene viewport seen in this command buffer, handed to g_dlss when its render pass ends
    bool hasSceneVp = false;
    VkViewport sceneVp{};
    std::shared_ptr<const std::unordered_set<uint64_t>> moving;  // g_movingSnap when the pass began
    // push constants as the game last set them (the reactive pass replays draws with them)
    struct PushRange { VkShaderStageFlags stages; uint32_t offset, size; };
    uint8_t push[256] = {};
    PushRange pushRanges[4] = {};
    uint32_t pushCount = 0;
    VkPipelineLayout pushLayout = VK_NULL_HANDLE;
};

// Render state of a pipeline, for the scene draw log (Ctrl+Shift+L).
struct PipeState {
    uint32_t colorMask = 0;      // OR of all attachments' write masks (0 = writes no colour)
    uint32_t attachments = 0;
    bool blend = false;
    bool depthTest = false, depthWrite = false, stencil = false;
    VkCompareOp depthOp = VK_COMPARE_OP_ALWAYS;
    VkStencilOp stencilPass = VK_STENCIL_OP_KEEP;
    VkCullModeFlags cull = 0;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    std::string vsNames;         // a few identifiers from the vertex shader, to recognise it
    std::string fsNames;         // the fragment shader's textures/uniforms
};
struct PipeViewport {
    bool dynamic = false;  // viewport is dynamic state (originally, or because we made it so)
    bool patched = false;  // we made it dynamic: the game never sets it, we set staticVp
    VkViewport staticVp{};
};

struct RenderPassInfo {
    std::vector<VkImageLayout> finalLayouts;
};

// What the draw hooks need to know about a pipeline; fixed once the pipeline is created.
struct PipeInfo {
    Kind kind = Kind::Other;
    PipeViewport vp;
    bool hasVtx = false;
    PipeVtx vtx;
    bool screenPass = false;  // no depth test: a full-screen pass (not jittered by default)
    uint8_t react = 0;                      // ReactClass of a 3D draw without motion vectors (macOS)
    VkPipeline reactPipe = VK_NULL_HANDLE;  // its copy for the reactive mask pass
};

// Guards everything below. Recursive: the FSR runtime may record through the loader (and so
// through our own hooks) while we hold it inside RecordUpscale.
std::recursive_mutex g_t;

// ---------------------------------------------------------------------------------------------
// Performance measurement (perf=1 in tf3_dlss.cfg), CPU side. Time spent in the layer's own code,
// by kind of call, read with the CPU's time-stamp counter (a few nanoseconds per reading). The
// driver's time isn't included: each scope ends before the call is passed on. Only the outermost
// scope on a thread counts (DLSS records through our hooks while we record it). The GPU side and
// the report are further down (GpuPerf, PerfPresent).
// ---------------------------------------------------------------------------------------------
enum PerfCat { kPerfDraw, kPerfBind, kPerfState, kPerfPass, kPerfDesc, kPerfUpscale, kPerfSubmit, kPerfPresent,
               kPerfCreate, kPerfCount };
const char* const kPerfNames[kPerfCount] = {"draws", "descriptor binds", "other commands", "render passes",
                                            "descriptor updates", "upscaler recording", "submit", "present",
                                            "shader/pipeline creation"};
struct PerfCounters {
    std::atomic<uint64_t> ticks[kPerfCount];
    std::atomic<uint64_t> calls[kPerfCount];
    std::atomic<uint64_t> lockWaits, lockWaitTicks;  // a thread had to wait for another one holding g_t
} g_perf;  // zero-initialised (global)
thread_local int t_perfDepth = 0;

struct PerfScope {
    PerfCat cat;
    uint64_t t0 = 0;
    bool active = false;
    explicit PerfScope(PerfCat c) : cat(c) {
        if (!g_cfg.perf) return;
        active = true;
        if (t_perfDepth++ == 0) t0 = TF3_TICKS();
    }
    ~PerfScope() {
        if (!active) return;
        --t_perfDepth;
        if (!t0) return;
        g_perf.ticks[cat].fetch_add(TF3_TICKS() - t0, std::memory_order_relaxed);
        g_perf.calls[cat].fetch_add(1, std::memory_order_relaxed);
    }
    PerfScope(const PerfScope&) = delete;
    PerfScope& operator=(const PerfScope&) = delete;
};

// Holds g_t. With perf=1 it also counts the times a thread had to wait for another one.
struct TLock {
    std::recursive_mutex& m;
    explicit TLock(std::recursive_mutex& mu) : m(mu) {
        if (m.try_lock()) return;
        uint64_t t0 = g_cfg.perf ? TF3_TICKS() : 0;
        m.lock();
        if (t0) {
            g_perf.lockWaits.fetch_add(1, std::memory_order_relaxed);
            g_perf.lockWaitTicks.fetch_add(TF3_TICKS() - t0, std::memory_order_relaxed);
        }
    }
    ~TLock() { m.unlock(); }
    TLock(const TLock&) = delete;
    TLock& operator=(const TLock&) = delete;
};
std::unordered_map<VkShaderModule, Kind> g_moduleKind;
std::unordered_map<VkPipeline, Kind> g_pipelineKind;
std::unordered_map<VkShaderModule, VsInfo> g_vsInfo;
std::unordered_map<VkPipeline, PipeVtx> g_pipeVtx;
std::unordered_map<VkPipeline, PipeViewport> g_pipeVp;
std::unordered_map<VkPipeline, PipeState> g_pipeState;
std::unordered_map<VkShaderModule, std::string> g_moduleTag;  // short description of each vertex shader
std::unordered_map<VkShaderModule, std::string> g_moduleRes;  // its textures/uniforms (ShaderResourceNames)
std::unordered_map<VkImage, ImageInfo> g_images;
std::unordered_map<VkImageView, VkImage> g_viewImage;
std::unordered_map<VkFramebuffer, std::vector<VkImageView>> g_fbViews;
std::unordered_map<VkRenderPass, RenderPassInfo> g_renderPasses;
std::unordered_map<VkImage, VkImageLayout> g_layout;  // last layout recorded anywhere
std::unordered_map<VkDescriptorSetLayout, std::vector<LayoutBinding>> g_setLayouts;
std::unordered_set<VkDescriptorSetLayout> g_updateAfterBindLayouts;  // their sets need an update-after-bind pool
std::unordered_map<VkDescriptorSet, VkDescriptorSetLayout> g_setLayoutOf;
std::unordered_map<VkDescriptorSet, std::map<uint32_t, std::vector<DescItem>>> g_setContents;
std::unordered_map<VkDescriptorUpdateTemplate, std::vector<TemplateEntry>> g_templates;
std::unordered_map<VkBuffer, BufferInfo> g_buffers;
std::unordered_map<VkDeviceMemory, Mapping> g_mapped;
std::unordered_map<VkCommandBuffer, CmdState> g_cmd;  // entries are never erased, so pointers to them stay valid

// The state of a command buffer. A thread records into one command buffer at a time, so it keeps the
// last one it looked up. Caller holds g_t.
// A command buffer is recorded by one thread at a time, so its CmdState needs no lock while it is being
// recorded; only finding it in g_cmd does (on a miss of this thread's small cache).
CmdState& CmdFor(VkCommandBuffer cb) {
    thread_local VkCommandBuffer cbs[4] = {};
    thread_local CmdState* states[4] = {};
    thread_local uint32_t next = 0;
    for (uint32_t i = 0; i < 4; ++i)
        if (cbs[i] == cb && states[i]) return *states[i];
    CmdState* s;
    {
        TLock lk(g_t);
        s = &g_cmd[cb];
    }
    cbs[next] = cb;
    states[next] = s;
    next = (next + 1) % 4;
    return *s;
}

// ---- Read by the per-command hooks without g_t ---------------------------------------------------
// The game records a frame on several threads. Taking g_t in every draw, bind and state call made those
// threads queue behind each other (thousands of waits per frame, several ms). Those hooks now work on
// their own command buffer's state, on the pipeline information below and on per-frame values that are
// written under g_t (at present) and read here as atomics; they take g_t only for the few draws that
// need more: the passes the layer replaces, moving objects, the first draws looking for the camera.
std::shared_mutex g_pipeInfoLock;  // guards g_pipeInfo (written under g_t as well)
std::unordered_map<VkPipeline, std::unique_ptr<PipeInfo>> g_pipeInfo;
std::atomic<uint32_t> g_pipeInfoGen{0};  // bumped when a pipeline is destroyed: per-thread caches start over

const PipeInfo* FindPipeInfo(VkPipeline p) {
    thread_local std::unordered_map<VkPipeline, const PipeInfo*> cache;
    thread_local uint32_t cacheGen = UINT32_MAX;
    uint32_t gen = g_pipeInfoGen.load(std::memory_order_acquire);
    if (gen != cacheGen) {
        cache.clear();
        cacheGen = gen;
    }
    auto c = cache.find(p);
    if (c != cache.end()) return c->second;
    const PipeInfo* info = nullptr;
    {
        std::shared_lock<std::shared_mutex> lk(g_pipeInfoLock);
        auto it = g_pipeInfo.find(p);
        if (it != g_pipeInfo.end()) info = it->second.get();
    }
    if (info) cache.emplace(p, info);  // (not cached while unknown: it may be registered a moment later)
    return info;
}

// Guards g_setLayoutOf and g_setLayouts for the descriptor-set bind hook (written under g_t as well).
std::shared_mutex g_descLock;

struct HotState {
    std::atomic<uint64_t> frame{0};            // g_frame
    std::atomic<bool> drawLog{false};          // the draw log (Ctrl+Shift+L) covers this frame
    std::atomic<bool> probe{false};            // ProbeFrame(g_frame): read the model matrices this frame
    std::atomic<bool> objects{false};          // object motion on and some meshes known to move
    std::atomic<bool> jitter{false};           // ViewportJitterOn()
    std::atomic<bool> jitterScreen{false};     // jitter_screen_passes
    std::atomic<bool> jitterOther{false};      // jitter_other_targets
    std::atomic<bool> react{false};            // reactive mask on: remember the draws that feed it
    std::atomic<uint32_t> inW{0}, inH{0};      // render size
    std::atomic<uint64_t> jitterFrame{UINT64_MAX};  // the frame jitterXY belongs to
    std::atomic<uint64_t> jitterXY{0};         // that frame's jitter, two floats
} g_hot;
// Meshes that moved recently (g_movingMeshes), rebuilt at present; a render pass takes a reference.
std::shared_ptr<const std::unordered_set<uint64_t>> g_movingSnap;
VkExtent2D g_swapExtent = {0, 0};
uint32_t g_sceneRenderW = 0;  // width of the scene depth buffer = render width

// The size the upscaler writes for an inW x inH render: the screen, or less when upscale_to asks for fewer lines
// or when MetalFX can't upscale that far (more than 3x). The game's blit samples the image with normalised
// coordinates and our linear sampler, so it stretches a smaller image over the screen, and the UI it draws next
// stays at the screen's resolution. Never smaller than the render size. Caller holds g_t.
VkExtent2D UpscaleExtent(uint32_t inW, uint32_t inH, bool metalFx) {
    VkExtent2D s = g_swapExtent;
    if (!s.width || !s.height || !inW || !inH) return s;
    double h = s.height;
    if (g_cfg.upscaleTo > 0) h = std::min(h, (double)g_cfg.upscaleTo);
#ifdef __APPLE__
    if (metalFx) {
        double k = mfx_max_scale();  // per axis
        h = std::min(h, std::floor(inH * k));
        h = std::min(h, std::floor(inW * k) * s.height / s.width);
    }
#else
    (void)metalFx;
#endif
    h = std::max(h, (double)inH);
    if (h >= s.height) return s;
    uint32_t w = (uint32_t)std::lround(h * s.width / s.height);
    return {std::max(w, inW), (uint32_t)h};
}

uint64_t g_frame = 0;
uint64_t g_firstFsrFrame = 0;
uint64_t g_lastViewLogFrame = UINT64_MAX;
uint64_t g_lastBlitLogFrame = UINT64_MAX;
uint64_t g_lastDlssFrame = UINT64_MAX;
uint64_t g_lastSwapFrame = UINT64_MAX;

bool LogFrame(uint64_t f) {
    if (g_firstFsrFrame && f >= g_firstFsrFrame && f < g_firstFsrFrame + 3) return true;
    return g_firstFsrFrame && g_cfg.logEvery > 0 && f % (uint64_t)g_cfg.logEvery == 0;
}

bool IsDepthFormat(VkFormat f) {
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT ||
           f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D16_UNORM_S8_UINT;
}
bool HasStencil(VkFormat f) {
    return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D16_UNORM_S8_UINT;
}
bool IsImageType(VkDescriptorType t) {
    return t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
           t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
}
bool IsBufferType(VkDescriptorType t) {
    return t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
           t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
}
bool IsDynamic(VkDescriptorType t) {
    return t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
}

const char* FormatName(VkFormat f) {
    switch (f) {
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return "R11G11B10_FLOAT";
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return "B10G10R10A2_UNORM";
        case VK_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case VK_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
        case VK_FORMAT_D24_UNORM_S8_UINT: return "D24S8";
        case VK_FORMAT_D32_SFLOAT: return "D32F";
        case VK_FORMAT_D16_UNORM: return "D16";
        case VK_FORMAT_R8_UNORM: return "R8_UNORM";
        case VK_FORMAT_R16G16B16A16_SFLOAT: return "R16G16B16A16_FLOAT";
        case VK_FORMAT_R16G16_SFLOAT: return "R16G16_FLOAT";
        default: return nullptr;
    }
}

std::string ImageDesc(VkImage img) {  // caller holds g_t
    auto it = g_images.find(img);
    if (it == g_images.end()) return H(img) + " (untracked)";
    char b[160];
    const char* fn = FormatName(it->second.fmt);
    std::snprintf(b, sizeof b, "%s %ux%u %s%s usage=0x%x%s", H(img).c_str(), it->second.w, it->second.h,
                  fn ? fn : "fmt#", fn ? "" : std::to_string((int)it->second.fmt).c_str(), (unsigned)it->second.usage,
                  it->second.samples > 1 ? (" samples=" + std::to_string((int)it->second.samples)).c_str() : "");
    return b;
}
VkImage ViewImage(VkImageView v) {
    auto it = g_viewImage.find(v);
    return it == g_viewImage.end() ? VK_NULL_HANDLE : it->second;
}
std::vector<VkImage> FbImages(VkFramebuffer fb) {
    std::vector<VkImage> out;
    auto it = g_fbViews.find(fb);
    if (it != g_fbViews.end())
        for (VkImageView v : it->second) out.push_back(ViewImage(v));
    return out;
}
void StoreDesc(VkDescriptorSet set, uint32_t binding, uint32_t element, const DescItem& d) {
    auto& v = g_setContents[set][binding];
    if (v.size() <= element) v.resize(element + 1);
    v[element] = d;
}
VkImageLayout CurrentLayout(const CmdState& cs, VkImage img) {
    auto a = cs.layouts.find(img);
    if (a != cs.layouts.end()) return a->second;
    auto b = g_layout.find(img);
    return b == g_layout.end() ? VK_IMAGE_LAYOUT_MAX_ENUM : b->second;
}
void SetLayout(CmdState& cs, VkImage img, VkImageLayout l) {
    cs.layouts[img] = l;
    g_layout[img] = l;
}

// =============================================================================================
// Camera UBO (std140 layout of "View" in shaders/pbr/view.h.glsl)
// =============================================================================================
constexpr size_t kViewSize = 464;
namespace ViewOff {
constexpr size_t projView = 0, projViewInverse = 64, proj = 128, view = 256, camPos = 384, near_ = 416,
                 far_ = 420, width = 428, height = 432, fov = 456;
}

// This frame's scene camera, for MetalFX frame interpolation (taken when the motion vectors are uploaded).
struct FrameCamera {
    uint64_t frame = UINT64_MAX;
    float nearPlane = 0.f, farPlane = 0.f, fovYDegrees = 0.f, aspect = 0.f;
} g_frameCam;

void MatMul(const float* a, const float* b, float* out) {  // column-major: out = a * b
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = s;
        }
}
float MaxDiff(const float* a, const float* b) {
    float m = 0;
    for (int i = 0; i < 16; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

void LogViewFromMemory(const CmdState& cs) {  // caller holds g_t
    auto bi = g_buffers.find(cs.viewBuf);
    if (bi == g_buffers.end()) return;
    auto mi = g_mapped.find(bi->second.memory);
    if (mi == g_mapped.end()) {
        Log("[view] camera UBO memory is not host-mapped");
        return;
    }
    VkDeviceSize at = bi->second.memOffset + cs.viewOff;
    if (at < mi->second.offset) return;
    const char* p = mi->second.ptr + (at - mi->second.offset);
    float pv[16], pvi[16], proj[16], view[16], tmp[16], ident[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::memcpy(pv, p + ViewOff::projView, 64);
    std::memcpy(pvi, p + ViewOff::projViewInverse, 64);
    std::memcpy(proj, p + ViewOff::proj, 64);
    std::memcpy(view, p + ViewOff::view, 64);
    float cam[3], nearF, farF, fov;
    int32_t width, height;
    std::memcpy(cam, p + ViewOff::camPos, 12);
    std::memcpy(&nearF, p + ViewOff::near_, 4);
    std::memcpy(&farF, p + ViewOff::far_, 4);
    std::memcpy(&width, p + ViewOff::width, 4);
    std::memcpy(&height, p + ViewOff::height, 4);
    std::memcpy(&fov, p + ViewOff::fov, 4);
    MatMul(proj, view, tmp);
    float e1 = MaxDiff(tmp, pv);
    MatMul(pv, pvi, tmp);
    float e2 = MaxDiff(tmp, ident);
    Log("[view] frame %llu: %dx%d near=%.3g far=%.3g fov=%.4g cam=(%.1f, %.1f, %.1f) %s",
        (unsigned long long)g_frame, width, height, nearF, farF, fov, cam[0], cam[1], cam[2],
        (e1 < 1e-2f && e2 < 1e-2f) ? "LOOKS RIGHT" : "MISMATCH");
}

// Camera UBO = set 0, binding 10, >= kViewSize bytes, taken from a draw into the main scene
// framebuffer. (Other passes - shadow cascades, environment cubemaps - use their own cameras.)
void CheckForView(CmdState& cs) {
    if (cs.hasView || cs.sets.empty() || !cs.fbIsScene) return;
    cs.viewGen = cs.setsGen;  // BeforeDraw looks again only after another bind
    const BoundSet& bs = cs.sets[0];
    auto c = g_setContents.find(bs.set);
    if (c == g_setContents.end()) return;
    auto b = c->second.find(10);
    if (b == c->second.end() || b->second.empty()) return;
    const DescItem& d = b->second[0];
    if (!(d.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || d.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)) return;
    if (d.range != VK_WHOLE_SIZE && d.range < kViewSize) return;
    VkDeviceSize off = d.offset;
    for (const auto& [binding, o] : bs.dyn)
        if (binding == 10) off += o;
    cs.hasView = true;
    cs.viewBuf = d.buffer;
    cs.viewOff = off;
    cs.viewRange = d.range;
}

// Returns {set index, binding} of the first bound image descriptor that views `img`, or {-1,-1}.
std::pair<int, int> FindSampled(const CmdState& cs, VkImage img) {
    for (size_t si = 0; si < cs.sets.size(); ++si) {
        auto c = g_setContents.find(cs.sets[si].set);
        if (c == g_setContents.end()) continue;
        for (auto& [binding, items] : c->second)
            for (auto& d : items)
                if (IsImageType(d.type) && d.view && ViewImage(d.view) == img) return {(int)si, (int)binding};
    }
    return {-1, -1};
}

// =============================================================================================
// DLSS state
// =============================================================================================
enum class Backend { None, Dlss, Fsr, Test, MetalFx };
const char* BackendName(Backend b) {
    return b == Backend::Dlss ? "DLSS" : b == Backend::Fsr ? "FSR 3.1" : b == Backend::Test ? "test upscaler"
         : b == Backend::MetalFx ? "MetalFX Temporal" : "none";
}
// TF3_DLSS_TEST_UPSCALER=1 (the test harness): a bilinear resize (test_upscale.comp) replaces DLSS and
// FSR 3.1, so everything around the upscaler can be tested on any GPU. Not a cfg setting on purpose.
bool TestUpscaler() {
    static int on = -1;
    if (on < 0) {
        const char* e = std::getenv("TF3_DLSS_TEST_UPSCALER");
        on = e && *e == '1' ? 1 : 0;
    }
    return on != 0;
}

// Shared upscaler state (the name is historical: it holds DLSS and everything common to both).
struct Dlss {
    bool ngxReady = false;       // NGX initialised and DLSS reported available
    bool enabled = true;         // Ctrl+Shift+D toggle
    bool failed = false;         // DLSS failed permanently (auto mode then uses FSR)
    bool ngxInited = false;
    bool imagesFailed = false;   // couldn't create our images (out of video memory?)
    std::string why = "not initialised";  // why DLSS isn't available
    Backend backend = Backend::None;      // the upscaler in use
    int override_ = -1;          // Ctrl+Shift+U: forced backend (Backend as int), -1 = follow the cfg
    DeviceData* dev = nullptr;
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle* handle = nullptr;
    uint32_t featInW = 0, featInH = 0, featOutW = 0, featOutH = 0;  // the DLSS feature's sizes
    uint32_t inW = 0, inH = 0, outW = 0, outH = 0;  // render and output size of our images
    // camera (last values seen at submit; used by FSR, which records before this frame's submit)
    float camNear = 0.5f, camFar = 10000.f, camFovY = 1.0f;
    // depth as R32_SFLOAT, written by the motion-vector pass (FSR reads this, not the D24S8 buffer)
    VkImage depth32Image = VK_NULL_HANDLE;
    VkImageView depth32View = VK_NULL_HANDLE;
    bool resetNext = true;
    // our resources
    VkImage outImage = VK_NULL_HANDLE;
    VkImageView outView = VK_NULL_HANDLE;
    VkImageLayout outLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage mvImage = VK_NULL_HANDLE;
    VkImageView mvView = VK_NULL_HANDLE;
    VkImageLayout mvLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    int presetInUse = -1;
    // motion-vector compute pass
    bool mvReady = false;
    VkDescriptorSetLayout mvSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout mvPipeLayout = VK_NULL_HANDLE;
    VkPipeline mvPipeline = VK_NULL_HANDLE;
    VkPipeline mvPipelineMs = VK_NULL_HANDLE;  // for a multisampled game depth buffer
    int mvDepthSamples = 0;                    // last logged sample count of the game's depth
    VkDescriptorPool mvPool = VK_NULL_HANDLE;
    VkDescriptorSet mvSet = VK_NULL_HANDLE;  // the one for the current depth view (from mvSets)
    // The game alternates between depth buffers (one per frame in flight), so keep a descriptor set
    // per view instead of rewriting one (which would need a full GPU wait every frame).
    std::vector<std::pair<VkImageView, VkDescriptorSet>> mvSets, objSets, dbgSets;
    VkSampler pointSampler = VK_NULL_HANDLE;
    VkBuffer mvBuffer = VK_NULL_HANDLE;   // packed half2 per pixel, copied into mvImage
    VkBuffer uboBuffer = VK_NULL_HANDLE;  // kMvSlots x kMvStride, host-visible
    char* uboPtr = nullptr;
    VkImageView mvDepthView = VK_NULL_HANDLE;  // depth view currently written into mvSet
    bool mvSetUsed = false;
    // camera history
    bool havePrev = false;
    float prevPV[16] = {};
    float prevCam[3] = {};
    uint64_t mvNoViewWarned = 0;
    uint64_t mvCounter = 0;
    uint64_t jitterIndex = 0;
    uint32_t jitterHold = 0;
    // One jitter per frame, shared by every command buffer that frame (see FrameJitter).
    uint64_t jitterFrame = UINT64_MAX;
    float frameJx = 0.f, frameJy = 0.f;
    uint32_t frameJitterIdx = 0, frameJitterPhases = 0;
    uint64_t mvBadViewWarned = 0;
    // debug view (motion vectors drawn into the output image)
    VkDescriptorSetLayout dbgSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout dbgPipeLayout = VK_NULL_HANDLE;
    VkPipeline dbgPipeline = VK_NULL_HANDLE;
    VkDescriptorPool dbgPool = VK_NULL_HANDLE;
    VkDescriptorSet dbgSet = VK_NULL_HANDLE;
    VkImageView dbgBoundMv = VK_NULL_HANDLE;
    VkBuffer dbgBoundCamBuf = VK_NULL_HANDLE;
    VkImageView dbgBoundView = VK_NULL_HANDLE;
    VkImageView dbgBoundColor = VK_NULL_HANDLE;
    VkSampler dbgSampler = VK_NULL_HANDLE;
    bool dbgSetUsed = false;
    bool dbgFailed = false;
    std::unordered_map<VkImage, VkImageView> depthViews;  // depth-aspect views we created
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::unordered_map<VkDescriptorSetLayout, VkDescriptorSet> blitSets;  // our sets, keyed by layout
    // sharpening after DLSS
    bool shpFailed = false, shpSetUsed = false;
    VkDescriptorSetLayout shpSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout shpPipeLayout = VK_NULL_HANDLE;
    VkPipeline shpPipeline = VK_NULL_HANDLE;
    VkDescriptorPool shpPool = VK_NULL_HANDLE;
    VkDescriptorSet shpSet = VK_NULL_HANDLE;
    VkImage shpImage = VK_NULL_HANDLE;   // temporary result, copied back into outImage
    VkImageView shpView = VK_NULL_HANDLE;
    VkImageView shpBoundOut = VK_NULL_HANDLE;
    VkImageView shpBoundMv = VK_NULL_HANDLE;  // motion vectors for motion-adaptive sharpening
    // test upscaler (TestUpscaler)
    bool tstFailed = false;
    VkDescriptorSetLayout tstSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout tstPipeLayout = VK_NULL_HANDLE;
    VkPipeline tstPipeline = VK_NULL_HANDLE;
    VkDescriptorPool tstPool = VK_NULL_HANDLE;
    VkSampler tstSampler = VK_NULL_HANDLE;
    VkImageView tstBoundOut = VK_NULL_HANDLE;  // the output view the cached sets point at
    std::vector<std::pair<VkImageView, VkDescriptorSet>> tstSets;  // one per input colour view
    bool tstSetUsed = false;
    // on-screen info box (overlay.comp)
    bool ovFailed = false, ovSetUsed = false;
    VkDescriptorSetLayout ovSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout ovPipeLayout = VK_NULL_HANDLE;
    VkPipeline ovPipeline = VK_NULL_HANDLE;
    VkDescriptorPool ovPool = VK_NULL_HANDLE;
    VkDescriptorSet ovSet = VK_NULL_HANDLE;
    VkImageView ovBoundOut = VK_NULL_HANDLE;
    VkBuffer ovBuffer = VK_NULL_HANDLE;  // font + kOvSlots text slots, host-visible
    char* ovPtr = nullptr;
    uint64_t ovCounter = 0;
    // object motion pass
    VkViewport sceneVp{};  // the game's viewport for the scene passes (without our jitter)
    bool sceneVpValid = false;
    bool objFailed = false;
    VkDescriptorSetLayout objSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout objPipeLayout = VK_NULL_HANDLE;
    VkDescriptorPool objPool = VK_NULL_HANDLE;
    VkDescriptorSet objSet = VK_NULL_HANDLE;
    VkImageView objDepthView = VK_NULL_HANDLE;
    bool objSetUsed = false;
    VkRenderPass objRenderPass = VK_NULL_HANDLE;
    VkFramebuffer objFb = VK_NULL_HANDLE;
    VkImageView objFbView = VK_NULL_HANDLE;
    VkShaderModule objVs = VK_NULL_HANDLE, objFs = VK_NULL_HANDLE;
    std::map<std::string, VkPipeline> objPipelines;  // keyed by vertex layout
    VkBuffer objBuf = VK_NULL_HANDLE;  // previous-frame model matrices, kObjSlots x kObjMax x 64 bytes
    char* objPtr = nullptr;
    uint64_t objCounter = 0;
    std::vector<VkDeviceMemory> memories;
} g_dlss;

void NGX_LogCallback(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    std::string m(message ? message : "");
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r')) m.pop_back();
    Log("[ngx] %s", m.c_str());
}

uint32_t FindMemoryType(DeviceData* d, uint32_t typeBits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties mp{};
    d->instance->GetPhysicalDeviceMemoryProperties(d->physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) return i;
    return UINT32_MAX;
}

bool CreateOwnImage(DeviceData* d, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImage* img,
                    VkImageView* view, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (d->CreateImage(d->device, &ci, nullptr, img) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    d->GetImageMemoryRequirements(d->device, *img, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX) return false;
    VkDeviceMemory mem;
    if (d->AllocateMemory(d->device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
    g_dlss.memories.push_back(mem);
    if (d->BindImageMemory(d->device, *img, mem, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = *img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = {aspect, 0, 1, 0, 1};
    return d->CreateImageView(d->device, &vi, nullptr, view) == VK_SUCCESS;
}

VkImageView DepthView(DeviceData* d, VkImage depth) {
    auto it = g_dlss.depthViews.find(depth);
    if (it != g_dlss.depthViews.end()) return it->second;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = depth;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = g_images[depth].fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (d->CreateImageView(d->device, &vi, nullptr, &v) != VK_SUCCESS) return VK_NULL_HANDLE;
    g_dlss.depthViews[depth] = v;
    return v;
}

void Barrier(DeviceData* d, VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect, VkImageLayout from,
             VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                          0, nullptr, 1, &b);
}

// ---------------------------------------------------------------------------------------------
// Motion-vector compute pass
// ---------------------------------------------------------------------------------------------
const uint32_t kMvSpirv[] = {
#include "camera_mv.spv.inc"
};
// The same shader built with DEPTH_MS: reads sample 0 of a multisampled depth buffer (TF3 with MSAA on).
// Sampling a multisampled image through a plain sampler2D is invalid; NVIDIA tolerates it, but on
// MoltenVK (macOS) it returns unrelated values, which made every motion vector wrong.
const uint32_t kMvMsSpirv[] = {
#include "camera_mv_ms.spv.inc"
};
// Per-frame camera data is written by the CPU at submit while the GPU may still be on earlier
// frames, so the ring must be longer than the frames that can be queued. 16 is far more than any
// swapchain allows (the old 4 relied on a GPU wait every frame that no longer exists).
constexpr uint32_t kMvSlots = 16;
constexpr uint32_t kMvStride = 256;   // max minUniformBufferOffsetAlignment allowed by the spec
struct MvUbo {
    float curInvViewProj[16];  // jittered (matches the depth buffer); with sizeAndScale[3] = 1 it is the
                               // whole reprojection (current clip -> previous clip, see UploadMvMatrices)
    float prevViewProj[16];    // unjittered
    float sizeAndScale[4];     // xy = render size, z = 1 for motion, w = 1 if curInvViewProj is the reprojection
    float jitterPx[4];         // this frame's jitter in render pixels (xy)
    // object motion pass (object_mv.vert/.frag)
    float curViewProj[16];     // unjittered (jitter comes from the viewport)
    float params[4];           // near, far, 1 = flip y in the shader
};
static_assert(sizeof(MvUbo) <= 256, "MvUbo must fit a ring slot");

// Motion-vector recording (part of the Ctrl+Shift+T test): copies MV, depth and colour of a few
// frames to host memory and writes them to disk with the matrices used, for offline checking.
struct CaptureFrame {
    uint64_t frame = 0;
    std::string phase;
    MvUbo ubo{};
    float curPV[16] = {};  // unjittered projView of this frame
    float jx = 0, jy = 0;
    // Copies of the readback buffers, taken right after the GPU finished, so the files can be written on
    // another thread: writing them on the game's thread stalled it and made the next frame's camera jump.
    std::vector<char> mv, mvFinal, depth, color, out, reactive;
    bool hasReactive = false;
    uint32_t w = 0, h = 0, outW = 0, outH = 0;
    int index = 0;
};
struct Capture {
    int requested = 0;  // frames still to capture
    std::string phase;
    int counter = 0;
    VkBuffer mv = VK_NULL_HANDLE, depth = VK_NULL_HANDLE, color = VK_NULL_HANDLE;
    char *mvPtr = nullptr, *depthPtr = nullptr, *colorPtr = nullptr;
    VkBuffer mvFinal = VK_NULL_HANDLE, out = VK_NULL_HANDLE;  // motion incl. objects; DLSS output
    char *mvFinalPtr = nullptr, *outPtr = nullptr;
    VkBuffer reactive = VK_NULL_HANDLE;  // the reactive mask (macOS), R32 float
    char* reactivePtr = nullptr;
    uint32_t outW = 0, outH = 0;
    bool outPending = false;  // this frame's DLSS output still to copy (after Evaluate)
    uint32_t w = 0, h = 0;
    bool failed = false;
    std::vector<CaptureFrame> pending;  // filled at submit, written after the GPU finishes
} g_cap;

// TF3_DLSS_CAPTURE=90,190 (test harness): record two frames in a row after each of these frames,
// as the Ctrl+Shift+T test does. TF3_DLSS_CAPTURE_REL=1: the numbers count from the first frame with the
// game's FSR pass (the map is on screen), not from the start of the process (menus take a varying time).
bool CaptureRequestedAt(uint64_t frame) {
    static const bool rel = [] {
        const char* e = std::getenv("TF3_DLSS_CAPTURE_REL");
        return e && *e == '1';
    }();
    if (rel) {
        if (!g_firstFsrFrame || frame < g_firstFsrFrame) return false;
        frame -= g_firstFsrFrame;
    }
    static std::vector<uint64_t> frames = [] {
        std::vector<uint64_t> v;
        const char* e = std::getenv("TF3_DLSS_CAPTURE");
        for (const char* p = e; p && *p;) {
            char* end = nullptr;
            unsigned long long f = std::strtoull(p, &end, 10);
            if (end == p) break;
            v.push_back(f);
            p = *end ? end + 1 : end;
        }
        return v;
    }();
    return std::find(frames.begin(), frames.end(), frame) != frames.end();
}
bool RecordingPossible() {
    static const bool env = [] {
        const char* e = std::getenv("TF3_DLSS_CAPTURE");
        return e && *e;
    }();
    return env || g_cfg.debugKeys;
}

// Frame trace (Ctrl+Shift+T): one log line per frame for kTraceFrames frames.
constexpr uint64_t kTraceFrames = 240;
uint64_t g_traceUntil = 0;

double NowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Frames per second for the info box: frames presented in the last half second or so.
float g_fps = 0.f;
void CountFrameForFps() {
    static double start = 0.0;
    static uint32_t frames = 0;
    double now = NowMs();
    if (start == 0.0) start = now;
    ++frames;
    if (now - start >= 500.0) {
        g_fps = (float)(frames * 1000.0 / (now - start));
        start = now;
        frames = 0;
    }
}

// ---------------------------------------------------------------------------------------------
// Automated camera test (Ctrl+Shift+T): drives the camera with synthetic input while the frame
// trace runs, so motion can be checked without anyone holding keys. TF3 controls:
// WASD pan, mouse wheel zoom, middle button + mouse up/down tilt.
// ---------------------------------------------------------------------------------------------
#ifdef _WIN32
enum class Act { None, Key, Wheel, Tilt, Wobble };
struct TestPhase {
    const char* name;
    Act act;
    WORD vk;      // for Key
    int amount;   // wheel notches per step / mouse pixels per frame
    double ms;
};
const TestPhase kTest[] = {
    {"still", Act::None, 0, 0, 1000},
    {"pan right (hold D)", Act::Key, 'D', 0, 1500},
    {"pause", Act::None, 0, 0, 600},
    {"pan left (hold A)", Act::Key, 'A', 0, 1500},
    {"pause", Act::None, 0, 0, 600},
    {"pan forward (hold W)", Act::Key, 'W', 0, 1500},
    {"pause", Act::None, 0, 0, 600},
    {"pan back (hold S)", Act::Key, 'S', 0, 1500},
    {"pause", Act::None, 0, 0, 600},
    {"zoom in (wheel)", Act::Wheel, 0, 1, 1200},
    {"pause", Act::None, 0, 0, 600},
    {"zoom out (wheel)", Act::Wheel, 0, -1, 1200},
    {"pause", Act::None, 0, 0, 600},
    {"tilt (middle drag up)", Act::Tilt, 0, -4, 1200},
    {"pause", Act::None, 0, 0, 600},
    {"tilt (middle drag down)", Act::Tilt, 0, 4, 1200},
    {"still", Act::None, 0, 0, 800},
};
// Ctrl+Shift+Y: small tilts back and forth (orbit wobble), recording 8 frames in a row mid-way.
const TestPhase kWobbleTest[] = {
    {"still", Act::None, 0, 0, 800},
    {"tilt wobble (quick middle drags up/down)", Act::Wobble, 0, 6, 3000},
    {"still", Act::None, 0, 0, 600},
};
constexpr int kWobbleFrames = 8;

struct AutoTest {
    const TestPhase* phases = nullptr;  // kTest or kWobbleTest
    int count = 0;
    int captureFrames = 2;  // consecutive frames recorded per phase
    bool armed = false;    // waiting for Ctrl/Shift to be released
    bool running = false;
    int phase = -1;
    double phaseStart = 0, lastWheel = 0;
    bool captured = false;  // this phase's recording was requested
} g_test;

void SendKey(WORD vk, bool down) {
    INPUT in{};
    in.type = INPUT_KEYBOARD;
    in.ki.wScan = (WORD)MapVirtualKeyW(vk, MAPVK_VK_TO_VSC);
    in.ki.dwFlags = KEYEVENTF_SCANCODE | (down ? 0 : KEYEVENTF_KEYUP);
    SendInput(1, &in, sizeof in);
}
void SendMouse(DWORD flags, LONG dx = 0, LONG dy = 0, DWORD data = 0) {
    INPUT in{};
    in.type = INPUT_MOUSE;
    in.mi.dwFlags = flags;
    in.mi.dx = dx;
    in.mi.dy = dy;
    in.mi.mouseData = data;
    SendInput(1, &in, sizeof in);
}
void EndPhase(const TestPhase& p) {
    if (p.act == Act::Key) SendKey(p.vk, false);
    if (p.act == Act::Tilt || p.act == Act::Wobble) SendMouse(MOUSEEVENTF_MIDDLEUP);
}
void StartPhase(const TestPhase& p) {
    if (p.act == Act::Key) SendKey(p.vk, true);
    if (p.act == Act::Tilt || p.act == Act::Wobble) SendMouse(MOUSEEVENTF_MIDDLEDOWN);
}
void StopTest(const char* why) {
    if (g_test.running && g_test.phase >= 0 && g_test.phase < g_test.count) EndPhase(g_test.phases[g_test.phase]);
    g_test = AutoTest{};
    g_traceUntil = 0;
    Log("[test] camera test finished (%s)", why);
}

// Called once per presented frame (caller holds g_t).
void AutoTestStep(bool focused, bool ctrlOrShiftHeld) {
    if (g_test.armed && !g_test.running) {
        if (ctrlOrShiftHeld) return;  // wait until the hotkey is released, or keys would combine
        g_test.running = true;
        g_test.phase = -1;
        g_test.phaseStart = 0;
        g_traceUntil = UINT64_MAX;
        Log("[test] camera test started (frame %llu)", (unsigned long long)g_frame);
    }
    if (!g_test.running) return;
    if (!focused) {
        StopTest("game window lost focus");
        return;
    }
    double now = NowMs();
    if (g_test.phase < 0 || now - g_test.phaseStart >= g_test.phases[g_test.phase].ms) {
        if (g_test.phase >= 0) EndPhase(g_test.phases[g_test.phase]);
        if (++g_test.phase >= g_test.count) {
            StopTest("completed");
            return;
        }
        g_test.phaseStart = now;
        g_test.lastWheel = 0;
        g_test.captured = false;
        StartPhase(g_test.phases[g_test.phase]);
        Log("[test] phase %d: %s (frame %llu)", g_test.phase, g_test.phases[g_test.phase].name,
            (unsigned long long)g_frame);
    }
    const TestPhase& p = g_test.phases[g_test.phase];
    if (!g_test.captured && std::strcmp(p.name, "pause") != 0 && now - g_test.phaseStart >= 0.4 * p.ms) {
        g_cap.requested = g_test.captureFrames;  // consecutive frames, mid-phase
        g_cap.phase = p.name;
        g_test.captured = true;
    }
    if (p.act == Act::Wheel && now - g_test.lastWheel >= 100) {  // one notch every 100 ms
        SendMouse(MOUSEEVENTF_WHEEL, 0, 0, (DWORD)(p.amount * WHEEL_DELTA));
        g_test.lastWheel = now;
    }
    if (p.act == Act::Tilt) SendMouse(MOUSEEVENTF_MOVE, 0, p.amount);
    if (p.act == Act::Wobble) {  // 6 px per frame (was 2), reversing every 200 ms (was 150): quicker, wider
        int dir = ((int)((now - g_test.phaseStart) / 200.0) % 2) ? 1 : -1;
        SendMouse(MOUSEEVENTF_MOVE, 0, dir * p.amount);
    }
}
#endif

// Expected motion (pixels) of the point at NDC (x, y, z) between prevPV and the current frame.
void MotionAt(const float* curInv, const float* prevPV, float x, float y, float z, float w, float h, float out[2]) {
    float ndc[4] = {x, y, z, 1.f}, world[4] = {}, prev[4] = {};
    for (int r = 0; r < 4; ++r)
        for (int k = 0; k < 4; ++k) world[r] += curInv[k * 4 + r] * ndc[k];
    for (int r = 0; r < 4; ++r) world[r] /= world[3];
    for (int r = 0; r < 4; ++r)
        for (int k = 0; k < 4; ++k) prev[r] += prevPV[k * 4 + r] * world[k];
    out[0] = (prev[0] / prev[3] - x) * 0.5f * w;
    out[1] = -(prev[1] / prev[3] - y) * 0.5f * h;  // pixel rows run top-down
}

bool CreateBuffer(DeviceData* d, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                  VkBuffer* buf, char** mapped) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    if (d->CreateBuffer(d->device, &bi, nullptr, buf) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    d->GetBufferMemoryRequirements(d->device, *buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, props);
    if (ai.memoryTypeIndex == UINT32_MAX) return false;
    VkDeviceMemory mem;
    if (d->AllocateMemory(d->device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
    g_dlss.memories.push_back(mem);
    if (d->BindBufferMemory(d->device, *buf, mem, 0) != VK_SUCCESS) return false;
    if (mapped) {
        void* p = nullptr;
        if (d->MapMemory(d->device, mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) return false;
        *mapped = static_cast<char*>(p);
    }
    return true;
}

// A descriptor set per image view from a small cache. Returns the cached set, or a new one with
// `isNew` set so the caller fills it. When the pool is full the oldest is reused after a GPU wait
// (only if the game cycles through more views than the pool holds).
VkDescriptorSet CachedSetRaw(DeviceData* d, std::vector<std::pair<VkImageView, VkDescriptorSet>>& cache,
                             VkImageView view, VkDescriptorPool pool, VkDescriptorSetLayout layout, bool used,
                             bool& isNew) {
    isNew = false;
    for (auto& e : cache)
        if (e.first == view) return e.second;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    if (d->AllocateDescriptorSets(d->device, &ai, &set) != VK_SUCCESS) {
        if (cache.empty()) return VK_NULL_HANDLE;
        if (used) d->DeviceWaitIdle(d->device);
        set = cache.front().second;
        cache.erase(cache.begin());
        Log("[dlss] more than %zu views in rotation; reusing a descriptor set", cache.size() + 1);
    }
    cache.push_back({view, set});
    isNew = true;
    return set;
}

void WriteMvSet(DeviceData* d, VkDescriptorSet set, VkImageView depthView) {
    VkDescriptorImageInfo ii{g_dlss.pointSampler, depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo sb{g_dlss.mvBuffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo ub{g_dlss.uboBuffer, 0, sizeof(MvUbo)};
    VkDescriptorImageInfo di{VK_NULL_HANDLE, g_dlss.depth32View, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[4] = {};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            &ii, nullptr, nullptr};
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr,
            &sb, nullptr};
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            nullptr, &ub, nullptr};
    w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &di,
            nullptr, nullptr};
    d->UpdateDescriptorSets(d->device, 4, w, 0, nullptr);
}

VkDescriptorSet CachedSet(DeviceData* d, std::vector<std::pair<VkImageView, VkDescriptorSet>>& cache,
                          VkImageView view, VkDescriptorPool pool, VkDescriptorSetLayout layout, bool used,
                          VkDescriptorSet /*current*/) {
    bool isNew = false;
    VkDescriptorSet set = CachedSetRaw(d, cache, view, pool, layout, used, isNew);
    if (set && isNew) WriteMvSet(d, set, view);
    return set;
}

// Drop every descriptor set that points at a game image view (depth for the motion passes, the
// DLSS input colour for the debug view). Called when the game destroys render targets (resolution
// change): the sets get rebuilt from the new images on the next frame. Caller holds g_t.
void InvalidateViewCaches(DeviceData* d) {
    d->DeviceWaitIdle(d->device);  // rare (resize); nothing may still use the sets
    if (g_dlss.mvPool) d->ResetDescriptorPool(d->device, g_dlss.mvPool, 0);
    if (g_dlss.objPool) d->ResetDescriptorPool(d->device, g_dlss.objPool, 0);
    if (g_dlss.dbgPool) d->ResetDescriptorPool(d->device, g_dlss.dbgPool, 0);
    g_dlss.mvSets.clear();
    g_dlss.objSets.clear();
    g_dlss.dbgSets.clear();
    g_dlss.mvSet = g_dlss.objSet = g_dlss.dbgSet = VK_NULL_HANDLE;
    g_dlss.mvDepthView = g_dlss.objDepthView = VK_NULL_HANDLE;
    g_dlss.mvSetUsed = g_dlss.objSetUsed = g_dlss.dbgSetUsed = false;
    g_dlss.dbgBoundMv = VK_NULL_HANDLE;  // forces the debug view to rebuild its sets too
    if (g_dlss.tstPool) d->ResetDescriptorPool(d->device, g_dlss.tstPool, 0);
    g_dlss.tstSets.clear();
    g_dlss.tstSetUsed = false;
}

// Pipeline objects (once) and the per-size buffer. Caller holds g_t.
bool CreateMvResources(DeviceData* d, uint32_t w, uint32_t h) {
    if (!g_dlss.mvPipeline) {
        VkDescriptorSetLayoutBinding b[4] = {};
        b[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[2] = {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};  // depth copy
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 4;
        li.pBindings = b;
        if (d->CreateDescriptorSetLayout(d->device, &li, nullptr, &g_dlss.mvSetLayout) != VK_SUCCESS) return false;
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &g_dlss.mvSetLayout;
        if (d->CreatePipelineLayout(d->device, &pli, nullptr, &g_dlss.mvPipeLayout) != VK_SUCCESS) return false;
        auto makePipeline = [&](const uint32_t* code, size_t size, VkPipeline* out) {
            VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            si.codeSize = size;
            si.pCode = code;
            VkShaderModule mod;
            if (d->CreateShaderModule(d->device, &si, nullptr, &mod) != VK_SUCCESS) return false;
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = mod;
            ci.stage.pName = "main";
            ci.layout = g_dlss.mvPipeLayout;
            VkResult r = d->CreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &ci, nullptr, out);
            d->DestroyShaderModule(d->device, mod, nullptr);
            return r == VK_SUCCESS;
        };
        if (!makePipeline(kMvSpirv, sizeof kMvSpirv, &g_dlss.mvPipeline)) return false;
        if (!makePipeline(kMvMsSpirv, sizeof kMvMsSpirv, &g_dlss.mvPipelineMs)) {
            g_dlss.mvPipelineMs = VK_NULL_HANDLE;
            Log("[mv] could not create the multisampled-depth variant; MSAA depth will give zero motion");
        }
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8},
                                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 8;
        pi.poolSizeCount = 4;
        pi.pPoolSizes = sizes;
        if (d->CreateDescriptorPool(d->device, &pi, nullptr, &g_dlss.mvPool) != VK_SUCCESS) return false;
        VkSamplerCreateInfo smp{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        smp.magFilter = smp.minFilter = VK_FILTER_NEAREST;
        smp.addressModeU = smp.addressModeV = smp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (d->CreateSampler(d->device, &smp, nullptr, &g_dlss.pointSampler) != VK_SUCCESS) return false;
        if (!CreateBuffer(d, kMvSlots * kMvStride, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          &g_dlss.uboBuffer, &g_dlss.uboPtr))
            return false;
        std::memset(g_dlss.uboPtr, 0, kMvSlots * kMvStride);
    }
    if (!CreateBuffer(d, (VkDeviceSize)w * h * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &g_dlss.mvBuffer, nullptr))
        return false;
    // new size: the sets (and the depth views in them) belong to the old render targets
    InvalidateViewCaches(d);
    g_dlss.mvReady = true;
    Log("[mv] motion-vector pass ready (%ux%u)", w, h);
    return true;
}

// Pick (or make) the compute pass's descriptor set for this depth view.
bool UpdateMvDepth(DeviceData* d, VkImageView depthView) {
    if (g_dlss.mvDepthView == depthView && g_dlss.mvSet) return true;
    VkDescriptorSet set = CachedSet(d, g_dlss.mvSets, depthView, g_dlss.mvPool, g_dlss.mvSetLayout,
                                    g_dlss.mvSetUsed, g_dlss.mvSet);
    if (!set) return false;
    g_dlss.mvSet = set;
    g_dlss.mvDepthView = depthView;
    return true;
}

const uint32_t kDbgSpirv[] = {
#include "mv_debug.spv.inc"
};
struct DbgPush {  // must match the push-constant block in mv_debug.comp (48 bytes)
    int32_t inSize[2];
    int32_t outSize[2];
    float scale;
    int32_t mode;
    int32_t phases;
    int32_t phaseIndex;
    float jitterApplied[2];
    float jitterSent[2];
    float objScale;
};
static_assert(sizeof(DbgPush) == 52, "DbgPush layout");

// Debug-view pipeline (once) and its descriptor set (re-pointed when our images change).
bool EnsureDebugResources(DeviceData* d, VkImageView colorView) {
    if (g_dlss.dbgFailed) return false;
    if (!g_dlss.dbgPipeline) {
        g_dlss.dbgFailed = true;  // cleared on success
        VkDescriptorSetLayoutBinding b[4] = {};
        b[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 4;
        li.pBindings = b;
        if (d->CreateDescriptorSetLayout(d->device, &li, nullptr, &g_dlss.dbgSetLayout) != VK_SUCCESS) return false;
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DbgPush)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &g_dlss.dbgSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pr;
        if (d->CreatePipelineLayout(d->device, &pli, nullptr, &g_dlss.dbgPipeLayout) != VK_SUCCESS) return false;
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = sizeof kDbgSpirv;
        si.pCode = kDbgSpirv;
        VkShaderModule mod;
        if (d->CreateShaderModule(d->device, &si, nullptr, &mod) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = g_dlss.dbgPipeLayout;
        VkResult r = d->CreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &ci, nullptr, &g_dlss.dbgPipeline);
        d->DestroyShaderModule(d->device, mod, nullptr);
        if (r != VK_SUCCESS) return false;
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8},
                                        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 8;
        pi.poolSizeCount = 3;
        pi.pPoolSizes = sizes;
        if (d->CreateDescriptorPool(d->device, &pi, nullptr, &g_dlss.dbgPool) != VK_SUCCESS) return false;
        VkSamplerCreateInfo smp{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        smp.magFilter = smp.minFilter = VK_FILTER_NEAREST;
        smp.addressModeU = smp.addressModeV = smp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (d->CreateSampler(d->device, &smp, nullptr, &g_dlss.dbgSampler) != VK_SUCCESS) return false;
        g_dlss.dbgFailed = false;
        Log("[debug] debug views ready");
    }
    // Our images rarely change (resize): then every cached set is stale; wait once and drop them.
    if (g_dlss.dbgBoundMv != g_dlss.mvView || g_dlss.dbgBoundCamBuf != g_dlss.mvBuffer ||
        g_dlss.dbgBoundView != g_dlss.outView) {
        if (!g_dlss.dbgSets.empty()) {
            if (g_dlss.dbgSetUsed) d->DeviceWaitIdle(d->device);
            for (auto& e : g_dlss.dbgSets) e.first = VK_NULL_HANDLE;  // force a rewrite below
        }
        g_dlss.dbgBoundMv = g_dlss.mvView;
        g_dlss.dbgBoundCamBuf = g_dlss.mvBuffer;
        g_dlss.dbgBoundView = g_dlss.outView;
    }
    // One set per game colour view (the game alternates between two).
    VkDescriptorSet set = VK_NULL_HANDLE;
    for (auto& e : g_dlss.dbgSets)
        if (e.first == colorView) set = e.second;
    if (!set) {
        for (auto& e : g_dlss.dbgSets)  // a stale one from a resize
            if (!e.first) {
                e.first = colorView;
                set = e.second;
                break;
            }
        if (!set) {
            bool isNew = false;
            set = CachedSetRaw(d, g_dlss.dbgSets, colorView, g_dlss.dbgPool, g_dlss.dbgSetLayout, g_dlss.dbgSetUsed,
                               isNew);
            if (!set) return false;
        }
        VkDescriptorImageInfo mi{g_dlss.pointSampler, g_dlss.mvView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo ii{VK_NULL_HANDLE, g_dlss.outView, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo ci{g_dlss.dbgSampler, colorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorBufferInfo cbi{g_dlss.mvBuffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w[4] = {};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &mi, nullptr, nullptr};
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ii,
                nullptr, nullptr};
        w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ci, nullptr, nullptr};
        w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr, &cbi, nullptr};
        d->UpdateDescriptorSets(d->device, 4, w, 0, nullptr);
    }
    g_dlss.dbgSet = set;
    return true;
}

// Overwrite the DLSS output (in GENERAL layout) with a debug picture (mode 1 motion, 2 jitter).
void RecordDebugView(DeviceData* d, VkCommandBuffer cb, const CmdState& cs) {
    if (!EnsureDebugResources(d, cs.colorView)) return;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                          nullptr, 0, nullptr);
    DbgPush pc{};
    pc.inSize[0] = (int32_t)g_dlss.inW;
    pc.inSize[1] = (int32_t)g_dlss.inH;
    pc.outSize[0] = (int32_t)g_dlss.outW;
    pc.outSize[1] = (int32_t)g_dlss.outH;
    pc.scale = g_cfg.debugScale;
    pc.mode = g_cfg.debugView;
    pc.phases = cs.jitterSet ? (int32_t)cs.jitterPhases : 0;
    pc.phaseIndex = cs.jitterSet ? (int32_t)cs.jitterIdx : 0;
    pc.jitterApplied[0] = cs.jitterSet ? cs.jx : 0.f;
    pc.jitterApplied[1] = cs.jitterSet ? cs.jy : 0.f;
    pc.jitterSent[0] = cs.jitterSet ? cs.jx * g_cfg.jitterSignX : 0.f;
    pc.jitterSent[1] = cs.jitterSet ? cs.jy * g_cfg.jitterSignY : 0.f;
    pc.objScale = g_cfg.debugObjectScale;
    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dlss.dbgPipeline);
    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dlss.dbgPipeLayout, 0, 1, &g_dlss.dbgSet, 0, nullptr);
    d->CmdPushConstants(cb, g_dlss.dbgPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    d->CmdDispatch(cb, (g_dlss.outW + 7) / 8, (g_dlss.outH + 7) / 8, 1);
    g_dlss.dbgSetUsed = true;
}

// ---------------------------------------------------------------------------------------------
// Jitter
// ---------------------------------------------------------------------------------------------
float Halton(uint32_t i, uint32_t base) {
    float f = 1.f, r = 0.f;
    while (i > 0) {
        f /= (float)base;
        r += f * (float)(i % base);
        i /= base;
    }
    return r;
}

// The jitter for the current frame. The game records a frame into several command buffers that
// all read the same camera buffer, so the offset must be per frame, not per command buffer:
// otherwise buffers without the DLSS pass write the unjittered camera back while others write
// the jittered one, and passes randomly see one or the other (flickering spots).
void FrameJitter(float& jx, float& jy, uint32_t* idxOut = nullptr, uint32_t* phasesOut = nullptr) {
    if (g_dlss.jitterFrame != g_frame) {
        g_dlss.jitterFrame = g_frame;
        float ratio = g_dlss.inW ? (float)g_dlss.outW / (float)g_dlss.inW : 1.f;
        uint32_t phases = g_cfg.jitterPhases > 0 ? (uint32_t)g_cfg.jitterPhases
                                                 : (uint32_t)std::max(8.f, std::ceil(8.f * ratio * ratio));
        // In the jitter debug view, hold each offset for 30 frames so it can be followed by eye.
        if (g_cfg.debugView != 2 || ++g_dlss.jitterHold >= 30) {
            g_dlss.jitterHold = 0;
            ++g_dlss.jitterIndex;
        }
        uint32_t idx = (uint32_t)(g_dlss.jitterIndex % phases) + 1;  // Halton starts at 1
        g_dlss.frameJx = Halton(idx, 2) - 0.5f;
        g_dlss.frameJy = Halton(idx, 3) - 0.5f;
        g_dlss.frameJitterIdx = idx;
        g_dlss.frameJitterPhases = phases;
        uint32_t bx, by;
        std::memcpy(&bx, &g_dlss.frameJx, 4);
        std::memcpy(&by, &g_dlss.frameJy, 4);
        g_hot.jitterXY.store(bx | (uint64_t)by << 32, std::memory_order_relaxed);
        g_hot.jitterFrame.store(g_frame, std::memory_order_release);
    }
    jx = g_dlss.frameJx;
    jy = g_dlss.frameJy;
    if (idxOut) *idxOut = g_dlss.frameJitterIdx;
    if (phasesOut) *phasesOut = g_dlss.frameJitterPhases;
}

// FrameJitter for the draw hooks: no lock once this frame's jitter has been worked out.
void HotFrameJitter(float& jx, float& jy) {
    if (g_hot.jitterFrame.load(std::memory_order_acquire) != g_hot.frame.load(std::memory_order_acquire)) {
        TLock lk(g_t);
        FrameJitter(jx, jy);
        return;
    }
    uint64_t xy = g_hot.jitterXY.load(std::memory_order_relaxed);
    uint32_t bx = (uint32_t)xy, by = (uint32_t)(xy >> 32);
    std::memcpy(&jx, &bx, 4);
    std::memcpy(&jy, &by, 4);
}

// Jitter is on while DLSS is running (it ran this frame or the previous one).
bool JitterActive() {
    return g_cfg.jitter && g_dlss.enabled && g_lastDlssFrame != UINT64_MAX && g_lastDlssFrame + 1 >= g_frame;
}

// Pointer to a camera UBO in mapped memory, or nullptr.
char* RefData(VkBuffer buf, VkDeviceSize off) {
    auto bi = g_buffers.find(buf);
    if (bi == g_buffers.end()) return nullptr;
    auto mi = g_mapped.find(bi->second.memory);
    if (mi == g_mapped.end()) return nullptr;
    VkDeviceSize at = bi->second.memOffset + off;
    if (at < mi->second.offset) return nullptr;
    return mi->second.ptr + (at - mi->second.offset);
}

// Jitter an inverse view-projection: Inv' = Inv * T^-1, T = translate(dx*w, dy*w) in NDC (only the
// fourth column changes). Used to reconstruct positions from the jittered depth buffer.
void JitterInverse(float* m, float dx, float dy) {
    for (int r = 0; r < 4; ++r) m[12 + r] -= dx * m[0 + r] + dy * m[4 + r];
}

// Column-major 4x4 helpers in double, for the camera reprojection.
void MulD(const double* a, const double* b, double* o) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            double t = 0;
            for (int k = 0; k < 4; ++k) t += a[k * 4 + r] * b[c * 4 + k];
            o[c * 4 + r] = t;
        }
}

// Gauss-Jordan with partial pivoting. False if m is singular.
bool InvertD(const double* m, double* inv) {
    double a[4][8];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            a[r][c] = m[c * 4 + r];
            a[r][4 + c] = r == c ? 1.0 : 0.0;
        }
    for (int c = 0; c < 4; ++c) {
        int piv = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
        if (std::fabs(a[piv][c]) < 1e-300) return false;
        if (piv != c)
            for (int k = 0; k < 8; ++k) std::swap(a[c][k], a[piv][k]);
        double d = a[c][c];
        for (int k = 0; k < 8; ++k) a[c][k] /= d;
        for (int r = 0; r < 4; ++r) {
            if (r == c || a[r][c] == 0.0) continue;
            double f = a[r][c];
            for (int k = 0; k < 8; ++k) a[r][k] -= f * a[c][k];
        }
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) inv[c * 4 + r] = a[r][4 + c];
    return true;
}

// Camera reprojection from this frame's jittered clip space to last frame's clip space:
// prevPV * (curPV)^-1 * T^-1 (T = the jitter translation, as in JitterInverse), in double.
// The game's own projViewInverse is float: far from the map origin its rounding alone tells
// DLSS that a still image moves by ~0.1 render pixels (0.12 px at 8 km). Inverting projView
// here in double and folding both matrices into one keeps the large world translation out of
// the float maths on the GPU, leaving ~0.0001 px (simulated; research/mv-precision).
bool ReprojectionD(const float* curPV, const float* prevPV, float dx, float dy, float* out) {
    double cur[16], prev[16], inv[16], r[16];
    for (int i = 0; i < 16; ++i) {
        cur[i] = curPV[i];
        prev[i] = prevPV[i];
    }
    if (!InvertD(cur, inv)) return false;
    for (int k = 0; k < 4; ++k) inv[12 + k] -= double(dx) * inv[0 + k] + double(dy) * inv[4 + k];
    MulD(prev, inv, r);
    for (int i = 0; i < 16; ++i) out[i] = (float)r[i];
    return true;
}

// Pointer to this command buffer's camera UBO contents in mapped memory, or nullptr.
const char* ViewData(const CmdState& cs) {
    if (!cs.hasView) return nullptr;
    auto bi = g_buffers.find(cs.viewBuf);
    if (bi == g_buffers.end()) return nullptr;
    auto mi = g_mapped.find(bi->second.memory);
    if (mi == g_mapped.end()) return nullptr;
    VkDeviceSize at = bi->second.memOffset + cs.viewOff;
    if (at < mi->second.offset) return nullptr;
    return mi->second.ptr + (at - mi->second.offset);
}

// TF3_DLSS_TRACE=start,count: [trace] lines for frames start..start+count-1 (the Ctrl+Shift+T trace,
// for platforms without the hotkeys). Also logs the camera position again at present ([trace-present]),
// to see whether the game changes the camera data after submitting the frame.
bool g_envTraceInit = false;
void EnvTraceUpdate() {
    static uint64_t start = 0, count = 0;
    if (!g_envTraceInit) {
        g_envTraceInit = true;
        if (const char* e = std::getenv("TF3_DLSS_TRACE")) {
            char* end = nullptr;
            start = std::strtoull(e, &end, 10);
            count = (end && *end == ',') ? std::strtoull(end + 1, nullptr, 10) : 240;
        }
    }
    static const bool rel = [] {
        const char* e = std::getenv("TF3_DLSS_CAPTURE_REL");
        return e && *e == '1';
    }();
    if (count && g_firstFsrFrame && rel && g_frame == g_firstFsrFrame + start) {
        g_traceUntil = g_frame + count;
        Log("[trace] TF3_DLSS_TRACE: tracing frames %llu..%llu (relative to the first FSR frame)",
            (unsigned long long)g_frame, (unsigned long long)(g_frame + count - 1));
    } else if (count && !rel && g_frame == start) {
        g_traceUntil = start + count;
        Log("[trace] TF3_DLSS_TRACE: tracing frames %llu..%llu", (unsigned long long)start,
            (unsigned long long)(start + count - 1));
    }
}
struct TraceCam { uint64_t frame; VkBuffer buf; VkDeviceSize off; float cam[3]; };
TraceCam g_traceHist[8] = {};  // camera buffers of the last frames, to see whether the game rewrites them
VkBuffer g_traceViewBuf = VK_NULL_HANDLE;
VkDeviceSize g_traceViewOff = 0;
float g_traceCamAtSubmit[3] = {};

// Called at submit, after the game has written this frame's camera data. Caller holds g_t.
void UploadMvMatrices(const CmdState& cs) {
    MvUbo u{};
    u.sizeAndScale[0] = (float)g_dlss.inW;
    u.sizeAndScale[1] = (float)g_dlss.inH;
    const char* v = ViewData(cs);
    if (!v) {
        if (g_dlss.mvNoViewWarned++ < 3) Log("[mv] camera data not available this frame; motion vectors set to zero");
        u.sizeAndScale[2] = 0.f;
    } else {
        int32_t vw = 0, vh = 0;
        std::memcpy(&vw, v + ViewOff::width, 4);
        std::memcpy(&vh, v + ViewOff::height, 4);
        if ((uint32_t)vw != g_dlss.inW || (uint32_t)vh != g_dlss.inH) {
            if (g_dlss.mvBadViewWarned++ < 10)
                Log("[mv] frame %llu: camera buffer is %dx%d, not the %ux%u scene camera; zero motion this frame",
                    (unsigned long long)g_frame, vw, vh, g_dlss.inW, g_dlss.inH);
            u.sizeAndScale[2] = 0.f;
            std::memcpy(g_dlss.uboPtr + cs.mvSlot * kMvStride, &u, sizeof u);
            return;
        }
        float pv[16], cam[3];  // the camera data is never jittered (jitter goes through the viewport)
        std::memcpy(pv, v + ViewOff::projView, 64);
        {
            float proj[16];
            std::memcpy(proj, v + ViewOff::proj, 64);
            std::memcpy(&g_frameCam.nearPlane, v + ViewOff::near_, 4);
            std::memcpy(&g_frameCam.farPlane, v + ViewOff::far_, 4);
            float sy = std::fabs(proj[5]), sx = std::fabs(proj[0]);  // 1 / tan(fov / 2), per axis
            g_frameCam.fovYDegrees = sy > 0.f ? 2.f * std::atan(1.f / sy) * 57.29578f : 0.f;
            g_frameCam.aspect = sx > 0.f ? sy / sx : 0.f;
            g_frameCam.frame = g_frame;
        }
        std::memcpy(u.curInvViewProj, v + ViewOff::projViewInverse, 64);
        std::memcpy(cam, v + ViewOff::camPos, 12);
        g_traceViewBuf = cs.viewBuf;
        g_traceViewOff = cs.viewOff;
        std::memcpy(g_traceCamAtSubmit, cam, 12);
        g_traceHist[g_frame % 8] = TraceCam{g_frame, cs.viewBuf, cs.viewOff, {cam[0], cam[1], cam[2]}};
        float jx = cs.jitterSet ? cs.jx : 0.f, jy = cs.jitterSet ? cs.jy : 0.f;
        if (jx != 0.f || jy != 0.f)  // depth was rendered jittered: reconstruct with the jittered inverse
            JitterInverse(u.curInvViewProj, 2.f * jx / (float)vw, -2.f * jy / (float)vh);
        u.jitterPx[0] = jx;
        u.jitterPx[1] = jy;
        bool jump = false;
        if (g_dlss.havePrev) {
            float dx = cam[0] - g_dlss.prevCam[0], dy = cam[1] - g_dlss.prevCam[1], dz = cam[2] - g_dlss.prevCam[2];
            jump = std::sqrt(dx * dx + dy * dy + dz * dz) > g_cfg.resetDistance;
        }
        if (jump) {
            g_dlss.resetNext = true;
            Log("[mv] camera jumped (frame %llu); resetting DLSS history", (unsigned long long)g_frame);
        }
        if (LogFrame(g_frame)) {
            float dx = cam[0] - g_dlss.prevCam[0], dy = cam[1] - g_dlss.prevCam[1], dz = cam[2] - g_dlss.prevCam[2];
            Log("[mv] frame %llu: slot %u, camera moved %.2f since last upload, buffer %s+%llu",
                (unsigned long long)g_frame, cs.mvSlot, std::sqrt(dx * dx + dy * dy + dz * dz), H(cs.viewBuf).c_str(),
                (unsigned long long)cs.viewOff);
        }
        std::memcpy(u.prevViewProj, (g_dlss.havePrev && !jump) ? g_dlss.prevPV : pv, 64);
        // mv_double=1: replace the float inverse with the whole reprojection, built in double. The object
        // pass keeps using prevViewProj; only the camera pass (camera_mv.comp) reads curInvViewProj.
        static const float kIdentity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const float* tracePrev = u.prevViewProj;
        if (g_cfg.mvDouble &&
            ReprojectionD(pv, u.prevViewProj, 2.f * jx / (float)vw, -2.f * jy / (float)vh, u.curInvViewProj)) {
            u.sizeAndScale[3] = 1.f;
            tracePrev = kIdentity;
        }
        if (g_frame < g_traceUntil) {
            static double lastMs = 0;
            double now = NowMs();
            float dx = cam[0] - g_dlss.prevCam[0], dy = cam[1] - g_dlss.prevCam[1], dz = cam[2] - g_dlss.prevCam[2];
            float mNear[2], mFar[2], mLeft[2];
            float W = (float)g_dlss.inW, Hh = (float)g_dlss.inH;
            MotionAt(u.curInvViewProj, tracePrev, 0.f, 0.f, 0.90f, W, Hh, mNear);  // screen centre, nearer
            MotionAt(u.curInvViewProj, tracePrev, 0.f, 0.f, 0.999f, W, Hh, mFar);  // screen centre, far
            MotionAt(u.curInvViewProj, tracePrev, -0.8f, 0.f, 0.99f, W, Hh, mLeft); // left edge
            Log("[trace] frame %llu dt=%.2fms slot=%u cam=(%.2f,%.2f,%.2f) d=%.3f mv_centre_near=(%.2f,%.2f) "
                "mv_centre_far=(%.2f,%.2f) mv_left=(%.2f,%.2f) mv=%s",
                (unsigned long long)g_frame, now - lastMs, cs.mvSlot, cam[0], cam[1], cam[2],
                std::sqrt(dx * dx + dy * dy + dz * dz), mNear[0], mNear[1], mFar[0], mFar[1], mLeft[0], mLeft[1],
                g_cfg.motionVectors ? "on" : "off");
            lastMs = now;
        }
        u.sizeAndScale[2] = g_cfg.motionVectors ? 1.f : 0.f;
        std::memcpy(u.curViewProj, pv, 64);
        std::memcpy(&u.params[0], v + ViewOff::near_, 4);
        std::memcpy(&u.params[1], v + ViewOff::far_, 4);
        {   // camera values for FSR (used from the next frame on: FSR is recorded before submit)
            float p11 = 0.f;
            std::memcpy(&p11, v + ViewOff::proj + 5 * 4, 4);
            if (u.params[0] > 0.f && u.params[1] > u.params[0]) {
                g_dlss.camNear = u.params[0];
                g_dlss.camFar = u.params[1];
            }
            if (std::fabs(p11) > 1e-6f) g_dlss.camFovY = 2.f * std::atan(1.f / std::fabs(p11));
        }
        u.params[2] = g_dlss.sceneVpValid && g_dlss.sceneVp.height > 0.f ? 1.f : 0.f;
        u.params[3] = g_cfg.objectDepthTest ? 0.f : 1.f;
        if (cs.capture) {
            CaptureFrame f;
            f.frame = g_frame;
            f.phase = cs.capPhase;
            f.ubo = u;
            f.hasReactive = cs.capReactive;
            std::memcpy(f.curPV, pv, 64);
            g_cap.pending.push_back(f);
        }
        std::memcpy(g_dlss.prevPV, pv, 64);
        std::memcpy(g_dlss.prevCam, cam, 12);
        g_dlss.havePrev = true;
    }
    std::memcpy(g_dlss.uboPtr + cs.mvSlot * kMvStride, &u, sizeof u);
}

// ---------------------------------------------------------------------------------------------
// Recording helpers
// ---------------------------------------------------------------------------------------------
std::filesystem::path CaptureDir() {
    // next to the log (logs\ in a development run, the mod folder or %TEMP% otherwise)
    std::filesystem::path base = g_logPath.empty() ? std::filesystem::path(RootDir())
                                                   : std::filesystem::path(g_logPath).parent_path();
    return base / "mvcap";
}

bool EnsureCaptureBuffers(DeviceData* d, uint32_t w, uint32_t h) {
    if (g_cap.failed) return false;
    if (g_cap.mv && g_cap.w == w && g_cap.h == h && g_cap.outW == g_dlss.outW && g_cap.outH == g_dlss.outH) return true;
    VkDeviceSize size = (VkDeviceSize)w * h * 4;
    // Host-cached memory reads back much faster; fall back to plain host-visible.
    for (VkMemoryPropertyFlags props :
         {VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
          VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)}) {
        VkDeviceSize outSize = (VkDeviceSize)g_dlss.outW * g_dlss.outH * 4;
        if (CreateBuffer(d, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, props, &g_cap.mv, &g_cap.mvPtr) &&
            CreateBuffer(d, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, props, &g_cap.depth, &g_cap.depthPtr) &&
            CreateBuffer(d, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, props, &g_cap.color, &g_cap.colorPtr) &&
            CreateBuffer(d, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, props, &g_cap.mvFinal, &g_cap.mvFinalPtr) &&
            CreateBuffer(d, outSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, props, &g_cap.out, &g_cap.outPtr) &&
            CreateBuffer(d, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, props, &g_cap.reactive, &g_cap.reactivePtr)) {
            g_cap.w = w;
            g_cap.h = h;
            g_cap.outW = g_dlss.outW;
            g_cap.outH = g_dlss.outH;
            return true;
        }
    }
    g_cap.failed = true;
    Log("[record] could not create readback buffers; recording disabled");
    return false;
}

// Copy this frame's motion vectors, depth and colour into the readback buffers.
// Called inside RecordUpscale after the MV pass, with depth and colour in SHADER_READ_ONLY.
void RecordCaptureCopies(DeviceData* d, VkCommandBuffer cb, CmdState& cs, VkImageAspectFlags depthAspects) {
    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, SRC = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkBufferCopy bc{0, 0, (VkDeviceSize)g_cap.w * g_cap.h * 4};
    d->CmdCopyBuffer(cb, g_dlss.mvBuffer, g_cap.mv, 1, &bc);
    VkBufferImageCopy ic{};
    ic.imageExtent = {g_cap.w, g_cap.h, 1};
    // Depth as the motion-vector pass read it (its R32F copy): valid for a multisampled game depth too,
    // which can't be copied to a buffer directly, and shows exactly what the layer saw.
    (void)depthAspects;
    Barrier(d, cb, g_dlss.depth32Image, VK_IMAGE_ASPECT_COLOR_BIT, RO, SRC);
    ic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    d->CmdCopyImageToBuffer(cb, g_dlss.depth32Image, SRC, g_cap.depth, 1, &ic);
    Barrier(d, cb, g_dlss.depth32Image, VK_IMAGE_ASPECT_COLOR_BIT, SRC, RO);
    Barrier(d, cb, cs.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, RO, SRC);
    ic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    d->CmdCopyImageToBuffer(cb, cs.colorImage, SRC, g_cap.color, 1, &ic);
    Barrier(d, cb, cs.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, SRC, RO);
    // final motion vectors (camera + object pass), as DLSS gets them
    Barrier(d, cb, g_dlss.mvImage, VK_IMAGE_ASPECT_COLOR_BIT, RO, SRC);
    d->CmdCopyImageToBuffer(cb, g_dlss.mvImage, SRC, g_cap.mvFinal, 1, &ic);
    Barrier(d, cb, g_dlss.mvImage, VK_IMAGE_ASPECT_COLOR_BIT, SRC, RO);
    g_cap.outPending = true;  // the DLSS output is copied right after Evaluate
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0,
                          nullptr);
    cs.capture = true;
    cs.capPhase = g_cap.phase;
}

// Snapshot the readback buffers into f (after the GPU finished). Caller holds g_t.
void SnapshotCapture(CaptureFrame& f) {
    size_t n = (size_t)g_cap.w * g_cap.h * 4;
    f.w = g_cap.w;
    f.h = g_cap.h;
    f.outW = g_cap.outW;
    f.outH = g_cap.outH;
    f.index = g_cap.counter++;
    f.mv.assign(g_cap.mvPtr, g_cap.mvPtr + n);
    f.mvFinal.assign(g_cap.mvFinalPtr, g_cap.mvFinalPtr + n);
    f.depth.assign(g_cap.depthPtr, g_cap.depthPtr + n);
    f.color.assign(g_cap.colorPtr, g_cap.colorPtr + n);
    if (g_cap.outPtr) f.out.assign(g_cap.outPtr, g_cap.outPtr + (size_t)g_cap.outW * g_cap.outH * 4);
    if (f.hasReactive && g_cap.reactivePtr) f.reactive.assign(g_cap.reactivePtr, g_cap.reactivePtr + n);
}

// Write one recorded frame (half resolution, every other pixel) plus a JSON description.
// Uses only f's own data (runs on a background thread).
void WriteCapture(const CaptureFrame& f) {
    std::error_code ec;
    std::filesystem::path dir = CaptureDir();
    std::filesystem::create_directories(dir, ec);
    std::string tag = f.phase;
    for (char& c : tag)
        if (!std::isalnum((unsigned char)c)) c = '_';
    char base[160];
    std::snprintf(base, sizeof base, "cap%03d_f%llu_%s", f.index, (unsigned long long)f.frame, tag.c_str());
    uint32_t w2 = f.w / 2, h2 = f.h / 2;
    std::vector<uint32_t> tmp((size_t)w2 * h2);
    auto dump = [&](const char* src, const char* ext, uint32_t sw, uint32_t sh) {  // every other pixel
        uint32_t dw = sw / 2, dh = sh / 2;
        std::vector<uint32_t> t((size_t)dw * dh);
        const uint32_t* s32 = reinterpret_cast<const uint32_t*>(src);
        for (uint32_t y = 0; y < dh; ++y)
            for (uint32_t x = 0; x < dw; ++x) t[(size_t)y * dw + x] = s32[(size_t)(y * 2) * sw + x * 2];
        std::ofstream o(dir / (std::string(base) + ext), std::ios::binary);
        o.write(reinterpret_cast<const char*>(t.data()), (std::streamsize)(t.size() * 4));
    };
    dump(f.mv.data(), ".mv.bin", f.w, f.h);              // camera motion: packed half2, render pixels
    dump(f.mvFinal.data(), ".mvfinal.bin", f.w, f.h);    // what DLSS gets (camera + objects)
    dump(f.depth.data(), ".depth.bin", f.w, f.h);        // depth as the MV pass read it, R32 float
    dump(f.color.data(), ".color.bin", f.w, f.h);        // DLSS input colour, R11G11B10 float
    if (!f.out.empty()) dump(f.out.data(), ".out.bin", f.outW, f.outH);  // DLSS output, RGBA8
    if (!f.reactive.empty()) {  // MetalFX's reactive mask, R32 float
        dump(f.reactive.data(), ".reactive.bin", f.w, f.h);
        const float* m = reinterpret_cast<const float*>(f.reactive.data());
        size_t n = (size_t)f.w * f.h, set = 0;
        float mx = 0.f;
        for (size_t i = 0; i < n; ++i) {
            if (m[i] > 0.f) ++set;
            mx = std::max(mx, m[i]);
        }
        Log("[record] reactive mask of frame %llu: %.2f%% of the pixels set, max %.3f", (unsigned long long)f.frame,
            100.0 * (double)set / (double)std::max<size_t>(n, 1), mx);
    }
    std::ofstream j(dir / (std::string(base) + ".json"));
    auto arr = [&](const float* v, int n) {
        std::string out = "[";
        char b[32];
        for (int i = 0; i < n; ++i) {
            std::snprintf(b, sizeof b, "%s%.9g", i ? "," : "", v[i]);
            out += b;
        }
        return out + "]";
    };
    j << "{\n  \"frame\": " << f.frame << ",\n  \"phase\": \"" << f.phase << "\",\n  \"render_w\": " << f.w
      << ",\n  \"render_h\": " << f.h << ",\n  \"file_w\": " << w2 << ",\n  \"file_h\": " << h2
      << ",\n  \"file_step\": 2,\n  \"out_file_w\": " << f.outW / 2 << ",\n  \"out_file_h\": " << f.outH / 2
      << ",\n  \"jitter_px\": " << arr(f.ubo.jitterPx, 2)
      << ",\n  \"mv_enabled\": " << f.ubo.sizeAndScale[2] << ",\n  \"depth_is_r32f\": 1"
      << ",\n  \"cur_inv_view_proj_jittered\": " << arr(f.ubo.curInvViewProj, 16)
      << ",\n  \"cur_inv_is_reprojection\": " << (f.ubo.sizeAndScale[3] == 1.f ? "true" : "false")
      << ",\n  \"prev_view_proj\": " << arr(f.ubo.prevViewProj, 16) << ",\n  \"cur_view_proj\": " << arr(f.curPV, 16)
      << ",\n  \"matrix_layout\": \"column-major (GLSL)\"\n}\n";
    Log("[record] wrote %s (%ux%u)", base, w2, h2);
}

// (Re)create our images for the given render and output sizes. Caller holds g_t.
bool EnsureImages(DeviceData* d, uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH) {
    if (g_dlss.imagesFailed) return false;  // don't retry (and leak) every frame
    if (!g_dlss.outImage || g_dlss.outW != outW || g_dlss.outH != outH) {
        if (g_dlss.outImage) Log("[upscale] output size changed to %ux%u", outW, outH);
        if (!CreateOwnImage(d, outW, outH, VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT |  // copied out by recordings
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,  // MetalFX output needs render-target usage
                            &g_dlss.outImage, &g_dlss.outView)) {
            Log("[upscale] failed to create the %ux%u output image; upscaling stops", outW, outH);
            g_dlss.outImage = VK_NULL_HANDLE;
            g_dlss.imagesFailed = true;
            return false;
        }
        g_dlss.outLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        g_dlss.blitSets.clear();  // they point at the old view
        g_dlss.outW = outW;
        g_dlss.outH = outH;
        g_dlss.resetNext = true;
    }
    if (!g_dlss.mvImage || g_dlss.inW != inW || g_dlss.inH != inH) {
        if (g_dlss.mvImage) Log("[upscale] render size changed to %ux%u", inW, inH);
        if (!CreateOwnImage(d, inW, inH, VK_FORMAT_R16G16_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,  // recordings
                            &g_dlss.mvImage, &g_dlss.mvView) ||
            !CreateOwnImage(d, inW, inH, VK_FORMAT_R32_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,  // recordings
                            &g_dlss.depth32Image, &g_dlss.depth32View)) {
            Log("[upscale] failed to create the motion-vector / depth images; upscaling stops");
            g_dlss.mvImage = VK_NULL_HANDLE;
            g_dlss.imagesFailed = true;
            return false;
        }
        g_dlss.mvLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        g_dlss.inW = inW;
        g_dlss.inH = inH;
        g_dlss.resetNext = true;
        if (!CreateMvResources(d, inW, inH)) {
            Log("[mv] failed to create the motion-vector pass; the upscaler will run with zero motion");
            g_dlss.mvReady = false;
        }
    }
    return true;
}

// (Re)create the DLSS feature for the current sizes and preset. Caller holds g_t.
bool EnsureDlssFeature(DeviceData* d, VkCommandBuffer cb) {
    if (g_dlss.handle && g_dlss.featInW == g_dlss.inW && g_dlss.featInH == g_dlss.inH &&
        g_dlss.featOutW == g_dlss.outW && g_dlss.featOutH == g_dlss.outH && g_dlss.presetInUse == g_cfg.preset)
        return true;
    if (g_dlss.handle) {
        Log("[dlss] size or preset changed, recreating feature");
        d->DeviceWaitIdle(d->device);
        NVSDK_NGX_VULKAN_ReleaseFeature(g_dlss.handle);
        g_dlss.handle = nullptr;
    }
    {
        const char* presetKeys[] = {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
                                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
                                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
                                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality};
        for (const char* k : presetKeys) NVSDK_NGX_Parameter_SetUI(g_dlss.params, k, (unsigned)g_cfg.preset);
    }
    NVSDK_NGX_DLSS_Create_Params cp{};
    cp.Feature.InWidth = g_dlss.inW;
    cp.Feature.InHeight = g_dlss.inH;
    cp.Feature.InTargetWidth = g_dlss.outW;
    cp.Feature.InTargetHeight = g_dlss.outH;
    cp.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;  // LDR input, MVs at render res
    NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT1(d->device, cb, 1, 1, &g_dlss.handle, g_dlss.params, &cp);
    if (NVSDK_NGX_FAILED(r)) {
        Log("[dlss] CreateFeature failed: 0x%08x", (unsigned)r);
        char b[96];
        std::snprintf(b, sizeof b, "DLSS could not be started (NGX error 0x%08x)", (unsigned)r);
        g_dlss.why = b;
        g_dlss.failed = true;
        g_dlss.handle = nullptr;
        return false;
    }
    g_dlss.featInW = g_dlss.inW;
    g_dlss.featInH = g_dlss.inH;
    g_dlss.featOutW = g_dlss.outW;
    g_dlss.featOutH = g_dlss.outH;
    g_dlss.resetNext = true;
    g_dlss.presetInUse = g_cfg.preset;
    Log("[dlss] feature created: %ux%u -> %ux%u, preset %d", g_dlss.inW, g_dlss.inH, g_dlss.outW, g_dlss.outH,
        g_cfg.preset);
    return true;
}

// Re-bind the game's graphics pipeline and descriptor sets after NGX touched the command buffer.
// ---------------------------------------------------------------------------------------------
// Sharpening after DLSS (optional)
// ---------------------------------------------------------------------------------------------
const uint32_t kShpSpirv[] = {
#include "sharpen.spv.inc"
};
struct ShpPush {
    int32_t size[2];
    int32_t motionSize[2];
    float sharpness;
    float motionSharpness;
    float motionThreshold;
    float motionLimit;
    int32_t type;
    int32_t debug;
};

bool EnsureSharpen(DeviceData* d) {
    if (g_dlss.shpFailed) return false;
    if (!g_dlss.shpPipeline) {
        g_dlss.shpFailed = true;  // cleared on success
        VkDescriptorSetLayoutBinding b[3] = {};
        b[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 3;
        li.pBindings = b;
        if (d->CreateDescriptorSetLayout(d->device, &li, nullptr, &g_dlss.shpSetLayout) != VK_SUCCESS) return false;
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShpPush)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &g_dlss.shpSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pr;
        if (d->CreatePipelineLayout(d->device, &pli, nullptr, &g_dlss.shpPipeLayout) != VK_SUCCESS) return false;
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = sizeof kShpSpirv;
        si.pCode = kShpSpirv;
        VkShaderModule mod;
        if (d->CreateShaderModule(d->device, &si, nullptr, &mod) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = g_dlss.shpPipeLayout;
        VkResult r = d->CreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &ci, nullptr, &g_dlss.shpPipeline);
        d->DestroyShaderModule(d->device, mod, nullptr);
        if (r != VK_SUCCESS) return false;
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 1;
        pi.poolSizeCount = 2;
        pi.pPoolSizes = sizes;
        if (d->CreateDescriptorPool(d->device, &pi, nullptr, &g_dlss.shpPool) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = g_dlss.shpPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &g_dlss.shpSetLayout;
        if (d->AllocateDescriptorSets(d->device, &ai, &g_dlss.shpSet) != VK_SUCCESS) return false;
        g_dlss.shpFailed = false;
        Log("[sharpen] sharpening pass ready");
    }
    if (g_dlss.shpBoundOut != g_dlss.outView) {  // output size changed (rare): new temporary image
        if (g_dlss.shpSetUsed) d->DeviceWaitIdle(d->device);
        if (!CreateOwnImage(d, g_dlss.outW, g_dlss.outH, VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &g_dlss.shpImage,
                            &g_dlss.shpView)) {
            g_dlss.shpFailed = true;
            Log("[sharpen] couldn't create the %ux%u image; sharpening off", g_dlss.outW, g_dlss.outH);
            return false;
        }
        VkDescriptorImageInfo src{g_dlss.pointSampler, g_dlss.outView, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo dst{VK_NULL_HANDLE, g_dlss.shpView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w[2] = {};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_dlss.shpSet, 0, 0, 1,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &src, nullptr, nullptr};
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_dlss.shpSet, 1, 0, 1,
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dst, nullptr, nullptr};
        d->UpdateDescriptorSets(d->device, 2, w, 0, nullptr);
        g_dlss.shpBoundOut = g_dlss.outView;
    }
    if (g_dlss.shpBoundMv != g_dlss.mvView) {  // render size changed: new motion-vector image
        if (g_dlss.shpSetUsed) d->DeviceWaitIdle(d->device);
        VkDescriptorImageInfo mv{g_dlss.pointSampler, g_dlss.mvView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_dlss.shpSet, 2, 0, 1,
                               VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &mv, nullptr, nullptr};
        d->UpdateDescriptorSets(d->device, 1, &w, 0, nullptr);
        g_dlss.shpBoundMv = g_dlss.mvView;
    }
    return true;
}

// After Evaluate, with outImage in GENERAL and mvImage in SHADER_READ_ONLY: sharpen into shpImage, copy back
// into outImage.
void RecordSharpen(DeviceData* d, VkCommandBuffer cb) {
    if ((g_cfg.sharpness <= 0.f && g_cfg.motionSharpness <= 0.f) || !g_dlss.pointSampler || !g_dlss.mvView ||
        !EnsureSharpen(d))
        return;
    VkImageMemoryBarrier b[2] = {};
    for (auto& x : b) {
        x.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        x.srcQueueFamilyIndex = x.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        x.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    // DLSS output written -> readable; temp image (contents not needed) -> writable
    b[0].image = g_dlss.outImage;
    b[0].oldLayout = b[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b[1].image = g_dlss.shpImage;
    b[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[1].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    b[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                          0, nullptr, 2, b);
    ShpPush pc{{(int32_t)g_dlss.outW, (int32_t)g_dlss.outH},
               {(int32_t)g_dlss.inW, (int32_t)g_dlss.inH},
               g_cfg.sharpness,
               g_cfg.motionSharpness,
               g_cfg.motionThreshold,
               g_cfg.motionLimit,
               g_cfg.sharpenType,
               g_cfg.sharpenDebug};
    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dlss.shpPipeline);
    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dlss.shpPipeLayout, 0, 1, &g_dlss.shpSet, 0, nullptr);
    d->CmdPushConstants(cb, g_dlss.shpPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    d->CmdDispatch(cb, (g_dlss.outW + 7) / 8, (g_dlss.outH + 7) / 8, 1);
    // copy back: temp -> source, output -> destination
    b[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                          nullptr, 2, b);
    VkImageCopy ic{};
    ic.srcSubresource = ic.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    ic.extent = {g_dlss.outW, g_dlss.outH, 1};
    d->CmdCopyImage(cb, g_dlss.shpImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_dlss.outImage,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &ic);
    b[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                          nullptr, 1, b);
    g_dlss.shpSetUsed = true;
}

// ---------------------------------------------------------------------------------------------
// Object motion pass (milestone 6)
// ---------------------------------------------------------------------------------------------
// Moving objects (vehicles...) only get camera motion from the compute pass. For meshes seen
// moving recently, the scene draws are replayed into the motion-vector image with object_mv.vert/
// .frag, which use this frame's model matrix (the game's instance data) and last frame's (matched
// on the CPU: nearest copy of the same mesh, within kObjMatchDist).
const uint32_t kObjVsSpirv[] = {
#include "object_mv_vert.spv.inc"
};
const uint32_t kObjFsSpirv[] = {
#include "object_mv_frag.spv.inc"
};
constexpr uint32_t kObjSlots = 16;  // see kMvSlots
constexpr uint32_t kObjMax = 16384;      // instances per frame
constexpr float kObjMatchDist = 2.f;     // world units per frame
constexpr uint64_t kObjMeshKeep = 300;   // frames a mesh stays "moving" after it last moved

std::unordered_map<uint64_t, uint64_t> g_movingMeshes;  // mesh -> last frame it was seen moving
struct ObjFrame {
    uint64_t frame = UINT64_MAX;
    std::vector<ObjDraw> draws;
    std::set<std::pair<VkBuffer, VkDeviceSize>> seen;  // depth prepass + colour pass draw the same
    uint32_t instances = 0;
} g_objFrame;
std::unordered_map<uint64_t, std::vector<std::array<float, 16>>> g_objPrev;  // mesh -> last frame's matrices
uint64_t g_objPrevFrame = UINT64_MAX;
struct ObjStats {
    uint64_t frames = 0, draws = 0, instances = 0, moved = 0, unmatched = 0;
} g_objStats;

std::string ObjLayoutKey(const PipeVtx& p) {
    char k[160];
    std::snprintf(k, sizeof k, "%d/%u/%u/%d|%u/%d/%u/%u/%u/%u|%d/%d", (int)p.posFormat, p.posOffset, p.posStride,
                  (int)p.posInstanceRate, p.stride, (int)p.fmt, p.colOffset[0], p.colOffset[1], p.colOffset[2],
                  p.colOffset[3], (int)p.topology, (int)p.primitiveRestart);
    return k;
}

bool EnsureObjResources(DeviceData* d) {
    if (g_dlss.objFailed) return false;
    if (g_dlss.objRenderPass) return true;
    g_dlss.objFailed = true;  // cleared on success
    VkDescriptorSetLayoutBinding b[2] = {};
    b[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 2;
    li.pBindings = b;
    if (d->CreateDescriptorSetLayout(d->device, &li, nullptr, &g_dlss.objSetLayout) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_dlss.objSetLayout;
    if (d->CreatePipelineLayout(d->device, &pli, nullptr, &g_dlss.objPipeLayout) != VK_SUCCESS) return false;
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
                                    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 8;
    pi.poolSizeCount = 2;
    pi.pPoolSizes = sizes;
    if (d->CreateDescriptorPool(d->device, &pi, nullptr, &g_dlss.objPool) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    si.codeSize = sizeof kObjVsSpirv;
    si.pCode = kObjVsSpirv;
    if (d->CreateShaderModule(d->device, &si, nullptr, &g_dlss.objVs) != VK_SUCCESS) return false;
    si.codeSize = sizeof kObjFsSpirv;
    si.pCode = kObjFsSpirv;
    if (d->CreateShaderModule(d->device, &si, nullptr, &g_dlss.objFs) != VK_SUCCESS) return false;
    if (!CreateBuffer(d, (VkDeviceSize)kObjSlots * kObjMax * 64, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &g_dlss.objBuf,
                      &g_dlss.objPtr))
        return false;
    VkAttachmentDescription att{};
    att.format = VK_FORMAT_R16G16_SFLOAT;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = att.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 1;
    rpi.pAttachments = &att;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sub;
    if (d->CreateRenderPass(d->device, &rpi, nullptr, &g_dlss.objRenderPass) != VK_SUCCESS) {
        g_dlss.objRenderPass = VK_NULL_HANDLE;
        return false;
    }
    g_dlss.objFailed = false;
    Log("[obj] object motion pass ready");
    return true;
}

VkPipeline ObjPipeline(DeviceData* d, const PipeVtx& p) {
    std::string key = ObjLayoutKey(p);
    auto it = g_dlss.objPipelines.find(key);
    if (it != g_dlss.objPipelines.end()) return it->second;
    VkVertexInputBindingDescription bind[3] = {
        {0, p.posStride, p.posInstanceRate ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX},
        {1, p.stride, VK_VERTEX_INPUT_RATE_INSTANCE},
        {2, 64, VK_VERTEX_INPUT_RATE_INSTANCE}};
    VkVertexInputAttributeDescription attr[9];
    attr[0] = {0, 0, p.posFormat, p.posOffset};
    for (uint32_t c = 0; c < 4; ++c) {
        attr[1 + c] = {1 + c, 1, p.fmt, p.colOffset[c]};
        attr[5 + c] = {5 + c, 2, VK_FORMAT_R32G32B32A32_SFLOAT, c * 16};
    }
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 3;
    vi.pVertexBindingDescriptions = bind;
    vi.vertexAttributeDescriptionCount = 9;
    vi.pVertexAttributeDescriptions = attr;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = p.topology;
    ia.primitiveRestartEnable = p.primitiveRestart;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 2;
    ds.pDynamicStates = dyn;
    VkPipelineShaderStageCreateInfo st[2] = {};
    st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, g_dlss.objVs,
             "main", nullptr};
    st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, g_dlss.objFs,
             "main", nullptr};
    VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gi.stageCount = 2;
    gi.pStages = st;
    gi.pVertexInputState = &vi;
    gi.pInputAssemblyState = &ia;
    gi.pViewportState = &vp;
    gi.pRasterizationState = &rs;
    gi.pMultisampleState = &ms;
    gi.pColorBlendState = &cb;
    gi.pDynamicState = &ds;
    gi.layout = g_dlss.objPipeLayout;
    gi.renderPass = g_dlss.objRenderPass;
    VkPipeline pipe = VK_NULL_HANDLE;
    // straight to the driver: our own CreateGraphicsPipelines hook must not patch or track it
    if (d->CreateGraphicsPipelines(d->device, VK_NULL_HANDLE, 1, &gi, nullptr, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
    g_dlss.objPipelines[key] = pipe;
    Log("[obj] pipeline for layout %s: %s", key.c_str(), pipe ? "created" : "FAILED");
    return pipe;
}

uint64_t Mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}
uint64_t MeshHash(const CmdState& cs, const PipeVtx& pv, bool indexed, uint32_t count, uint32_t first,
                  int32_t vertexOffset) {
    uint64_t h = indexed ? Mix((uint64_t)(uintptr_t)cs.ib, cs.ibOff) : 0;
    h = Mix(h, ((uint64_t)first << 32) | count);
    h = Mix(h, (uint64_t)(int64_t)vertexOffset);
    if (pv.posBinding < 16) h = Mix(Mix(h, (uint64_t)(uintptr_t)cs.vb[pv.posBinding].first), cs.vb[pv.posBinding].second);
    return h;
}
uint32_t FormatComps(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 4;
        case VK_FORMAT_R32G32B32_SFLOAT: return 3;
        default: return 0;
    }
}
bool ViewportJitterOn() { return JitterActive(); }

// Which draws get the jitter. Only 3D geometry: full-screen passes (reflections, ambient occlusion,
// bloom, ...) work per pixel on images that are already jittered, and shifting them as well made the
// screen-space reflections flicker black on water. 3D draws are jittered in every render-size target,
// so passes that the scene reads back (the water normals) line up with it. No lock needed.
bool JitterThisDraw(const CmdState& cs) {
    if (!g_hot.jitter.load(std::memory_order_relaxed)) return false;
    bool screenPass = cs.pipe && cs.pipe->screenPass;
    if (screenPass && !g_hot.jitterScreen.load(std::memory_order_relaxed)) return false;
    if (cs.fbIsScene) return true;
    uint32_t w = g_hot.inW.load(std::memory_order_relaxed), h = g_hot.inH.load(std::memory_order_relaxed);
    return g_hot.jitterOther.load(std::memory_order_relaxed) && !screenPass && w && cs.fbW == w && cs.fbH == h;
}
// Solid geometry only: overlay lines/points drawn in 3D (guides, routes...) aren't objects.
bool IsTriangles(VkPrimitiveTopology t) {
    return t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP ||
           t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
}

// Every scene draw, without a lock: the mesh if this draw is of a mesh that moved recently, else 0.
uint64_t ObjCandidate(const CmdState& cs, bool indexed, uint32_t count, uint32_t instanceCount, uint32_t first,
                      int32_t vertexOffset) {
    if (!g_hot.objects.load(std::memory_order_relaxed) || !cs.fbIsScene || !instanceCount || !cs.moving ||
        !cs.pipe || !cs.pipe->hasVtx)
        return 0;
    const PipeVtx& pv = cs.pipe->vtx;
    if (!pv.hasMdl || !pv.instanceRate || !IsTriangles(pv.topology) || pv.cols != 4 ||
        pv.fmt != VK_FORMAT_R32G32B32A32_SFLOAT || pv.mdlBinding >= 16 ||
        pv.posBinding >= 16 || !FormatComps(pv.posFormat) || pv.posInstanceRate)
        return 0;
    uint64_t mesh = MeshHash(cs, pv, indexed, count, first, vertexOffset);
    return cs.moving->count(mesh) ? mesh : 0;
}

// For a draw ObjCandidate picked: remember it for the object motion pass. Caller holds g_t.
void ObjRecordDraw(CmdState& cs, uint64_t mesh, bool indexed, uint32_t count, uint32_t instanceCount,
                   uint32_t first, int32_t vertexOffset, uint32_t firstInstance) {
    const PipeVtx& pv = cs.pipe->vtx;
    if (g_objFrame.frame != g_frame) {
        g_objFrame = ObjFrame{};
        g_objFrame.frame = g_frame;
    }
    ObjDraw od;
    od.layout = pv;
    od.posBuf = cs.vb[pv.posBinding].first;
    od.posOff = cs.vb[pv.posBinding].second;
    od.mdlBuf = cs.vb[pv.mdlBinding].first;
    od.mdlOff = cs.vb[pv.mdlBinding].second + (VkDeviceSize)firstInstance * pv.stride;
    if (!od.posBuf || !od.mdlBuf || (indexed && !cs.ib)) return;
    if (!g_objFrame.seen.insert({od.mdlBuf, od.mdlOff}).second) return;
    if (g_objFrame.instances + instanceCount > kObjMax) return;
    od.ib = cs.ib;
    od.ibOff = cs.ibOff;
    od.ibType = cs.ibType;
    od.indexed = indexed;
    od.count = count;
    od.first = first;
    od.instances = instanceCount;
    od.vertexOffset = vertexOffset;
    od.mesh = mesh;
    g_objFrame.instances += instanceCount;
    g_objFrame.draws.push_back(od);
}

// In RecordUpscale, after the camera motion vectors are in mvImage (TRANSFER_DST layout).
// Returns true if the pass was recorded (mvImage is then in COLOR_ATTACHMENT layout).
bool RecordObjectPass(DeviceData* d, VkCommandBuffer cb, CmdState& cs, VkImageView depthView, uint32_t dyn) {
    if (!g_cfg.objectMotion || g_objFrame.frame != g_frame || g_objFrame.draws.empty() ||
        !g_dlss.sceneVpValid)
        return false;
    if (!EnsureObjResources(d)) return false;
    if (g_dlss.objDepthView != depthView || !g_dlss.objSet) {
        bool isNew = false;
        VkDescriptorSet set = CachedSetRaw(d, g_dlss.objSets, depthView, g_dlss.objPool, g_dlss.objSetLayout,
                                           g_dlss.objSetUsed, isNew);
        if (!set) return false;
        if (isNew) {
            VkDescriptorImageInfo ii{g_dlss.pointSampler, depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorBufferInfo ub{g_dlss.uboBuffer, 0, sizeof(MvUbo)};
            VkWriteDescriptorSet w[2] = {};
            w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii, nullptr, nullptr};
            w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &ub, nullptr};
            d->UpdateDescriptorSets(d->device, 2, w, 0, nullptr);
        }
        g_dlss.objSet = set;
        g_dlss.objDepthView = depthView;
    }
    if (g_dlss.objFbView != g_dlss.mvView) {
        if (g_dlss.objFb) {
            d->DeviceWaitIdle(d->device);
            d->DestroyFramebuffer(d->device, g_dlss.objFb, nullptr);
        }
        VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fi.renderPass = g_dlss.objRenderPass;
        fi.attachmentCount = 1;
        fi.pAttachments = &g_dlss.mvView;
        fi.width = g_dlss.inW;
        fi.height = g_dlss.inH;
        fi.layers = 1;
        g_dlss.objFb = VK_NULL_HANDLE;
        if (d->CreateFramebuffer(d->device, &fi, nullptr, &g_dlss.objFb) != VK_SUCCESS) {
            g_dlss.objFb = VK_NULL_HANDLE;
            g_dlss.objFailed = true;
            Log("[obj] couldn't create the framebuffer; object motion off");
            return false;
        }
        g_dlss.objFbView = g_dlss.mvView;
    }
    uint32_t slot = (uint32_t)(g_dlss.objCounter++ % kObjSlots);
    Barrier(d, cb, g_dlss.mvImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderPassBeginInfo rb{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rb.renderPass = g_dlss.objRenderPass;
    rb.framebuffer = g_dlss.objFb;
    rb.renderArea = {{0, 0}, {g_dlss.inW, g_dlss.inH}};
    d->CmdBeginRenderPass(cb, &rb, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vp = g_dlss.sceneVp;
    if (ViewportJitterOn()) {
        float jx, jy;
        FrameJitter(jx, jy);
        vp.x += jx;
        vp.y += jy;
    }
    VkRect2D sc{{0, 0}, {g_dlss.inW, g_dlss.inH}};
    d->CmdSetViewport(cb, 0, 1, &vp);
    d->CmdSetScissor(cb, 0, 1, &sc);
    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_dlss.objPipeLayout, 0, 1, &g_dlss.objSet, 1, &dyn);
    cs.objDraws.clear();
    uint32_t next = 0;
    VkPipeline bound = VK_NULL_HANDLE;
    for (ObjDraw od : g_objFrame.draws) {
        VkPipeline pipe = ObjPipeline(d, od.layout);
        if (!pipe) continue;
        if (pipe != bound) {
            d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            bound = pipe;
        }
        od.dst = next;
        next += od.instances;
        VkBuffer bufs[3] = {od.posBuf, od.mdlBuf, g_dlss.objBuf};
        VkDeviceSize offs[3] = {od.posOff, od.mdlOff, ((VkDeviceSize)slot * kObjMax + od.dst) * 64};
        d->CmdBindVertexBuffers(cb, 0, 3, bufs, offs);
        if (od.indexed) {
            d->CmdBindIndexBuffer(cb, od.ib, od.ibOff, od.ibType);
            d->CmdDrawIndexed(cb, od.count, od.instances, od.first, od.vertexOffset, 0);
        } else
            d->CmdDraw(cb, od.count, od.instances, od.first, 0);
        cs.objDraws.push_back(od);
    }
    d->CmdEndRenderPass(cb);
    g_dlss.objSetUsed = true;
    cs.objSlot = slot;
    cs.objPending = !cs.objDraws.empty();
    return true;
}

// At submit: fill last frame's model matrix for every replayed instance (or this frame's when the
// object can't be matched, which leaves it with camera motion only).
void ObjFillPrev(CmdState& cs) {
    if (!cs.objPending || !g_dlss.objPtr) return;
    cs.objPending = false;
    bool havePrev = g_objPrevFrame != UINT64_MAX && g_objPrevFrame + 1 == g_frame;
    std::unordered_map<uint64_t, std::vector<std::array<float, 16>>> next;
    uint64_t moved = 0, unmatched = 0, total = 0;
    char* base = g_dlss.objPtr + (size_t)cs.objSlot * kObjMax * 64;
    // last frame's copies by exact position: parked/stopped copies are found without a search
    auto exactKey = [](uint64_t mesh, const float* m) {
        uint32_t x, y, z;
        std::memcpy(&x, &m[12], 4);
        std::memcpy(&y, &m[13], 4);
        std::memcpy(&z, &m[14], 4);
        return Mix(Mix(Mix(mesh, x), y), z);
    };
    std::unordered_map<uint64_t, const float*> exact;
    if (havePrev)
        for (const auto& [mesh, list] : g_objPrev)
            for (const auto& m : list) exact.emplace(exactKey(mesh, m.data()), m.data());
    for (const ObjDraw& od : cs.objDraws) {
        char* src = RefData(od.mdlBuf, od.mdlOff);
        auto& list = next[od.mesh];
        auto pp = havePrev ? g_objPrev.find(od.mesh) : g_objPrev.end();
        for (uint32_t i = 0; i < od.instances; ++i) {
            float cur[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            if (src)
                for (int c = 0; c < 4; ++c)
                    std::memcpy(cur + c * 4, src + (size_t)i * od.layout.stride + od.layout.colOffset[c], 16);
            const float* prev = cur;
            auto ex = src && havePrev ? exact.find(exactKey(od.mesh, cur)) : exact.end();
            if (ex != exact.end()) prev = ex->second;
            else if (src && pp != g_objPrev.end()) {
                float best = kObjMatchDist * kObjMatchDist;
                for (const auto& m : pp->second) {
                    float dx = m[12] - cur[12], dy = m[13] - cur[13], dz = m[14] - cur[14];
                    float d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 < best) best = d2, prev = m.data();
                }
            }
            if (prev == cur) ++unmatched;
            else if (std::memcmp(prev, cur, 64) != 0) ++moved;
            std::memcpy(base + (size_t)(od.dst + i) * 64, prev, 64);
            std::array<float, 16> a;
            std::memcpy(a.data(), cur, 64);
            list.push_back(a);
            ++total;
        }
    }
    g_objPrev = std::move(next);
    g_objPrevFrame = g_frame;
    ++g_objStats.frames;
    g_objStats.draws += cs.objDraws.size();
    g_objStats.instances += total;
    g_objStats.moved += moved;
    g_objStats.unmatched += unmatched;
    if (LogFrame(g_frame))
        Log("[obj] frame %llu: %zu draws, %llu instances of %zu moving meshes; %llu moved, %llu unmatched "
            "(totals: %llu frames, %llu instances, %llu moved)",
            (unsigned long long)g_frame, cs.objDraws.size(), (unsigned long long)total, g_movingMeshes.size(),
            (unsigned long long)moved, (unsigned long long)unmatched, (unsigned long long)g_objStats.frames,
            (unsigned long long)g_objStats.instances, (unsigned long long)g_objStats.moved);
}

// ---------------------------------------------------------------------------------------------
// FSR 3.1 (AMD FidelityFX API, amd_fidelityfx_vk.dll next to the layer)
// ---------------------------------------------------------------------------------------------
struct Fsr {
#ifdef _WIN32
    HMODULE lib = nullptr;
#endif
    bool loadTried = false, available = false, failed = false;
    bool deviceReady = false;  // the game's device was created with FSR's features
    std::string why = "not loaded";
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    PfnFfxQuery query = nullptr;
    ffxContext ctx = nullptr;
    VkPhysicalDevice appPhysicalDevice = VK_NULL_HANDLE;  // the loader's handle for our GPU (see AppPhysicalDevice)
    uint32_t inW = 0, inH = 0, outW = 0, outH = 0;
    int debugInUse = -1, srgbInUse = -1;
    double lastMs = 0;
    uint64_t dispatches = 0;
} g_fsr;

void FfxMessage(uint32_t type, const wchar_t* msg) {
    static int count = 0;
    if (count++ > 200) return;  // debug checking repeats itself every frame
    std::wstring m(msg ? msg : L"");
    while (!m.empty() && (m.back() == L'\n' || m.back() == L'\r')) m.pop_back();
    Log("[fsr] %s: %s", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", Narrow(m).c_str());
}

// Load amd_fidelityfx_vk.dll once. Caller holds g_t.
bool FsrLoad() {
    if (!g_fsr.deviceReady) return false;  // g_fsr.why says why
    if (g_fsr.loadTried) return g_fsr.available;
    g_fsr.loadTried = true;
#ifdef _WIN32
    std::wstring path = ModuleDir() + L"\\amd_fidelityfx_vk.dll";
    g_fsr.lib = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g_fsr.lib) {
        g_fsr.why = "amd_fidelityfx_vk.dll is missing from the mod's bin folder";
        Log("[fsr] couldn't load %s (error %lu)", Narrow(path).c_str(), (unsigned long)GetLastError());
        return false;
    }
    g_fsr.create = (PfnFfxCreateContext)(void*)GetProcAddress(g_fsr.lib, "ffxCreateContext");
    g_fsr.destroy = (PfnFfxDestroyContext)(void*)GetProcAddress(g_fsr.lib, "ffxDestroyContext");
    g_fsr.dispatch = (PfnFfxDispatch)(void*)GetProcAddress(g_fsr.lib, "ffxDispatch");
    g_fsr.query = (PfnFfxQuery)(void*)GetProcAddress(g_fsr.lib, "ffxQuery");
    if (!g_fsr.create || !g_fsr.destroy || !g_fsr.dispatch || !g_fsr.query) {
        g_fsr.why = "amd_fidelityfx_vk.dll is not a FidelityFX API runtime (wrong file?)";
        Log("[fsr] %s", g_fsr.why.c_str());
        return false;
    }
    g_fsr.available = true;
    Log("[fsr] loaded %s", Narrow(path).c_str());
#else
    g_fsr.why = "FSR is only available on Windows";
#endif
    return g_fsr.available;
}

// The FSR runtime calls instance functions (vkGetPhysicalDeviceProperties, ...) through vulkan-1.dll,
// i.e. the loader's trampolines, which only accept the application's physical-device handles. The
// handle a layer sees in vkCreateDevice is the loader's unwrapped one; passing that to FSR makes the
// loader abort the process. So ask the loader for its handles and pick the one for our GPU (by UUID).
// Caller holds g_t.
VkPhysicalDevice AppPhysicalDevice(DeviceData* d) {
#ifdef _WIN32
    if (g_fsr.appPhysicalDevice) return g_fsr.appPhysicalDevice;
    HMODULE vk = GetModuleHandleW(L"vulkan-1.dll");
    auto enumerate = vk ? (PFN_vkEnumeratePhysicalDevices)(void*)GetProcAddress(vk, "vkEnumeratePhysicalDevices") : nullptr;
    auto loaderProps2 =
        vk ? (PFN_vkGetPhysicalDeviceProperties2)(void*)GetProcAddress(vk, "vkGetPhysicalDeviceProperties2") : nullptr;
    InstanceData* inst = d->instance;
    auto ourProps2 = inst ? (PFN_vkGetPhysicalDeviceProperties2)inst->GetInstanceProcAddr(
                                inst->instance, "vkGetPhysicalDeviceProperties2")
                          : nullptr;
    if (!enumerate || !loaderProps2 || !ourProps2) {
        Log("[fsr] can't reach the Vulkan loader's functions (vulkan-1.dll %s)", vk ? "loaded" : "not loaded");
        return VK_NULL_HANDLE;
    }
    auto uuid = [](PFN_vkGetPhysicalDeviceProperties2 fn, VkPhysicalDevice pd, uint8_t out[VK_UUID_SIZE]) {
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p.pNext = &id;
        fn(pd, &p);
        std::memcpy(out, id.deviceUUID, VK_UUID_SIZE);
    };
    uint8_t ours[VK_UUID_SIZE];
    uuid(ourProps2, d->physicalDevice, ours);
    uint32_t n = 0;
    enumerate(inst->instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    if (n) enumerate(inst->instance, &n, pds.data());
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t theirs[VK_UUID_SIZE];
        uuid(loaderProps2, pds[i], theirs);
        if (!std::memcmp(ours, theirs, VK_UUID_SIZE)) {
            g_fsr.appPhysicalDevice = pds[i];
            Log("[fsr] GPU handle for the FSR runtime: %s (layer handle %s, %u GPUs)", H(pds[i]).c_str(),
                H(d->physicalDevice).c_str(), n);
            return pds[i];
        }
    }
    Log("[fsr] none of the loader's %u GPUs matches ours", n);
#endif
    return VK_NULL_HANDLE;
}

// (Re)create the FSR context for the current sizes. Caller holds g_t.
bool FsrEnsure(DeviceData* d) {
    int dbg = g_cfg.fsrDebug ? 1 : 0;
    if (g_fsr.ctx && g_fsr.inW == g_dlss.inW && g_fsr.inH == g_dlss.inH && g_fsr.outW == g_dlss.outW &&
        g_fsr.outH == g_dlss.outH && g_fsr.debugInUse == dbg && g_fsr.srgbInUse == g_cfg.fsrColorSrgb)
        return true;
    if (g_fsr.failed || !FsrLoad()) return false;
    if (g_fsr.ctx) {
        Log("[fsr] size or settings changed, recreating the context");
        d->DeviceWaitIdle(d->device);
        g_fsr.destroy(&g_fsr.ctx, nullptr);
        g_fsr.ctx = nullptr;
    }
    VkPhysicalDevice appPd = AppPhysicalDevice(d);
    if (!appPd) {
        g_fsr.why = "FSR 3.1 could not find the GPU through the Vulkan loader";
        g_fsr.failed = true;
        Log("[fsr] %s", g_fsr.why.c_str());
        return false;
    }
    ffxCreateBackendVKDesc be{};
    be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    be.vkDevice = d->device;
    be.vkPhysicalDevice = appPd;  // the loader's handle: FSR calls instance functions through vulkan-1.dll
    be.vkDeviceProcAddr = d->GetDeviceProcAddr;  // the next layer's: FSR's commands bypass our hooks
    ffxCreateContextDescUpscale cd{};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    cd.header.pNext = &be.header;
    // Depth is OpenGL style (near 0, far 1: not inverted); the input is tone-mapped LDR colour.
    cd.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    if (g_cfg.fsrColorSrgb) cd.flags |= FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE;
    if (dbg) cd.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
    cd.maxRenderSize = {g_dlss.inW, g_dlss.inH};
    cd.maxUpscaleSize = {g_dlss.outW, g_dlss.outH};
    cd.fpMessage = &FfxMessage;
    Log("[fsr] creating the context (%ux%u -> %ux%u)", g_dlss.inW, g_dlss.inH, g_dlss.outW, g_dlss.outH);
    ffxReturnCode_t r = g_fsr.create(&g_fsr.ctx, &cd.header, nullptr);
    if (r != FFX_API_RETURN_OK || !g_fsr.ctx) {
        char b[96];
        std::snprintf(b, sizeof b, "FSR 3.1 could not be started (FidelityFX error %u)", (unsigned)r);
        g_fsr.why = b;
        g_fsr.failed = true;
        g_fsr.ctx = nullptr;
        Log("[fsr] %s", b);
        return false;
    }
    g_fsr.inW = g_dlss.inW;
    g_fsr.inH = g_dlss.inH;
    g_fsr.outW = g_dlss.outW;
    g_fsr.outH = g_dlss.outH;
    g_fsr.debugInUse = dbg;
    g_fsr.srgbInUse = g_cfg.fsrColorSrgb;
    g_dlss.resetNext = true;
    ffxQueryGetProviderVersion pv{};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    const char* ver = g_fsr.query(&g_fsr.ctx, &pv.header) == FFX_API_RETURN_OK && pv.versionName ? pv.versionName : "?";
    Log("[fsr] context created: %ux%u -> %ux%u, FSR version %s, colour %s%s", g_dlss.inW, g_dlss.inH, g_dlss.outW,
        g_dlss.outH, ver, g_cfg.fsrColorSrgb ? "sRGB" : "linear", dbg ? ", debug checking on" : "");
    return true;
}

FfxApiResource FfxImage(VkImage img, VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage, uint32_t state) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.usage = usage;
    uint32_t extra = state == FFX_API_RESOURCE_STATE_UNORDERED_ACCESS ? (uint32_t)FFX_API_RESOURCE_USAGE_UAV : 0u;
    return ffxApiGetResourceVK((void*)img, ffxApiGetImageResourceDescriptionVK(img, ci, extra), state);
}

// Record FSR. Inputs are in SHADER_READ_ONLY_OPTIMAL, the output in GENERAL. Caller holds g_t.
bool RecordFsr(VkCommandBuffer cb, const CmdState& cs, const ImageInfo& ci) {
    ffxDispatchDescUpscale dd{};
    dd.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    dd.commandList = (void*)cb;
    const uint32_t RD = FFX_API_RESOURCE_STATE_COMPUTE_READ;
    dd.color = FfxImage(cs.colorImage, ci.fmt, ci.w, ci.h, ci.usage, RD);
    dd.depth = FfxImage(g_dlss.depth32Image, VK_FORMAT_R32_SFLOAT, g_dlss.inW, g_dlss.inH,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, RD);
    dd.motionVectors = FfxImage(g_dlss.mvImage, VK_FORMAT_R16G16_SFLOAT, g_dlss.inW, g_dlss.inH,
                                VK_IMAGE_USAGE_SAMPLED_BIT, RD);
    dd.output = FfxImage(g_dlss.outImage, VK_FORMAT_R8G8B8A8_UNORM, g_dlss.outW, g_dlss.outH,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    // Same conventions as for DLSS: jitter in render pixels, motion = previous - current in pixels.
    dd.jitterOffset = {cs.jitterSet ? cs.jx * g_cfg.fsrJitterSignX : 0.f, cs.jitterSet ? cs.jy * g_cfg.fsrJitterSignY : 0.f};
    dd.motionVectorScale = {g_cfg.mvScaleX, g_cfg.mvScaleY};
    dd.renderSize = {ci.w, ci.h};
    dd.upscaleSize = {g_dlss.outW, g_dlss.outH};
    dd.enableSharpening = g_cfg.fsrSharpness > 0.f;  // FSR's own RCAS instead of our sharpening pass
    dd.sharpness = g_cfg.fsrSharpness;
    double now = NowMs();
    dd.frameTimeDelta = g_fsr.lastMs > 0 ? (float)std::min(std::max(now - g_fsr.lastMs, 1.0), 100.0) : 16.7f;
    g_fsr.lastMs = now;
    dd.preExposure = 1.f;
    dd.reset = g_dlss.resetNext;
    dd.cameraNear = g_dlss.camNear;
    dd.cameraFar = g_dlss.camFar;
    dd.cameraFovAngleVertical = g_dlss.camFovY;
    dd.viewSpaceToMetersFactor = 1.f;  // the game works in metres
    dd.flags = (g_cfg.fsrColorSrgb ? (uint32_t)FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB : 0u) |
               (g_cfg.fsrDebug >= 2 ? (uint32_t)FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW : 0u);
    ffxReturnCode_t r = g_fsr.dispatch(&g_fsr.ctx, &dd.header);
    if (r != FFX_API_RETURN_OK) {
        static int logged = 0;
        if (logged++ < 5) Log("[fsr] dispatch failed: %u (frame %llu)", (unsigned)r, (unsigned long long)g_frame);
        return false;
    }
    ++g_fsr.dispatches;
    return true;
}

void DestroyFsr(DeviceData* d) {
    if (g_fsr.ctx) {
        d->DeviceWaitIdle(d->device);
        g_fsr.destroy(&g_fsr.ctx, nullptr);
        g_fsr.ctx = nullptr;
    }
    g_fsr.inW = g_fsr.inH = g_fsr.outW = g_fsr.outH = 0;
    g_fsr.appPhysicalDevice = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------------------------
// Performance measurement (perf=1), GPU side and the report
// ---------------------------------------------------------------------------------------------
// GPU timestamps around the work the layer adds to a frame (motion vectors, the upscaler, sharpening,
// the rest), and from the game's FSR 1 passes (EASU + RCAS, whose image the upscaler replaces) to
// the start of ours. A ring of kGpuSlots frames: a slot's results are read back, without waiting,
// just before it is reused; the GPU finished that frame long before.
constexpr uint32_t kGpuSlots = 8;
enum GpuMarkId { kGmGameFsr, kGmStart, kGmMotion, kGmUpscaler, kGmSharpen, kGmEnd, kGmCount };
constexpr uint32_t kGpuParts = kGmEnd - kGmStart;  // the intervals between kGmStart .. kGmEnd
const char* const kGpuPartNames[kGpuParts] = {"motion vectors", "upscaler", "sharpening",
                                              "info box, debug view, barriers"};
struct GpuPerf {
    VkQueryPool pool = VK_NULL_HANDLE;
    bool failed = false;
    double msPerTick = 0.0;
    uint32_t slot = 0;
    uint32_t written[kGpuSlots] = {};       // marks written into each slot since its reset, one bit per mark
    uint64_t gameFsrFrame[kGpuSlots] = {};  // frame of the slot's kGmGameFsr mark ...
    uint64_t startFrame[kGpuSlots] = {};    // ... and of its kGmStart mark: same frame = usable interval
    // sums over the report window
    double partMs[kGpuParts] = {}, totalMs = 0, gameFsrMs = 0;
    uint64_t frames = 0, gameFsrFrames = 0;
} g_gpu;

// Creates the query pool on first use and resets all of it in this command buffer (which must be
// outside a render pass). Caller holds g_t.
bool GpuPerfEnsure(DeviceData* d, VkCommandBuffer cb) {
    if (!g_cfg.perf || g_gpu.failed) return false;
    if (g_gpu.pool) return true;
    g_gpu.failed = true;  // cleared on success
    VkPhysicalDeviceProperties props{};
    if (d->instance && d->instance->GetPhysicalDeviceProperties)
        d->instance->GetPhysicalDeviceProperties(d->physicalDevice, &props);
    if (!props.limits.timestampComputeAndGraphics || props.limits.timestampPeriod <= 0.f) {
        Log("[perf] this GPU can't time graphics work (no timestamps); no GPU timings");
        return false;
    }
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = kGpuSlots * kGmCount;
    if (d->CreateQueryPool(d->device, &qi, nullptr, &g_gpu.pool) != VK_SUCCESS) {
        g_gpu.pool = VK_NULL_HANDLE;
        Log("[perf] couldn't create the timestamp query pool; no GPU timings");
        return false;
    }
    d->CmdResetQueryPool(cb, g_gpu.pool, 0, qi.queryCount);
    g_gpu.msPerTick = props.limits.timestampPeriod * 1e-6;
    g_gpu.slot = 0;
    std::memset(g_gpu.written, 0, sizeof g_gpu.written);
    g_gpu.failed = false;
    Log("[perf] GPU timings ready (timestamp period %.3g ns)", props.limits.timestampPeriod);
    return true;
}

// A timestamp into the current slot, once per slot. Caller holds g_t.
void GpuMark(DeviceData* d, VkCommandBuffer cb, GpuMarkId m) {
    if (!g_cfg.perf || !g_gpu.pool) return;
    uint32_t bit = 1u << m;
    if (g_gpu.written[g_gpu.slot] & bit) return;  // a query is written once between resets
    d->CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_gpu.pool, g_gpu.slot * kGmCount + m);
    g_gpu.written[g_gpu.slot] |= bit;
    if (m == kGmGameFsr) g_gpu.gameFsrFrame[g_gpu.slot] = g_frame;
    if (m == kGmStart) g_gpu.startFrame[g_gpu.slot] = g_frame;
}

// End of the layer's GPU work for a frame (outside a render pass): read the next slot back, add it to
// the window, reset it and make it current. Caller holds g_t.
void GpuFrameEnd(DeviceData* d, VkCommandBuffer cb) {
    if (!g_cfg.perf || !g_gpu.pool) return;
    uint32_t next = (g_gpu.slot + 1) % kGpuSlots;
    uint32_t w = g_gpu.written[next];
    if (w) {
        uint64_t q[kGmCount][2] = {};  // value, availability
        d->GetQueryPoolResults(d->device, g_gpu.pool, next * kGmCount, kGmCount, sizeof q, q, sizeof q[0],
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        auto have = [&](int m) { return ((w >> m) & 1u) && q[m][1]; };
        auto ms = [&](int a, int b) { return (double)(int64_t)(q[b][0] - q[a][0]) * g_gpu.msPerTick; };
        bool all = true;
        for (int m = kGmStart; m <= kGmEnd; ++m) all = all && have(m);
        double total = all ? ms(kGmStart, kGmEnd) : -1.0;
        if (total >= 0.0 && total < 1000.0) {
            for (uint32_t i = 0; i < kGpuParts; ++i) g_gpu.partMs[i] += ms(kGmStart + i, kGmStart + i + 1);
            g_gpu.totalMs += total;
            ++g_gpu.frames;
            double game = have(kGmGameFsr) && g_gpu.gameFsrFrame[next] == g_gpu.startFrame[next]
                              ? ms(kGmGameFsr, kGmStart) : -1.0;
            if (game >= 0.0 && game < 1000.0) {
                g_gpu.gameFsrMs += game;
                ++g_gpu.gameFsrFrames;
            }
        }
    }
    d->CmdResetQueryPool(cb, g_gpu.pool, next * kGmCount, kGmCount);
    g_gpu.written[next] = 0;
    g_gpu.slot = next;
}

void DestroyGpuPerf(DeviceData* d) {
    if (g_gpu.pool) d->DestroyQueryPool(d->device, g_gpu.pool, nullptr);
    g_gpu = GpuPerf{};
}

// The report window: frame times (present to present) and the counters above, logged every
// perf_seconds and summed up in the info box.
struct PerfWindow {
    bool running = false;
    double startMs = 0, lastMs = 0;
    uint64_t startTsc = 0;
    std::vector<float> frameMs;
    double scopeTicks = 0;     // what one PerfScope costs, roughly (PerfScopeTicks)
    std::string overlayLine;   // for the info box
} g_pw;

// Rough cost of one measurement (two counter readings and two atomic adds), so the report can say
// how much the measuring itself adds.
double PerfScopeTicks() {
    std::atomic<uint64_t> a{0}, b{0};
    const int n = 4096;
    uint64_t t0 = TF3_TICKS();
    for (int i = 0; i < n; ++i) {
        uint64_t t = TF3_TICKS();
        a.fetch_add(TF3_TICKS() - t, std::memory_order_relaxed);
        b.fetch_add(1, std::memory_order_relaxed);
    }
    return (double)(TF3_TICKS() - t0) / n;
}

void PerfReset() {
    for (int c = 0; c < kPerfCount; ++c) {
        g_perf.ticks[c].store(0, std::memory_order_relaxed);
        g_perf.calls[c].store(0, std::memory_order_relaxed);
    }
    g_perf.lockWaits.store(0, std::memory_order_relaxed);
    g_perf.lockWaitTicks.store(0, std::memory_order_relaxed);
    for (double& m : g_gpu.partMs) m = 0.0;
    g_gpu.totalMs = g_gpu.gameFsrMs = 0.0;
    g_gpu.frames = g_gpu.gameFsrFrames = 0;
#ifdef __APPLE__
    mfx_gpu_window(nullptr, nullptr, nullptr, nullptr);  // MetalFX's own GPU times: start a new window too
    mfx_fg_gpu_window(nullptr, nullptr, nullptr, nullptr);
#endif
}

// Once per presented frame. Caller holds g_t.
void PerfPresent() {
    if (!g_cfg.perf) {
        if (g_pw.running) Log("[perf] measuring stopped");
        g_pw = PerfWindow{};
        return;
    }
    double now = NowMs();
    uint64_t tsc = TF3_TICKS();
    if (!g_pw.running) {
        PerfReset();
        g_pw = PerfWindow{};
        g_pw.running = true;
        g_pw.startMs = g_pw.lastMs = now;
        g_pw.startTsc = tsc;
        g_pw.scopeTicks = PerfScopeTicks();
        g_pw.overlayLine = "PERF: MEASURING...";
#ifdef __APPLE__
        mfx_sys_start();
#endif
        Log("[perf] measuring the layer's cost: a report every %g s (perf_seconds)", g_cfg.perfSeconds);
        return;
    }
    g_pw.frameMs.push_back((float)(now - g_pw.lastMs));
    g_pw.lastMs = now;
    double winMs = now - g_pw.startMs;
    if (winMs < g_cfg.perfSeconds * 1000.0 || g_pw.frameMs.empty() || tsc <= g_pw.startTsc) return;

    double tpm = (double)(tsc - g_pw.startTsc) / winMs;  // counter ticks per millisecond
    double n = (double)g_pw.frameMs.size();
    std::vector<float> sorted = g_pw.frameMs;
    std::sort(sorted.begin(), sorted.end());
    float p99 = sorted[(size_t)(0.99 * (double)(sorted.size() - 1))];
    double avg = winMs / n;
    bool running = g_dlss.enabled && g_lastDlssFrame != UINT64_MAX && g_lastDlssFrame + 1 >= g_frame;
    char up[96];
    if (!running) std::snprintf(up, sizeof up, "upscaler not running (the game's FSR 1)");
    else if (g_dlss.backend == Backend::Dlss && g_cfg.preset > 0)
        std::snprintf(up, sizeof up, "DLSS preset %c %ux%u -> %ux%u", 'A' + g_cfg.preset - 1, g_dlss.inW, g_dlss.inH,
                      g_dlss.outW, g_dlss.outH);
    else std::snprintf(up, sizeof up, "%s %ux%u -> %ux%u", BackendName(g_dlss.backend), g_dlss.inW, g_dlss.inH,
                       g_dlss.outW, g_dlss.outH);
    if (running && g_dlss.outH != g_swapExtent.height) {
        size_t len = std::strlen(up);
        std::snprintf(up + len, sizeof up - len, " (screen %ux%u)", g_swapExtent.width, g_swapExtent.height);
    }
    Log("[perf] %.1f s, %.0f frames: %.2f ms per frame (%.1f fps), slowest 1%%: %.2f ms | %s | skip_game_fsr=%d",
        winMs / 1000.0, n, avg, 1000.0 / avg, p99, up, g_cfg.skipGameFsr);

    double cpuMs = 0.0, scopes = 0.0;
    std::string parts;
    for (int c = 0; c < kPerfCount; ++c) {
        uint64_t t = g_perf.ticks[c].load(std::memory_order_relaxed), k = g_perf.calls[c].load(std::memory_order_relaxed);
        if (!k) continue;
        double ms = (double)t / tpm / n;
        cpuMs += ms;
        scopes += (double)k;
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %.3f (%.0f calls)", parts.empty() ? "" : ", ", kPerfNames[c], ms, (double)k / n);
        parts += b;
    }
    Log("[perf]   layer CPU: %.3f ms per frame (%.1f%% of the frame): %s", cpuMs, 100.0 * cpuMs / avg,
        parts.empty() ? "nothing measured" : parts.c_str());
    uint64_t waits = g_perf.lockWaits.load(std::memory_order_relaxed);
    Log("[perf]   measuring itself adds about %.3f ms per frame; game threads waited for the layer's lock %.0f times "
        "per frame, %.3f ms in all",
        g_pw.scopeTicks * scopes / tpm / n, (double)waits / n,
        (double)g_perf.lockWaitTicks.load(std::memory_order_relaxed) / tpm / n);
    double gpuMs = g_gpu.frames ? g_gpu.totalMs / (double)g_gpu.frames : 0.0;
    double upMs = g_gpu.frames ? g_gpu.partMs[1] / (double)g_gpu.frames : 0.0;
    if (g_gpu.frames) {
        std::string gp;
        for (uint32_t i = 0; i < kGpuParts; ++i) {
            char b[96];
            std::snprintf(b, sizeof b, "%s%s %.3f", i ? ", " : "", kGpuPartNames[i], g_gpu.partMs[i] / (double)g_gpu.frames);
            gp += b;
        }
        char game[96] = "not measured";
        if (g_gpu.gameFsrFrames)
            std::snprintf(game, sizeof game, "%.3f ms", g_gpu.gameFsrMs / (double)g_gpu.gameFsrFrames);
        Log("[perf]   GPU: the layer's passes %.3f ms per frame (%s); the game's FSR 1 passes before them %s",
            gpuMs, gp.c_str(), game);
    } else
        Log("[perf]   GPU: %s", g_gpu.failed ? "no timings on this GPU" : "nothing measured (no upscaler ran)");
    const char* upName = g_dlss.backend == Backend::Fsr ? "FSR" : g_dlss.backend == Backend::MetalFx ? "MFX" : "DLSS";
#ifdef __APPLE__
    // MetalFX runs on its own Metal queue: the layer's Vulkan "upscaler" part above is the colour copy plus
    // the wait for MetalFX, so its own command buffer times are the cost of MetalFX itself.
    // The median is the typical cost: now and then a command buffer's time also covers some of the game's work.
    double mfxMed = 0.0, mfxSum = 0.0, mfxMax = 0.0;
    uint32_t mfxN = 0;
    mfx_gpu_window(&mfxMed, &mfxSum, &mfxMax, &mfxN);
    if (mfxN) {
        upMs = mfxMed;
        Log("[perf]   MetalFX GPU: %.3f ms per frame (median, %.1f%% of the frame); mean %.3f, slowest %.3f ms, "
            "%u frames | %ux%u -> %ux%u",
            mfxMed, 100.0 * mfxMed / avg, mfxSum / mfxN, mfxMax, mfxN, g_dlss.inW, g_dlss.inH, g_dlss.outW,
            g_dlss.outH);
    }
    double fgMed = 0.0, fgSum = 0.0, fgMax = 0.0;
    uint32_t fgN = 0;
    mfx_fg_gpu_window(&fgMed, &fgSum, &fgMax, &fgN);
    if (fgN)
        Log("[perf]   frame generation GPU: %.3f ms per generated frame (median, %.1f%% of the game's frame); mean "
            "%.3f, slowest %.3f ms, %u frames generated",
            fgMed, 100.0 * fgMed / avg, fgSum / fgN, fgMax, fgN);
    // What limits the frame rate: a GPU that is busy nearly all the time, or one game thread that is.
    double gpuBusy = -1.0, procCpu = 0.0, topThread = 0.0;
    char threads[192] = {};
    mfx_sys_window(&gpuBusy, &procCpu, &topThread, threads, sizeof threads);
    double fps = 1000.0 / avg;
    const char* limit = fps > 57.0 && fps < 61.0                ? "the display's 60 Hz (vsync)"
                      : gpuBusy >= 90.0                            ? "the GPU"
                      : topThread >= 85.0                          ? "the CPU (the busiest game thread)"
                                                                   : "unclear (neither the GPU nor one thread is maxed)";
    Log("[perf]   system: GPU busy %.0f%% (driver statistic, all apps), game CPU %.0f%% (100 = one core); busiest "
        "threads: %s | limit: %s",
        gpuBusy, procCpu, threads, limit);
#endif
    char ov[96];
#ifdef __APPLE__
    if (gpuBusy >= 0.0)
        std::snprintf(ov, sizeof ov, "GPU BUSY %.0f%%  TOP THREAD %.0f%%  %s %.1f MS", gpuBusy, topThread, upName,
                      upMs);
    else
#endif
    if (g_gpu.frames)
        std::snprintf(ov, sizeof ov, "LAYER CPU %.2f MS  GPU %.2f MS  %s %.2f", cpuMs, gpuMs, upName, upMs);
    else std::snprintf(ov, sizeof ov, "LAYER CPU %.2f MS  GPU -", cpuMs);
    g_pw.overlayLine = ov;

    PerfReset();
    g_pw.startMs = now;
    g_pw.startTsc = tsc;
    g_pw.frameMs.clear();
}

// ---------------------------------------------------------------------------------------------
// On-screen info box (overlay=1, or Ctrl+Shift+O with debug_keys=1)
// ---------------------------------------------------------------------------------------------
// A few lines of text drawn into the upscaled image by overlay.comp, so settings can be compared
// while tuning them with the hotkeys. Only shown while an upscaler runs (it draws into our output).
const uint32_t kOvSpirv[] = {
#include "overlay.spv.inc"
};
constexpr uint32_t kOvLines = 7, kOvCols = 44;    // at most; the 7th line is perf=1's
constexpr uint32_t kOvFontWords = 128;           // 64 glyphs (ASCII 32..95) x 2 words
constexpr uint32_t kOvSlotWords = 128;           // text of one frame: kOvLines x kOvCols bytes, rounded up
constexpr uint32_t kOvSlots = 8;                 // frames in flight never reach this
static_assert(kOvLines * kOvCols <= kOvSlotWords * 4, "overlay text must fit a slot");
struct OvPush {
    int32_t origin[2], size[2];
    int32_t scale, lines, cols, textBase;
};

// 5x7 font: 7 rows of 5 pixels per character, top row first, separated by '|'. Characters not
// listed draw blank; lower case is drawn as upper case.
struct Glyph {
    char c;
    const char* rows;
};
const Glyph kFont[] = {
    {'0', ".###.|#...#|#..##|#.#.#|##..#|#...#|.###."}, {'1', "..#..|.##..|..#..|..#..|..#..|..#..|.###."},
    {'2', ".###.|#...#|....#|...#.|..#..|.#...|#####"}, {'3', "#####|...#.|..#..|...#.|....#|#...#|.###."},
    {'4', "...#.|..##.|.#.#.|#..#.|#####|...#.|...#."}, {'5', "#####|#....|####.|....#|....#|#...#|.###."},
    {'6', "..##.|.#...|#....|####.|#...#|#...#|.###."}, {'7', "#####|....#|...#.|..#..|.#...|.#...|.#..."},
    {'8', ".###.|#...#|#...#|.###.|#...#|#...#|.###."}, {'9', ".###.|#...#|#...#|.####|....#|...#.|.##.."},
    {'A', ".###.|#...#|#...#|#####|#...#|#...#|#...#"}, {'B', "####.|#...#|#...#|####.|#...#|#...#|####."},
    {'C', ".###.|#...#|#....|#....|#....|#...#|.###."}, {'D', "###..|#..#.|#...#|#...#|#...#|#..#.|###.."},
    {'E', "#####|#....|#....|####.|#....|#....|#####"}, {'F', "#####|#....|#....|####.|#....|#....|#...."},
    {'G', ".###.|#...#|#....|#.###|#...#|#...#|.####"}, {'H', "#...#|#...#|#...#|#####|#...#|#...#|#...#"},
    {'I', ".###.|..#..|..#..|..#..|..#..|..#..|.###."}, {'J', "..###|...#.|...#.|...#.|...#.|#..#.|.##.."},
    {'K', "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#"}, {'L', "#....|#....|#....|#....|#....|#....|#####"},
    {'M', "#...#|##.##|#.#.#|#.#.#|#...#|#...#|#...#"}, {'N', "#...#|#...#|##..#|#.#.#|#..##|#...#|#...#"},
    {'O', ".###.|#...#|#...#|#...#|#...#|#...#|.###."}, {'P', "####.|#...#|#...#|####.|#....|#....|#...."},
    {'Q', ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#"}, {'R', "####.|#...#|#...#|####.|#.#..|#..#.|#...#"},
    {'S', ".####|#....|#....|.###.|....#|....#|####."}, {'T', "#####|..#..|..#..|..#..|..#..|..#..|..#.."},
    {'U', "#...#|#...#|#...#|#...#|#...#|#...#|.###."}, {'V', "#...#|#...#|#...#|#...#|#...#|.#.#.|..#.."},
    {'W', "#...#|#...#|#...#|#.#.#|#.#.#|#.#.#|.#.#."}, {'X', "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#"},
    {'Y', "#...#|#...#|.#.#.|..#..|..#..|..#..|..#.."}, {'Z', "#####|....#|...#.|..#..|.#...|#....|#####"},
    {'.', ".....|.....|.....|.....|.....|.##..|.##.."}, {',', ".....|.....|.....|.....|.##..|..#..|.#..."},
    {':', ".....|.##..|.##..|.....|.##..|.##..|....."}, {'-', ".....|.....|.....|#####|.....|.....|....."},
    {'+', ".....|..#..|..#..|#####|..#..|..#..|....."}, {'(', "...#.|..#..|.#...|.#...|.#...|..#..|...#."},
    {')', ".#...|..#..|...#.|...#.|...#.|..#..|.#..."}, {'[', ".###.|.#...|.#...|.#...|.#...|.#...|.###."},
    {']', ".###.|...#.|...#.|...#.|...#.|...#.|.###."}, {'/', ".....|....#|...#.|..#..|.#...|#....|....."},
    {'>', ".#...|..#..|...#.|....#|...#.|..#..|.#..."}, {'%', "##...|##..#|...#.|..#..|.#...|#..##|...##"},
    {'=', ".....|.....|#####|.....|#####|.....|....."}, {'!', "..#..|..#..|..#..|..#..|..#..|.....|..#.."},
    {'?', ".###.|#...#|....#|...#.|..#..|.....|..#.."},
};

// Packs kFont for overlay.comp: glyph g = ASCII 32 + g, rows 0-3 in word 2g, rows 4-6 in word 2g+1,
// bit row * 5 + (4 - column).
void PackFont(uint32_t* words) {
    std::memset(words, 0, kOvFontWords * 4);
    for (const Glyph& g : kFont) {
        std::string rows;
        for (const char* r = g.rows; *r; ++r)
            if (*r == '#' || *r == '.') rows += *r;  // '|' and other separators are only for reading
        if (rows.size() != 35) {
            Log("[overlay] font: glyph '%c' has %zu pixels, not 35; drawn blank", g.c, rows.size());
            continue;
        }
        uint32_t idx = (uint32_t)(g.c - 32);
        for (int row = 0; row < 7; ++row)
            for (int col = 0; col < 5; ++col)
                if (rows[row * 5 + col] == '#')
                    words[idx * 2 + (row < 4 ? 0 : 1)] |= 1u << ((row < 4 ? row : row - 4) * 5 + (4 - col));
    }
}

bool EnsureOverlay(DeviceData* d) {
    if (g_dlss.ovFailed) return false;
    if (!g_dlss.ovPipeline) {
        g_dlss.ovFailed = true;  // cleared on success
        VkDescriptorSetLayoutBinding b[2] = {};
        b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 2;
        li.pBindings = b;
        if (d->CreateDescriptorSetLayout(d->device, &li, nullptr, &g_dlss.ovSetLayout) != VK_SUCCESS) return false;
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(OvPush)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &g_dlss.ovSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pr;
        if (d->CreatePipelineLayout(d->device, &pli, nullptr, &g_dlss.ovPipeLayout) != VK_SUCCESS) return false;
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = sizeof kOvSpirv;
        si.pCode = kOvSpirv;
        VkShaderModule mod;
        if (d->CreateShaderModule(d->device, &si, nullptr, &mod) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = g_dlss.ovPipeLayout;
        VkResult r = d->CreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &ci, nullptr, &g_dlss.ovPipeline);
        d->DestroyShaderModule(d->device, mod, nullptr);
        if (r != VK_SUCCESS) return false;
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 1;
        pi.poolSizeCount = 2;
        pi.pPoolSizes = sizes;
        if (d->CreateDescriptorPool(d->device, &pi, nullptr, &g_dlss.ovPool) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = g_dlss.ovPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &g_dlss.ovSetLayout;
        if (d->AllocateDescriptorSets(d->device, &ai, &g_dlss.ovSet) != VK_SUCCESS) return false;
        VkDeviceSize size = (kOvFontWords + kOvSlots * kOvSlotWords) * 4;
        if (!CreateBuffer(d, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &g_dlss.ovBuffer,
                          &g_dlss.ovPtr))
            return false;
        PackFont(reinterpret_cast<uint32_t*>(g_dlss.ovPtr));
        VkDescriptorBufferInfo bi{g_dlss.ovBuffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_dlss.ovSet, 1, 0, 1,
                               VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi, nullptr};
        d->UpdateDescriptorSets(d->device, 1, &w, 0, nullptr);
        g_dlss.ovFailed = false;
        Log("[overlay] info box ready");
    }
    if (g_dlss.ovBoundOut != g_dlss.outView) {  // output image recreated (size change)
        if (g_dlss.ovSetUsed) d->DeviceWaitIdle(d->device);
        VkDescriptorImageInfo img{VK_NULL_HANDLE, g_dlss.outView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_dlss.ovSet, 0, 0, 1,
                               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img, nullptr, nullptr};
        d->UpdateDescriptorSets(d->device, 1, &w, 0, nullptr);
        g_dlss.ovBoundOut = g_dlss.outView;
    }
    return true;
}

// The info box's lines for this frame.
#ifdef __APPLE__
bool FgActive();
#endif

std::vector<std::string> OverlayLines(Backend backend, const CmdState& cs, uint32_t inW, uint32_t inH) {
    char b[kOvLines][96];
    bool fsr = backend == Backend::Fsr;
    if (fsr) std::snprintf(b[0], sizeof b[0], "TF3-DLSS " TF3_DLSS_VERSION "  FSR 3.1");
    else if (backend == Backend::MetalFx) std::snprintf(b[0], sizeof b[0], "TF3-DLSS " TF3_DLSS_VERSION "  METALFX TEMPORAL");
    else if (backend == Backend::Test) std::snprintf(b[0], sizeof b[0], "TF3-DLSS " TF3_DLSS_VERSION "  TEST UPSCALER");
    else if (g_cfg.preset > 0)
        std::snprintf(b[0], sizeof b[0], "TF3-DLSS " TF3_DLSS_VERSION "  DLSS, PRESET %c", 'A' + g_cfg.preset - 1);
    else std::snprintf(b[0], sizeof b[0], "TF3-DLSS " TF3_DLSS_VERSION "  DLSS, PRESET AUTO");
    if (g_dlss.outH && g_dlss.outH != g_swapExtent.height)  // upscaled to less than the screen; the blit stretches it
        std::snprintf(b[1], sizeof b[1], "%ux%u -> %ux%u -> %uP", inW, inH, g_dlss.outW, g_dlss.outH,
                      g_swapExtent.height);
    else std::snprintf(b[1], sizeof b[1], "%ux%u -> %ux%u (%d%%)", inW, inH, g_dlss.outW, g_dlss.outH,
                       g_dlss.outW ? (int)std::lround(100.0 * inW / g_dlss.outW) : 0);
#ifdef __APPLE__
    if (backend == Backend::MetalFx && mfx_gpu_ms() > 0.f) {  // MetalFX's own GPU time, median of the last frames
        size_t n = std::strlen(b[1]);
        std::snprintf(b[1] + n, sizeof b[1] - n, "  GPU %.1f MS", mfx_gpu_ms());
    }
    bool fg = FgActive();
    if (fg) {
        size_t n = std::strlen(b[1]);
        std::snprintf(b[1] + n, sizeof b[1] - n, "  FG %.1f MS", mfx_fg_gpu_ms());
    }
#else
    const bool fg = false;
#endif
    if (fsr) std::snprintf(b[2], sizeof b[2], "SHARPNESS %.1f (FSR RCAS)  [CTRL+SHIFT+UP/DOWN]", g_cfg.fsrSharpness);
    else if (g_cfg.motionSharpness != 0.f)
        std::snprintf(b[2], sizeof b[2], "SHARP %.1f %s, MOTION %+.1f  [UP/DN S A]", g_cfg.sharpness,
                      g_cfg.sharpenType == 1 ? "CAS" : "RCAS", g_cfg.motionSharpness);
    else std::snprintf(b[2], sizeof b[2], "SHARP %.1f %s, NO MOTION  [UP/DN S A]", g_cfg.sharpness,
                       g_cfg.sharpenType == 1 ? "CAS" : "RCAS");
    float sx = fsr ? g_cfg.fsrJitterSignX : g_cfg.jitterSignX, sy = fsr ? g_cfg.fsrJitterSignY : g_cfg.jitterSignY;
    if (cs.jitterSet)
        std::snprintf(b[3], sizeof b[3], "JITTER %u/%u, SIGN X%+.0f Y%+.0f   [K, J]", cs.jitterIdx + 1,
                      cs.jitterPhases, sx, sy);
    else std::snprintf(b[3], sizeof b[3], "JITTER OFF, SIGN X%+.0f Y%+.0f       [K, J]", sx, sy);
    std::snprintf(b[4], sizeof b[4], "MOTION %s",
                  !g_cfg.motionVectors ? "OFF                     [M]"
                  : g_cfg.objectMotion ? "CAMERA + OBJECTS        [M]"
                                       : "CAMERA ONLY             [M]");
    if (fg)
        std::snprintf(b[5], sizeof b[5], "FPS %.0f -> %.0f SHOWN  [F]FG [D]OFF [O]HIDE", g_fps, 2.f * g_fps);
    else
        std::snprintf(b[5], sizeof b[5], "FPS %.0f (%.1f MS)  [U]SWITCH [D]OFF [O]HIDE", g_fps,
                      g_fps > 0.f ? 1000.f / g_fps : 0.f);
    std::snprintf(b[6], sizeof b[6], "%s", g_pw.overlayLine.c_str());
    return std::vector<std::string>(std::begin(b), std::begin(b) + (g_cfg.perf ? 7 : 6));
}

// Draws the info box into outImage (in GENERAL, after the upscaler and sharpening).
void RecordOverlay(DeviceData* d, VkCommandBuffer cb, Backend backend, const CmdState& cs, uint32_t inW,
                   uint32_t inH) {
    if (!g_cfg.overlay || !EnsureOverlay(d)) return;
    uint32_t slot = (uint32_t)(g_dlss.ovCounter++ % kOvSlots);
    uint32_t base = kOvFontWords + slot * kOvSlotWords;
    char* text = g_dlss.ovPtr + base * 4;
    std::memset(text, ' ', kOvSlotWords * 4);
    std::vector<std::string> lines = OverlayLines(backend, cs, inW, inH);
    uint32_t nLines = (uint32_t)std::min<size_t>(lines.size(), kOvLines);
    for (uint32_t l = 0; l < nLines; ++l)
        for (uint32_t c = 0; c < kOvCols && c < lines[l].size(); ++c)
            text[l * kOvCols + c] = (char)std::toupper((unsigned char)lines[l][c]);
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.image = g_dlss.outImage;
    b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                          0, nullptr, 1, &b);
    int32_t scale = std::max<int32_t>(1, (int32_t)g_dlss.outH / 720);  // 3 at 4K, 2 at 1440p
    uint32_t w = (kOvCols * 6 + 3) * scale, h = (nLines * 9 + 3) * scale;
    // overlay_x / overlay_y: position in percent of the screen, kept fully on screen
    int32_t x = (int32_t)std::lround(g_cfg.overlayX / 100.0 * g_dlss.outW);
    int32_t y = (int32_t)std::lround(g_cfg.overlayY / 100.0 * g_dlss.outH);
    x = std::max(0, std::min(x, (int32_t)g_dlss.outW - (int32_t)w));
    y = std::max(0, std::min(y, (int32_t)g_dlss.outH - (int32_t)h));
    OvPush pc{{x, y}, {(int32_t)g_dlss.outW, (int32_t)g_dlss.outH}, scale, (int32_t)nLines, (int32_t)kOvCols,
              (int32_t)base};
    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dlss.ovPipeline);
    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dlss.ovPipeLayout, 0, 1, &g_dlss.ovSet, 0, nullptr);
    d->CmdPushConstants(cb, g_dlss.ovPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    d->CmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                          0, nullptr, 1, &b);
    g_dlss.ovSetUsed = true;
}

// ---------------------------------------------------------------------------------------------
// Test upscaler (TestUpscaler): bilinear resize of the input colour into outImage
// ---------------------------------------------------------------------------------------------
const uint32_t kTstSpirv[] = {
#include "test_upscale.spv.inc"
};
constexpr uint32_t kTstSets = 8;

bool EnsureTestUpscale(DeviceData* d) {
    if (g_dlss.tstFailed) return false;
    if (!g_dlss.tstPipeline) {
        g_dlss.tstFailed = true;  // cleared on success
        VkDescriptorSetLayoutBinding b[2] = {};
        b[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 2;
        li.pBindings = b;
        if (d->CreateDescriptorSetLayout(d->device, &li, nullptr, &g_dlss.tstSetLayout) != VK_SUCCESS) return false;
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &g_dlss.tstSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pr;
        if (d->CreatePipelineLayout(d->device, &pli, nullptr, &g_dlss.tstPipeLayout) != VK_SUCCESS) return false;
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = sizeof kTstSpirv;
        si.pCode = kTstSpirv;
        VkShaderModule mod;
        if (d->CreateShaderModule(d->device, &si, nullptr, &mod) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = g_dlss.tstPipeLayout;
        VkResult r = d->CreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &ci, nullptr, &g_dlss.tstPipeline);
        d->DestroyShaderModule(d->device, mod, nullptr);
        if (r != VK_SUCCESS) return false;
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kTstSets},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kTstSets}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = kTstSets;
        pi.poolSizeCount = 2;
        pi.pPoolSizes = sizes;
        if (d->CreateDescriptorPool(d->device, &pi, nullptr, &g_dlss.tstPool) != VK_SUCCESS) return false;
        VkSamplerCreateInfo smp{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        smp.magFilter = smp.minFilter = VK_FILTER_LINEAR;
        smp.addressModeU = smp.addressModeV = smp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (d->CreateSampler(d->device, &smp, nullptr, &g_dlss.tstSampler) != VK_SUCCESS) return false;
        g_dlss.tstFailed = false;
        Log("[test] test upscaler ready (TF3_DLSS_TEST_UPSCALER=1)");
    }
    if (g_dlss.tstBoundOut != g_dlss.outView) {  // output image recreated: the sets point at the old one
        if (g_dlss.tstSetUsed) d->DeviceWaitIdle(d->device);
        d->ResetDescriptorPool(d->device, g_dlss.tstPool, 0);
        g_dlss.tstSets.clear();
        g_dlss.tstSetUsed = false;
        g_dlss.tstBoundOut = g_dlss.outView;
    }
    return true;
}

// Input colour in SHADER_READ_ONLY_OPTIMAL, outImage in GENERAL. Caller holds g_t.
#ifdef __APPLE__
// ---------------------------------------------------------------------------------------------
// MetalFX Temporal (macOS / MoltenVK)
// ---------------------------------------------------------------------------------------------
// The game's command buffer gets vkCmdSetEvent(in[slot]) once our inputs are written and
// vkCmdWaitEvents(out[slot]) before anything reads the output. MoltenVK turns those into
// encodeSignalEvent / encodeWaitForEvent on the events' MTLSharedEvents with value "current + 1",
// computed on the CPU when it encodes the command buffer (at vkQueueSubmit). At submit we make both
// values even (MoltenVK's "not set"), let the game's submission through, then commit our own Metal
// command buffer: wait(in, v+1) -> MetalFX -> signal(out, v+1), on our own queue.
constexpr int kMfxSlots = 16;  // must match metalfx_bridge.mm
struct MetalFx {
    bool tried = false, ready = false;
    std::string why = "not set up";
    PFN_vkExportMetalObjectsEXT exportObjects = nullptr;
    PFN_vkCreateEvent createEvent = nullptr;
    PFN_vkDestroyEvent destroyEvent = nullptr;
    PFN_vkCmdSetEvent cmdSetEvent = nullptr;
    PFN_vkCmdWaitEvents cmdWaitEvents = nullptr;
    VkEvent evIn[kMfxSlots] = {}, evOut[kMfxSlots] = {};
    void* mtlIn[kMfxSlots] = {};
    void* mtlOut[kMfxSlots] = {};
    uint32_t counter = 0;
    // our copy of the colour input (the game's image may rotate; MetalFX gets a stable texture)
    VkImage colorImage = VK_NULL_HANDLE;
    VkImageView colorView = VK_NULL_HANDLE;
    VkFormat colorFmt = VK_FORMAT_UNDEFINED;
    VkImageLayout colorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t cw = 0, ch = 0;
    // scaler configuration
    uint32_t inW = 0, inH = 0, outW = 0, outH = 0;
    VkFormat confFmt = VK_FORMAT_UNDEFINED;
    bool deviceExtEnabled = false;  // VK_EXT_metal_objects was enabled at device creation
    bool confReactive = false;      // the scaler was created with a reactive mask
    bool maskThisFrame = false;     // RecordReactive made this frame's mask
    int encodeErrors = 0;
    // the last frame recorded, for frame generation at present
    uint64_t lastFrame = UINT64_MAX;
    float lastJx = 0.f, lastJy = 0.f;
    bool lastReset = false;
} g_mfx;

void* MtlTexture(DeviceData* d, VkImage img) {
    VkExportMetalTextureInfoEXT ti{VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT};
    ti.image = img;
    ti.plane = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VkExportMetalObjectsInfoEXT oi{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
    oi.pNext = &ti;
    g_mfx.exportObjects(d->device, &oi);
    return (void*)ti.mtlTexture;
}

// ---------------------------------------------------------------------------------------------
// Reactive mask (MetalFX): where the image has no usable motion vectors
// ---------------------------------------------------------------------------------------------
// TF3 has no motion vectors of its own. The layer computes them for the camera and for moving vehicles,
// but smoke and other particles, rain, people and animals (skinned meshes) and water move on their own, so
// MetalFX reprojects them with the camera's motion and drags their old image along: trails. A reactive mask
// tells MetalFX, per pixel, to trust the current frame more than its history (0 = normal, 1 = no history).
//
// The mask is drawn from the game's own draws. When the game creates a pipeline for one of these kinds of
// 3D draws (ReactClassify), the layer creates a copy of it (ReactClone) for its own render pass: an RGBA8
// target at the render size plus a depth copy of the scene, depth test on and no depth writes, so hidden
// parts don't count. Particles and blended draws keep the game's fragment shader, whose alpha is their
// coverage; skinned meshes and water use reactive_const.frag (alpha 1). The blend adds alpha x the kind's
// strength (blend constant alpha). While the game records the frame, every draw with such a pipeline is
// remembered with its state (descriptor sets, vertex and index buffers, push constants), and after the scene
// is finished the draws are replayed (RecordReactive) with this frame's jitter. reactive_mask.comp then
// copies the alpha into the R32 float mask MetalFX reads.
enum ReactClass : uint8_t { kRcNone, kRcParticle, kRcAnimated, kRcWater, kRcTransparent, kRcCount };
const char* const kRcNames[kRcCount] = {"none", "particles", "animated", "water", "transparent"};
const uint32_t kReactConstFsSpirv[] = {
#include "reactive_const_frag.spv.inc"
};
const uint32_t kReactMaskSpirv[] = {
#include "reactive_mask.spv.inc"
};
constexpr size_t kReactMax = 8192;  // replayed draws per frame

std::unordered_map<VkShaderModule, uint8_t> g_moduleReact;  // fragment shaders recognised by their names

float ReactStrength(uint8_t cls) {
    switch (cls) {
        case kRcParticle: return g_cfg.reactiveParticles;
        case kRcAnimated: return g_cfg.reactiveAnimated;
        case kRcWater: return g_cfg.reactiveWater;
        case kRcTransparent: return g_cfg.reactiveTransparent;
        default: return 0.f;
    }
}

// From the SPIR-V debug names of a fragment shader: particle.fs (noise0Tex), rain.fs (tex + noise), the
// water shaders (waterDepthTex).
uint8_t ReactModuleClass(const uint32_t* w, size_t words) {
    if (words < 5 || w[0] != 0x07230203u) return kRcNone;
    bool fragment = false, noise0 = false, noise = false, tex = false, water = false;
    for (size_t i = 5; i < words;) {
        uint32_t op = w[i] & 0xffffu, n = w[i] >> 16;
        if (!n || i + n > words) break;
        if (op == 15 && n >= 2 && w[i + 1] == 4) fragment = true;  // OpEntryPoint Fragment
        else if (op == 5 && n >= 3) {                              // OpName
            const char* str = reinterpret_cast<const char*>(&w[i + 2]);
            std::string nm(str, strnlen(str, (n - 2) * 4));
            if (nm == "noise0Tex") noise0 = true;
            else if (nm == "noise") noise = true;
            else if (nm == "tex") tex = true;
            else if (nm == "waterDepthTex") water = true;
        }
        i += n;
    }
    if (!fragment) return kRcNone;
    if (noise0 || (noise && tex)) return kRcParticle;
    if (water) return kRcWater;
    return kRcNone;
}

struct ReactDraw {
    VkPipeline pipe = VK_NULL_HANDLE;  // the copy
    uint8_t cls = kRcNone;
    std::vector<BoundSet> sets;
    std::pair<VkBuffer, VkDeviceSize> vb[16] = {};
    VkBuffer ib = VK_NULL_HANDLE;
    VkDeviceSize ibOff = 0;
    VkIndexType ibType = VK_INDEX_TYPE_UINT16;
    bool indexed = false;
    uint32_t count = 0, instances = 0, first = 0, firstInstance = 0;
    int32_t vertexOffset = 0;
    VkPipelineLayout pushLayout = VK_NULL_HANDLE;
    uint32_t pushCount = 0;
    CmdState::PushRange pushRanges[4] = {};
    uint8_t push[256] = {};
};

struct Reactive {
    bool baseTried = false, baseOk = false;
    VkRenderPass rp = VK_NULL_HANDLE;
    VkShaderModule constFs = VK_NULL_HANDLE;
    uint32_t clones = 0, cloneFailures = 0;
    uint32_t clonesByClass[kRcCount] = {};
    // per render size
    bool failed = false;
    uint32_t w = 0, h = 0;
    VkImage rt = VK_NULL_HANDLE, depth = VK_NULL_HANDLE, mask = VK_NULL_HANDLE;
    VkImageView rtView = VK_NULL_HANDLE, depthView = VK_NULL_HANDLE, maskView = VK_NULL_HANDLE;
    VkBuffer staging = VK_NULL_HANDLE;
    VkFramebuffer fb = VK_NULL_HANDLE;
    // reactive_mask.comp
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline maskPipe = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkImageView boundOut = VK_NULL_HANDLE;
    bool used = false;
    // this frame's draws (from every command buffer of the frame)
    uint64_t frame = UINT64_MAX;
    std::vector<ReactDraw> draws;
    uint32_t drawsByClass[kRcCount] = {};
    uint64_t maskFrame = UINT64_MAX;  // the frame the mask was last drawn for (reactive_debug)
    std::atomic<uint64_t> outside{0};  // draws of copied pipelines outside the main 3D pass (not replayed)
} g_react;

// The render pass and the constant fragment shader, needed before the first copy. Caller holds g_t.
bool ReactEnsureBase(DeviceData* d) {
    if (g_react.baseTried) return g_react.baseOk;
    g_react.baseTried = true;
    if (!mfx_reactive_supported()) {
        Log("[reactive] MetalFX's reactive mask needs macOS 14.4 or newer; off");
        return false;
    }
    VkAttachmentDescription att[2] = {};
    att[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    att[1] = att[0];
    att[1].format = VK_FORMAT_D32_SFLOAT;
    att[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].initialLayout = att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &color;
    sub.pDepthStencilAttachment = &depth;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 2;
    rpi.pAttachments = att;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sub;
    if (d->CreateRenderPass(d->device, &rpi, nullptr, &g_react.rp) != VK_SUCCESS) {
        g_react.rp = VK_NULL_HANDLE;
        Log("[reactive] couldn't create the render pass; off");
        return false;
    }
    VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    si.codeSize = sizeof kReactConstFsSpirv;
    si.pCode = kReactConstFsSpirv;
    if (d->CreateShaderModule(d->device, &si, nullptr, &g_react.constFs) != VK_SUCCESS) {
        Log("[reactive] couldn't create the fragment shader; off");
        return false;
    }
    g_react.baseOk = true;
    Log("[reactive] reactive mask ready (strengths: particles %g, animated %g, water %g, transparent %g)",
        g_cfg.reactiveParticles, g_cfg.reactiveAnimated, g_cfg.reactiveWater, g_cfg.reactiveTransparent);
    return true;
}

// Which kind of reactive draw a game pipeline makes (kRcNone: not one). Only 3D draws into colour with a
// depth test and no tessellation (terrain). Caller holds g_t.
uint8_t ReactClassify(const VkGraphicsPipelineCreateInfo& ci) {
    if (!ci.renderPass || !ci.pDepthStencilState || !ci.pDepthStencilState->depthTestEnable || !ci.pColorBlendState ||
        !ci.pColorBlendState->attachmentCount || !ci.pColorBlendState->pAttachments[0].colorWriteMask ||
        (ci.pRasterizationState && ci.pRasterizationState->rasterizerDiscardEnable))
        return kRcNone;
    if (ci.pInputAssemblyState && !IsTriangles(ci.pInputAssemblyState->topology)) return kRcNone;
    VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
    for (uint32_t k = 0; k < ci.stageCount; ++k) {
        VkShaderStageFlagBits st = ci.pStages[k].stage;
        if (st == VK_SHADER_STAGE_VERTEX_BIT) vs = ci.pStages[k].module;
        else if (st == VK_SHADER_STAGE_FRAGMENT_BIT) fs = ci.pStages[k].module;
        else return kRcNone;  // tessellation / geometry shaders: terrain, grass
    }
    if (!vs || !fs) return kRcNone;
    auto m = g_moduleReact.find(fs);
    if (m != g_moduleReact.end()) return m->second;
    auto v = g_vsInfo.find(vs);
    if (v != g_vsInfo.end() && v->second.skinned) return kRcAnimated;
    if (ci.pColorBlendState->pAttachments[0].blendEnable && !ci.pDepthStencilState->depthWriteEnable)
        return kRcTransparent;
    return kRcNone;
}

// The copy of a game pipeline for the reactive pass (the create info as the driver got it). Call without g_t.
VkPipeline ReactClone(DeviceData* d, const VkGraphicsPipelineCreateInfo& game, uint8_t cls) {
    {
        TLock lk(g_t);
        if (!ReactEnsureBase(d)) return VK_NULL_HANDLE;
    }
    VkGraphicsPipelineCreateInfo ci = game;
    ci.pNext = nullptr;
    ci.flags = 0;
    ci.basePipelineHandle = VK_NULL_HANDLE;
    ci.basePipelineIndex = -1;
    ci.renderPass = g_react.rp;
    ci.subpass = 0;
    ci.pTessellationState = nullptr;
    std::vector<VkPipelineShaderStageCreateInfo> stages(game.pStages, game.pStages + game.stageCount);
    bool constFs = cls == kRcAnimated || cls == kRcWater;  // their own colour doesn't matter, only coverage
    for (auto& st : stages) {
        st.pNext = nullptr;
        st.flags = 0;
        if (st.stage == VK_SHADER_STAGE_FRAGMENT_BIT && constFs) {
            st.module = g_react.constFs;
            st.pName = "main";
            st.pSpecializationInfo = nullptr;
        }
    }
    ci.pStages = stages.data();
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    ci.pViewportState = &vp;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    if (game.pRasterizationState) rs = *game.pRasterizationState;
    rs.pNext = nullptr;
    rs.rasterizerDiscardEnable = VK_FALSE;
    rs.lineWidth = 1.f;
    ci.pRasterizationState = &rs;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    ci.pMultisampleState = &ms;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_FALSE;
    VkCompareOp op = game.pDepthStencilState->depthCompareOp;
    ds.depthCompareOp = op == VK_COMPARE_OP_GREATER || op == VK_COMPARE_OP_GREATER_OR_EQUAL
                            ? VK_COMPARE_OP_GREATER_OR_EQUAL
                            : VK_COMPARE_OP_LESS_OR_EQUAL;  // the same surface drawn again must pass
    ci.pDepthStencilState = &ds;
    VkPipelineColorBlendAttachmentState ba{};
    ba.blendEnable = VK_TRUE;
    ba.srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.colorBlendOp = VK_BLEND_OP_ADD;
    ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;  // the kind's strength
    ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.alphaBlendOp = VK_BLEND_OP_ADD;
    ba.colorWriteMask = VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &ba;
    ci.pColorBlendState = &cb;
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dsi.dynamicStateCount = 3;
    dsi.pDynamicStates = dyn;
    ci.pDynamicState = &dsi;
    VkPipeline pipe = VK_NULL_HANDLE;
    // straight to the driver: our own CreateGraphicsPipelines hook must not patch or track it
    if (d->CreateGraphicsPipelines(d->device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe) != VK_SUCCESS)
        pipe = VK_NULL_HANDLE;
    TLock lk(g_t);
    if (pipe) {
        ++g_react.clones;
        ++g_react.clonesByClass[cls];
        if (g_react.clones <= 4 || g_react.clones % 100 == 0)
            Log("[reactive] pipeline copy %u (%s, %s fragment shader); by kind: particles %u, animated %u, water %u, "
                "transparent %u", g_react.clones, kRcNames[cls], constFs ? "constant" : "the game's",
                g_react.clonesByClass[kRcParticle], g_react.clonesByClass[kRcAnimated], g_react.clonesByClass[kRcWater],
                g_react.clonesByClass[kRcTransparent]);
    } else if (g_react.cloneFailures++ < 5)
        Log("[reactive] couldn't copy a %s pipeline; its draws stay out of the mask", kRcNames[cls]);
    return pipe;
}

// A game draw with a copied pipeline in the main 3D pass: remember it for the replay. Caller holds g_t.
void ReactRecordDraw(const CmdState& cs, bool indexed, uint32_t count, uint32_t instances, uint32_t first,
                     int32_t vertexOffset, uint32_t firstInstance) {
    if (g_react.frame != g_frame) {
        g_react.frame = g_frame;
        g_react.draws.clear();
        for (auto& n : g_react.drawsByClass) n = 0;
    }
    if (g_react.draws.size() >= kReactMax || ReactStrength(cs.pipe->react) <= 0.f) return;
    ReactDraw& rd = g_react.draws.emplace_back();
    rd.pipe = cs.pipe->reactPipe;
    rd.cls = cs.pipe->react;
    rd.sets = cs.sets;
    std::copy(std::begin(cs.vb), std::end(cs.vb), rd.vb);
    rd.ib = cs.ib;
    rd.ibOff = cs.ibOff;
    rd.ibType = cs.ibType;
    rd.indexed = indexed;
    rd.count = count;
    rd.instances = instances;
    rd.first = first;
    rd.vertexOffset = vertexOffset;
    rd.firstInstance = firstInstance;
    rd.pushLayout = cs.pushLayout;
    rd.pushCount = cs.pushCount;
    std::copy(std::begin(cs.pushRanges), std::end(cs.pushRanges), rd.pushRanges);
    std::memcpy(rd.push, cs.push, sizeof rd.push);
    ++g_react.drawsByClass[rd.cls];
}

// Images, framebuffer and the mask compute pass for the render size. Caller holds g_t.
bool ReactEnsureImages(DeviceData* d, uint32_t w, uint32_t h) {
    if (g_react.failed || !g_react.baseOk) return false;
    if (!g_react.maskPipe) {
        g_react.failed = true;  // cleared on success
        VkDescriptorSetLayoutBinding b[3] = {};
        b[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 3;
        li.pBindings = b;
        if (d->CreateDescriptorSetLayout(d->device, &li, nullptr, &g_react.setLayout) != VK_SUCCESS) return false;
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &g_react.setLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pr;
        if (d->CreatePipelineLayout(d->device, &pli, nullptr, &g_react.pipeLayout) != VK_SUCCESS) return false;
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = sizeof kReactMaskSpirv;
        si.pCode = kReactMaskSpirv;
        VkShaderModule mod;
        if (d->CreateShaderModule(d->device, &si, nullptr, &mod) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = g_react.pipeLayout;
        VkResult r = d->CreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &ci, nullptr, &g_react.maskPipe);
        d->DestroyShaderModule(d->device, mod, nullptr);
        if (r != VK_SUCCESS) {
            g_react.maskPipe = VK_NULL_HANDLE;
            return false;
        }
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 1;
        pi.poolSizeCount = 2;
        pi.pPoolSizes = sizes;
        if (d->CreateDescriptorPool(d->device, &pi, nullptr, &g_react.pool) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = g_react.pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &g_react.setLayout;
        if (d->AllocateDescriptorSets(d->device, &ai, &g_react.set) != VK_SUCCESS) return false;
        g_react.failed = false;
    }
    if (g_react.w != w || g_react.h != h || !g_react.mask) {
        if (g_react.used) d->DeviceWaitIdle(d->device);
        if (g_react.fb) d->DestroyFramebuffer(d->device, g_react.fb, nullptr);
        for (VkImageView v : {g_react.rtView, g_react.depthView, g_react.maskView})
            if (v) d->DestroyImageView(d->device, v, nullptr);
        for (VkImage i : {g_react.rt, g_react.depth, g_react.mask})
            if (i) d->DestroyImage(d->device, i, nullptr);
        g_react.fb = VK_NULL_HANDLE;
        g_react.rt = g_react.depth = g_react.mask = VK_NULL_HANDLE;
        g_react.rtView = g_react.depthView = g_react.maskView = VK_NULL_HANDLE;
        bool ok =
            CreateOwnImage(d, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &g_react.rt,
                           &g_react.rtView) &&
            CreateOwnImage(d, w, h, VK_FORMAT_D32_SFLOAT,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           &g_react.depth, &g_react.depthView, VK_IMAGE_ASPECT_DEPTH_BIT) &&
            CreateOwnImage(d, w, h, VK_FORMAT_R32_SFLOAT,
                           VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,  // cleared when unused; recordings
                           &g_react.mask, &g_react.maskView) &&
            CreateBuffer(d, (VkDeviceSize)w * h * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &g_react.staging, nullptr);
        if (ok) {
            VkImageView views[2] = {g_react.rtView, g_react.depthView};
            VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fi.renderPass = g_react.rp;
            fi.attachmentCount = 2;
            fi.pAttachments = views;
            fi.width = w;
            fi.height = h;
            fi.layers = 1;
            ok = d->CreateFramebuffer(d->device, &fi, nullptr, &g_react.fb) == VK_SUCCESS;
        }
        if (!ok) {
            g_react.failed = true;
            g_react.mask = VK_NULL_HANDLE;
            Log("[reactive] couldn't create the %ux%u images; reactive mask off", w, h);
            return false;
        }
        VkDescriptorImageInfo rt{g_dlss.pointSampler, g_react.rtView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo mk{VK_NULL_HANDLE, g_react.maskView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet wr[2] = {};
        wr[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_react.set, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rt, nullptr, nullptr};
        wr[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_react.set, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mk, nullptr, nullptr};
        d->UpdateDescriptorSets(d->device, 2, wr, 0, nullptr);
        g_react.w = w;
        g_react.h = h;
        Log("[reactive] mask images %ux%u", w, h);
    }
    if (g_react.boundOut != g_dlss.outView) {
        if (g_react.used) d->DeviceWaitIdle(d->device);
        VkDescriptorImageInfo out{VK_NULL_HANDLE, g_dlss.outView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_react.set, 2, 0, 1,
                                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &out, nullptr, nullptr};
        d->UpdateDescriptorSets(d->device, 1, &wr, 0, nullptr);
        g_react.boundOut = g_dlss.outView;
    }
    return true;
}

// Draws this frame's reactive mask (or clears it) into g_react.mask, left in GENERAL layout. Inside
// RecordMetalFx, after the motion-vector pass (depth32Image in SHADER_READ_ONLY). False: no mask this frame.
// Caller holds g_t.
bool RecordReactive(DeviceData* d, VkCommandBuffer cb, CmdState& cs) {
    if (!g_react.baseOk || !g_dlss.pointSampler || !ReactEnsureImages(d, g_dlss.inW, g_dlss.inH)) return false;
    const uint32_t w = g_dlss.inW, h = g_dlss.inH;
    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, GEN = VK_IMAGE_LAYOUT_GENERAL;
    bool haveDraws = g_cfg.reactive && g_react.frame == g_frame && !g_react.draws.empty() && cs.mvPending &&
                     g_dlss.sceneVpValid && g_dlss.depth32Image;
    g_react.used = true;
    if (!haveDraws) {
        Barrier(d, cb, g_react.mask, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue zero{};
        VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        d->CmdClearColorImage(cb, g_react.mask, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
        Barrier(d, cb, g_react.mask, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, GEN);
        return true;
    }
    // the scene's depth (the motion-vector pass's R32F copy) -> our D32 depth buffer, through a buffer
    Barrier(d, cb, g_dlss.depth32Image, VK_IMAGE_ASPECT_COLOR_BIT, RO, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy bc{};
    bc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    bc.imageExtent = {w, h, 1};
    d->CmdCopyImageToBuffer(cb, g_dlss.depth32Image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_react.staging, 1, &bc);
    Barrier(d, cb, g_dlss.depth32Image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, RO);
    VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = g_react.staging;
    bb.size = VK_WHOLE_SIZE;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bb, 0,
                          nullptr);
    Barrier(d, cb, g_react.depth, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    bc.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    d->CmdCopyBufferToImage(cb, g_react.staging, g_react.depth, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
    Barrier(d, cb, g_react.depth, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

    VkClearValue clear[2] = {};
    VkRenderPassBeginInfo rb{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rb.renderPass = g_react.rp;
    rb.framebuffer = g_react.fb;
    rb.renderArea = {{0, 0}, {w, h}};
    rb.clearValueCount = 2;
    rb.pClearValues = clear;
    d->CmdBeginRenderPass(cb, &rb, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vp = g_dlss.sceneVp;  // the game's, with this frame's jitter (as its draws had)
    if (ViewportJitterOn()) {
        float jx, jy;
        FrameJitter(jx, jy);
        vp.x += jx;
        vp.y += jy;
    }
    VkRect2D sc{{0, 0}, {w, h}};
    d->CmdSetViewport(cb, 0, 1, &vp);
    d->CmdSetScissor(cb, 0, 1, &sc);
    VkPipeline bound = VK_NULL_HANDLE;
    uint8_t boundCls = kRcNone;
    for (const ReactDraw& rd : g_react.draws) {
        if (rd.pipe != bound) {
            d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rd.pipe);
            bound = rd.pipe;
            boundCls = kRcNone;  // dynamic state is kept, but set it again with each pipeline
        }
        if (rd.cls != boundCls) {
            float k[4] = {0.f, 0.f, 0.f, ReactStrength(rd.cls)};
            d->CmdSetBlendConstants(cb, k);
            boundCls = rd.cls;
        }
        for (uint32_t i = 0; i < rd.sets.size(); ++i) {
            const BoundSet& bs = rd.sets[i];
            if (!bs.set || !bs.layout) continue;
            d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, bs.layout, i, 1, &bs.set,
                                     (uint32_t)bs.dynOffsets.size(), bs.dynOffsets.data());
        }
        for (uint32_t i = 0; i < 16; ++i)
            if (rd.vb[i].first) d->CmdBindVertexBuffers(cb, i, 1, &rd.vb[i].first, &rd.vb[i].second);
        if (rd.pushLayout)
            for (uint32_t i = 0; i < rd.pushCount; ++i)
                d->CmdPushConstants(cb, rd.pushLayout, rd.pushRanges[i].stages, rd.pushRanges[i].offset,
                                    rd.pushRanges[i].size, rd.push + rd.pushRanges[i].offset);
        if (rd.indexed) {
            if (!rd.ib) continue;
            d->CmdBindIndexBuffer(cb, rd.ib, rd.ibOff, rd.ibType);
            d->CmdDrawIndexed(cb, rd.count, rd.instances, rd.first, rd.vertexOffset, rd.firstInstance);
        } else
            d->CmdDraw(cb, rd.count, rd.instances, rd.first, rd.firstInstance);
    }
    d->CmdEndRenderPass(cb);
    // alpha of the replay target -> the R32F mask
    Barrier(d, cb, g_react.mask, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, GEN);
    int32_t pc[5] = {(int32_t)w, (int32_t)h, (int32_t)w, (int32_t)h, 0};
    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_react.maskPipe);
    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_react.pipeLayout, 0, 1, &g_react.set, 0, nullptr);
    d->CmdPushConstants(cb, g_react.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    d->CmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    Barrier(d, cb, g_react.mask, VK_IMAGE_ASPECT_COLOR_BIT, GEN, GEN);
    if (cs.capture && g_cap.reactive && g_cap.w == w && g_cap.h == h) {  // recording: the mask too
        Barrier(d, cb, g_react.mask, VK_IMAGE_ASPECT_COLOR_BIT, GEN, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        bc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        d->CmdCopyImageToBuffer(cb, g_react.mask, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_cap.reactive, 1, &bc);
        Barrier(d, cb, g_react.mask, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, GEN);
        cs.capReactive = true;
    }
    g_react.maskFrame = g_frame;
    if (LogFrame(g_frame))
        Log("[reactive] frame %llu: %zu draws replayed (particles %u, animated %u, water %u, transparent %u); "
            "%llu draws of copied pipelines outside the main 3D pass so far",
            (unsigned long long)g_frame, g_react.draws.size(), g_react.drawsByClass[kRcParticle],
            g_react.drawsByClass[kRcAnimated], g_react.drawsByClass[kRcWater], g_react.drawsByClass[kRcTransparent],
            (unsigned long long)g_react.outside.load(std::memory_order_relaxed));
    return true;
}

// reactive_debug=1: tint the upscaled image (GENERAL) red where this frame's mask is set. Caller holds g_t.
void RecordReactiveDebug(DeviceData* d, VkCommandBuffer cb) {
    if (!g_cfg.reactiveDebug || g_react.maskFrame != g_frame || !g_react.maskPipe || g_react.boundOut != g_dlss.outView)
        return;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                          nullptr, 0, nullptr);
    int32_t pc[5] = {(int32_t)g_dlss.outW, (int32_t)g_dlss.outH, (int32_t)g_react.w, (int32_t)g_react.h, 1};
    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_react.maskPipe);
    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_react.pipeLayout, 0, 1, &g_react.set, 0, nullptr);
    d->CmdPushConstants(cb, g_react.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    d->CmdDispatch(cb, (g_dlss.outW + 7) / 8, (g_dlss.outH + 7) / 8, 1);
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                          nullptr, 0, nullptr);
}

// Once per device. Caller holds g_t.
bool MfxEnsure(DeviceData* d) {
    if (g_mfx.ready) return true;
    if (g_mfx.tried) return false;
    g_mfx.tried = true;
    if (!g_mfx.deviceExtEnabled) {
        g_mfx.why = "the Vulkan driver has no VK_EXT_metal_objects (MoltenVK needed)";
        return false;
    }
    // The event hand-off relies on MoltenVK encoding command buffers at vkQueueSubmit (its default).
    // Prefilled command buffers are encoded while recording, with event values we can't line up.
    if (const char* pf = std::getenv("MVK_CONFIG_PREFILL_METAL_COMMAND_BUFFERS")) {
        if (*pf && *pf != '0') {
            g_mfx.why = "MVK_CONFIG_PREFILL_METAL_COMMAND_BUFFERS is set; MetalFX hand-off disabled";
            return false;
        }
    }
    auto gdpa = d->GetDeviceProcAddr;
    g_mfx.exportObjects = (PFN_vkExportMetalObjectsEXT)gdpa(d->device, "vkExportMetalObjectsEXT");
    g_mfx.createEvent = (PFN_vkCreateEvent)gdpa(d->device, "vkCreateEvent");
    g_mfx.destroyEvent = (PFN_vkDestroyEvent)gdpa(d->device, "vkDestroyEvent");
    g_mfx.cmdSetEvent = (PFN_vkCmdSetEvent)gdpa(d->device, "vkCmdSetEvent");
    g_mfx.cmdWaitEvents = (PFN_vkCmdWaitEvents)gdpa(d->device, "vkCmdWaitEvents");
    if (!g_mfx.exportObjects || !g_mfx.createEvent || !g_mfx.cmdSetEvent || !g_mfx.cmdWaitEvents) {
        g_mfx.why = "vkExportMetalObjectsEXT / event functions not available";
        return false;
    }
    VkExportMetalDeviceInfoEXT di{VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT};
    VkExportMetalObjectsInfoEXT oi{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
    oi.pNext = &di;
    g_mfx.exportObjects(d->device, &oi);
    char err[256] = {};
    if (!mfx_init((void*)di.mtlDevice, err, sizeof err)) {
        g_mfx.why = err;
        return false;
    }
    for (int i = 0; i < kMfxSlots; ++i) {
        VkEventCreateInfo ei{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
        if (g_mfx.createEvent(d->device, &ei, nullptr, &g_mfx.evIn[i]) != VK_SUCCESS ||
            g_mfx.createEvent(d->device, &ei, nullptr, &g_mfx.evOut[i]) != VK_SUCCESS) {
            g_mfx.why = "vkCreateEvent failed";
            return false;
        }
        for (int k = 0; k < 2; ++k) {
            VkExportMetalSharedEventInfoEXT se{VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT};
            se.event = k ? g_mfx.evOut[i] : g_mfx.evIn[i];
            VkExportMetalObjectsInfoEXT o{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
            o.pNext = &se;
            g_mfx.exportObjects(d->device, &o);
            (k ? g_mfx.mtlOut : g_mfx.mtlIn)[i] = (void*)se.mtlSharedEvent;
        }
        if (!g_mfx.mtlIn[i] || !g_mfx.mtlOut[i]) {
            g_mfx.why = "MoltenVK's VkEvents are not MTLSharedEvents on this GPU";
            return false;
        }
    }
    g_mfx.ready = true;
    g_mfx.why = "ready";
    Log("[mfx] MetalFX Temporal ready (own Metal queue, %d event slots)", kMfxSlots);
    return true;
}

void MfxDestroy(DeviceData* d) {
    if (!g_mfx.ready && !g_mfx.tried) return;
    mfx_shutdown();
    for (int i = 0; i < kMfxSlots; ++i) {
        if (g_mfx.evIn[i] && g_mfx.destroyEvent) g_mfx.destroyEvent(d->device, g_mfx.evIn[i], nullptr);
        if (g_mfx.evOut[i] && g_mfx.destroyEvent) g_mfx.destroyEvent(d->device, g_mfx.evOut[i], nullptr);
    }
    g_mfx = MetalFx{};
}

// Records the MetalFX hand-off into the game's command buffer. Inputs: cs.colorImage (copied), the
// camera pass's R32F depth, the motion-vector image; output: g_dlss.outImage (GENERAL). Caller holds g_t.
bool RecordMetalFx(DeviceData* d, VkCommandBuffer cb, CmdState& cs, const ImageInfo& ci) {
    if (!MfxEnsure(d) || !g_dlss.mvReady || !g_dlss.depth32Image) return false;
    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // colour copy (same format; MetalFX reads B10G11R11 directly)
    if (!g_mfx.colorImage || g_mfx.cw != ci.w || g_mfx.ch != ci.h || g_mfx.colorFmt != ci.fmt) {
        if (g_mfx.colorImage) {
            d->DeviceWaitIdle(d->device);
            d->DestroyImageView(d->device, g_mfx.colorView, nullptr);
            d->DestroyImage(d->device, g_mfx.colorImage, nullptr);
        }
        if (!CreateOwnImage(d, ci.w, ci.h, ci.fmt, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            &g_mfx.colorImage, &g_mfx.colorView)) {
            g_mfx.colorImage = VK_NULL_HANDLE;
            Log("[mfx] could not create the colour copy");
            return false;
        }
        g_mfx.cw = ci.w;
        g_mfx.ch = ci.h;
        g_mfx.colorFmt = ci.fmt;
        g_mfx.colorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }
    // the reactive mask for this frame (cleared if nothing feeds it); a scaler without one if it can't be made
    bool mask = RecordReactive(d, cb, cs);
    if (g_mfx.inW != ci.w || g_mfx.inH != ci.h || g_mfx.outW != g_dlss.outW || g_mfx.outH != g_dlss.outH ||
        g_mfx.confFmt != ci.fmt || g_mfx.confReactive != mask) {
        char err[256] = {};
        if (!mfx_configure(ci.w, ci.h, g_dlss.outW, g_dlss.outH, (uint32_t)ci.fmt, mask, err, sizeof err)) {
            Log("[mfx] scaler setup failed: %s", err);
            g_mfx.ready = false;
            g_mfx.why = err;
            return false;
        }
        g_mfx.inW = ci.w;
        g_mfx.inH = ci.h;
        g_mfx.outW = g_dlss.outW;
        g_mfx.outH = g_dlss.outH;
        g_mfx.confFmt = ci.fmt;
        g_mfx.confReactive = mask;
        g_dlss.resetNext = true;
        Log("[mfx] temporal scaler %ux%u -> %ux%u (colour %s, %s)", ci.w, ci.h, g_dlss.outW, g_dlss.outH,
            FormatName(ci.fmt) ? FormatName(ci.fmt) : "?", mask ? "with a reactive mask" : "no reactive mask");
    }
    // the game's colour (RO) -> our copy
    Barrier(d, cb, cs.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, RO, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    Barrier(d, cb, g_mfx.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy cp{};
    cp.srcSubresource = cp.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    cp.extent = {ci.w, ci.h, 1};
    d->CmdCopyImage(cb, cs.colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_mfx.colorImage,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    Barrier(d, cb, g_mfx.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, RO);
    Barrier(d, cb, cs.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, RO);

    int slot = (int)(g_mfx.counter++ % kMfxSlots);
    // everything written so far -> visible to MetalFX; MetalFX's output -> visible to what follows
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                          nullptr, 0, nullptr);
    g_mfx.cmdSetEvent(cb, g_mfx.evIn[slot], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    g_mfx.cmdWaitEvents(cb, 1, &g_mfx.evOut[slot], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 1, &mb, 0, nullptr, 0, nullptr);
    cs.mfxPending = true;
    cs.mfxSlot = slot;
    cs.mfxJx = cs.jitterSet ? cs.jx * g_cfg.mfxJitterSignX : 0.f;
    cs.mfxJy = cs.jitterSet ? cs.jy * g_cfg.mfxJitterSignY : 0.f;
    cs.mfxReset = g_dlss.resetNext;
    cs.mfxInW = ci.w;
    cs.mfxInH = ci.h;
    g_mfx.lastFrame = g_frame;
    g_mfx.lastJx = cs.mfxJx;
    g_mfx.lastJy = cs.mfxJy;
    g_mfx.lastReset = cs.mfxReset;
    return true;
}

struct MfxJob {
    int slot;
    uint64_t vin, vout;
    MfxFrame frame;
};

// Before the game's submission (caller holds g_t): free the slot, make the events "not set" and fill
// in the Metal objects for this frame.
bool MfxPrepare(DeviceData* d, const CmdState& cs, MfxJob& job) {
    if (!g_mfx.ready) return false;
    job.slot = cs.mfxSlot;
    mfx_slot_wait(job.slot);
    job.vin = mfx_event_unset(g_mfx.mtlIn[job.slot]);
    job.vout = mfx_event_unset(g_mfx.mtlOut[job.slot]);
    MfxFrame& f = job.frame;
    f = MfxFrame{};
    f.color = MtlTexture(d, g_mfx.colorImage);
    f.depth = MtlTexture(d, g_dlss.depth32Image);
    f.motion = MtlTexture(d, g_dlss.mvImage);
    f.output = MtlTexture(d, g_dlss.outImage);
    f.reactive = g_mfx.confReactive && g_react.mask ? MtlTexture(d, g_react.mask) : nullptr;
    f.eventIn = g_mfx.mtlIn[job.slot];
    f.waitValue = job.vin + 1;
    f.eventOut = g_mfx.mtlOut[job.slot];
    f.signalValue = job.vout + 1;
    f.jitterX = cs.mfxJx;
    f.jitterY = cs.mfxJy;
    f.mvScaleX = g_cfg.mvScaleX;
    f.mvScaleY = g_cfg.mvScaleY;
    f.reset = cs.mfxReset;
    f.depthReversed = false;  // GL-style depth: near 0, far 1
    f.inW = cs.mfxInW;
    f.inH = cs.mfxInH;
    return true;
}

// After the game's submission: commit the MetalFX work. If the submission failed, release the events so
// nothing waits forever.
void MfxCommit(MfxJob& job, bool submitted) {
    if (!submitted) {
        mfx_event_set(job.frame.eventOut, job.frame.signalValue);
        return;
    }
    char err[256] = {};
    if (!mfx_encode(job.slot, &job.frame, err, sizeof err)) {
        if (g_mfx.encodeErrors++ < 5) Log("[mfx] encode failed: %s", err);
        mfx_event_set(job.frame.eventOut, job.frame.signalValue);
    }
}

// ---------------------------------------------------------------------------------------------
// Frame generation (frame_gen=1): MetalFX frame interpolation at present
// ---------------------------------------------------------------------------------------------
// At the present of each upscaled frame N, the layer acquires a second swapchain image, has MetalFX make the
// frame halfway between N-1 and N in it, and presents it before N. One command buffer of ours, on the game's
// queue, waits for the game's frame (the present's semaphores), copies the finished frame (UI included) for
// MetalFX, hands over to MetalFX through a VkEvent pair as the upscaler does, copies the result into the
// extra image and keeps this frame's upscaled image for the next interpolation. MetalFX gets the image
// without UI (the upscaler's output, which the game's blit copies to the screen before drawing its UI) and
// the finished frame, and takes the UI out of the latter itself (uiTextureComposited).
// With FIFO presentation the two images are shown at successive refreshes, so the game runs at most at half
// the refresh rate: 30 fps -> 60 on a 60 Hz display.
constexpr int kFgSlots = 8;  // must match metalfx_bridge.mm
// ---------------------------------------------------------------------------------------------
// Frame rate lock (fps_limit=30): every frame on screen for the same number of refreshes
// ---------------------------------------------------------------------------------------------
// TF3 presents without vsync (IMMEDIATE) at a frame rate that changes from frame to frame (37-47 fps on an
// M4). On a 60 Hz screen without VRR each frame then stays up for 1 or 2 refreshes in an uneven pattern, which
// looks like jumps, and the image can tear. With fps_limit the swapchain uses FIFO (vsync) and the layer hands
// each frame to the driver half a refresh before every n-th refresh (n = refresh rate / fps_limit, 2 for 30 on
// 60 Hz), so FIFO shows it at that refresh. The refresh times come from a CVDisplayLink on the main display
// (mfx_vsync). Holding the present also holds the game's thread, so it doesn't run ahead and add latency.
// Measured outside the game: MoltenVK's VK_GOOGLE_display_timing doesn't help here. presentAtTime wasn't
// honoured (frames came ~24 ms before the asked time), and the "actual" present times it reports are the
// hand-over times (~1 ms after the present), not refreshes.
struct FpsLock {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;  // the swapchain set up with FIFO for the lock
    uint64_t lastTarget = 0;  // the refresh the last frame was handed over for
    uint64_t presents = 0, lateFrames = 0, sleptNs = 0, noClock = 0;
    uint64_t gaps[4] = {};  // refreshes between consecutive targets: 1, 2, 3, 4+
} g_fpsLock;

uint64_t NowNs() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

// At swapchain creation (caller holds g_t).
void FpsLockSwapchain(VkSwapchainKHR swapchain, bool fifo) {
    g_fpsLock.swapchain = fifo ? swapchain : VK_NULL_HANDLE;
    g_fpsLock.lastTarget = 0;
    if (!fifo) return;
    uint64_t v = 0, r = 0;
    bool clock = mfx_vsync(&v, &r);  // starts the display link
    Log("[fps] frame rate lock to %d fps: FIFO; refresh clock %s", g_cfg.fpsLimit,
        clock ? (std::to_string(r / 1000) + " us per refresh").c_str() : "starting");
}

// Instead of the next layer's vkQueuePresentKHR when frame generation is off.
VkResult PacedPresent(DeviceData* d, VkQueue queue, const VkPresentInfoKHR* pInfo) {
    uint64_t sleepUntil = 0;
    {
        TLock lk(g_t);
        FpsLock& L = g_fpsLock;
        if (g_cfg.fpsLimit > 0 && L.swapchain && pInfo->swapchainCount == 1 && pInfo->pSwapchains[0] == L.swapchain) {
            uint64_t vsync = 0, R = 0, now = NowNs();
            if (!mfx_vsync(&vsync, &R)) {
                ++L.noClock;  // no refresh clock (yet): FIFO alone
            } else {
                uint64_t n = std::max<uint64_t>(1, (uint64_t)std::llround(1e9 / ((double)R * g_cfg.fpsLimit)));
                // the refresh n after the last frame's, on the display's grid; a late frame takes the first
                // refresh it can still make, and the count goes on from there
                uint64_t want = L.lastTarget ? L.lastTarget + n * R : now + R;
                want = want > vsync ? vsync + (want - vsync + R / 2) / R * R : vsync + R;
                if (want < now + R / 2) {
                    want = vsync + ((now + R / 2 - vsync) + R - 1) / R * R;
                    ++L.lateFrames;
                } else
                    sleepUntil = want - R / 2;
                if (L.lastTarget && want > L.lastTarget) ++L.gaps[std::min<uint64_t>((want - L.lastTarget + R / 2) / R, 4) - 1];
                L.lastTarget = want;
                ++L.presents;
                if (L.presents % 300 == 0)  // every 10 s at 30 fps, also in the menus (before the upscaler starts)
                    Log("[fps] lock at %d fps (every %llu refreshes of %.3f ms): frames %llu, %llu late; refreshes "
                        "between frames 1: %llu, 2: %llu, 3: %llu, 4+: %llu; game thread held %.2f ms per frame",
                        g_cfg.fpsLimit, (unsigned long long)n, R / 1e6, (unsigned long long)L.presents,
                        (unsigned long long)L.lateFrames, (unsigned long long)L.gaps[0], (unsigned long long)L.gaps[1],
                        (unsigned long long)L.gaps[2], (unsigned long long)L.gaps[3],
                        L.sleptNs / 1e6 / (double)L.presents);
            }
        }
    }
    if (sleepUntil) {
        uint64_t now = NowNs();
        if (sleepUntil > now) {
            uint64_t ns = std::min<uint64_t>(sleepUntil - now, 100000000);  // never more than 100 ms
            std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
            TLock lk(g_t);
            g_fpsLock.sleptNs += ns;
        }
    }
    return d->QueuePresentKHR(queue, pInfo);
}

struct FrameGen {
    // the swapchain, set up at creation when frame_gen=1 (transfer usage, 3 images, FIFO)
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapFmt = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    std::vector<VkImage> swapImages;
    // our objects
    bool tried = false, ready = false;
    std::string why = "not set up";
    uint32_t family = UINT32_MAX;
    PFN_vkCreateCommandPool createCommandPool = nullptr;
    PFN_vkDestroyCommandPool destroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers allocateCommandBuffers = nullptr;
    PFN_vkEndCommandBuffer endCommandBuffer = nullptr;
    PFN_vkCreateFence createFence = nullptr;
    PFN_vkDestroyFence destroyFence = nullptr;
    PFN_vkWaitForFences waitForFences = nullptr;
    PFN_vkResetFences resetFences = nullptr;
    PFN_vkCreateSemaphore createSemaphore = nullptr;
    PFN_vkDestroySemaphore destroySemaphore = nullptr;
    PFN_vkAcquireNextImageKHR acquireNextImage = nullptr;
    PFN_vkGetSwapchainImagesKHR getSwapchainImages = nullptr;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb[kFgSlots] = {};
    VkFence fence[kFgSlots] = {};
    bool fenceUsed[kFgSlots] = {};
    VkSemaphore semAcquire[kFgSlots] = {}, semGen[kFgSlots] = {}, semGame[kFgSlots] = {};
    VkEvent evIn[kFgSlots] = {}, evOut[kFgSlots] = {};
    void* mtlIn[kFgSlots] = {};
    void* mtlOut[kFgSlots] = {};
    uint64_t counter = 0;
    // the finished frame (UI included), the last upscaled frame, the generated frame
    VkImage uiImage = VK_NULL_HANDLE, prevImage = VK_NULL_HANDLE, genImage = VK_NULL_HANDLE;
    VkImageView uiView = VK_NULL_HANDLE, prevView = VK_NULL_HANDLE, genView = VK_NULL_HANDLE;
    VkFormat imgFmt = VK_FORMAT_UNDEFINED;
    uint32_t w = 0, h = 0;
    bool genInGeneral = false;
    uint32_t confInW = 0, confInH = 0;  // the interpolator's depth / motion size
    // history
    bool havePrev = false;
    uint64_t prevFrame = UINT64_MAX;  // g_frame of the upscaled image in prevImage
    std::chrono::steady_clock::time_point prevTime;
    // counters, for the log and the overlay
    uint64_t generated = 0, skipped = 0;
    std::string skipWhy;
    int errors = 0;
    // TF3_FG_DUMP (test harness): readback of the previous frame, the generated one and the finished one
    VkBuffer dumpBuf[3] = {};
    char* dumpPtr[3] = {};
    bool dumpFailed = false;
    uint64_t dumpKeyFrame = UINT64_MAX;  // Ctrl+Shift+G: dump this frame and the next
} g_fg;

// TF3_FG_DUMP=150,151: write the frames around these frames' generated frames to the capture folder. With
// TF3_DLSS_CAPTURE_REL=1 the numbers count from the first frame with the game's FSR pass, as for TF3_DLSS_CAPTURE.
bool FgDumpAt(uint64_t frame) {
    if (g_fg.dumpKeyFrame != UINT64_MAX && (frame == g_fg.dumpKeyFrame || frame == g_fg.dumpKeyFrame + 1)) return true;
    static const bool rel = [] {
        const char* e = std::getenv("TF3_DLSS_CAPTURE_REL");
        return e && *e == '1';
    }();
    if (rel) {
        if (!g_firstFsrFrame || frame < g_firstFsrFrame) return false;
        frame -= g_firstFsrFrame;
    }
    static std::vector<uint64_t> frames = [] {
        std::vector<uint64_t> v;
        const char* e = std::getenv("TF3_FG_DUMP");
        for (const char* p = e; p && *p;) {
            char* end = nullptr;
            unsigned long long f = std::strtoull(p, &end, 10);
            if (end == p) break;
            v.push_back(f);
            p = *end ? end + 1 : end;
        }
        return v;
    }();
    return std::find(frames.begin(), frames.end(), frame) != frames.end();
}

// Writes w x h 4-byte pixels as PPM (bgra: blue first in memory).
void FgWritePpm(const std::filesystem::path& path, const char* px, uint32_t w, uint32_t h, bool bgra) {
    std::ofstream o(path, std::ios::binary);
    o << "P6\n" << w << " " << h << "\n255\n";
    std::vector<char> row((size_t)w * 3);
    for (uint32_t y = 0; y < h; ++y) {
        const char* src = px + (size_t)y * w * 4;
        for (uint32_t x = 0; x < w; ++x) {
            row[x * 3 + 0] = src[x * 4 + (bgra ? 2 : 0)];
            row[x * 3 + 1] = src[x * 4 + 1];
            row[x * 3 + 2] = src[x * 4 + (bgra ? 0 : 2)];
        }
        o.write(row.data(), (std::streamsize)row.size());
    }
}

// Is frame generation set up and switched on (for the overlay). Caller holds g_t.
bool FgActive() { return g_cfg.frameGen && g_fg.ready && g_fg.swapchain && g_fg.generated > 0; }

void FgDestroyImages(DeviceData* d) {
    for (VkImageView* v : {&g_fg.uiView, &g_fg.prevView, &g_fg.genView})
        if (*v) d->DestroyImageView(d->device, *v, nullptr), *v = VK_NULL_HANDLE;
    for (VkImage* i : {&g_fg.uiImage, &g_fg.prevImage, &g_fg.genImage})
        if (*i) d->DestroyImage(d->device, *i, nullptr), *i = VK_NULL_HANDLE;
    g_fg.w = g_fg.h = 0;
    g_fg.havePrev = false;
    g_fg.genInGeneral = false;
}

// Once per device: command pool, per-slot command buffers, fences, semaphores, events. Caller holds g_t.
bool FgEnsure(DeviceData* d, VkQueue queue) {
    if (g_fg.ready) return true;
    if (g_fg.tried) return false;
    g_fg.tried = true;
    if (!MfxEnsure(d)) {
        g_fg.why = "MetalFX isn't running (" + g_mfx.why + ")";
        return false;
    }
    if (!mfx_fg_supported()) {
        g_fg.why = "MetalFX frame interpolation needs macOS 26 and a GPU that supports it";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(g_lock);
        auto it = g_queueFamily.find(queue);
        if (it != g_queueFamily.end()) g_fg.family = it->second;
    }
    if (g_fg.family == UINT32_MAX) {
        g_fg.why = "the present queue's family is unknown";
        return false;
    }
    auto gdpa = d->GetDeviceProcAddr;
#define FG_FN(field, name) g_fg.field = (decltype(g_fg.field))gdpa(d->device, name);
    FG_FN(createCommandPool, "vkCreateCommandPool")
    FG_FN(destroyCommandPool, "vkDestroyCommandPool")
    FG_FN(allocateCommandBuffers, "vkAllocateCommandBuffers")
    FG_FN(endCommandBuffer, "vkEndCommandBuffer")
    FG_FN(createFence, "vkCreateFence")
    FG_FN(destroyFence, "vkDestroyFence")
    FG_FN(waitForFences, "vkWaitForFences")
    FG_FN(resetFences, "vkResetFences")
    FG_FN(createSemaphore, "vkCreateSemaphore")
    FG_FN(destroySemaphore, "vkDestroySemaphore")
    FG_FN(acquireNextImage, "vkAcquireNextImageKHR")
#undef FG_FN
    if (!g_fg.createCommandPool || !g_fg.allocateCommandBuffers || !g_fg.endCommandBuffer || !g_fg.createFence ||
        !g_fg.waitForFences || !g_fg.resetFences || !g_fg.createSemaphore || !g_fg.acquireNextImage) {
        g_fg.why = "Vulkan functions missing";
        return false;
    }
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = g_fg.family;
    if (g_fg.createCommandPool(d->device, &pci, nullptr, &g_fg.pool) != VK_SUCCESS) {
        g_fg.why = "vkCreateCommandPool failed";
        return false;
    }
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_fg.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = kFgSlots;
    if (g_fg.allocateCommandBuffers(d->device, &ai, g_fg.cb) != VK_SUCCESS) {
        g_fg.why = "vkAllocateCommandBuffers failed";
        return false;
    }
    for (int i = 0; i < kFgSlots; ++i) {
        // Command buffers we allocate ourselves don't get the loader's dispatch pointer: set it.
        *reinterpret_cast<void**>(g_fg.cb[i]) = *reinterpret_cast<void**>(d->device);
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkEventCreateInfo ei{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
        if (g_fg.createFence(d->device, &fi, nullptr, &g_fg.fence[i]) != VK_SUCCESS ||
            g_fg.createSemaphore(d->device, &si, nullptr, &g_fg.semAcquire[i]) != VK_SUCCESS ||
            g_fg.createSemaphore(d->device, &si, nullptr, &g_fg.semGen[i]) != VK_SUCCESS ||
            g_fg.createSemaphore(d->device, &si, nullptr, &g_fg.semGame[i]) != VK_SUCCESS ||
            g_mfx.createEvent(d->device, &ei, nullptr, &g_fg.evIn[i]) != VK_SUCCESS ||
            g_mfx.createEvent(d->device, &ei, nullptr, &g_fg.evOut[i]) != VK_SUCCESS) {
            g_fg.why = "could not create fences, semaphores or events";
            return false;
        }
        for (int k = 0; k < 2; ++k) {
            VkExportMetalSharedEventInfoEXT se{VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT};
            se.event = k ? g_fg.evOut[i] : g_fg.evIn[i];
            VkExportMetalObjectsInfoEXT o{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
            o.pNext = &se;
            g_mfx.exportObjects(d->device, &o);
            (k ? g_fg.mtlOut : g_fg.mtlIn)[i] = (void*)se.mtlSharedEvent;
        }
        if (!g_fg.mtlIn[i] || !g_fg.mtlOut[i]) {
            g_fg.why = "MoltenVK's VkEvents are not MTLSharedEvents on this GPU";
            return false;
        }
    }
    g_fg.ready = true;
    g_fg.why = "ready";
    Log("[fg] frame generation ready (queue family %u, %d slots)", g_fg.family, kFgSlots);
    return true;
}

void FgDestroy(DeviceData* d) {
    if (!g_fg.tried) return;
    for (int i = 0; i < kFgSlots; ++i) mfx_fg_slot_wait(i);
    FgDestroyImages(d);
    for (int i = 0; i < kFgSlots; ++i) {
        if (g_fg.fence[i] && g_fg.destroyFence) g_fg.destroyFence(d->device, g_fg.fence[i], nullptr);
        for (VkSemaphore s : {g_fg.semAcquire[i], g_fg.semGen[i], g_fg.semGame[i]})
            if (s && g_fg.destroySemaphore) g_fg.destroySemaphore(d->device, s, nullptr);
        for (VkEvent e : {g_fg.evIn[i], g_fg.evOut[i]})
            if (e && g_mfx.destroyEvent) g_mfx.destroyEvent(d->device, e, nullptr);
    }
    if (g_fg.pool && g_fg.destroyCommandPool) g_fg.destroyCommandPool(d->device, g_fg.pool, nullptr);
    g_fg = FrameGen{};
}

// Images at the swapchain's size, and the interpolator for this render size. Caller holds g_t.
bool FgEnsureImages(DeviceData* d, uint32_t inW, uint32_t inH) {
    uint32_t w = g_fg.extent.width, h = g_fg.extent.height;
    if (!g_fg.uiImage || g_fg.w != w || g_fg.h != h || g_fg.imgFmt != g_fg.swapFmt) {
        if (g_fg.uiImage) {
            d->DeviceWaitIdle(d->device);
            FgDestroyImages(d);
        }
        const VkImageUsageFlags u = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (!CreateOwnImage(d, w, h, g_fg.swapFmt, u, &g_fg.uiImage, &g_fg.uiView) ||
            !CreateOwnImage(d, w, h, VK_FORMAT_R8G8B8A8_UNORM, u, &g_fg.prevImage, &g_fg.prevView) ||
            !CreateOwnImage(d, w, h, g_fg.swapFmt,  // MetalFX writes it as a render target
                            u | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &g_fg.genImage,
                            &g_fg.genView)) {
            FgDestroyImages(d);
            g_fg.ready = false;
            g_fg.why = "could not create the frame generation images";
            Log("[fg] %s", g_fg.why.c_str());
            return false;
        }
        g_fg.w = w;
        g_fg.h = h;
        g_fg.imgFmt = g_fg.swapFmt;
        g_fg.confInW = g_fg.confInH = 0;
    }
    if (g_fg.confInW != inW || g_fg.confInH != inH) {
        char err[256] = {};
        if (!mfx_fg_configure(inW, inH, w, h, (uint32_t)g_fg.swapFmt, err, sizeof err)) {
            g_fg.ready = false;
            g_fg.why = err;
            Log("[fg] interpolator setup failed: %s", err);
            return false;
        }
        g_fg.confInW = inW;
        g_fg.confInH = inH;
        g_fg.havePrev = false;
        Log("[fg] frame interpolator: depth and motion %ux%u, frames %ux%u (%s)", inW, inH, w, h,
            FormatName(g_fg.swapFmt) ? FormatName(g_fg.swapFmt) : "?");
    }
    return true;
}

void FgBarrier(DeviceData* d, VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
               VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.image = img;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                          nullptr, 1, &b);
}

void FgCopy(DeviceData* d, VkCommandBuffer cb, VkImage src, VkImage dst, uint32_t w, uint32_t h) {
    VkImageCopy c{};
    c.srcSubresource = c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.extent = {w, h, 1};
    d->CmdCopyImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
}

void FgSkip(uint64_t frame, const char* why) {  // caller holds g_t
    ++g_fg.skipped;
    if (g_fg.skipWhy != why) {
        g_fg.skipWhy = why;
        Log("[fg] frame %llu: no generated frame: %s", (unsigned long long)frame, why);
    }
}

// Called at present instead of the next layer's vkQueuePresentKHR. frame: the number of the frame being
// presented (g_frame has moved on by then); upscaled: its upscaled image was swapped into the blit.
VkResult FgPresent(DeviceData* d, VkQueue queue, const VkPresentInfoKHR* pInfo, uint64_t frame, bool upscaled) {
    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, SRC = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        DST = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, PRESENT = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    auto passThrough = [&] { return PacedPresent(d, queue, pInfo); };
    uint32_t slot = 0, gameIndex = 0, w = 0, h = 0;
    VkImage outImage = VK_NULL_HANDLE, depth = VK_NULL_HANDLE, motion = VK_NULL_HANDLE, gameImage = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapImages;
    bool dump = false, dumpPrev = false;
    MfxFgFrame job{};
    auto now = std::chrono::steady_clock::now();
    // Under g_t: decide, and gather what this frame needs. The presents themselves run without the lock
    // (FIFO presentation and the acquire can block until the screen refreshes).
    auto prepare = [&]() -> bool {
        if (!g_cfg.frameGen) {
            g_fg.havePrev = false;
            return false;
        }
        if (!g_fg.swapchain) {
            FgSkip(frame, "the swapchain wasn't set up for it (frame_gen=1 needs a restart of the game)");
            return false;
        }
        if (pInfo->swapchainCount != 1 || pInfo->pSwapchains[0] != g_fg.swapchain) {
            FgSkip(frame, "this present isn't for the game's swapchain");
            return false;
        }
        if (!upscaled || g_dlss.backend != Backend::MetalFx || g_mfx.lastFrame != frame) {
            g_fg.havePrev = false;  // menus, loading screens, upscaler off: start again afterwards
            FgSkip(frame, "the MetalFX upscaler didn't run this frame");
            return false;
        }
        if (g_dlss.outW != g_fg.extent.width || g_dlss.outH != g_fg.extent.height) {
            FgSkip(frame, "the upscaler's output isn't the size of the screen");
            return false;
        }
        if (!FgEnsure(d, queue) || !FgEnsureImages(d, g_dlss.inW, g_dlss.inH)) {
            FgSkip(frame, g_fg.why.c_str());
            return false;
        }
        gameIndex = pInfo->pImageIndices[0];
        if (gameIndex >= g_fg.swapImages.size()) {
            FgSkip(frame, "unknown swapchain image");
            return false;
        }
        slot = (uint32_t)(g_fg.counter++ % kFgSlots);
        outImage = g_dlss.outImage;
        depth = g_dlss.depth32Image;
        motion = g_dlss.mvImage;
        gameImage = g_fg.swapImages[gameIndex];
        swapchain = g_fg.swapchain;
        swapImages = g_fg.swapImages;
        w = g_fg.w;
        h = g_fg.h;
        bool continuous = g_fg.havePrev && g_fg.prevFrame + 1 == frame;
        dump = FgDumpAt(frame) && !g_fg.dumpFailed;
        if (dump && !g_fg.dumpBuf[0]) {
            for (int i = 0; i < 3 && !g_fg.dumpFailed; ++i)
                if (!CreateBuffer(d, (VkDeviceSize)g_fg.w * g_fg.h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  &g_fg.dumpBuf[i], &g_fg.dumpPtr[i]))
                    g_fg.dumpFailed = true;
            dump = !g_fg.dumpFailed;
        }
        dumpPrev = dump && continuous;
        job.color = MtlTexture(d, outImage);
        job.prevColor = MtlTexture(d, continuous ? g_fg.prevImage : outImage);
        job.depth = MtlTexture(d, depth);
        job.motion = MtlTexture(d, motion);
        job.ui = MtlTexture(d, g_fg.uiImage);
        job.output = MtlTexture(d, g_fg.genImage);
        job.jitterX = g_mfx.lastJx;
        job.jitterY = g_mfx.lastJy;
        job.mvScaleX = g_cfg.mvScaleX;
        job.mvScaleY = g_cfg.mvScaleY;
        double dt = continuous ? std::chrono::duration<double>(now - g_fg.prevTime).count() : 1.0 / 30.0;
        job.deltaTime = (float)std::min(std::max(dt, 0.001), 0.25);
        bool cam = g_frameCam.frame == frame;
        job.nearPlane = cam ? g_frameCam.nearPlane : 4.f;
        job.farPlane = cam ? g_frameCam.farPlane : 7200.f;
        job.fovDegrees = cam && g_frameCam.fovYDegrees > 0.f ? g_frameCam.fovYDegrees : 60.f;
        job.aspect = cam && g_frameCam.aspect > 0.f ? g_frameCam.aspect : (float)g_fg.extent.width / g_fg.extent.height;
        job.reset = !continuous || g_mfx.lastReset;
        job.depthReversed = false;  // GL-style depth: near 0, far 1
        if (LogFrame(frame))
            Log("[fg] frame %llu: slot %u, dt %.1f ms, near %.3g far %.3g fov %.1f aspect %.3f%s",
                (unsigned long long)frame, slot, job.deltaTime * 1000.f, job.nearPlane, job.farPlane,
                job.fovDegrees, job.aspect, job.reset ? ", history reset" : "");
        return true;
    };
    bool go;
    {
        TLock lk(g_t);
        go = prepare();
    }
    if (!go) return passThrough();
    // The slot's last use: its command buffer (and the MetalFX work it waited for) must be done.
    if (g_fg.fenceUsed[slot]) {
        if (g_fg.waitForFences(d->device, 1, &g_fg.fence[slot], VK_TRUE, 2000000000ull) != VK_SUCCESS) {
            {
                TLock lk(g_t);
                Log("[fg] frame %llu: an earlier generated frame didn't finish within 2 s; frame generation off",
                    (unsigned long long)frame);
                g_cfg.frameGen = 0;  // the slot would hang again on every round
                g_fg.havePrev = false;
            }
            return passThrough();
        }
        g_fg.resetFences(d->device, 1, &g_fg.fence[slot]);
        g_fg.fenceUsed[slot] = false;
    }
    mfx_fg_slot_wait((int)slot);
    // The extra image. FIFO with 3 images: one on screen, the game's, and this one once the screen moves on.
    uint32_t genIndex = 0;
    VkResult ar = g_fg.acquireNextImage(d->device, swapchain, 500000000ull, g_fg.semAcquire[slot], VK_NULL_HANDLE,
                                        &genIndex);
    if ((ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) || genIndex >= swapImages.size()) {
        {
            TLock lk(g_t);
            char why[96];
            std::snprintf(why, sizeof why, "no second swapchain image (vkAcquireNextImageKHR: %d)", (int)ar);
            FgSkip(frame, why);
            g_fg.havePrev = false;
        }
        return passThrough();
    }
    VkImage genSwap = swapImages[genIndex];

    VkCommandBuffer cb = g_fg.cb[slot];
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    d->BeginCommandBuffer(cb, &bi);
    // the finished frame (UI included) -> ours
    FgBarrier(d, cb, gameImage, PRESENT, SRC, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    FgBarrier(d, cb, g_fg.uiImage, VK_IMAGE_LAYOUT_UNDEFINED, DST, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    FgCopy(d, cb, gameImage, g_fg.uiImage, w, h);
    FgBarrier(d, cb, gameImage, SRC, PRESENT, VK_ACCESS_TRANSFER_READ_BIT, 0);
    FgBarrier(d, cb, g_fg.uiImage, DST, RO, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    if (!g_fg.genInGeneral) {
        FgBarrier(d, cb, g_fg.genImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
                  VK_ACCESS_MEMORY_WRITE_BIT);
        g_fg.genInGeneral = true;
    }
    // MetalFX: wait for "in", interpolate, signal "out"
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                          nullptr, 0, nullptr);
    g_mfx.cmdSetEvent(cb, g_fg.evIn[slot], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    g_mfx.cmdWaitEvents(cb, 1, &g_fg.evOut[slot], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        1, &mb, 0, nullptr, 0, nullptr);
    // the generated frame -> the extra swapchain image
    FgBarrier(d, cb, g_fg.genImage, VK_IMAGE_LAYOUT_GENERAL, SRC, VK_ACCESS_MEMORY_WRITE_BIT,
              VK_ACCESS_TRANSFER_READ_BIT);
    if (dump) {
        VkBufferImageCopy bc{};
        bc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        bc.imageExtent = {w, h, 1};
        if (dumpPrev) {
            FgBarrier(d, cb, g_fg.prevImage, RO, SRC, 0, VK_ACCESS_TRANSFER_READ_BIT);
            d->CmdCopyImageToBuffer(cb, g_fg.prevImage, SRC, g_fg.dumpBuf[0], 1, &bc);
            FgBarrier(d, cb, g_fg.prevImage, SRC, RO, VK_ACCESS_TRANSFER_READ_BIT, 0);
        }
        d->CmdCopyImageToBuffer(cb, g_fg.genImage, SRC, g_fg.dumpBuf[1], 1, &bc);
        FgBarrier(d, cb, g_fg.uiImage, RO, SRC, 0, VK_ACCESS_TRANSFER_READ_BIT);
        d->CmdCopyImageToBuffer(cb, g_fg.uiImage, SRC, g_fg.dumpBuf[2], 1, &bc);
        FgBarrier(d, cb, g_fg.uiImage, SRC, RO, VK_ACCESS_TRANSFER_READ_BIT, 0);
    }
    FgBarrier(d, cb, genSwap, VK_IMAGE_LAYOUT_UNDEFINED, DST, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    FgCopy(d, cb, g_fg.genImage, genSwap, w, h);
    FgBarrier(d, cb, genSwap, DST, PRESENT, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
    FgBarrier(d, cb, g_fg.genImage, SRC, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
              VK_ACCESS_MEMORY_WRITE_BIT);
    // this frame's upscaled image becomes the next interpolation's previous frame
    FgBarrier(d, cb, outImage, RO, SRC, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    FgBarrier(d, cb, g_fg.prevImage, VK_IMAGE_LAYOUT_UNDEFINED, DST, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    FgCopy(d, cb, outImage, g_fg.prevImage, w, h);
    FgBarrier(d, cb, outImage, SRC, RO, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT);
    FgBarrier(d, cb, g_fg.prevImage, DST, RO, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    g_fg.endCommandBuffer(cb);

    // Events "not set" before the submission (MoltenVK encodes set/wait with value + 1 at submit).
    uint64_t vin = mfx_event_unset(g_fg.mtlIn[slot]), vout = mfx_event_unset(g_fg.mtlOut[slot]);
    job.eventIn = g_fg.mtlIn[slot];
    job.waitValue = vin + 1;
    job.eventOut = g_fg.mtlOut[slot];
    job.signalValue = vout + 1;
    std::vector<VkSemaphore> waits(pInfo->pWaitSemaphores, pInfo->pWaitSemaphores + pInfo->waitSemaphoreCount);
    waits.push_back(g_fg.semAcquire[slot]);
    std::vector<VkPipelineStageFlags> stages(waits.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkSemaphore signals[2] = {g_fg.semGen[slot], g_fg.semGame[slot]};
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = (uint32_t)waits.size();
    si.pWaitSemaphores = waits.data();
    si.pWaitDstStageMask = stages.data();
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 2;
    si.pSignalSemaphores = signals;
    VkResult sr = d->QueueSubmit(queue, 1, &si, g_fg.fence[slot]);
    if (sr != VK_SUCCESS) {
        // Nothing was submitted: the game's semaphores weren't waited on, so its present still works. The extra
        // image stays acquired (swapchains have no way to give it back): present it as it is.
        mfx_event_set(job.eventOut, job.signalValue);
        {
            TLock lk(g_t);
            if (g_fg.errors++ < 5) Log("[fg] vkQueueSubmit failed (%d); frame generation off", (int)sr);
            g_cfg.frameGen = 0;
            g_fg.havePrev = false;
        }
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &g_fg.semAcquire[slot];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &genIndex;
        d->QueuePresentKHR(queue, &pi);
        return passThrough();
    }
    g_fg.fenceUsed[slot] = true;
    char err[256] = {};
    bool encoded = mfx_fg_encode((int)slot, &job, err, sizeof err);
    if (dump && g_fg.waitForFences(d->device, 1, &g_fg.fence[slot], VK_TRUE, 2000000000ull) == VK_SUCCESS) {
        std::error_code ec;
        std::filesystem::path dir = CaptureDir();
        std::filesystem::create_directories(dir, ec);
        char base[64];
        std::snprintf(base, sizeof base, "fg_f%llu", (unsigned long long)frame);
        bool bgra = g_fg.swapFmt == VK_FORMAT_B8G8R8A8_UNORM;
        if (dumpPrev) FgWritePpm(dir / (std::string(base) + "_prev.ppm"), g_fg.dumpPtr[0], w, h, false);
        FgWritePpm(dir / (std::string(base) + "_gen.ppm"), g_fg.dumpPtr[1], w, h, bgra);
        FgWritePpm(dir / (std::string(base) + "_cur.ppm"), g_fg.dumpPtr[2], w, h, bgra);
        TLock lk(g_t);
        Log("[fg] wrote %s_{%sgen,cur}.ppm (%ux%u)", base, dumpPrev ? "prev," : "", w, h);
    }
    {
        TLock lk(g_t);
        if (!encoded && g_fg.errors++ < 5) Log("[fg] frame %llu: %s", (unsigned long long)frame, err);
        g_fg.havePrev = true;
        g_fg.prevFrame = frame;
        g_fg.prevTime = now;
        ++g_fg.generated;
        if (g_fg.generated == 1) Log("[fg] first generated frame (frame %llu)", (unsigned long long)frame);
        g_fg.skipWhy.clear();
    }
    // generated frame first, then the game's
    VkPresentInfoKHR pg{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pg.waitSemaphoreCount = 1;
    pg.pWaitSemaphores = &g_fg.semGen[slot];
    pg.swapchainCount = 1;
    pg.pSwapchains = &swapchain;
    pg.pImageIndices = &genIndex;
    VkResult r1 = d->QueuePresentKHR(queue, &pg);
    VkPresentInfoKHR pgame = *pInfo;
    pgame.waitSemaphoreCount = 1;
    pgame.pWaitSemaphores = &g_fg.semGame[slot];
    VkResult r2 = d->QueuePresentKHR(queue, &pgame);
    if (r1 != VK_SUCCESS && r1 != VK_SUBOPTIMAL_KHR) {
        TLock lk(g_t);
        if (g_fg.errors++ < 5) Log("[fg] presenting the generated frame failed (%d)", (int)r1);
    }
    return r2;
}
#endif  // __APPLE__

bool RecordTestUpscale(DeviceData* d, VkCommandBuffer cb, const CmdState& cs) {
    if (!EnsureTestUpscale(d)) return false;
    bool isNew = false;
    VkDescriptorSet set = CachedSetRaw(d, g_dlss.tstSets, cs.colorView, g_dlss.tstPool, g_dlss.tstSetLayout,
                                       g_dlss.tstSetUsed, isNew);
    if (!set) return false;
    if (isNew) {
        VkDescriptorImageInfo src{g_dlss.tstSampler, cs.colorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo dst{VK_NULL_HANDLE, g_dlss.outView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w[2] = {};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                &src, nullptr, nullptr};
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dst,
                nullptr, nullptr};
        d->UpdateDescriptorSets(d->device, 2, w, 0, nullptr);
    }
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.image = g_dlss.outImage;
    b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                          0, nullptr, 1, &b);
    int32_t size[2] = {(int32_t)g_dlss.outW, (int32_t)g_dlss.outH};
    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dlss.tstPipeline);
    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dlss.tstPipeLayout, 0, 1, &set, 0, nullptr);
    d->CmdPushConstants(cb, g_dlss.tstPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof size, size);
    d->CmdDispatch(cb, (g_dlss.outW + 7) / 8, (g_dlss.outH + 7) / 8, 1);
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                          0, nullptr, 1, &b);
    g_dlss.tstSetUsed = true;
    return true;
}

// ---------------------------------------------------------------------------------------------
// Which upscaler runs
// ---------------------------------------------------------------------------------------------
UpscalerPref ActivePref() {
    if (g_cfg.upscaler == UpscalerPref::Off) return UpscalerPref::Off;  // Ctrl+Shift+U can't override "off"
    if (g_dlss.override_ == (int)Backend::Dlss) return UpscalerPref::Dlss;
    if (g_dlss.override_ == (int)Backend::Fsr) return UpscalerPref::Fsr;
    return g_cfg.upscaler;
}

// Why nothing runs, for the log and TF3-DLSS-DID-NOT-START.txt.
std::string NoUpscalerReason() {
    UpscalerPref p = ActivePref();
    if (p == UpscalerPref::Off) return "upscaler=off in tf3_dlss.cfg";
    if (p == UpscalerPref::Dlss) return "upscaler=dlss, but " + g_dlss.why;
    if (p == UpscalerPref::Fsr) return "upscaler=fsr, but " + g_fsr.why;
#ifdef __APPLE__
    if (p == UpscalerPref::MetalFx) return "upscaler=metalfx, but " + g_mfx.why;
    return "MetalFX: " + g_mfx.why;
#else
    if (p == UpscalerPref::MetalFx) return "upscaler=metalfx, but MetalFX exists only on macOS";
    return "DLSS: " + g_dlss.why + "; FSR: " + g_fsr.why;
#endif
}

// Picks the upscaler for this frame from the setting and what works. Caller holds g_t.
Backend ChooseBackend() {
    UpscalerPref p = ActivePref();
    bool dlssOk = g_dlss.ngxReady && !g_dlss.failed;
    auto fsrOk = [] { return !g_fsr.failed && FsrLoad(); };
    Backend b = Backend::None;
    if (TestUpscaler()) p = p == UpscalerPref::Off ? p : UpscalerPref::Auto;
#ifdef __APPLE__
    // macOS: no DLSS, no AMD runtime; MetalFX is the upscaler (MfxEnsure runs on the first recorded frame)
    auto mfxOk = [] { return g_mfx.ready || !g_mfx.tried; };
#else
    auto mfxOk = [] { return false; };
#endif
    switch (p) {
        case UpscalerPref::Auto:
            b = TestUpscaler() ? Backend::Test : mfxOk() ? Backend::MetalFx : dlssOk ? Backend::Dlss
              : fsrOk() ? Backend::Fsr : Backend::None;
            break;
        case UpscalerPref::Dlss: b = dlssOk ? Backend::Dlss : Backend::None; break;
        case UpscalerPref::Fsr: b = fsrOk() ? Backend::Fsr : Backend::None; break;
        case UpscalerPref::MetalFx: b = mfxOk() ? Backend::MetalFx : Backend::None; break;
        case UpscalerPref::Off: break;
    }
    if (b != g_dlss.backend) {
        if (b == Backend::None) Log("[upscale] no upscaler running: %s", NoUpscalerReason().c_str());
        else Log("[upscale] using %s (upscaler=%s%s)", BackendName(b), PrefName(g_cfg.upscaler),
                 g_dlss.override_ >= 0 ? ", switched with Ctrl+Shift+U" : "");
        g_dlss.backend = b;
        g_dlss.resetNext = true;
    }
    return b;
}

// True if an upscaler is expected to run (decides the texture LOD bias at sampler creation).
bool UpscalerExpected() {
    if (!g_cfg.enabled) return false;
    UpscalerPref p = g_cfg.upscaler;
    if (p == UpscalerPref::Off) return false;
    if (p != UpscalerPref::Fsr && g_dlss.ngxReady && !g_dlss.failed) return true;
    return p != UpscalerPref::Dlss && !g_fsr.failed && FsrLoad();
}

void RestoreGameState(DeviceData* d, VkCommandBuffer cb, const CmdState& cs) {
    if (cs.pipeline) d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, cs.pipeline);
    for (uint32_t i = 0; i < 16; ++i)
        if (cs.vb[i].first) d->CmdBindVertexBuffers(cb, i, 1, &cs.vb[i].first, &cs.vb[i].second);
    if (cs.ib) d->CmdBindIndexBuffer(cb, cs.ib, cs.ibOff, cs.ibType);
    if (cs.hasCurVp) d->CmdSetViewport(cb, 0, 1, &cs.curVp);
    if (cs.hasScissor) d->CmdSetScissor(cb, 0, 1, &cs.scissor);
    for (uint32_t i = 0; i < cs.sets.size(); ++i) {
        const BoundSet& bs = cs.sets[i];
        if (!bs.set || !bs.layout) continue;
        d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, bs.layout, i, 1, &bs.set,
                                 (uint32_t)bs.dynOffsets.size(), bs.dynOffsets.data());
    }
    if (cs.pushLayout)  // the reactive pass pushes the replayed draws' constants
        for (uint32_t i = 0; i < cs.pushCount; ++i)
            d->CmdPushConstants(cb, cs.pushLayout, cs.pushRanges[i].stages, cs.pushRanges[i].offset,
                                cs.pushRanges[i].size, cs.push + cs.pushRanges[i].offset);
}

// Record the upscaler (DLSS or FSR) into the game's command buffer, outside any render pass.
// Caller holds g_t.
void RecordUpscale(DeviceData* d, VkCommandBuffer cb, CmdState& cs) {
    if (!g_dlss.enabled || cs.dlssDone) return;
    Backend backend = ChooseBackend();
    if (backend == Backend::None) return;
    bool verbose = LogFrame(g_frame);
    if (!cs.colorImage || !cs.depthImage || !g_swapExtent.width) {
        if (verbose)
            Log("[dlss] frame %llu: missing input (color=%s depth=%s swap=%ux%u)", (unsigned long long)g_frame,
                H(cs.colorImage).c_str(), H(cs.depthImage).c_str(), g_swapExtent.width, g_swapExtent.height);
        return;
    }
    const ImageInfo& ci = g_images[cs.colorImage];
    const ImageInfo& di = g_images[cs.depthImage];
    if (!(di.usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        if (verbose) Log("[dlss] depth image lacks SAMPLED usage; skipping");
        return;
    }
    VkImageLayout colorLayout = CurrentLayout(cs, cs.colorImage);
    VkImageLayout depthLayout = CurrentLayout(cs, cs.depthImage);
    if (colorLayout == VK_IMAGE_LAYOUT_MAX_ENUM || depthLayout == VK_IMAGE_LAYOUT_MAX_ENUM) {
        if (verbose) Log("[dlss] frame %llu: unknown input layout (color %d, depth %d); skipping",
                         (unsigned long long)g_frame, (int)colorLayout, (int)depthLayout);
        return;
    }
    VkExtent2D out = UpscaleExtent(ci.w, ci.h, backend == Backend::MetalFx);
    static VkExtent2D loggedOut{}, loggedSwap{};
    if (out.width != loggedOut.width || out.height != loggedOut.height || g_swapExtent.width != loggedSwap.width ||
        g_swapExtent.height != loggedSwap.height) {
        loggedOut = out;
        loggedSwap = g_swapExtent;
        if (out.height != g_swapExtent.height)
            Log("[upscale] upscaling %ux%u to %ux%u; the game's blit stretches that to the screen (%ux%u)%s", ci.w,
                ci.h, out.width, out.height, g_swapExtent.width, g_swapExtent.height,
                g_cfg.upscaleTo > 0 && out.height <= (uint32_t)g_cfg.upscaleTo ? " (upscale_to)"
                                                                               : " (MetalFX upscales 3x at most)");
    }
    if (!EnsureImages(d, ci.w, ci.h, out.width, out.height)) return;
    if (backend == Backend::Dlss && !EnsureDlssFeature(d, cb)) return;  // auto mode moves on to FSR next frame
    if (backend == Backend::Fsr && (!g_dlss.mvReady || !FsrEnsure(d))) {
        if (!g_dlss.mvReady && verbose) Log("[fsr] the motion-vector pass isn't available; FSR needs it");
        return;
    }
    VkImageView depthView = DepthView(d, cs.depthImage);
    if (!depthView) return;

    VkImageAspectFlags depthAspects =
        VK_IMAGE_ASPECT_DEPTH_BIT | (HasStencil(di.fmt) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (GpuPerfEnsure(d, cb)) GpuMark(d, cb, kGmStart);  // perf=1: GPU timings

    // inputs -> read-only, output -> general
    if (colorLayout != RO) Barrier(d, cb, cs.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, colorLayout, RO);
    if (depthLayout != RO) Barrier(d, cb, cs.depthImage, depthAspects, depthLayout, RO);
    // Motion vectors: compute into a buffer, then copy into the R16G16 image DLSS reads.
    if (g_dlss.mvReady && UpdateMvDepth(d, depthView)) {
        uint32_t slot = (uint32_t)(g_dlss.mvCounter++ % kMvSlots);  // own counter: frames can overlap
        uint32_t dyn = slot * kMvStride;
        {   // the previous frame's copy may still be reading mvBuffer: wait before overwriting it
            VkBufferMemoryBarrier war{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            war.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
            war.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            war.srcQueueFamilyIndex = war.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            war.buffer = g_dlss.mvBuffer;
            war.size = VK_WHOLE_SIZE;
            d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                  nullptr, 1, &war, 0, nullptr);
        }
        // the depth copy (old contents not needed) -> writable
        Barrier(d, cb, g_dlss.depth32Image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        int samples = (int)di.samples;
        if (samples != g_dlss.mvDepthSamples) {
            g_dlss.mvDepthSamples = samples;
            Log("[mv] game depth %s has %d sample(s)%s", ImageDesc(cs.depthImage).c_str(), samples,
                samples > 1 ? ": reading sample 0 (multisampled variant)" : "");
        }
        static const bool msVariant = [] {  // TF3_DLSS_MS_DEPTH=0: old behaviour, for comparing in the harness
            const char* e = std::getenv("TF3_DLSS_MS_DEPTH");
            return !(e && *e == '0');
        }();
        VkPipeline mvPipe = samples > 1 && msVariant ? g_dlss.mvPipelineMs : g_dlss.mvPipeline;
        if (!mvPipe) mvPipe = g_dlss.mvPipeline;
        d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, mvPipe);
        d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dlss.mvPipeLayout, 0, 1, &g_dlss.mvSet, 1, &dyn);
        d->CmdDispatch(cb, (ci.w + 7) / 8, (ci.h + 7) / 8, 1);
        Barrier(d, cb, g_dlss.depth32Image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, RO);
        VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer = g_dlss.mvBuffer;
        bb.size = VK_WHOLE_SIZE;
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                              &bb, 0, nullptr);
        Barrier(d, cb, g_dlss.mvImage, VK_IMAGE_ASPECT_COLOR_BIT, g_dlss.mvLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {ci.w, ci.h, 1};
        d->CmdCopyBufferToImage(cb, g_dlss.mvBuffer, g_dlss.mvImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        // A multisampled game depth can't be read through a plain sampler2D (MoltenVK returns unrelated
        // values): the object pass then compares against the camera pass's single-sample R32F copy.
        VkImageView objDepth = (di.samples > 1 && g_dlss.depth32View) ? g_dlss.depth32View : depthView;
        if (g_cfg.motionVectors && RecordObjectPass(d, cb, cs, objDepth, dyn))
            Barrier(d, cb, g_dlss.mvImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RO);
        else
            Barrier(d, cb, g_dlss.mvImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, RO);
        g_dlss.mvLayout = RO;
        g_dlss.mvSetUsed = true;
        cs.mvSlot = slot;
        cs.mvPending = true;
        if (g_cap.requested > 0 && !((ci.usage & di.usage) & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            Log("[record] the game's colour/depth images can't be copied (set debug_keys=1 and restart the game)");
            g_cap.requested = 0;
        }
        if (g_cap.requested > 0 && EnsureCaptureBuffers(d, ci.w, ci.h)) {
            RecordCaptureCopies(d, cb, cs, depthAspects);
            --g_cap.requested;
        }
    } else if (backend == Backend::Fsr) {  // FSR needs the depth copy from the motion pass
        if (depthLayout != RO) Barrier(d, cb, cs.depthImage, depthAspects, RO, depthLayout);
        if (colorLayout != RO) Barrier(d, cb, cs.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, RO, colorLayout);
        GpuFrameEnd(d, cb);
        RestoreGameState(d, cb, cs);
        return;
    } else if (g_dlss.mvLayout != RO) {  // no motion pass: zero motion vectors
        Barrier(d, cb, g_dlss.mvImage, VK_IMAGE_ASPECT_COLOR_BIT, g_dlss.mvLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue zero{};
        VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        d->CmdClearColorImage(cb, g_dlss.mvImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
        Barrier(d, cb, g_dlss.mvImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, RO);
        g_dlss.mvLayout = RO;
    }
    GpuMark(d, cb, kGmMotion);
    Barrier(d, cb, g_dlss.outImage, VK_IMAGE_ASPECT_COLOR_BIT, g_dlss.outLayout, VK_IMAGE_LAYOUT_GENERAL);

    if (g_cfg.jitter) {
        FrameJitter(cs.jx, cs.jy, &cs.jitterIdx, &cs.jitterPhases);
        cs.jitterSet = true;
    }
    bool ok = false;
    unsigned dlssResult = 0;
    if (backend == Backend::Dlss) {
        VkImageSubresourceRange colorRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageSubresourceRange depthRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        NVSDK_NGX_Resource_VK color = NVSDK_NGX_Create_ImageView_Resource_VK(cs.colorView, cs.colorImage, colorRange,
                                                                              ci.fmt, ci.w, ci.h, false);
        NVSDK_NGX_Resource_VK depth =
            NVSDK_NGX_Create_ImageView_Resource_VK(depthView, cs.depthImage, depthRange, di.fmt, di.w, di.h, false);
        NVSDK_NGX_Resource_VK mv = NVSDK_NGX_Create_ImageView_Resource_VK(g_dlss.mvView, g_dlss.mvImage, colorRange,
                                                                           VK_FORMAT_R16G16_SFLOAT, ci.w, ci.h, false);
        NVSDK_NGX_Resource_VK out = NVSDK_NGX_Create_ImageView_Resource_VK(
            g_dlss.outView, g_dlss.outImage, colorRange, VK_FORMAT_R8G8B8A8_UNORM, g_dlss.outW, g_dlss.outH, true);

        NVSDK_NGX_VK_DLSS_Eval_Params ep{};
        ep.Feature.pInColor = &color;
        ep.Feature.pInOutput = &out;
        ep.pInDepth = &depth;
        ep.pInMotionVectors = &mv;
        ep.InJitterOffsetX = cs.jitterSet ? cs.jx * g_cfg.jitterSignX : 0.f;
        ep.InJitterOffsetY = cs.jitterSet ? cs.jy * g_cfg.jitterSignY : 0.f;
        ep.InRenderSubrectDimensions = {ci.w, ci.h};
        ep.InReset = g_dlss.resetNext ? 1 : 0;
        ep.InMVScaleX = g_cfg.mvScaleX;
        ep.InMVScaleY = g_cfg.mvScaleY;
        NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cb, g_dlss.handle, g_dlss.params, &ep);
        ok = NVSDK_NGX_SUCCEED(r);
        dlssResult = (unsigned)r;
    } else if (backend == Backend::Fsr) {
        ok = RecordFsr(cb, cs, ci);
    } else if (backend == Backend::MetalFx) {
#ifdef __APPLE__
        ok = RecordMetalFx(d, cb, cs, ci);
        if (!ok && verbose) Log("[mfx] frame %llu: not recorded (%s)", (unsigned long long)g_frame, g_mfx.why.c_str());
#endif
    } else {
        ok = RecordTestUpscale(d, cb, cs);
    }
    GpuMark(d, cb, kGmUpscaler);

    if (g_cap.outPending && g_cap.out && ok) {  // recording: the upscaler's output itself
        Barrier(d, cb, g_dlss.outImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy oc{};
        oc.imageExtent = {g_cap.outW, g_cap.outH, 1};
        oc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        d->CmdCopyImageToBuffer(cb, g_dlss.outImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_cap.out, 1, &oc);
        Barrier(d, cb, g_dlss.outImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL);
    }
    g_cap.outPending = false;
    if (ok && backend != Backend::Fsr) RecordSharpen(d, cb);  // FSR sharpens itself (RCAS)
#ifdef __APPLE__
    if (ok && backend == Backend::MetalFx) RecordReactiveDebug(d, cb);
#endif
    GpuMark(d, cb, kGmSharpen);
    if (g_cfg.debugView && ok && (g_cfg.debugView == 2 || cs.mvPending)) RecordDebugView(d, cb, cs);
    if (ok) RecordOverlay(d, cb, backend, cs, ci.w, ci.h);

    // back to the game's layouts; our output becomes sampleable for the blit
    Barrier(d, cb, g_dlss.outImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, RO);
    g_dlss.outLayout = RO;
    if (depthLayout != RO) Barrier(d, cb, cs.depthImage, depthAspects, RO, depthLayout);
    if (colorLayout != RO) Barrier(d, cb, cs.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, RO, colorLayout);
    GpuMark(d, cb, kGmEnd);
    GpuFrameEnd(d, cb);
    RestoreGameState(d, cb, cs);

    if (!ok) {
        if (backend == Backend::Dlss)
            Log("[dlss] Evaluate failed: 0x%08x (frame %llu)", dlssResult, (unsigned long long)g_frame);
        return;
    }
    cs.dlssDone = true;
    g_dlss.resetNext = false;
    g_lastDlssFrame = g_frame;
    if (verbose)
        Log("[upscale] frame %llu: %s %ux%u -> %ux%u (color %s layout %d, depth layout %d, motion %s)",
            (unsigned long long)g_frame, BackendName(backend), ci.w, ci.h, g_dlss.outW, g_dlss.outH,
            FormatName(ci.fmt) ? FormatName(ci.fmt) : "?", (int)colorLayout, (int)depthLayout,
            cs.mvPending ? (g_cfg.motionVectors ? (g_cfg.objectMotion ? "camera + objects" : "camera") : "off by cfg")
                         : "zero");
}

// Our copy of the blit's descriptor set, with binding `binding` pointing at the DLSS output.
VkDescriptorSet BlitSet(DeviceData* d, VkDescriptorSet gameSet, uint32_t binding) {
    auto lo = g_setLayoutOf.find(gameSet);
    if (lo == g_setLayoutOf.end()) return VK_NULL_HANDLE;
    auto cached = g_dlss.blitSets.find(lo->second);
    if (cached != g_dlss.blitSets.end()) return cached->second;

    auto lb = g_setLayouts.find(lo->second);
    if (lb == g_setLayouts.end()) return VK_NULL_HANDLE;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    std::string layoutDesc;
    for (const auto& b : lb->second) {
        layoutDesc += " b" + std::to_string(b.binding) + ":t" + std::to_string((int)b.type) + "x" +
                      std::to_string(b.count);
        if (b.binding == binding) type = b.type;
    }
    if (type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && type != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
        Log("[blit] unexpected descriptor type %d at binding %u (layout:%s)", (int)type, binding, layoutDesc.c_str());
        return VK_NULL_HANDLE;
    }
    if (!g_dlss.pool) {
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
                                        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64},
                                        {VK_DESCRIPTOR_TYPE_SAMPLER, 64},
                                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64},
                                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 64},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 16;
        pi.poolSizeCount = (uint32_t)(sizeof sizes / sizeof sizes[0]);
        pi.pPoolSizes = sizes;
        // The game's blit layout may be update-after-bind; such sets must come from a pool with that flag
        // (a pool with it also serves ordinary layouts).
        if (g_updateAfterBindLayouts.count(lo->second)) pi.flags |= VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        if (d->CreateDescriptorPool(d->device, &pi, nullptr, &g_dlss.pool) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.f;
        if (d->CreateSampler(d->device, &si, nullptr, &g_dlss.sampler) != VK_SUCCESS) return VK_NULL_HANDLE;
    }
    VkDescriptorSetLayout layout = lo->second;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = g_dlss.pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    VkDescriptorSet ours = VK_NULL_HANDLE;
    if (d->AllocateDescriptorSets(d->device, &ai, &ours) != VK_SUCCESS) return VK_NULL_HANDLE;

    // Copy every binding from the game's set (keeps samplers/buffers), then replace ours.
    std::vector<VkCopyDescriptorSet> copies;
    for (const auto& b : lb->second) {
        if (b.binding == binding) continue;
        VkCopyDescriptorSet c{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET};
        c.srcSet = gameSet;
        c.srcBinding = b.binding;
        c.dstSet = ours;
        c.dstBinding = b.binding;
        c.descriptorCount = b.count;
        copies.push_back(c);
    }
    VkDescriptorImageInfo ii{g_dlss.sampler, g_dlss.outView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = ours;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = type;
    w.pImageInfo = &ii;
    d->UpdateDescriptorSets(d->device, 1, &w, (uint32_t)copies.size(), copies.data());
    g_dlss.blitSets[layout] = ours;
    Log("[blit] created replacement set %s for layout %s (binding %u type %d; layout:%s)", H(ours).c_str(),
        H(layout).c_str(), binding, (int)type, layoutDesc.c_str());
    return ours;
}

// =============================================================================================
// Per-draw logic. Returns a set to bind temporarily (blit swap), or VK_NULL_HANDLE.
// =============================================================================================
struct SwapPlan {
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint32_t index = 0;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::vector<uint32_t> dyn;  // dynamic offsets for the replacement set
};

// Caller holds g_t.
SwapPlan OnDraw(DeviceData* d, CmdState& cs) {
    SwapPlan plan;
    if (cs.kind == Kind::FsrEasu && !g_firstFsrFrame) {
        g_firstFsrFrame = g_frame + 1;
        Log("[layer] FSR pass first seen on frame %llu", (unsigned long long)g_frame);
    }
    CheckForView(cs);

    switch (cs.kind) {
        case Kind::Compose:  // the tonemap pass's framebuffer contains the main depth buffer
            for (VkImage img : FbImages(cs.fb)) {
                auto it = g_images.find(img);
                if (it != g_images.end() && IsDepthFormat(it->second.fmt)) cs.depthImage = img;
            }
            return plan;
        case Kind::FsrEasu: {  // EASU's set 0 binding 0 is the post-tonemap colour = DLSS input
            if (!cs.sets.empty()) {
                auto c = g_setContents.find(cs.sets[0].set);
                if (c != g_setContents.end()) {
                    auto b = c->second.find(0);
                    if (b != c->second.end() && !b->second.empty() && b->second[0].view) {
                        cs.colorView = b->second[0].view;
                        cs.colorImage = ViewImage(cs.colorView);
                    }
                }
            }
            return plan;
        }
        case Kind::FsrRcas: {
            auto imgs = FbImages(cs.fb);
            if (!imgs.empty()) cs.rcasOut = imgs[0];
            cs.rcasInThisPass = true;
            cs.blitGen = UINT32_MAX;
            return plan;
        }
        default: break;
    }

    // The blit: first draw after RCAS that samples RCAS's output.
    if (!cs.rcasOut) return plan;
    // Every draw after RCAS (the whole UI) gets here; search the bound sets again only if they changed.
    if (cs.blitGen != cs.setsGen) {
        cs.blitHit = FindSampled(cs, cs.rcasOut);
        cs.blitGen = cs.setsGen;
    }
    auto [si, binding] = cs.blitHit;
    if (si < 0) return plan;
    if (LogFrame(g_frame) && g_lastBlitLogFrame != g_frame) {
        g_lastBlitLogFrame = g_frame;
        Log("[blit] frame %llu: pipeline %s samples RCAS output via set %d binding %d (dlss %s)",
            (unsigned long long)g_frame, H(cs.pipeline).c_str(), si, binding, cs.dlssDone ? "ready" : "not run");
    }
    if (!cs.dlssDone) return plan;
    VkDescriptorSet ours = BlitSet(d, cs.sets[si].set, (uint32_t)binding);
    if (!ours || !cs.sets[si].layout) return plan;
    plan.set = ours;
    plan.index = (uint32_t)si;
    plan.layout = cs.sets[si].layout;
    g_lastSwapFrame = g_frame;
    return plan;
}

}  // namespace

// =============================================================================================
// Instance
// =============================================================================================
static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateInstance(const VkInstanceCreateInfo* pCreateInfo,
                                                         const VkAllocationCallbacks* pAllocator,
                                                         VkInstance* pInstance) {
    auto* chain = (VkLayerInstanceCreateInfo*)pCreateInfo->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerInstanceCreateInfo*)chain->pNext;
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    VkLayerInstanceLink* nextLink = chain->u.pLayerInfo;  // layers below advance it again; restored for a retry
    auto createFn = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");

    // Add the instance extensions NGX needs (harmless on other GPUs: the loader just enables them).
    std::vector<const char*> exts(pCreateInfo->ppEnabledExtensionNames,
                                  pCreateInfo->ppEnabledExtensionNames + pCreateInfo->enabledExtensionCount);
    unsigned int ni = 0, nd = 0;
    const char** iexts = nullptr;
    const char** dexts = nullptr;
    bool wantDlss = g_cfg.upscaler != UpscalerPref::Fsr && g_cfg.upscaler != UpscalerPref::Off;
    if (wantDlss && NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_RequiredExtensions(&ni, &iexts, &nd, &dexts)))
        for (unsigned i = 0; i < ni; ++i)
            if (std::none_of(exts.begin(), exts.end(), [&](const char* e) { return !std::strcmp(e, iexts[i]); })) {
                exts.push_back(iexts[i]);
                Log("[instance] + %s (for DLSS)", iexts[i]);
            }
    VkInstanceCreateInfo ci = *pCreateInfo;
    ci.enabledExtensionCount = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    VkResult r = createFn(&ci, pAllocator, pInstance);
    bool extrasOk = true;
    if (r != VK_SUCCESS) {
        Log("[instance] creation with DLSS extensions failed (%d); retrying without", (int)r);
        chain->u.pLayerInfo = nextLink;
        r = createFn(pCreateInfo, pAllocator, pInstance);
        extrasOk = false;
    }
    if (r != VK_SUCCESS) return r;
    if (!extrasOk) exts.assign(pCreateInfo->ppEnabledExtensionNames,
                               pCreateInfo->ppEnabledExtensionNames + pCreateInfo->enabledExtensionCount);

    InstanceData d;
    d.instance = *pInstance;
    d.GetInstanceProcAddr = gipa;
    d.DestroyInstance = (PFN_vkDestroyInstance)gipa(*pInstance, "vkDestroyInstance");
    d.GetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)gipa(*pInstance, "vkGetPhysicalDeviceProperties");
    d.GetPhysicalDeviceMemoryProperties =
        (PFN_vkGetPhysicalDeviceMemoryProperties)gipa(*pInstance, "vkGetPhysicalDeviceMemoryProperties");
    d.EnumerateDeviceExtensionProperties =
        (PFN_vkEnumerateDeviceExtensionProperties)gipa(*pInstance, "vkEnumerateDeviceExtensionProperties");
    const VkApplicationInfo* appInfo = pCreateInfo->pApplicationInfo;
    d.apiVersion = appInfo && appInfo->apiVersion ? appInfo->apiVersion : VK_API_VERSION_1_0;
    if (d.apiVersion >= VK_API_VERSION_1_1)
        d.GetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)gipa(*pInstance, "vkGetPhysicalDeviceFeatures2");
    if (!d.GetPhysicalDeviceFeatures2)  // Vulkan 1.0 instance: only via the extension, if the game enabled it
        for (const char* e : exts)
            if (!std::strcmp(e, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
                d.GetPhysicalDeviceFeatures2 =
                    (PFN_vkGetPhysicalDeviceFeatures2)gipa(*pInstance, "vkGetPhysicalDeviceFeatures2KHR");
    {
        std::lock_guard<std::mutex> lk(g_lock);
        g_instances[Key(*pInstance)] = d;
    }
    const VkApplicationInfo* app = pCreateInfo->pApplicationInfo;
    Log("[instance] created. app='%s' apiVersion=%u.%u",
        app && app->pApplicationName ? app->pApplicationName : "?", app ? VK_API_VERSION_MAJOR(app->apiVersion) : 0,
        app ? VK_API_VERSION_MINOR(app->apiVersion) : 0);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL tf3_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator) {
    InstanceData* d = GetInstance(Key(instance));
    if (!d) return;
    PFN_vkDestroyInstance fn = d->DestroyInstance;
    {
        std::lock_guard<std::mutex> lk(g_lock);
        g_instances.erase(Key(instance));
    }
    fn(instance, pAllocator);
}

// =============================================================================================
// Device
// =============================================================================================
static void InitNgx(DeviceData* d) {
    std::wstring dir = ModuleDir();
    std::wstring dllDir = g_cfg.dlssDev ? dir + TF3_SEP L"dev" : dir;  // dev build has debug overlays/hotkeys
    const wchar_t* paths[] = {dllDir.c_str()};
    Log("[dlss] loading %s DLSS runtime", g_cfg.dlssDev ? "developer" : "release");
    NVSDK_NGX_FeatureCommonInfo info{};
    info.PathListInfo.Path = paths;
    info.PathListInfo.Length = 1;
    info.LoggingInfo.LoggingCallback = &NGX_LogCallback;
    info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
    info.LoggingInfo.DisableOtherLoggingSinks = false;

    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        "a0f57b54-1daf-4934-90ae-c4035c19df04", NVSDK_NGX_ENGINE_TYPE_CUSTOM, TF3_DLSS_VERSION, dir.c_str(),
        d->instance->instance, d->physicalDevice, d->device, d->instance->GetInstanceProcAddr, d->GetDeviceProcAddr,
        &info);
    if (NVSDK_NGX_FAILED(r)) {
        char b[96];
        std::snprintf(b, sizeof b, "NVIDIA NGX could not start (0x%08x; is the driver up to date?)", (unsigned)r);
        g_dlss.why = b;
        Log("[dlss] NGX init failed: 0x%08x", (unsigned)r);
        return;
    }
    g_dlss.ngxInited = true;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&g_dlss.params))) {
        g_dlss.why = "NVIDIA NGX didn't report DLSS capabilities";
        Log("[dlss] GetCapabilityParameters failed");
        return;
    }
    int available = 0, needsDriver = 0;
    NVSDK_NGX_Parameter_GetI(g_dlss.params, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    NVSDK_NGX_Parameter_GetI(g_dlss.params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsDriver);
    Log("[dlss] NGX initialised. DLSS available=%d needsUpdatedDriver=%d", available, needsDriver);
    g_dlss.ngxReady = available != 0;
    if (!available)
        g_dlss.why = needsDriver ? "DLSS needs a newer NVIDIA driver" : "DLSS isn't supported on this GPU (needs RTX)";
}

// FSR 3.1's Vulkan shaders use these optional device features when the GPU has them; switch them
// on at device creation (the game doesn't). Game structures changed in place are put back after.
struct FeaturePatch {
    VkPhysicalDeviceFeatures core{};
    VkPhysicalDeviceShaderFloat16Int8Features f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice16BitStorageFeatures s16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceSubgroupSizeControlFeatures ssc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    std::vector<std::pair<VkBool32*, VkBool32>> undo;
    std::string summary;
    void Set(VkBool32* p) {
        if (*p) return;
        undo.push_back({p, *p});
        *p = VK_TRUE;
    }
    void Restore() {
        for (auto it = undo.rbegin(); it != undo.rend(); ++it) *it->first = it->second;
        undo.clear();
    }
};

VkBaseOutStructure* FindInChain(const void* pNext, VkStructureType t) {
    for (auto* p = (VkBaseOutStructure*)const_cast<void*>(pNext); p; p = p->pNext)
        if (p->sType == t) return p;
    return nullptr;
}

// Returns false if the features couldn't be set up (FSR is then not offered: its runtime picks shader
// variants from what the GPU supports, which must also be enabled on the device).
bool AddFsrFeatures(InstanceData* inst, VkPhysicalDevice pd, VkDeviceCreateInfo& ci, std::vector<const char*>& exts,
                    const std::set<std::string>& supported, FeaturePatch& fp) {
    if (!inst->GetPhysicalDeviceFeatures2 || !inst->GetPhysicalDeviceProperties) {
        Log("[device] can't query device features (the game's Vulkan instance is 1.0); FSR not available");
        return false;
    }
    VkPhysicalDeviceProperties props{};
    inst->GetPhysicalDeviceProperties(pd, &props);
    uint32_t api = std::min(inst->apiVersion, props.apiVersion);
    auto has = [&](const char* e) { return supported.count(e) != 0; };
    auto enabled = [&](const char* e) {
        return std::any_of(exts.begin(), exts.end(), [&](const char* x) { return !std::strcmp(x, e); });
    };
    bool f16Avail = api >= VK_API_VERSION_1_2 || has(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    bool sscAvail = api >= VK_API_VERSION_1_3 || has(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

    // what the GPU supports
    VkPhysicalDeviceShaderFloat16Int8Features qf16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice16BitStorageFeatures qs16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceSubgroupSizeControlFeatures qssc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    q.pNext = &qs16;
    void** tail = &qs16.pNext;
    if (f16Avail) { *tail = &qf16; tail = &qf16.pNext; }
    if (sscAvail) { *tail = &qssc; tail = &qssc.pNext; }
    inst->GetPhysicalDeviceFeatures2(pd, &q);

    // 1. image load/store without a format qualifier (core features)
    if (q.features.shaderStorageImageReadWithoutFormat || q.features.shaderStorageImageWriteWithoutFormat) {
        VkPhysicalDeviceFeatures* core = nullptr;
        if (ci.pEnabledFeatures) {
            fp.core = *ci.pEnabledFeatures;
            ci.pEnabledFeatures = &fp.core;
            core = &fp.core;
        } else if (auto* f2 = FindInChain(ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)) {
            core = &reinterpret_cast<VkPhysicalDeviceFeatures2*>(f2)->features;
        } else {
            ci.pEnabledFeatures = &fp.core;
            core = &fp.core;
        }
        if (q.features.shaderStorageImageReadWithoutFormat) fp.Set(&core->shaderStorageImageReadWithoutFormat);
        if (q.features.shaderStorageImageWriteWithoutFormat) fp.Set(&core->shaderStorageImageWriteWithoutFormat);
        fp.summary += " imageWithoutFormat";
    }
    // 2. 16-bit storage buffers (core in Vulkan 1.1)
    if (qs16.storageBuffer16BitAccess && api < VK_API_VERSION_1_1) {
        Log("[device] the game's Vulkan version is 1.0; FSR not available");
        return false;
    }
    if (qs16.storageBuffer16BitAccess) {
        if (auto* v11 = FindInChain(ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES))
            fp.Set(&reinterpret_cast<VkPhysicalDeviceVulkan11Features*>(v11)->storageBuffer16BitAccess);
        else if (auto* s = FindInChain(ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES))
            fp.Set(&reinterpret_cast<VkPhysicalDevice16BitStorageFeatures*>(s)->storageBuffer16BitAccess);
        else {
            fp.s16.storageBuffer16BitAccess = VK_TRUE;
            fp.s16.pNext = const_cast<void*>(ci.pNext);
            ci.pNext = &fp.s16;
        }
        fp.summary += " storageBuffer16Bit";
    }
    // 3. half-precision maths (core in Vulkan 1.2, else VK_KHR_shader_float16_int8)
    if (f16Avail && qf16.shaderFloat16) {
        if (auto* v12 = FindInChain(ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES))
            fp.Set(&reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(v12)->shaderFloat16);
        else if (auto* s = FindInChain(ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES))
            fp.Set(&reinterpret_cast<VkPhysicalDeviceShaderFloat16Int8Features*>(s)->shaderFloat16);
        else {
            fp.f16.shaderFloat16 = VK_TRUE;
            fp.f16.pNext = const_cast<void*>(ci.pNext);
            ci.pNext = &fp.f16;
        }
        if (api < VK_API_VERSION_1_2 && !enabled(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME))
            exts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        fp.summary += " shaderFloat16";
    }
    // 4. subgroup size control (core in Vulkan 1.3, else VK_EXT_subgroup_size_control); used on AMD
    if (sscAvail && qssc.subgroupSizeControl) {
        auto* v13 = api >= VK_API_VERSION_1_3 ? FindInChain(ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
                                              : nullptr;
        if (v13) {
            auto* f = reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(v13);
            fp.Set(&f->subgroupSizeControl);
            if (qssc.computeFullSubgroups) fp.Set(&f->computeFullSubgroups);
        } else if (auto* s = FindInChain(ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES)) {
            auto* f = reinterpret_cast<VkPhysicalDeviceSubgroupSizeControlFeatures*>(s);
            fp.Set(&f->subgroupSizeControl);
            if (qssc.computeFullSubgroups) fp.Set(&f->computeFullSubgroups);
        } else {
            fp.ssc.subgroupSizeControl = VK_TRUE;
            fp.ssc.computeFullSubgroups = qssc.computeFullSubgroups;
            fp.ssc.pNext = const_cast<void*>(ci.pNext);
            ci.pNext = &fp.ssc;
        }
        if (api < VK_API_VERSION_1_3 && !enabled(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
            exts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        fp.summary += " subgroupSizeControl";
    }
    return true;
}

std::string DriverVersion(const VkPhysicalDeviceProperties& p) {
    char b[48];
    uint32_t v = p.driverVersion;
    if (p.vendorID == 0x10DE)  // NVIDIA: 10.8.8.6 bits
        std::snprintf(b, sizeof b, "%u.%02u", (v >> 22) & 0x3ff, (v >> 14) & 0xff);
    else
        std::snprintf(b, sizeof b, "%u.%u.%u", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
    return b;
}

static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateDevice(VkPhysicalDevice physicalDevice,
                                                       const VkDeviceCreateInfo* pCreateInfo,
                                                       const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
    auto* chain = (VkLayerDeviceCreateInfo*)pCreateInfo->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerDeviceCreateInfo*)chain->pNext;
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    VkLayerDeviceLink* nextLink = chain->u.pLayerInfo;  // layers below advance it again; restored for a retry

    InstanceData* inst = GetInstance(Key(physicalDevice));
    auto createFn = (PFN_vkCreateDevice)gipa(inst ? inst->instance : VK_NULL_HANDLE, "vkCreateDevice");

    VkPhysicalDeviceProperties props{};
    if (inst && inst->GetPhysicalDeviceProperties) inst->GetPhysicalDeviceProperties(physicalDevice, &props);
    bool nvidia = props.vendorID == 0x10DE;
    UpscalerPref pref = g_cfg.upscaler;
    bool wantDlss = nvidia && pref != UpscalerPref::Fsr && pref != UpscalerPref::Off;
    bool wantFsr = pref != UpscalerPref::Dlss && pref != UpscalerPref::Off;
    Log("[device] GPU '%s' (vendor 0x%04x, driver %s, Vulkan %u.%u)", props.deviceName, props.vendorID,
        DriverVersion(props).c_str(), VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion));

    std::vector<const char*> exts(pCreateInfo->ppEnabledExtensionNames,
                                  pCreateInfo->ppEnabledExtensionNames + pCreateInfo->enabledExtensionCount);
    std::set<std::string> supported;
    if (inst && inst->EnumerateDeviceExtensionProperties) {
        uint32_t n = 0;
        inst->EnumerateDeviceExtensionProperties(physicalDevice, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> eprops(n);
        inst->EnumerateDeviceExtensionProperties(physicalDevice, nullptr, &n, eprops.data());
        for (auto& p : eprops) supported.insert(p.extensionName);
    }
    VkDeviceCreateInfo ci = *pCreateInfo;
    // Add the device extensions NGX needs (only those the GPU supports).
    bool addedBda = false, addedBdaExt = false;
    unsigned int ni = 0, nd = 0;
    const char** iexts = nullptr;
    const char** dexts = nullptr;
    if (wantDlss && NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_RequiredExtensions(&ni, &iexts, &nd, &dexts)))
        for (unsigned i = 0; i < nd; ++i) {
            if (std::any_of(exts.begin(), exts.end(), [&](const char* e) { return !std::strcmp(e, dexts[i]); }))
                continue;
            if (!supported.count(dexts[i])) {
                Log("[device] DLSS wants %s but the GPU doesn't offer it", dexts[i]);
                continue;
            }
            exts.push_back(dexts[i]);
            if (!std::strcmp(dexts[i], VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) addedBda = true;
            if (!std::strcmp(dexts[i], VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) addedBdaExt = true;
            Log("[device] + %s (for DLSS)", dexts[i]);
        }
    // buffer_device_address needs its feature switched on as well
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    if (addedBda) {
        bool present = false;
        for (auto* p = (const VkBaseInStructure*)pCreateInfo->pNext; p; p = p->pNext)
            if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES ||
                p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                present = true;
        if (!present) {
            bda.bufferDeviceAddress = VK_TRUE;
            bda.pNext = const_cast<void*>(ci.pNext);
            ci.pNext = &bda;
        }
    }
    // NGX asks for the EXT version on a Vulkan 1.1 game; it has its own feature structure (another sType),
    // and NGX calls vkGetBufferDeviceAddressEXT, which needs the feature on (validation error otherwise).
    VkPhysicalDeviceBufferDeviceAddressFeaturesEXT bdaExt{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT};
    if (addedBdaExt && !addedBda &&
        !FindInChain(pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT) &&
        !FindInChain(pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES) &&
        !FindInChain(pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)) {
        bdaExt.bufferDeviceAddress = VK_TRUE;
        bdaExt.pNext = const_cast<void*>(ci.pNext);
        ci.pNext = &bdaExt;
        Log("[device] bufferDeviceAddress feature switched on (for DLSS)");
    }
#ifdef __APPLE__
    // MetalFX needs MoltenVK's Metal objects (device, textures, shared events)
    bool addedMetalObjects = false;
    if (supported.count(VK_EXT_METAL_OBJECTS_EXTENSION_NAME) &&
        !std::any_of(exts.begin(), exts.end(), [](const char* e) { return !std::strcmp(e, VK_EXT_METAL_OBJECTS_EXTENSION_NAME); })) {
        exts.push_back(VK_EXT_METAL_OBJECTS_EXTENSION_NAME);
        addedMetalObjects = true;
        Log("[device] + %s (for MetalFX)", VK_EXT_METAL_OBJECTS_EXTENSION_NAME);
    } else if (supported.count(VK_EXT_METAL_OBJECTS_EXTENSION_NAME)) {
        addedMetalObjects = true;  // the game enabled it itself
    }
#endif
    FeaturePatch fp;
    size_t extsBeforeFsr = exts.size();
    bool fsrFeatures = wantFsr && inst && AddFsrFeatures(inst, physicalDevice, ci, exts, supported, fp);
    for (size_t i = extsBeforeFsr; i < exts.size(); ++i) Log("[device] + %s (for FSR)", exts[i]);
    if (!fp.summary.empty()) Log("[device] features for FSR:%s", fp.summary.c_str());
    ci.enabledExtensionCount = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();

    VkResult r = createFn(physicalDevice, &ci, pAllocator, pDevice);
    fp.Restore();
    bool extrasOk = true;
    if (r != VK_SUCCESS) {
        Log("[device] creation with the upscalers' extensions/features failed (%d); retrying without", (int)r);
        chain->u.pLayerInfo = nextLink;
        r = createFn(physicalDevice, pCreateInfo, pAllocator, pDevice);
        extrasOk = false;
    }
    if (r != VK_SUCCESS) return r;

    DeviceData* d;
    {
        std::lock_guard<std::mutex> lk(g_lock);
        d = &g_devices[Key(*pDevice)];
        g_deviceGen.fetch_add(1, std::memory_order_release);
    }
    d->device = *pDevice;
    d->physicalDevice = physicalDevice;
    d->instance = inst;
    d->GetDeviceProcAddr = gdpa;
#define X(name) d->name = (PFN_vk##name)gdpa(*pDevice, "vk" #name);
    DEVICE_HOOKS(X)
    DEVICE_CALLS(X)
#undef X
    {
        TLock lk(g_t);
        if (!g_dlss.dev) {
            g_dlss.dev = d;
            const char* off = std::getenv("TF3_DLSS_OFF");
            g_dlss.enabled = g_cfg.enabled != 0 && !(off && *off == '1');
            if (!nvidia) g_dlss.why = "not an NVIDIA GPU";
            else if (pref == UpscalerPref::Fsr || pref == UpscalerPref::Off) g_dlss.why = "not used (upscaler setting)";
            else if (!extrasOk || !inst) g_dlss.why = "the Vulkan extensions DLSS needs couldn't be enabled";
            else InitNgx(d);
#ifdef __APPLE__
            g_mfx.deviceExtEnabled = addedMetalObjects && extrasOk;
#endif
            // FSR only on a device created with its features (see AddFsrFeatures)
            g_fsr.deviceReady = fsrFeatures && extrasOk;
            if (!wantFsr) g_fsr.why = "not set up: the game was started with upscaler=dlss or off";
            else if (!g_fsr.deviceReady) g_fsr.why = "the device features FSR needs couldn't be enabled";
            Backend b = ChooseBackend();
            Log("[upscale] upscaler setting '%s' -> %s; upscaling is %s (Ctrl+Shift+D toggles with debug_keys=1)", PrefName(pref),
                BackendName(b), g_dlss.enabled ? "ON" : "OFF");
            if (b == Backend::None) WriteStatus(NoUpscalerReason());
        }
    }
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL tf3_DestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator) {
    DeviceData* d = GetDevice(Key(device));
    if (!d) return;
    PFN_vkDestroyDevice fn = d->DestroyDevice;
    {
        TLock lk(g_t);
        if (g_dlss.dev == d) {
            d->DeviceWaitIdle(device);
            DestroyFsr(d);
#ifdef __APPLE__
            FgDestroy(d);
            MfxDestroy(d);
#endif
            DestroyGpuPerf(d);
            if (g_dlss.handle) NVSDK_NGX_VULKAN_ReleaseFeature(g_dlss.handle);
            if (g_dlss.params) NVSDK_NGX_VULKAN_DestroyParameters(g_dlss.params);
            if (g_dlss.ngxInited) NVSDK_NGX_VULKAN_Shutdown1(device);
            g_dlss = Dlss{};
        }
    }
    Log("[device] destroyed after %llu frames (last upscaled frame %lld, FSR dispatches %llu)",
        (unsigned long long)g_frame, g_lastDlssFrame == UINT64_MAX ? -1LL : (long long)g_lastDlssFrame,
        (unsigned long long)g_fsr.dispatches);
    {
        std::lock_guard<std::mutex> lk(g_lock);
        g_devices.erase(Key(device));
        g_deviceGen.fetch_add(1, std::memory_order_release);
    }
    fn(device, pAllocator);
}

// =============================================================================================
// Shaders and pipelines
// =============================================================================================
// Which vertex inputs (Location-decorated Input variables) the value written to gl_Position depends
// on. A conservative backward data-flow walk over the SPIR-V: loads, stores, access chains (with
// their indices), function calls (arguments, parameters, return values) and all value operands.
// Used to find the position attribute when the shader doesn't name it attrPosition, or to check the
// name. Returns false if the shader doesn't write gl_Position.
bool TracePositionInputs(const uint32_t* w, size_t words, const std::unordered_map<uint32_t, std::string>& names,
                         const std::unordered_map<uint32_t, int>& locs, std::vector<std::pair<int, std::string>>& out) {
    const uint32_t kPos = 0xffffffffu;  // pseudo node: gl_Position
    std::unordered_map<uint32_t, uint32_t> varStorage;          // OpVariable -> storage class
    std::unordered_map<uint32_t, uint32_t> varType;             // OpVariable -> its pointer type
    std::unordered_map<uint32_t, uint32_t> ptrPointee;          // OpTypePointer -> pointee type
    std::unordered_map<uint32_t, uint32_t> constVal;            // 32-bit OpConstant values
    std::unordered_map<uint32_t, uint32_t> root;                // pointer -> the variable it points into
    std::unordered_map<uint32_t, uint32_t> firstIndex;          // access chain -> its first index id
    std::unordered_map<uint32_t, std::vector<uint32_t>> acIdx;  // access chain -> all index ids on the way
    std::unordered_map<uint32_t, std::vector<uint32_t>> deps;   // node -> nodes its value comes from
    std::unordered_map<uint32_t, std::vector<uint32_t>> params, rets;
    struct Call { uint32_t result, func; std::vector<uint32_t> args; };
    std::vector<Call> calls;
    uint32_t posVar = 0;                   // variable decorated BuiltIn Position (old style)
    uint32_t perVertexType = 0, posMember = 0;  // gl_PerVertex block and its Position member
    uint32_t curFunc = 0;
    bool inBody = false;
    auto rootOf = [&](uint32_t id) {
        auto it = root.find(id);
        return it == root.end() ? id : it->second;
    };
    static const std::unordered_set<uint32_t> kNoResult = {0,   8,   62,  63,  64,  99,  218, 219, 220, 221, 224,
                                                           225, 228, 246, 247, 249, 250, 251, 252, 253, 254, 255,
                                                           256, 257, 317, 4416, 5380};
    for (size_t i = 5; i < words;) {
        uint32_t op = w[i] & 0xffffu, n = w[i] >> 16;
        if (!n || i + n > words) return false;
        const uint32_t* a = w + i;
        switch (op) {
            case 71:  // OpDecorate id BuiltIn Position
                if (n >= 4 && a[2] == 11 && a[3] == 0) posVar = a[1];
                break;
            case 72:  // OpMemberDecorate struct member BuiltIn Position
                if (n >= 5 && a[3] == 11 && a[4] == 0) { perVertexType = a[1]; posMember = a[2]; }
                break;
            case 32:  // OpTypePointer id storage type
                if (n >= 4) ptrPointee[a[1]] = a[3];
                break;
            case 43:  // OpConstant type id value
                if (n == 4) constVal[a[2]] = a[3];
                break;
            case 54:  // OpFunction
                inBody = true;
                curFunc = a[2];
                break;
            case 56:  // OpFunctionEnd
                inBody = false;
                break;
            default: break;
        }
        if (op == 59 && n >= 4) {  // OpVariable type id storage
            varStorage[a[2]] = a[3];
            varType[a[2]] = a[1];
            root[a[2]] = a[2];
        }
        if (!inBody || op == 54 || op == 56 || op == 59) { i += n; continue; }
        switch (op) {
            case 55: params[curFunc].push_back(a[2]); root[a[2]] = a[2]; break;  // OpFunctionParameter
            case 65: case 66: case 67: case 70: {  // access chains: type id base indices...
                uint32_t base = a[3];
                root[a[2]] = rootOf(base);
                std::vector<uint32_t> idx = acIdx.count(base) ? acIdx[base] : std::vector<uint32_t>{};
                uint32_t firstAt = (op == 67 || op == 70) ? 5 : 4;  // Ptr chains have an element operand first
                for (uint32_t k = 4; k < n; ++k) idx.push_back(a[k]);
                acIdx[a[2]] = idx;
                if (firstIndex.count(base)) firstIndex[a[2]] = firstIndex[base];
                else if (n > firstAt) firstIndex[a[2]] = a[firstAt];
                break;
            }
            case 61: {  // OpLoad type id ptr
                auto& dv = deps[a[2]];
                dv.push_back(rootOf(a[3]));
                if (acIdx.count(a[3])) dv.insert(dv.end(), acIdx[a[3]].begin(), acIdx[a[3]].end());
                break;
            }
            case 62: {  // OpStore ptr obj
                uint32_t r = rootOf(a[1]), node = r;
                if (r == posVar) node = kPos;
                else if (perVertexType && varStorage.count(r) && varStorage[r] == 3 &&
                         ptrPointee[varType[r]] == perVertexType) {  // the gl_PerVertex output block
                    auto fi = firstIndex.find(a[1]);
                    if (fi != firstIndex.end()) {
                        auto c = constVal.find(fi->second);
                        if (c != constVal.end() && c->second == posMember) node = kPos;
                    } else if (a[1] == r) node = kPos;  // whole block stored
                }
                auto& dv = deps[node];
                dv.push_back(a[2]);
                if (acIdx.count(a[1])) dv.insert(dv.end(), acIdx[a[1]].begin(), acIdx[a[1]].end());
                break;
            }
            case 63: deps[rootOf(a[1])].push_back(rootOf(a[2])); break;  // OpCopyMemory target source
            case 57: {  // OpFunctionCall type id func args...
                Call c{a[2], a[3], {}};
                for (uint32_t k = 4; k < n; ++k) c.args.push_back(a[k]);
                calls.push_back(c);
                break;
            }
            case 254: rets[curFunc].push_back(a[1]); break;  // OpReturnValue
            case 81: deps[a[2]].push_back(a[3]); break;     // OpCompositeExtract (rest are literals)
            case 82: deps[a[2]].push_back(a[3]); deps[a[2]].push_back(a[4]); break;  // OpCompositeInsert
            case 79: deps[a[2]].push_back(a[3]); deps[a[2]].push_back(a[4]); break;  // OpVectorShuffle
            case 12: for (uint32_t k = 5; k < n; ++k) deps[a[2]].push_back(a[k]); break;  // OpExtInst
            case 245: for (uint32_t k = 3; k + 1 < n; k += 2) deps[a[2]].push_back(a[k]); break;  // OpPhi
            case 248: break;  // OpLabel
            default:
                if (!kNoResult.count(op) && n >= 3)  // type id operands...: every operand may carry data
                    for (uint32_t k = 3; k < n; ++k) deps[a[2]].push_back(a[k]);
                break;
        }
        i += n;
    }
    for (const auto& c : calls) {
        auto& dv = deps[c.result];
        auto r = rets.find(c.func);
        if (r != rets.end()) dv.insert(dv.end(), r->second.begin(), r->second.end());
        auto p = params.find(c.func);
        if (p == params.end()) continue;
        for (size_t k = 0; k < c.args.size() && k < p->second.size(); ++k) {
            uint32_t arg = rootOf(c.args[k]), par = p->second[k];
            deps[par].push_back(arg);
            deps[arg].push_back(par);  // out/inout parameters write back into the argument
            if (acIdx.count(c.args[k])) deps[par].insert(deps[par].end(), acIdx[c.args[k]].begin(), acIdx[c.args[k]].end());
        }
    }
    if (!deps.count(kPos)) return false;
    std::unordered_set<uint32_t> seen{kPos};
    std::vector<uint32_t> todo{kPos};
    while (!todo.empty()) {
        uint32_t id = todo.back();
        todo.pop_back();
        auto vs = varStorage.find(id);
        if (vs != varStorage.end() && vs->second == 1) {  // Input variable
            auto l = locs.find(id);
            if (l != locs.end()) {
                auto nm = names.find(id);
                out.push_back({l->second, nm == names.end() ? std::string() : nm->second});
            }
        }
        auto d = deps.find(id);
        if (d == deps.end()) continue;
        for (uint32_t x : d->second)
            if (seen.insert(x).second) todo.push_back(x);
    }
    std::sort(out.begin(), out.end());
    return true;
}

// Finds the input locations of instAttrModel / attrPosition in a vertex shader (debug names are kept).
VsInfo ParseVertexShader(const uint32_t* w, size_t words) {
    VsInfo vi;
    if (words < 5 || w[0] != 0x07230203u) return vi;
    std::unordered_map<uint32_t, std::string> names;
    std::unordered_map<uint32_t, int> locs;
    bool vertex = false;
    for (size_t i = 5; i < words;) {
        uint32_t op = w[i] & 0xffffu, n = w[i] >> 16;
        if (!n || i + n > words) break;
        if (op == 15 && n >= 2 && w[i + 1] == 0) vertex = true;  // OpEntryPoint Vertex
        else if (op == 5 && n >= 3) {                              // OpName
            const char* str = reinterpret_cast<const char*>(&w[i + 2]);
            names[w[i + 1]] = std::string(str, strnlen(str, (n - 2) * 4));
        } else if (op == 71 && n >= 4 && w[i + 2] == 30)  // OpDecorate Location
            locs[w[i + 1]] = (int)w[i + 3];
        i += n;
    }
    if (!vertex) return vi;
    {   // a short tag: the "interesting" input/output names, to tell shaders apart in the draw log
        std::string tag;
        for (auto& [id, nm] : names) {
            if (locs.find(id) == locs.end()) continue;
            if (nm.rfind("attr", 0) == 0 || nm.rfind("instAttr", 0) == 0) {
                if (tag.size() < 120) tag += (tag.empty() ? "" : ",") + nm;
            }
        }
        vi.tag = tag;
    }
    static int namesLogged = 0;
    bool hasModel = false;
    for (auto& [id, nm] : names) if (nm == "instAttrModel") hasModel = true;
    if (hasModel && namesLogged < 2) {
        ++namesLogged;
        std::string list;
        for (auto& [id, nm] : names) {
            auto l = locs.find(id);
            if (l != locs.end()) list += " " + nm + "@" + std::to_string(l->second);
        }
        Log("[obj] vertex inputs/outputs with locations:%s", list.c_str());
    }
    for (auto& [id, nm] : names) {
        if (nm == "u_skinned" || nm == "Skinned") vi.skinned = true;
        auto l = locs.find(id);
        if (l == locs.end()) continue;
        if (nm == "instAttrModel") vi.mdlLoc = l->second;
        else if (nm == "attrPosition") vi.posLoc = l->second;
    }
    if (vi.mdlLoc >= 0 || vi.skinned) vi.traced = TracePositionInputs(w, words, names, locs, vi.posInputs);
    return vi;
}


// Names of a shader's textures, uniform blocks and push constants (from the SPIR-V debug names), to
// recognise fragment shaders in the draw log, e.g. "colorTex,waterDepthTex,u_water,...". Empty if
// the module has no names.
std::string ShaderResourceNames(const uint32_t* w, size_t words) {
    if (words < 5 || w[0] != 0x07230203) return {};
    std::unordered_map<uint32_t, std::string> names;
    std::vector<uint32_t> vars;
    for (size_t i = 5; i < words;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (n == 0 || i + n > words) break;
        if (op == 5 && n >= 3)  // OpName id "name"
            names[w[i + 1]] = std::string(reinterpret_cast<const char*>(&w[i + 2]),
                                          strnlen(reinterpret_cast<const char*>(&w[i + 2]), (n - 2) * 4));
        else if (op == 59 && n >= 4) {  // OpVariable type id storage
            uint32_t sc = w[i + 3];
            if (sc == 0 || sc == 2 || sc == 9 || sc == 12) vars.push_back(w[i + 2]);  // uniform/push/storage
        }
        i += n;
    }
    std::string out;
    int count = 0;
    for (uint32_t id : vars) {
        auto it = names.find(id);
        if (it == names.end() || it->second.empty()) continue;
        if (count++ == 8) { out += ",..."; break; }
        out += (out.empty() ? "" : ",") + it->second;
    }
    return out;
}

static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pInfo,
                                                             const VkAllocationCallbacks* pAllocator,
                                                             VkShaderModule* pModule) {
    VkResult r = GetDevice(Key(device))->CreateShaderModule(device, pInfo, pAllocator, pModule);
    if (r == VK_SUCCESS) {
        PerfScope ps(kPerfCreate);
        VsInfo vi = ParseVertexShader(pInfo->pCode, pInfo->codeSize / 4);
        if (!vi.tag.empty()) {
            TLock lk(g_t);
            g_moduleTag[*pModule] = vi.tag;
        }
        std::string res = ShaderResourceNames(pInfo->pCode, pInfo->codeSize / 4);
        if (!res.empty()) {
            TLock lk(g_t);
            g_moduleRes[*pModule] = res;
        }
        if (vi.mdlLoc >= 0 || vi.skinned) {
            TLock lk(g_t);
            g_vsInfo[*pModule] = vi;
            static int logged = 0;
            if (logged++ < 4)
                Log("[probe] vertex shader %s: instAttrModel at location %d, attrPosition at location %d%s",
                    H(*pModule).c_str(), vi.mdlLoc, vi.posLoc, vi.skinned ? ", skinned" : "");
        }
#ifdef __APPLE__
        if (uint8_t rc = ReactModuleClass(pInfo->pCode, pInfo->codeSize / 4)) {
            TLock lk(g_t);
            g_moduleReact[*pModule] = rc;
        }
#endif
        Kind k = Classify(pInfo->pCode, pInfo->codeSize);
        if (k != Kind::Other) {
            TLock lk(g_t);
            g_moduleKind[*pModule] = k;
            Log("[shader] module %s = %s", H(*pModule).c_str(), KindName(k));
        }
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL tf3_DestroyShaderModule(VkDevice device, VkShaderModule module,
                                                          const VkAllocationCallbacks* pAllocator) {
    {
        TLock lk(g_t);
        g_moduleKind.erase(module);
    }
    GetDevice(Key(device))->DestroyShaderModule(device, module, pAllocator);
}

static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateGraphicsPipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                                  const VkGraphicsPipelineCreateInfo* pInfos,
                                                                  const VkAllocationCallbacks* pAllocator,
                                                                  VkPipeline* pPipelines) {
    // Viewport jitter needs the viewport to be dynamic state; make it so where the game baked it in.
    std::optional<PerfScope> ps(std::in_place, kPerfCreate);
    std::vector<VkGraphicsPipelineCreateInfo> infos(pInfos, pInfos + count);
    std::vector<VkPipelineDynamicStateCreateInfo> dyn(count);
    std::vector<std::vector<VkDynamicState>> dynList(count);
    std::vector<PipeViewport> pvp(count);
    std::vector<VkPipelineDepthStencilStateCreateInfo> dss(count);
    for (uint32_t i = 0; i < count; ++i) {
        auto& ci = infos[i];
        // The hover highlight (fill + outline lines) is plain coloured geometry that writes stencil and
        // also depth, stamping its shape into the depth buffer the motion vectors are computed from:
        // the ground around a highlighted object then gets the wrong motion and DLSS smears it.
        // Such overlays don't need depth writes, so turn them off.
        if (g_cfg.fixHighlightDepth && ci.pDepthStencilState && ci.pDepthStencilState->depthWriteEnable &&
            ci.pDepthStencilState->stencilTestEnable &&
            (ci.pDepthStencilState->front.passOp != VK_STENCIL_OP_KEEP ||
             ci.pDepthStencilState->back.passOp != VK_STENCIL_OP_KEEP)) {
            std::string tag;
            {
                TLock lk(g_t);
                for (uint32_t k = 0; k < ci.stageCount; ++k)
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_VERTEX_BIT) {
                        auto t = g_moduleTag.find(ci.pStages[k].module);
                        if (t != g_moduleTag.end()) tag = t->second;
                    }
            }
            if (tag.find("attrColor") != std::string::npos && tag.find("instAttrModel") == std::string::npos) {
                dss[i] = *ci.pDepthStencilState;
                dss[i].depthWriteEnable = VK_FALSE;
                ci.pDepthStencilState = &dss[i];
                Log("[highlight] overlay pipeline (vs %s): depth writes turned off", tag.c_str());
            }
        }
        bool dynVp = false;
        if (ci.pDynamicState)
            for (uint32_t k = 0; k < ci.pDynamicState->dynamicStateCount; ++k)
                if (ci.pDynamicState->pDynamicStates[k] == VK_DYNAMIC_STATE_VIEWPORT) dynVp = true;
        pvp[i].dynamic = dynVp;
        bool discard = ci.pRasterizationState && ci.pRasterizationState->rasterizerDiscardEnable;
        if (!dynVp && !discard && ci.pViewportState && ci.pViewportState->viewportCount == 1 &&
            ci.pViewportState->pViewports) {
            if (ci.pDynamicState) {
                dyn[i] = *ci.pDynamicState;
                dynList[i].assign(ci.pDynamicState->pDynamicStates,
                                  ci.pDynamicState->pDynamicStates + ci.pDynamicState->dynamicStateCount);
            } else
                dyn[i] = VkPipelineDynamicStateCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynList[i].push_back(VK_DYNAMIC_STATE_VIEWPORT);
            dyn[i].dynamicStateCount = (uint32_t)dynList[i].size();
            dyn[i].pDynamicStates = dynList[i].data();
            ci.pDynamicState = &dyn[i];
            pvp[i].dynamic = pvp[i].patched = true;
            pvp[i].staticVp = ci.pViewportState->pViewports[0];
        }
    }
    ps.reset();  // the driver's time isn't ours
    DeviceData* dev = GetDevice(Key(device));
    VkResult r = dev->CreateGraphicsPipelines(device, cache, count, infos.data(), pAllocator, pPipelines);
    if (r != VK_SUCCESS) return r;
    // Reactive mask (MetalFX): a copy of each pipeline that draws things without motion vectors.
    std::vector<uint8_t> reactCls(count, kRcNoneValue);
    std::vector<VkPipeline> reactPipe(count, VK_NULL_HANDLE);
#ifdef __APPLE__
    {
        TLock lk(g_t);
        for (uint32_t i = 0; i < count; ++i)
            if (pPipelines[i] && g_cfg.reactive) reactCls[i] = ReactClassify(infos[i]);
    }
    for (uint32_t i = 0; i < count; ++i)
        if (reactCls[i] && ReactStrength(reactCls[i]) > 0.f) reactPipe[i] = ReactClone(dev, infos[i], reactCls[i]);
#endif
    ps.emplace(kPerfCreate);
    TLock lk(g_t);
    for (uint32_t i = 0; i < count; ++i)
        if (pPipelines[i]) g_pipeVp[pPipelines[i]] = pvp[i];
    for (uint32_t i = 0; i < count; ++i) {
        if (!pPipelines[i]) continue;
        const auto& ci = pInfos[i];
        PipeState st;
        if (ci.pColorBlendState)
            for (uint32_t a = 0; a < ci.pColorBlendState->attachmentCount; ++a) {
                st.colorMask |= ci.pColorBlendState->pAttachments[a].colorWriteMask;
                st.blend |= ci.pColorBlendState->pAttachments[a].blendEnable != 0;
                ++st.attachments;
            }
        if (ci.pDepthStencilState) {
            st.depthTest = ci.pDepthStencilState->depthTestEnable;
            st.depthWrite = ci.pDepthStencilState->depthWriteEnable;
            st.depthOp = ci.pDepthStencilState->depthCompareOp;
            st.stencil = ci.pDepthStencilState->stencilTestEnable;
            st.stencilPass = ci.pDepthStencilState->front.passOp;
        }
        if (ci.pRasterizationState) st.cull = ci.pRasterizationState->cullMode;
        if (ci.pInputAssemblyState) st.topology = ci.pInputAssemblyState->topology;
        for (uint32_t k = 0; k < ci.stageCount; ++k)
            if (ci.pStages[k].stage == VK_SHADER_STAGE_VERTEX_BIT) {
                auto t = g_moduleTag.find(ci.pStages[k].module);
                if (t != g_moduleTag.end()) st.vsNames = t->second;
            } else if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
                auto t = g_moduleRes.find(ci.pStages[k].module);
                if (t != g_moduleRes.end()) st.fsNames = t->second;
            }
        g_pipeState[pPipelines[i]] = st;
    }
    static int vtxLogged = 0;
    for (uint32_t i = 0; i < count; ++i)
        for (uint32_t s = 0; s < pInfos[i].stageCount; ++s) {
            const auto& st = pInfos[i].pStages[s];
            if (st.stage == VK_SHADER_STAGE_VERTEX_BIT) {
                auto vi = g_vsInfo.find(st.module);
                const VkPipelineVertexInputStateCreateInfo* vis = pInfos[i].pVertexInputState;
                if (vi == g_vsInfo.end() || !vis) continue;
                PipeVtx pv;
                pv.skinned = vi->second.skinned;
                if (pInfos[i].pInputAssemblyState) {
                    pv.topology = pInfos[i].pInputAssemblyState->topology;
                    pv.primitiveRestart = pInfos[i].pInputAssemblyState->primitiveRestartEnable;
                }
                // Position: an input the shader's gl_Position actually depends on (SPIR-V trace) that is a
                // per-vertex float3/float4 attribute outside the model matrix. attrPosition if it's one of
                // them, else the only one named *pos*, else the only one. Ambiguous -> no object motion
                // for this pipeline (a wrong guess draws stray triangles into the motion vectors).
                int posLoc = vi->second.posLoc;
                auto perVertexFloat = [&](int loc) {
                    if (vi->second.mdlLoc >= 0 && loc >= vi->second.mdlLoc && loc <= vi->second.mdlLoc + 3) return false;
                    for (uint32_t a = 0; a < vis->vertexAttributeDescriptionCount; ++a) {
                        const auto& ad = vis->pVertexAttributeDescriptions[a];
                        if ((int)ad.location != loc || FormatComps(ad.format) < 3) continue;
                        for (uint32_t b = 0; b < vis->vertexBindingDescriptionCount; ++b)
                            if (vis->pVertexBindingDescriptions[b].binding == ad.binding)
                                return vis->pVertexBindingDescriptions[b].inputRate == VK_VERTEX_INPUT_RATE_VERTEX;
                    }
                    return false;
                };
                if (vi->second.traced) {
                    std::vector<std::pair<int, std::string>> cands;
                    for (const auto& c : vi->second.posInputs)
                        if (perVertexFloat(c.first)) cands.push_back(c);
                    int byName = -1, posNamed = -1, posNamedCount = 0;
                    for (const auto& c : cands) {
                        if (c.second == "attrPosition") byName = c.first;
                        std::string lower = c.second;
                        for (auto& ch : lower) ch = (char)std::tolower((unsigned char)ch);
                        if (lower.find("pos") != std::string::npos) { posNamed = c.first; ++posNamedCount; }
                    }
                    int chosen = byName >= 0 ? byName : posNamedCount == 1 ? posNamed
                                                      : cands.size() == 1 ? cands[0].first : -1;
                    static int traceLogged = 0;
                    if ((chosen != posLoc || chosen < 0) && traceLogged < 12) {
                        ++traceLogged;
                        std::string list;
                        for (const auto& c : vi->second.posInputs)
                            list += " " + (c.second.empty() ? std::string("?") : c.second) + "@" + std::to_string(c.first);
                        Log("[obj] pipeline %s: gl_Position reads%s; position %s (by name: %d)", H(pPipelines[i]).c_str(),
                            list.empty() ? " nothing from vertex inputs" : list.c_str(),
                            chosen >= 0 ? ("at location " + std::to_string(chosen)).c_str() : "unclear, no object motion",
                            posLoc);
                    }
                    posLoc = chosen;
                } else if (posLoc < 0) {
                    for (uint32_t a = 0; a < vis->vertexAttributeDescriptionCount; ++a) {
                        const auto& ad = vis->pVertexAttributeDescriptions[a];
                        int col = (int)ad.location - vi->second.mdlLoc;
                        if (vi->second.mdlLoc >= 0 && col >= 0 && col <= 3) continue;
                        if (!FormatComps(ad.format)) continue;
                        bool perVertex = false;
                        for (uint32_t b = 0; b < vis->vertexBindingDescriptionCount; ++b)
                            if (vis->pVertexBindingDescriptions[b].binding == ad.binding)
                                perVertex = vis->pVertexBindingDescriptions[b].inputRate == VK_VERTEX_INPUT_RATE_VERTEX;
                        if (perVertex && (posLoc < 0 || (int)ad.location < posLoc)) posLoc = (int)ad.location;
                    }
                }
                for (uint32_t a = 0; a < vis->vertexAttributeDescriptionCount; ++a) {
                    const auto& ad = vis->pVertexAttributeDescriptions[a];
                    if ((int)ad.location == posLoc) {
                        pv.posBinding = ad.binding;
                        pv.posFormat = ad.format;
                        pv.posOffset = ad.offset;
                    }
                    int col = (int)ad.location - vi->second.mdlLoc;
                    if (vi->second.mdlLoc < 0 || col < 0 || col > 3) continue;
                    if (col == 0) {
                        pv.hasMdl = true;
                        pv.mdlBinding = ad.binding;
                        pv.fmt = ad.format;
                        pv.comps = FormatComps(ad.format);
                    }
                    pv.colOffset[col] = ad.offset;
                    pv.cols = std::max(pv.cols, (uint32_t)col + 1);
                }
                for (uint32_t b = 0; b < vis->vertexBindingDescriptionCount; ++b) {
                    const auto& bd = vis->pVertexBindingDescriptions[b];
                    if (bd.binding == pv.posBinding) {
                        pv.posStride = bd.stride;
                        pv.posInstanceRate = bd.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE;
                    }
                    if (pv.hasMdl && bd.binding == pv.mdlBinding) {
                        pv.stride = bd.stride;
                        pv.instanceRate = bd.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE;
                    }
                }
                g_pipeVtx[pPipelines[i]] = pv;
                if (vtxLogged < 8 && (pv.hasMdl || pv.skinned)) {
                    ++vtxLogged;
                    Log("[probe] pipeline %s: model matrix %s (binding %u, %s, stride %u, %u cols, format %d, "
                        "col offsets %u/%u/%u/%u), position binding %d%s",
                        H(pPipelines[i]).c_str(), pv.hasMdl ? "found" : "none", pv.mdlBinding,
                        pv.instanceRate ? "per instance" : "per vertex", pv.stride, pv.cols, (int)pv.fmt,
                        pv.colOffset[0], pv.colOffset[1], pv.colOffset[2], pv.colOffset[3],
                        pv.posBinding == UINT32_MAX ? -1 : (int)pv.posBinding, pv.skinned ? ", skinned" : "");
                }
                continue;
            }
            if (st.stage != VK_SHADER_STAGE_FRAGMENT_BIT) continue;
            auto it = g_moduleKind.find(st.module);
            if (it == g_moduleKind.end()) continue;
            g_pipelineKind[pPipelines[i]] = it->second;
            Log("[pipeline] %s uses %s", H(pPipelines[i]).c_str(), KindName(it->second));
        }
    std::unique_lock<std::shared_mutex> pl(g_pipeInfoLock);
    for (uint32_t i = 0; i < count; ++i) {
        if (!pPipelines[i]) continue;
        auto info = std::make_unique<PipeInfo>();
        auto k = g_pipelineKind.find(pPipelines[i]);
        if (k != g_pipelineKind.end()) info->kind = k->second;
        info->vp = pvp[i];
        auto v = g_pipeVtx.find(pPipelines[i]);
        if (v != g_pipeVtx.end()) info->hasVtx = true, info->vtx = v->second;
        auto st = g_pipeState.find(pPipelines[i]);
        info->screenPass = st != g_pipeState.end() && !st->second.depthTest;
        info->react = reactPipe[i] ? reactCls[i] : kRcNoneValue;
        info->reactPipe = reactPipe[i];
        g_pipeInfo[pPipelines[i]] = std::move(info);
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL tf3_DestroyPipeline(VkDevice device, VkPipeline pipeline,
                                                     const VkAllocationCallbacks* pAllocator) {
    {
        TLock lk(g_t);
        g_pipelineKind.erase(pipeline);
        g_pipeVtx.erase(pipeline);
        g_pipeVp.erase(pipeline);
        g_pipeState.erase(pipeline);
        std::unique_lock<std::shared_mutex> pl(g_pipeInfoLock);
        auto it = g_pipeInfo.find(pipeline);
        if (it != g_pipeInfo.end() && it->second->reactPipe)  // its reactive-mask copy (never in use without it)
            GetDevice(Key(device))->DestroyPipeline(device, it->second->reactPipe, nullptr);
        g_pipeInfo.erase(pipeline);
        g_pipeInfoGen.fetch_add(1, std::memory_order_release);
    }
    GetDevice(Key(device))->DestroyPipeline(device, pipeline, pAllocator);
}

// =============================================================================================
// Texture LOD bias
// =============================================================================================
// DLSS renders at a lower resolution, so the GPU picks blurrier mip levels than the 4K output
// needs. "auto" uses log2(render width / display width) (-1 at 50%): NVIDIA's recommendation has an
// extra -1, but in TF3 that made cut-out textures (fences) shimmer under jitter. Applied to the game's
// material samplers when they are created: mipmapped, filtering, repeating texture coordinates
// (UI, shadow and post-processing samplers clamp or don't use mips, and are left alone).
static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateSampler(VkDevice device, const VkSamplerCreateInfo* pInfo,
                                                       const VkAllocationCallbacks* pAllocator, VkSampler* pSampler) {
    VkSamplerCreateInfo info = *pInfo;
    float bias = 0.f;
    uint32_t lodOutW = 0;
    {
        TLock lk(g_t);
        // Only with an upscaler: the game's own FSR 1 doesn't recover the detail, it would just shimmer.
        if (!UpscalerExpected()) bias = 0.f;
        else if (g_cfg.mipBiasAuto) {
            if (g_sceneRenderW && g_swapExtent.width) {
                // relative to what the upscaler writes, which is less than the screen with upscale_to
                bool mfx = ActivePref() != UpscalerPref::Dlss && ActivePref() != UpscalerPref::Fsr;
                uint32_t renderH = (uint32_t)((uint64_t)g_sceneRenderW * g_swapExtent.height / g_swapExtent.width);
                lodOutW = UpscaleExtent(g_sceneRenderW, renderH, mfx).width;
                if (g_sceneRenderW < lodOutW) bias = std::log2((float)g_sceneRenderW / (float)lodOutW);
            }
        } else bias = g_cfg.mipBias;
        auto repeats = [](VkSamplerAddressMode m) {
            return m == VK_SAMPLER_ADDRESS_MODE_REPEAT || m == VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        };
        bool material = info.maxLod > 0.5f && !info.compareEnable && info.minFilter == VK_FILTER_LINEAR &&
                        (repeats(info.addressModeU) || repeats(info.addressModeV) || info.anisotropyEnable);
        static uint64_t biased = 0, skipped = 0;
        if (bias != 0.f && material) {
            info.mipLodBias += bias;
            if (biased++ < 3 || biased % 100 == 0)
                Log("[lod] texture sampler %llu: LOD bias %+.2f (render %u / upscaled %u)", (unsigned long long)biased,
                    bias, g_sceneRenderW, lodOutW);
        } else if (++skipped <= 3 && bias != 0.f)
            Log("[lod] sampler left alone (maxLod %.1f, address %d/%d, aniso %d)", info.maxLod, (int)info.addressModeU,
                (int)info.addressModeV, (int)info.anisotropyEnable);
    }
    return GetDevice(Key(device))->CreateSampler(device, &info, pAllocator, pSampler);
}

// =============================================================================================
// Images, views, framebuffers, render passes, swapchain
// =============================================================================================
static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateImage(VkDevice device, const VkImageCreateInfo* pInfo,
                                                      const VkAllocationCallbacks* pAllocator, VkImage* pImage) {
    VkImageCreateInfo info = *pInfo;
    bool addedSampled = false;
    // DLSS reads depth in a shader, so large depth buffers need the SAMPLED usage bit.
    if (IsDepthFormat(info.format) && info.imageType == VK_IMAGE_TYPE_2D && info.extent.width >= 1000 &&
        (info.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) && !(info.usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        addedSampled = true;
    }
    // Recordings (Ctrl+Shift+T, TF3_DLSS_CAPTURE) copy the depth and the upscaler's input colour out, which
    // needs TRANSFER_SRC. Only added when recordings are possible: players' images stay as the game made them.
    if (RecordingPossible() && info.imageType == VK_IMAGE_TYPE_2D && info.extent.width >= 1000 &&
        (info.usage & (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) &&
        (info.usage & VK_IMAGE_USAGE_SAMPLED_BIT))
        info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    DeviceData* d = GetDevice(Key(device));
    VkResult r = d->CreateImage(device, &info, pAllocator, pImage);
    if (r != VK_SUCCESS && addedSampled) {
        addedSampled = false;
        info = *pInfo;
        r = d->CreateImage(device, &info, pAllocator, pImage);
    }
    if (r == VK_SUCCESS && info.imageType == VK_IMAGE_TYPE_2D) {
        TLock lk(g_t);
        if (IsDepthFormat(info.format) && info.extent.width >= 1000 && info.extent.width != info.extent.height)
            g_sceneRenderW = info.extent.width;  // the game's render width (for the texture LOD bias)
        g_images[*pImage] = ImageInfo{info.extent.width, info.extent.height, info.format, info.usage, info.samples};
        g_layout[*pImage] = info.initialLayout;
        if (info.extent.width >= 1800 && info.arrayLayers == 1 && (info.usage & 0x30))  // render targets only
            Log("[image] %s%s", ImageDesc(*pImage).c_str(), addedSampled ? "  (+SAMPLED)" : "");
    }
    return r;
}

// The game destroys its render targets when the resolution changes. Our views of them and every
// descriptor set using those views must go first: the driver may hand the same handle to the new
// image, and a stale view would then point at freed memory.
static VKAPI_ATTR void VKAPI_CALL tf3_DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator) {
    DeviceData* d = GetDevice(Key(device));
    {
        TLock lk(g_t);
        auto ii = g_images.find(image);
        bool renderTarget = ii != g_images.end() && ii->second.w >= 1000 && ii->second.w != ii->second.h;
        auto dv = g_dlss.depthViews.find(image);
        if (dv != g_dlss.depthViews.end() || (renderTarget && !g_dlss.mvSets.empty())) {
            static int logged = 0;
            if (logged++ < 4) Log("[dlss] game destroyed render target %s; dropping our views of it", H(image).c_str());
            InvalidateViewCaches(d);
            if (dv != g_dlss.depthViews.end()) {
                d->DestroyImageView(d->device, dv->second, nullptr);
                g_dlss.depthViews.erase(dv);
            }
        }
        if (ii != g_images.end()) g_images.erase(ii);
        g_layout.erase(image);
    }
    d->DestroyImage(device, image, pAllocator);
}

static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateImageView(VkDevice device, const VkImageViewCreateInfo* pInfo,
                                                          const VkAllocationCallbacks* pAllocator, VkImageView* pView) {
    VkResult r = GetDevice(Key(device))->CreateImageView(device, pInfo, pAllocator, pView);
    if (r == VK_SUCCESS) {
        TLock lk(g_t);
        g_viewImage[*pView] = pInfo->image;
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo* pInfo,
                                                            const VkAllocationCallbacks* pAllocator,
                                                            VkFramebuffer* pFb) {
    VkResult r = GetDevice(Key(device))->CreateFramebuffer(device, pInfo, pAllocator, pFb);
    if (r == VK_SUCCESS) {
        TLock lk(g_t);
        g_fbViews[*pFb].assign(pInfo->pAttachments, pInfo->pAttachments + pInfo->attachmentCount);
    }
    return r;
}

// DLSS and the motion-vector pass read depth after the game's last depth render pass. If that pass
// ends with storeOp DONT_CARE the contents are undefined afterwards, so keep depth/stencil stored.
template <typename Desc>
std::vector<Desc> KeepDepth(const Desc* att, uint32_t count, int* changed) {
    std::vector<Desc> v(att, att + count);
    for (auto& a : v)
        if (IsDepthFormat(a.format) &&
            (a.storeOp != VK_ATTACHMENT_STORE_OP_STORE || (HasStencil(a.format) && a.stencilStoreOp != VK_ATTACHMENT_STORE_OP_STORE))) {
            a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            if (HasStencil(a.format)) a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
            ++*changed;
        }
    return v;
}

static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* pInfo,
                                                           const VkAllocationCallbacks* pAllocator, VkRenderPass* pRp) {
    int changed = 0;
    auto atts = KeepDepth(pInfo->pAttachments, pInfo->attachmentCount, &changed);
    VkRenderPassCreateInfo info = *pInfo;
    info.pAttachments = atts.data();
    VkResult r = GetDevice(Key(device))->CreateRenderPass(device, &info, pAllocator, pRp);
    if (changed && r == VK_SUCCESS) Log("[renderpass] %s: depth storeOp forced to STORE", H(*pRp).c_str());
    if (r == VK_SUCCESS) {
        TLock lk(g_t);
        auto& rp = g_renderPasses[*pRp];
        rp.finalLayouts.clear();
        for (uint32_t i = 0; i < pInfo->attachmentCount; ++i) rp.finalLayouts.push_back(pInfo->pAttachments[i].finalLayout);
    }
    return r;
}
static void RecordRp2(const VkRenderPassCreateInfo2* pInfo, VkRenderPass rpH) {
    TLock lk(g_t);
    auto& rp = g_renderPasses[rpH];
    rp.finalLayouts.clear();
    for (uint32_t i = 0; i < pInfo->attachmentCount; ++i) rp.finalLayouts.push_back(pInfo->pAttachments[i].finalLayout);
}
static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateRenderPass2(VkDevice device, const VkRenderPassCreateInfo2* pInfo,
                                                            const VkAllocationCallbacks* pAllocator, VkRenderPass* pRp) {
    int changed = 0;
    auto atts = KeepDepth(pInfo->pAttachments, pInfo->attachmentCount, &changed);
    VkRenderPassCreateInfo2 info = *pInfo;
    info.pAttachments = atts.data();
    VkResult r = GetDevice(Key(device))->CreateRenderPass2(device, &info, pAllocator, pRp);
    if (r == VK_SUCCESS) RecordRp2(pInfo, *pRp);
    if (changed && r == VK_SUCCESS) Log("[renderpass] %s: depth storeOp forced to STORE", H(*pRp).c_str());
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateRenderPass2KHR(VkDevice device, const VkRenderPassCreateInfo2* pInfo,
                                                               const VkAllocationCallbacks* pAllocator,
                                                               VkRenderPass* pRp) {
    int changed = 0;
    auto atts = KeepDepth(pInfo->pAttachments, pInfo->attachmentCount, &changed);
    VkRenderPassCreateInfo2 info = *pInfo;
    info.pAttachments = atts.data();
    VkResult r = GetDevice(Key(device))->CreateRenderPass2KHR(device, &info, pAllocator, pRp);
    if (r == VK_SUCCESS) RecordRp2(pInfo, *pRp);
    if (changed && r == VK_SUCCESS) Log("[renderpass] %s: depth storeOp forced to STORE", H(*pRp).c_str());
    return r;
}

static const char* PresentModeName(VkPresentModeKHR m) {
    switch (m) {
        case VK_PRESENT_MODE_IMMEDIATE_KHR: return "immediate";
        case VK_PRESENT_MODE_MAILBOX_KHR: return "mailbox";
        case VK_PRESENT_MODE_FIFO_KHR: return "fifo";
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "fifo relaxed";
        default: return "other";
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* pInfo,
                                                             const VkAllocationCallbacks* pAllocator,
                                                             VkSwapchainKHR* pSwapchain) {
    DeviceData* d = GetDevice(Key(device));
    VkSwapchainCreateInfoKHR ci = *pInfo;
    bool forFg = false;
#ifdef __APPLE__
    // Frame generation presents a second image per frame: it copies into swapchain images (transfer usage),
    // needs a free image while the game holds one and one is on screen (3 images), and relies on FIFO to show
    // the two at successive refreshes.
    std::string fgWhy;
    if (g_cfg.frameGen && d->instance) {
        auto caps = (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)d->instance->GetInstanceProcAddr(
            d->instance->instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        VkSurfaceCapabilitiesKHR sc{};
        const VkImageUsageFlags need = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (!caps || caps(d->physicalDevice, pInfo->surface, &sc) != VK_SUCCESS) fgWhy = "no surface capabilities";
        else if ((sc.supportedUsageFlags & need) != need) fgWhy = "swapchain images can't be copied";
        else if (sc.maxImageCount && sc.maxImageCount < 3) fgWhy = "the swapchain can't have 3 images";
        else if (pInfo->imageFormat != VK_FORMAT_B8G8R8A8_UNORM && pInfo->imageFormat != VK_FORMAT_R8G8B8A8_UNORM)
            fgWhy = "unsupported swapchain format";
        else {
            ci.imageUsage |= need;
            ci.minImageCount = std::max(ci.minImageCount, 3u);
            ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
            forFg = true;
        }
    }
    if (g_cfg.fpsLimit > 0) ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // always supported
#endif
    VkResult r = d->CreateSwapchainKHR(device, &ci, pAllocator, pSwapchain);
    if (r != VK_SUCCESS && forFg) {  // try again as the game asked
        Log("[fg] swapchain with frame generation settings failed (%d); creating it as the game asked", (int)r);
        ci = *pInfo;
        forFg = false;
        r = d->CreateSwapchainKHR(device, &ci, pAllocator, pSwapchain);
    }
    if (r == VK_SUCCESS) {
        TLock lk(g_t);
        g_swapExtent = pInfo->imageExtent;
        Log("[swapchain] %ux%u format %d, %u images asked (%u used), present mode %s%s%s, usage 0x%x",
            pInfo->imageExtent.width, pInfo->imageExtent.height, (int)pInfo->imageFormat, pInfo->minImageCount,
            ci.minImageCount, PresentModeName(pInfo->presentMode),
            ci.presentMode != pInfo->presentMode ? " -> " : "",
            ci.presentMode != pInfo->presentMode ? PresentModeName(ci.presentMode) : "", (unsigned)ci.imageUsage);
#ifdef __APPLE__
        if (g_fg.swapchain) {  // replaced: our generated frames may still use the old one's images
            d->DeviceWaitIdle(device);
            g_fg.swapchain = VK_NULL_HANDLE;
            g_fg.swapImages.clear();
            g_fg.havePrev = false;
        }
        if (forFg) {
            auto getImages = (PFN_vkGetSwapchainImagesKHR)d->GetDeviceProcAddr(device, "vkGetSwapchainImagesKHR");
            uint32_t n = 0;
            if (getImages && getImages(device, *pSwapchain, &n, nullptr) == VK_SUCCESS && n) {
                g_fg.swapImages.resize(n);
                getImages(device, *pSwapchain, &n, g_fg.swapImages.data());
                g_fg.swapchain = *pSwapchain;
                g_fg.swapFmt = pInfo->imageFormat;
                g_fg.extent = pInfo->imageExtent;
                Log("[fg] swapchain set up for frame generation (%u images, FIFO)", n);
            }
        } else if (g_cfg.frameGen)
            Log("[fg] swapchain not set up for frame generation: %s", fgWhy.empty() ? "creation failed" : fgWhy.c_str());
        FpsLockSwapchain(*pSwapchain, g_cfg.fpsLimit > 0 && ci.presentMode == VK_PRESENT_MODE_FIFO_KHR && !forFg);
#endif
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL tf3_DestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain,
                                                          const VkAllocationCallbacks* pAllocator) {
    DeviceData* d = GetDevice(Key(device));
#ifdef __APPLE__
    {
        TLock lk(g_t);
        if (swapchain && swapchain == g_fg.swapchain) {
            d->DeviceWaitIdle(device);  // our generated frames copy into its images
            g_fg.swapchain = VK_NULL_HANDLE;
            g_fg.swapImages.clear();
            g_fg.havePrev = false;
        }
    }
#endif
    d->DestroySwapchainKHR(device, swapchain, pAllocator);
}

static VKAPI_ATTR void VKAPI_CALL tf3_GetDeviceQueue(VkDevice device, uint32_t family, uint32_t index, VkQueue* pQueue) {
    GetDevice(Key(device))->GetDeviceQueue(device, family, index, pQueue);
    if (pQueue && *pQueue) {
        std::lock_guard<std::mutex> lk(g_lock);
        g_queueFamily[*pQueue] = family;
    }
}

static VKAPI_ATTR void VKAPI_CALL tf3_GetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2* pInfo, VkQueue* pQueue) {
    GetDevice(Key(device))->GetDeviceQueue2(device, pInfo, pQueue);
    if (pInfo && pQueue && *pQueue) {
        std::lock_guard<std::mutex> lk(g_lock);
        g_queueFamily[*pQueue] = pInfo->queueFamilyIndex;
    }
}

// =============================================================================================
// Buffers and mapped memory
// =============================================================================================
static VKAPI_ATTR VkResult VKAPI_CALL tf3_BindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                                           VkDeviceSize offset) {
    VkResult r = GetDevice(Key(device))->BindBufferMemory(device, buffer, memory, offset);
    if (r == VK_SUCCESS) {
        TLock lk(g_t);
        g_buffers[buffer] = BufferInfo{memory, offset};
    }
    return r;
}
static void RecordBinds2(uint32_t count, const VkBindBufferMemoryInfo* p) {
    TLock lk(g_t);
    for (uint32_t i = 0; i < count; ++i) g_buffers[p[i].buffer] = BufferInfo{p[i].memory, p[i].memoryOffset};
}
static VKAPI_ATTR VkResult VKAPI_CALL tf3_BindBufferMemory2(VkDevice device, uint32_t count,
                                                            const VkBindBufferMemoryInfo* p) {
    VkResult r = GetDevice(Key(device))->BindBufferMemory2(device, count, p);
    if (r == VK_SUCCESS) RecordBinds2(count, p);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL tf3_BindBufferMemory2KHR(VkDevice device, uint32_t count,
                                                               const VkBindBufferMemoryInfo* p) {
    VkResult r = GetDevice(Key(device))->BindBufferMemory2KHR(device, count, p);
    if (r == VK_SUCCESS) RecordBinds2(count, p);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL tf3_MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                                                    VkDeviceSize size, VkMemoryMapFlags flags, void** ppData) {
    VkResult r = GetDevice(Key(device))->MapMemory(device, memory, offset, size, flags, ppData);
    if (r == VK_SUCCESS) {
        TLock lk(g_t);
        g_mapped[memory] = Mapping{static_cast<char*>(*ppData), offset, size};
    }
    return r;
}
static VKAPI_ATTR void VKAPI_CALL tf3_UnmapMemory(VkDevice device, VkDeviceMemory memory) {
    {
        TLock lk(g_t);
        g_mapped.erase(memory);
    }
    GetDevice(Key(device))->UnmapMemory(device, memory);
}

// =============================================================================================
// Descriptor sets
// =============================================================================================
static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateDescriptorSetLayout(VkDevice device,
                                                                    const VkDescriptorSetLayoutCreateInfo* pInfo,
                                                                    const VkAllocationCallbacks* pAllocator,
                                                                    VkDescriptorSetLayout* pLayout) {
    VkResult r = GetDevice(Key(device))->CreateDescriptorSetLayout(device, pInfo, pAllocator, pLayout);
    if (r == VK_SUCCESS) {
        std::vector<LayoutBinding> b;
        for (uint32_t i = 0; i < pInfo->bindingCount; ++i)
            b.push_back({pInfo->pBindings[i].binding, pInfo->pBindings[i].descriptorType,
                         pInfo->pBindings[i].descriptorCount});
        std::sort(b.begin(), b.end(), [](const LayoutBinding& x, const LayoutBinding& y) { return x.binding < y.binding; });
        TLock lk(g_t);
        std::unique_lock<std::shared_mutex> dl(g_descLock);
        g_setLayouts[*pLayout] = std::move(b);
        if (pInfo->flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT)
            g_updateAfterBindLayouts.insert(*pLayout);
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL tf3_AllocateDescriptorSets(VkDevice device,
                                                                 const VkDescriptorSetAllocateInfo* pInfo,
                                                                 VkDescriptorSet* pSets) {
    VkResult r = GetDevice(Key(device))->AllocateDescriptorSets(device, pInfo, pSets);
    if (r == VK_SUCCESS) {
        PerfScope ps(kPerfDesc);
        TLock lk(g_t);
        std::unique_lock<std::shared_mutex> dl(g_descLock);
        for (uint32_t i = 0; i < pInfo->descriptorSetCount; ++i) {
            g_setLayoutOf[pSets[i]] = pInfo->pSetLayouts[i];
            g_setContents.erase(pSets[i]);
        }
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL tf3_UpdateDescriptorSets(VkDevice device, uint32_t writeCount,
                                                           const VkWriteDescriptorSet* pWrites, uint32_t copyCount,
                                                           const VkCopyDescriptorSet* pCopies) {
    {
        PerfScope ps(kPerfDesc);
        TLock lk(g_t);
        for (uint32_t i = 0; i < writeCount; ++i) {
            const auto& w = pWrites[i];
            for (uint32_t e = 0; e < w.descriptorCount; ++e) {
                DescItem d;
                d.type = w.descriptorType;
                if (IsImageType(w.descriptorType) && w.pImageInfo) d.view = w.pImageInfo[e].imageView;
                else if (IsBufferType(w.descriptorType) && w.pBufferInfo) {
                    d.buffer = w.pBufferInfo[e].buffer;
                    d.offset = w.pBufferInfo[e].offset;
                    d.range = w.pBufferInfo[e].range;
                } else continue;
                StoreDesc(w.dstSet, w.dstBinding, w.dstArrayElement + e, d);
            }
        }
        for (uint32_t i = 0; i < copyCount; ++i) {
            const auto& c = pCopies[i];
            auto src = g_setContents.find(c.srcSet);
            if (src == g_setContents.end()) continue;
            auto b = src->second.find(c.srcBinding);
            if (b == src->second.end()) continue;
            for (uint32_t e = 0; e < c.descriptorCount && c.srcArrayElement + e < b->second.size(); ++e)
                StoreDesc(c.dstSet, c.dstBinding, c.dstArrayElement + e, b->second[c.srcArrayElement + e]);
        }
    }
    GetDevice(Key(device))->UpdateDescriptorSets(device, writeCount, pWrites, copyCount, pCopies);
}

static void RecordTemplate(const VkDescriptorUpdateTemplateCreateInfo* pInfo, VkDescriptorUpdateTemplate t) {
    std::vector<TemplateEntry> v;
    for (uint32_t i = 0; i < pInfo->descriptorUpdateEntryCount; ++i) {
        const auto& e = pInfo->pDescriptorUpdateEntries[i];
        v.push_back({e.dstBinding, e.dstArrayElement, e.descriptorCount, e.descriptorType, e.offset, e.stride});
    }
    TLock lk(g_t);
    g_templates[t] = std::move(v);
}
static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateDescriptorUpdateTemplate(
    VkDevice device, const VkDescriptorUpdateTemplateCreateInfo* pInfo, const VkAllocationCallbacks* pAllocator,
    VkDescriptorUpdateTemplate* pT) {
    VkResult r = GetDevice(Key(device))->CreateDescriptorUpdateTemplate(device, pInfo, pAllocator, pT);
    if (r == VK_SUCCESS) RecordTemplate(pInfo, *pT);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL tf3_CreateDescriptorUpdateTemplateKHR(
    VkDevice device, const VkDescriptorUpdateTemplateCreateInfo* pInfo, const VkAllocationCallbacks* pAllocator,
    VkDescriptorUpdateTemplate* pT) {
    VkResult r = GetDevice(Key(device))->CreateDescriptorUpdateTemplateKHR(device, pInfo, pAllocator, pT);
    if (r == VK_SUCCESS) RecordTemplate(pInfo, *pT);
    return r;
}
static void ApplyTemplate(VkDescriptorSet set, VkDescriptorUpdateTemplate t, const void* pData) {
    PerfScope ps(kPerfDesc);
    TLock lk(g_t);
    auto it = g_templates.find(t);
    if (it == g_templates.end()) return;
    const char* base = static_cast<const char*>(pData);
    for (const auto& e : it->second)
        for (uint32_t i = 0; i < e.count; ++i) {
            const char* p = base + e.offset + i * e.stride;
            DescItem d;
            d.type = e.type;
            if (IsImageType(e.type)) d.view = reinterpret_cast<const VkDescriptorImageInfo*>(p)->imageView;
            else if (IsBufferType(e.type)) {
                const auto* bi = reinterpret_cast<const VkDescriptorBufferInfo*>(p);
                d.buffer = bi->buffer;
                d.offset = bi->offset;
                d.range = bi->range;
            } else continue;
            StoreDesc(set, e.binding, e.arrayElement + i, d);
        }
}
static VKAPI_ATTR void VKAPI_CALL tf3_UpdateDescriptorSetWithTemplate(VkDevice device, VkDescriptorSet set,
                                                                      VkDescriptorUpdateTemplate t, const void* pData) {
    ApplyTemplate(set, t, pData);
    GetDevice(Key(device))->UpdateDescriptorSetWithTemplate(device, set, t, pData);
}
static VKAPI_ATTR void VKAPI_CALL tf3_UpdateDescriptorSetWithTemplateKHR(VkDevice device, VkDescriptorSet set,
                                                                         VkDescriptorUpdateTemplate t,
                                                                         const void* pData) {
    ApplyTemplate(set, t, pData);
    GetDevice(Key(device))->UpdateDescriptorSetWithTemplateKHR(device, set, t, pData);
}

// =============================================================================================
// Command recording
// =============================================================================================
static VKAPI_ATTR VkResult VKAPI_CALL tf3_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* pInfo) {
    {
        PerfScope ps(kPerfState);
        CmdFor(cb) = CmdState{};
    }
    return GetDevice(Key(cb))->BeginCommandBuffer(cb, pInfo);
}

static VKAPI_ATTR void VKAPI_CALL tf3_CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline pipeline) {
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        PerfScope ps(kPerfState);
        auto& cs = CmdFor(cb);
        cs.pipe = FindPipeInfo(pipeline);
        cs.kind = cs.pipe ? cs.pipe->kind : Kind::Other;
        cs.pipeline = pipeline;
        if (!cs.pipe || !cs.pipe->vp.dynamic) cs.hasCurVp = false;  // static viewport replaces ours
    }
    GetDevice(Key(cb))->CmdBindPipeline(cb, bp, pipeline);
}

static VKAPI_ATTR void VKAPI_CALL tf3_CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp,
                                                            VkPipelineLayout layout, uint32_t firstSet, uint32_t count,
                                                            const VkDescriptorSet* sets, uint32_t dynCount,
                                                            const uint32_t* dynOffsets) {
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        PerfScope ps(kPerfBind);
        auto& cs = CmdFor(cb);
        if (cs.sets.size() < firstSet + count) cs.sets.resize(firstSet + count);
        ++cs.setsGen;
        uint32_t di = 0;  // dynamic offsets are consumed in set order, then binding order
        for (uint32_t i = 0; i < count; ++i) {
            BoundSet& bs = cs.sets[firstSet + i];  // filled in place: keeps the vectors' memory
            bs.set = sets[i];
            bs.layout = layout;
            bs.dyn.clear();
            bs.dynOffsets.clear();
            if (di >= dynCount) continue;  // no dynamic offsets left (most binds have none)
            std::shared_lock<std::shared_mutex> dl(g_descLock);
            auto lo = g_setLayoutOf.find(sets[i]);
            if (lo != g_setLayoutOf.end()) {
                auto lb = g_setLayouts.find(lo->second);
                if (lb != g_setLayouts.end())
                    for (const auto& b : lb->second)
                        if (IsDynamic(b.type))
                            for (uint32_t k = 0; k < b.count; ++k, ++di)
                                if (di < dynCount) {
                                    if (k == 0) bs.dyn.push_back({b.binding, dynOffsets[di]});
                                    bs.dynOffsets.push_back(dynOffsets[di]);
                                }
            }
        }
    }
    GetDevice(Key(cb))->CmdBindDescriptorSets(cb, bp, layout, firstSet, count, sets, dynCount, dynOffsets);
}

static VKAPI_ATTR void VKAPI_CALL tf3_CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src,
                                                         VkPipelineStageFlags dst, VkDependencyFlags dep,
                                                         uint32_t memCount, const VkMemoryBarrier* mem,
                                                         uint32_t bufCount, const VkBufferMemoryBarrier* buf,
                                                         uint32_t imgCount, const VkImageMemoryBarrier* img) {
    if (imgCount) {
        PerfScope ps(kPerfState);
        TLock lk(g_t);
        auto& cs = CmdFor(cb);
        for (uint32_t i = 0; i < imgCount; ++i) SetLayout(cs, img[i].image, img[i].newLayout);
    }
    GetDevice(Key(cb))->CmdPipelineBarrier(cb, src, dst, dep, memCount, mem, bufCount, buf, imgCount, img);
}

static void BeginRp(VkCommandBuffer cb, const VkRenderPassBeginInfo* p) {
    PerfScope ps(kPerfPass);
    TLock lk(g_t);
    auto& cs = CmdFor(cb);
    cs.fb = p->framebuffer;
    cs.rp = p->renderPass;
    cs.rcasInThisPass = false;
    cs.fbIsScene = false;
    cs.fbW = cs.fbH = 0;
    cs.viewGen = UINT32_MAX;
    cs.moving = g_movingSnap;
    for (VkImage img : FbImages(p->framebuffer)) {
        auto it = g_images.find(img);
        if (it != g_images.end() && !cs.fbW) {
            cs.fbW = it->second.w;
            cs.fbH = it->second.h;
        }
        if (it != g_images.end() && IsDepthFormat(it->second.fmt) && it->second.w >= 1000 &&
            it->second.w != it->second.h)
            cs.fbIsScene = true;
    }
}
static VKAPI_ATTR void VKAPI_CALL tf3_CmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* p,
                                                         VkSubpassContents c) {
    BeginRp(cb, p);
    GetDevice(Key(cb))->CmdBeginRenderPass(cb, p, c);
}
static VKAPI_ATTR void VKAPI_CALL tf3_CmdBeginRenderPass2(VkCommandBuffer cb, const VkRenderPassBeginInfo* p,
                                                          const VkSubpassBeginInfo* s) {
    BeginRp(cb, p);
    GetDevice(Key(cb))->CmdBeginRenderPass2(cb, p, s);
}
static VKAPI_ATTR void VKAPI_CALL tf3_CmdBeginRenderPass2KHR(VkCommandBuffer cb, const VkRenderPassBeginInfo* p,
                                                             const VkSubpassBeginInfo* s) {
    BeginRp(cb, p);
    GetDevice(Key(cb))->CmdBeginRenderPass2KHR(cb, p, s);
}

// After a render pass ends: apply its final layouts, and if it was RCAS's pass, run DLSS.
static void AfterEndRp(DeviceData* d, VkCommandBuffer cb) {
    TLock lk(g_t);
    auto& cs = CmdFor(cb);
    {
        PerfScope ps(kPerfPass);
        if (cs.hasSceneVp) {  // seen by ApplyViewport in this pass
            const VkViewport& v = cs.sceneVp;
            if (!g_dlss.sceneVpValid)
                Log("[obj] scene viewport: x=%g y=%g w=%g h=%g depth %g..%g", v.x, v.y, v.width, v.height,
                    v.minDepth, v.maxDepth);
            g_dlss.sceneVp = v;
            g_dlss.sceneVpValid = true;
            cs.hasSceneVp = false;
        }
        auto rp = g_renderPasses.find(cs.rp);
        if (rp != g_renderPasses.end()) {
            auto imgs = FbImages(cs.fb);
            for (size_t i = 0; i < imgs.size() && i < rp->second.finalLayouts.size(); ++i)
                if (imgs[i]) SetLayout(cs, imgs[i], rp->second.finalLayouts[i]);
        }
    }
    if (cs.rcasInThisPass) {
        cs.rcasInThisPass = false;
        PerfScope ps(kPerfUpscale);
        RecordUpscale(d, cb, cs);
    }
}
static VKAPI_ATTR void VKAPI_CALL tf3_CmdEndRenderPass(VkCommandBuffer cb) {
    DeviceData* d = GetDevice(Key(cb));
    d->CmdEndRenderPass(cb);
    AfterEndRp(d, cb);
}
static VKAPI_ATTR void VKAPI_CALL tf3_CmdEndRenderPass2(VkCommandBuffer cb, const VkSubpassEndInfo* p) {
    DeviceData* d = GetDevice(Key(cb));
    d->CmdEndRenderPass2(cb, p);
    AfterEndRp(d, cb);
}
static VKAPI_ATTR void VKAPI_CALL tf3_CmdEndRenderPass2KHR(VkCommandBuffer cb, const VkSubpassEndInfo* p) {
    DeviceData* d = GetDevice(Key(cb));
    d->CmdEndRenderPass2KHR(cb, p);
    AfterEndRp(d, cb);
}

// Bind our replacement set (if any) for one draw, then put the game's set back.
static void SwapBind(DeviceData* d, VkCommandBuffer cb, const SwapPlan& plan) {
    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, plan.layout, plan.index, 1, &plan.set,
                             (uint32_t)plan.dyn.size(), plan.dyn.data());
}
static void SwapRestore(DeviceData* d, VkCommandBuffer cb, const SwapPlan& plan) {
    PerfScope ps(kPerfDraw);
    TLock lk(g_t);
    auto& cs = CmdFor(cb);
    if (plan.index < cs.sets.size()) {
        const BoundSet& bs = cs.sets[plan.index];
        d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, bs.layout, plan.index, 1, &bs.set,
                                 (uint32_t)bs.dynOffsets.size(), bs.dynOffsets.data());
    }
}

// ---- Viewport jitter ---------------------------------------------------------------------------
// Jitter by shifting the viewport of draws into the scene framebuffer by the sub-pixel offset.
// Only what DLSS upscales is jittered; the camera data stays untouched, so passes that aren't
// upscaled (the selection mask behind the hover outline, terrain tile caches, ...) are unaffected.
// No lock needed (the scene viewport goes to g_dlss when the render pass ends).
void ApplyViewport(DeviceData* d, VkCommandBuffer cb, CmdState& cs) {
    if (!cs.pipe || !cs.pipe->vp.dynamic) return;
    VkViewport want;
    if (cs.pipe->vp.patched) want = cs.pipe->vp.staticVp;
    else if (cs.hasGameVp) want = cs.gameVp;
    else return;
    if (cs.fbIsScene) {
        cs.sceneVp = want;
        cs.hasSceneVp = true;
    }
    if (JitterThisDraw(cs)) {
        float jx, jy;
        HotFrameJitter(jx, jy);
        want.x += jx;  // jitter is in top-down pixels, like framebuffer coordinates
        want.y += jy;
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true))
            Log("[jitter] viewport jitter active (frame %llu)", (unsigned long long)g_hot.frame.load());
    }
    if (cs.hasCurVp && std::memcmp(&want, &cs.curVp, sizeof want) == 0) return;
    d->CmdSetViewport(cb, 0, 1, &want);
    cs.curVp = want;
    cs.hasCurVp = true;
}

static VKAPI_ATTR void VKAPI_CALL tf3_CmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t count,
                                                     const VkViewport* pViewports) {
    {
        PerfScope ps(kPerfState);
        CmdState& cs = CmdFor(cb);
        if (first == 0 && count >= 1) {
            cs.gameVp = cs.curVp = pViewports[0];
            cs.hasGameVp = cs.hasCurVp = true;
        }
    }
    GetDevice(Key(cb))->CmdSetViewport(cb, first, count, pViewports);
}

static VKAPI_ATTR void VKAPI_CALL tf3_CmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t count,
                                                    const VkRect2D* pScissors) {
    {
        PerfScope ps(kPerfState);
        CmdState& cs = CmdFor(cb);
        if (first == 0 && count >= 1) {
            cs.scissor = pScissors[0];
            cs.hasScissor = true;
        }
    }
    GetDevice(Key(cb))->CmdSetScissor(cb, first, count, pScissors);
}

// ---- Draw log (Ctrl+Shift+L): every draw of one frame, with its pass and whether it is jittered -----
struct DrawLog {
    uint64_t frame = UINT64_MAX;  // frame being logged
    bool requested = false;
    int index = 0;
    VkRenderPass lastRp = VK_NULL_HANDLE;
    VkFramebuffer lastFb = VK_NULL_HANDLE;
} g_drawLog;

// Caller holds g_t.
void LogSceneDraw(VkCommandBuffer cb, CmdState& cs, bool indexed, uint32_t count, uint32_t instances,
                  uint32_t firstInstance) {
    if (g_drawLog.frame != g_frame) return;
    if (cs.fb != g_drawLog.lastFb || cs.rp != g_drawLog.lastRp) {
        g_drawLog.lastFb = cs.fb;
        g_drawLog.lastRp = cs.rp;
        std::string atts;
        for (VkImage img : FbImages(cs.fb)) {
            auto ii = g_images.find(img);
            if (ii != g_images.end()) atts += " " + std::to_string(ii->second.w) + "x" + std::to_string(ii->second.h) + ":" +
                                             std::to_string((int)ii->second.fmt);
        }
        Log("[draws] -- render pass %s, framebuffer %s, attachments (format ids):%s%s", H(cs.rp).c_str(),
            H(cs.fb).c_str(), atts.c_str(), cs.fbIsScene ? " (main 3D pass)" : "");
    }
    auto ps = g_pipeState.find(cs.pipeline);
    PipeState st = ps == g_pipeState.end() ? PipeState{} : ps->second;
    auto pv = g_pipeVtx.find(cs.pipeline);
    uint64_t mesh = pv != g_pipeVtx.end() ? MeshHash(cs, pv->second, indexed, count, 0, 0) : 0;
    Log("[draws] %4d %s pipe %s cmd %s: %s%u x%u inst | colour mask %x%s (%u att) | depth %s%s op %d | stencil %s pass %d | "
        "cull %u topo %d | mesh %016llx | vs %s | fs %s",
        g_drawLog.index++, JitterThisDraw(cs) ? "JIT" : "---", H(cs.pipeline).c_str(), H(cb).c_str(), indexed ? "idx " : "", count, instances, st.colorMask,
        st.blend ? " blend" : "", st.attachments, st.depthTest ? "test" : "-", st.depthWrite ? "+write" : "",
        (int)st.depthOp, st.stencil ? "on" : "off", (int)st.stencilPass, (unsigned)st.cull, (int)st.topology,
        (unsigned long long)mesh, st.vsNames.c_str(), st.fsNames.c_str());
}

// ---- Moving-mesh discovery ---------------------------------------------------------------------
// Every kProbeEvery frames, read the model matrix of every scene instance on two consecutive frames.
// A mesh with a copy that moved a little (and has no copy left exactly where it was) is "moving";
// the object motion pass then replays its draws for a while (kObjMeshKeep).
// Cheap by design: instances found at exactly the same position as last frame (almost all of
// them) are settled with one hash lookup.
constexpr uint64_t kProbeEvery = 30;
bool ProbeFrame(uint64_t f) {
    return g_cfg.objectMotion && g_firstFsrFrame && f > g_firstFsrFrame + 10 &&
           (f % kProbeEvery == 0 || f % kProbeEvery == 1);
}

struct ProbePos {
    uint64_t mesh;
    float x, y, z;
};
struct ProbeFrameData {
    uint64_t frame = UINT64_MAX;
    std::vector<ProbePos> inst;
    std::unordered_set<uint64_t> seen;  // instance slots already read (depth prepass + colour pass)
};
ProbeFrameData g_probeCur, g_probePrev;

// No lock needed: fills this command buffer's list, read at submit.
void ProbeDraw(CmdState& cs, bool indexed, uint32_t count, uint32_t instanceCount, uint32_t first,
               int32_t vertexOffset, uint32_t firstInstance) {
    if (!g_hot.probe.load(std::memory_order_relaxed)) return;
    if (!cs.fbIsScene || !cs.pipe || !cs.pipe->hasVtx) return;
    const PipeVtx& pv = cs.pipe->vtx;
    if (!pv.hasMdl || !pv.instanceRate || !IsTriangles(pv.topology) || pv.mdlBinding >= 16 || pv.cols < 4 ||
        pv.comps < 3)
        return;
    ObjProbe p;
    p.mesh = MeshHash(cs, pv, indexed, count, first, vertexOffset);
    p.buf = cs.vb[pv.mdlBinding].first;
    p.base = cs.vb[pv.mdlBinding].second;
    p.stride = pv.stride;
    p.first = firstInstance;
    p.count = instanceCount;
    p.cols = pv.cols;
    p.comps = pv.comps;
    std::memcpy(p.colOffset, pv.colOffset, sizeof p.colOffset);
    if (p.buf && cs.probes.size() < 20000) cs.probes.push_back(p);
}

// At submit, when the game has finished writing this frame's instance data: just the translations.
void ProbeCollect(CmdState& cs) {
    if (cs.probes.empty()) return;
    if (g_probeCur.frame != g_frame) {
        g_probeCur = ProbeFrameData{};
        g_probeCur.frame = g_frame;
    }
    for (const ObjProbe& p : cs.probes) {
        char* base = RefData(p.buf, p.base);
        auto bi = g_buffers.find(p.buf);
        if (!base || bi == g_buffers.end()) continue;
        const Mapping& mp = g_mapped[bi->second.memory];
        for (uint32_t i = 0; i < p.count && i < 4096; ++i) {
            VkDeviceSize off = (VkDeviceSize)(p.first + i) * p.stride;
            if (!g_probeCur.seen.insert(Mix((uint64_t)(uintptr_t)p.buf, p.base + off)).second) continue;
            VkDeviceSize end = bi->second.memOffset + p.base + off + p.colOffset[3] + 12;
            if (mp.size != VK_WHOLE_SIZE && end > mp.offset + mp.size) break;
            float t[3];
            std::memcpy(t, base + off + p.colOffset[3], 12);  // translation = 4th column
            g_probeCur.inst.push_back({p.mesh, t[0], t[1], t[2]});
        }
    }
    cs.probes.clear();
}

uint64_t PosKey(const ProbePos& p) {
    uint32_t x, y, z;
    std::memcpy(&x, &p.x, 4);
    std::memcpy(&y, &p.y, 4);
    std::memcpy(&z, &p.z, 4);
    return Mix(Mix(Mix(p.mesh, x), y), z);
}

// At present: find meshes with a copy that moved between the two probed frames.
void ProbeReport() {
    if (g_probeCur.frame != g_frame) return;
    if (g_probePrev.frame + 1 == g_frame) {
        for (auto it = g_movingMeshes.begin(); it != g_movingMeshes.end();)
            if (it->second + kObjMeshKeep < g_frame) it = g_movingMeshes.erase(it);
            else ++it;
        std::unordered_set<uint64_t> prevExact;
        prevExact.reserve(g_probePrev.inst.size() * 2);
        for (const auto& p : g_probePrev.inst) prevExact.insert(PosKey(p));
        std::unordered_map<uint64_t, std::vector<const ProbePos*>> prevByMesh;  // built lazily
        bool byMeshBuilt = false;
        uint32_t still = 0, moved = 0;
        for (const auto& c : g_probeCur.inst) {
            if (prevExact.count(PosKey(c))) {
                ++still;
                continue;
            }
            if (!byMeshBuilt) {
                for (const auto& p : g_probePrev.inst) prevByMesh[p.mesh].push_back(&p);
                byMeshBuilt = true;
            }
            auto pm = prevByMesh.find(c.mesh);
            if (pm == prevByMesh.end()) continue;
            float best = 1e30f;
            for (const ProbePos* p : pm->second) {
                float dx = p->x - c.x, dy = p->y - c.y, dz = p->z - c.z;
                best = std::min(best, dx * dx + dy * dy + dz * dz);
            }
            if (best > 1e-8f && best < kObjMatchDist * kObjMatchDist) {
                g_movingMeshes[c.mesh] = g_frame;
                ++moved;
            }
        }
        if (g_frame % 600 < kProbeEvery)
            Log("[obj] frame %llu: %zu instances, %u in place, %u moved; %zu meshes tracked as moving",
                (unsigned long long)g_frame, g_probeCur.inst.size(), still, moved, g_movingMeshes.size());
    }
    g_probePrev = std::move(g_probeCur);
    g_probeCur = ProbeFrameData{};
}

// The per-frame values the draw hooks read without g_t. Caller holds g_t; at present, for the next frame.
void UpdateHot() {
    g_hot.frame.store(g_frame, std::memory_order_release);
    g_hot.drawLog.store(g_drawLog.frame == g_frame, std::memory_order_relaxed);
    g_hot.probe.store(ProbeFrame(g_frame), std::memory_order_relaxed);
    auto snap = std::make_shared<std::unordered_set<uint64_t>>();
    for (const auto& [mesh, last] : g_movingMeshes)
        if (last + kObjMeshKeep >= g_frame) snap->insert(mesh);
    if (!g_movingSnap || *g_movingSnap != *snap) g_movingSnap = std::move(snap);
    g_hot.objects.store(g_cfg.objectMotion && g_cfg.motionVectors && !g_movingSnap->empty(), std::memory_order_relaxed);
    g_hot.jitter.store(ViewportJitterOn(), std::memory_order_relaxed);
    g_hot.jitterScreen.store(g_cfg.jitterScreenPasses != 0, std::memory_order_relaxed);
    g_hot.jitterOther.store(g_cfg.jitterOtherTargets != 0, std::memory_order_relaxed);
    g_hot.react.store(g_cfg.reactive != 0, std::memory_order_relaxed);
    g_hot.inW.store(g_dlss.inW, std::memory_order_relaxed);
    g_hot.inH.store(g_dlss.inH, std::memory_order_relaxed);
}

static VKAPI_ATTR void VKAPI_CALL tf3_CmdBindVertexBuffers(VkCommandBuffer cb, uint32_t firstBinding,
                                                           uint32_t bindingCount, const VkBuffer* pBuffers,
                                                           const VkDeviceSize* pOffsets) {
    {
        PerfScope ps(kPerfState);
        CmdState& cs = CmdFor(cb);
        for (uint32_t i = 0; i < bindingCount; ++i)
            if (firstBinding + i < 16) cs.vb[firstBinding + i] = {pBuffers[i], pOffsets[i]};
    }
    GetDevice(Key(cb))->CmdBindVertexBuffers(cb, firstBinding, bindingCount, pBuffers, pOffsets);
}

static VKAPI_ATTR void VKAPI_CALL tf3_CmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                                         VkIndexType type) {
    {
        PerfScope ps(kPerfState);
        CmdState& cs = CmdFor(cb);
        cs.ib = buffer;
        cs.ibOff = offset;
        cs.ibType = type;
    }
    GetDevice(Key(cb))->CmdBindIndexBuffer(cb, buffer, offset, type);
}

static VKAPI_ATTR void VKAPI_CALL tf3_CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout,
                                                       VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                                                       const void* pValues) {
    {
        PerfScope ps(kPerfState);
        CmdState& cs = CmdFor(cb);
        if ((uint64_t)offset + size <= sizeof cs.push) {
            std::memcpy(cs.push + offset, pValues, size);
            if (cs.pushLayout != layout) cs.pushCount = 0;
            cs.pushLayout = layout;
            uint32_t i = 0;
            while (i < cs.pushCount && !(cs.pushRanges[i].stages == stages && cs.pushRanges[i].offset == offset &&
                                         cs.pushRanges[i].size == size))
                ++i;
            if (i == cs.pushCount) {
                if (cs.pushCount < 4) ++cs.pushCount;
                else i = 3;
                cs.pushRanges[i] = {stages, offset, size};
            }
        }
    }
    GetDevice(Key(cb))->CmdPushConstants(cb, layout, stages, offset, size, pValues);
}

// skip_game_fsr=1: the game's FSR 1 passes (EASU, RCAS) draw an image that nobody sees while DLSS or
// FSR 3.1 runs (the blit shows ours instead), so their draws can be left out. Only while the upscaler
// ran last frame; if it then fails, the screen shows a stale image for that frame. Caller holds g_t.
bool SkipGameFsrDraw(const CmdState& cs) {
    return g_cfg.skipGameFsr && (cs.kind == Kind::FsrEasu || cs.kind == Kind::FsrRcas) && g_dlss.enabled &&
           g_lastDlssFrame != UINT64_MAX && g_lastDlssFrame + 1 >= g_frame;
}

// Everything the layer does before a game draw. Fills `plan` with the descriptor set to swap in for this
// draw (the blit), and returns false if the draw is to be left out. Most draws need no lock; g_t only
// for the draws OnDraw has work for, moving objects and the draw log.
bool BeforeDraw(DeviceData* d, VkCommandBuffer cb, bool indexed, uint32_t count, uint32_t instances, uint32_t first,
                int32_t vertexOffset, uint32_t firstInstance, SwapPlan& plan) {
    PerfScope ps(kPerfDraw);
    CmdState& cs = CmdFor(cb);
    ProbeDraw(cs, indexed, count, instances, first, vertexOffset, firstInstance);
    uint64_t mesh = ObjCandidate(cs, indexed, count, instances, first, vertexOffset);
    ApplyViewport(d, cb, cs);
    bool react = cs.pipe && cs.pipe->reactPipe && instances && g_hot.react.load(std::memory_order_relaxed);
#ifdef __APPLE__
    if (react && !cs.fbIsScene) {
        g_react.outside.fetch_add(1, std::memory_order_relaxed);
        react = false;
    }
#endif
    bool slow = mesh || react || cs.kind != Kind::Other || g_hot.drawLog.load(std::memory_order_relaxed) ||
                // OnDraw: still looking for the camera buffer, and a set was bound since the last look
                (cs.fbIsScene && !cs.hasView && !cs.sets.empty() && cs.viewGen != cs.setsGen) ||
                // OnDraw: after RCAS, the blit search or the swap itself
                (cs.rcasOut && (cs.blitGen != cs.setsGen || cs.blitHit.first >= 0));
    if (!slow) return true;
    TLock lk(g_t);
    LogSceneDraw(cb, cs, indexed, count, instances, firstInstance);
    if (mesh) ObjRecordDraw(cs, mesh, indexed, count, instances, first, vertexOffset, firstInstance);
#ifdef __APPLE__
    if (react) ReactRecordDraw(cs, indexed, count, instances, first, vertexOffset, firstInstance);
#endif
    plan = OnDraw(d, cs);
    if (cs.kind == Kind::FsrEasu) GpuMark(d, cb, kGmGameFsr);
    return !SkipGameFsrDraw(cs);
}

static VKAPI_ATTR void VKAPI_CALL tf3_CmdDraw(VkCommandBuffer cb, uint32_t vertexCount, uint32_t instanceCount,
                                              uint32_t firstVertex, uint32_t firstInstance) {
    DeviceData* d = GetDevice(Key(cb));
    SwapPlan plan;
    if (!BeforeDraw(d, cb, false, vertexCount, instanceCount, firstVertex, 0, firstInstance, plan)) return;
    if (plan.set) SwapBind(d, cb, plan);
    d->CmdDraw(cb, vertexCount, instanceCount, firstVertex, firstInstance);
    if (plan.set) SwapRestore(d, cb, plan);
}

static VKAPI_ATTR void VKAPI_CALL tf3_CmdDrawIndexed(VkCommandBuffer cb, uint32_t indexCount, uint32_t instanceCount,
                                                     uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    DeviceData* d = GetDevice(Key(cb));
    SwapPlan plan;
    if (!BeforeDraw(d, cb, true, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, plan)) return;
    if (plan.set) SwapBind(d, cb, plan);
    d->CmdDrawIndexed(cb, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    if (plan.set) SwapRestore(d, cb, plan);
}

// =============================================================================================
// Submission and present
// =============================================================================================
static VKAPI_ATTR VkResult VKAPI_CALL tf3_QueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits,
                                                      VkFence fence) {
    {
        PerfScope ps(kPerfSubmit);
        TLock lk(g_t);
        for (uint32_t s = 0; s < submitCount; ++s)
            for (uint32_t c = 0; c < pSubmits[s].commandBufferCount; ++c) {
                auto it = g_cmd.find(pSubmits[s].pCommandBuffers[c]);
                if (it == g_cmd.end()) continue;
                CmdState& cs = it->second;
                if (cs.mvPending && g_dlss.uboPtr) {
                    UploadMvMatrices(cs);
                    cs.mvPending = false;
                }
                ObjFillPrev(cs);
                ProbeCollect(cs);
            }
        if (LogFrame(g_frame) && g_lastViewLogFrame != g_frame)
            for (uint32_t s = 0; s < submitCount && g_lastViewLogFrame != g_frame; ++s)
                for (uint32_t c = 0; c < pSubmits[s].commandBufferCount; ++c) {
                    auto it = g_cmd.find(pSubmits[s].pCommandBuffers[c]);
                    if (it != g_cmd.end() && it->second.hasView) {
                        LogViewFromMemory(it->second);
                        g_lastViewLogFrame = g_frame;
                        break;
                    }
                }
    }
    DeviceData* d = GetDevice(Key(queue));
#ifdef __APPLE__
    std::vector<MfxJob> mfxJobs;
    {
        TLock lk(g_t);
        for (uint32_t s = 0; s < submitCount; ++s)
            for (uint32_t c = 0; c < pSubmits[s].commandBufferCount; ++c) {
                auto it = g_cmd.find(pSubmits[s].pCommandBuffers[c]);
                if (it == g_cmd.end() || !it->second.mfxPending) continue;
                MfxJob job;
                if (MfxPrepare(d, it->second, job)) mfxJobs.push_back(job);
            }
    }
#endif
    VkResult r = d->QueueSubmit(queue, submitCount, pSubmits, fence);
#ifdef __APPLE__
    if (!mfxJobs.empty()) {
        TLock lk(g_t);
        for (auto& job : mfxJobs) MfxCommit(job, r == VK_SUCCESS);
    }
#endif
    std::vector<CaptureFrame> toWrite;
    {
        TLock lk(g_t);
        toWrite.swap(g_cap.pending);
    }
    if (!toWrite.empty() && r == VK_SUCCESS) {
        d->QueueWaitIdle(queue);  // recording frames only: wait so the copies are complete
        {
            TLock lk(g_t);
            for (auto& f : toWrite) SnapshotCapture(f);
        }
        std::thread([frames = std::move(toWrite)] {
            for (const auto& f : frames) WriteCapture(f);
        }).detach();
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL tf3_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pInfo) {
    uint64_t presentFrame = 0;
    bool upscaledNow = false;
    {
        PerfScope ps(kPerfPresent);
        TLock lk(g_t);
        EnvTraceUpdate();
        if (g_frame < g_traceUntil && g_traceViewBuf) {
            if (const char* v = RefData(g_traceViewBuf, g_traceViewOff)) {
                float cam[3];
                std::memcpy(cam, v + ViewOff::camPos, 12);
                float dx = cam[0] - g_traceCamAtSubmit[0], dy = cam[1] - g_traceCamAtSubmit[1],
                      dz = cam[2] - g_traceCamAtSubmit[2];
                Log("[trace-present] frame %llu cam=(%.2f,%.2f,%.2f) changed since submit by %.3f, buffer %s+%llu",
                    (unsigned long long)g_frame, cam[0], cam[1], cam[2], std::sqrt(dx * dx + dy * dy + dz * dz),
                    H(g_traceViewBuf).c_str(), (unsigned long long)g_traceViewOff);
            }
            // Older frames' camera buffers: did the game write new data into them since we read them?
            char hist[256];
            int len = 0;
            for (uint64_t back = 1; back <= 6 && back <= g_frame; ++back) {
                const TraceCam& h = g_traceHist[(g_frame - back) % 8];
                if (h.frame != g_frame - back || !h.buf) continue;
                const char* v = RefData(h.buf, h.off);
                if (!v) continue;
                float c[3];
                std::memcpy(c, v + ViewOff::camPos, 12);
                float dd = std::sqrt((c[0] - h.cam[0]) * (c[0] - h.cam[0]) + (c[1] - h.cam[1]) * (c[1] - h.cam[1]) +
                                     (c[2] - h.cam[2]) * (c[2] - h.cam[2]));
                len += std::snprintf(hist + len, sizeof hist - len, " -%llu:%.2f", (unsigned long long)back, dd);
                if (len >= (int)sizeof hist - 32) break;
            }
            Log("[trace-reuse] frame %llu buffers of earlier frames changed by:%s", (unsigned long long)g_frame, hist);
        }
#ifdef _WIN32
        // Hotkeys, only while the game window has focus and only with debug_keys=1 in tf3_dlss.cfg (players
        // choose with the cfg, so a stray key press can't switch the upscaler off):
        //   Ctrl+Shift+D  upscaler on/off (compare with the game's own FSR 1)
        //   Ctrl+Shift+U  switch between DLSS and FSR 3.1 (when both work)
        //   Ctrl+Shift+M  motion vectors: camera + objects -> camera only -> off (until the cfg file changes)
        //   Ctrl+Shift+V  cycle debug views: off -> motion vectors -> jitter
        //   Ctrl+Shift+T  automated camera test with a per-frame trace; press again to cancel
        //   Ctrl+Shift+Y  automated small-tilt wobble test, records 8 frames in a row
        //   Ctrl+Shift+L  log every draw of one frame (pass, jitter, pipeline state, shader names)
        //   Ctrl+Shift+J  jitter on/off
        //   Ctrl+Shift+O  on-screen info box on/off
        //   Ctrl+Shift+Up/Down  sharpness of the running upscaler +/- 0.1 (sharpness / fsr_sharpness)
        //   Ctrl+Shift+K  jitter sign of the running upscaler: X+Y+ -> X-Y+ -> X+Y- -> X-Y-
        // Changes made with keys last until tf3_dlss.cfg is saved again.
        DWORD fgPid = 0;
        GetWindowThreadProcessId(GetForegroundWindow(), &fgPid);
        bool focused = fgPid == GetCurrentProcessId();
        bool ctrlShift = focused && (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SHIFT) & 0x8000);
        bool dev = g_cfg.debugKeys != 0;
        auto pressed = [&](int vk, bool& wasDown) {  // true once per key press
            bool down = ctrlShift && (GetAsyncKeyState(vk) & 0x8000);
            bool edge = down && !wasDown;
            wasDown = down;
            return edge;
        };
        auto devPressed = [&](int vk, bool& wasDown) { return pressed(vk, wasDown) && dev; };
        static bool dWasDown = false, mWasDown = false, vWasDown = false, tWasDown = false, jWasDown = false,
                    uWasDown = false;
        if (devPressed('U', uWasDown)) {
            // the last choice, not the running backend: that one doesn't change while the upscaler is off
            Backend now = g_dlss.override_ >= 0 ? (Backend)g_dlss.override_ : g_dlss.backend;
            Backend want = now == Backend::Fsr ? Backend::Dlss : Backend::Fsr;
            bool possible = want == Backend::Dlss ? (g_dlss.ngxReady && !g_dlss.failed) : (!g_fsr.failed && FsrLoad());
            if (possible) {
                g_dlss.override_ = (int)want;
                Log("[upscale] Ctrl+Shift+U: switching to %s (frame %llu)", BackendName(want), (unsigned long long)g_frame);
            } else
                Log("[upscale] Ctrl+Shift+U: %s isn't available (%s)", BackendName(want),
                    want == Backend::Dlss ? g_dlss.why.c_str() : g_fsr.why.c_str());
        }
        if (devPressed('J', jWasDown)) {
            g_cfg.jitter = g_cfg.jitter ? 0 : 1;
            g_dlss.resetNext = true;
            Log("[jitter] Ctrl+Shift+J: jitter %s (frame %llu)", g_cfg.jitter ? "ON" : "OFF", (unsigned long long)g_frame);
        }
        static bool oWasDown = false, upWasDown = false, downWasDown = false, kWasDown = false;
        if (devPressed('O', oWasDown)) {
            g_cfg.overlay = g_cfg.overlay ? 0 : 1;
            Log("[overlay] Ctrl+Shift+O: info box %s (frame %llu)", g_cfg.overlay ? "ON" : "OFF",
                (unsigned long long)g_frame);
        }
        bool up = devPressed(VK_UP, upWasDown), down = devPressed(VK_DOWN, downWasDown);
        if (up || down) {
            bool fsr = g_dlss.backend == Backend::Fsr;
            float& sh = fsr ? g_cfg.fsrSharpness : g_cfg.sharpness;
            float s = std::lround((sh + (up ? 0.1f : -0.1f)) * 10.f) / 10.f;
            sh = std::min(std::max(s, 0.f), 1.f);
            Log("[sharpen] Ctrl+Shift+%s: %s sharpness %.1f (frame %llu)", up ? "Up" : "Down", fsr ? "FSR" : "DLSS", sh,
                (unsigned long long)g_frame);
        }
        if (devPressed('K', kWasDown)) {  // cycle the sign pair of the upscaler that is running
            bool fsr = g_dlss.backend == Backend::Fsr;
            float& sx = fsr ? g_cfg.fsrJitterSignX : g_cfg.jitterSignX;
            float& sy = fsr ? g_cfg.fsrJitterSignY : g_cfg.jitterSignY;
            int i = (sx < 0 ? 1 : 0) + (sy < 0 ? 2 : 0);  // 0: ++, 1: -+, 2: +-, 3: --
            i = (i + 1) % 4;
            sx = (i & 1) ? -1.f : 1.f;
            sy = (i & 2) ? -1.f : 1.f;
            g_dlss.resetNext = true;
            Log("[jitter] Ctrl+Shift+K: %s jitter sign x=%g y=%g (frame %llu)", fsr ? "FSR" : "DLSS", sx, sy,
                (unsigned long long)g_frame);
        }
        static bool lWasDown = false;
        if (devPressed('L', lWasDown)) {
            g_drawLog = DrawLog{};
            g_drawLog.frame = g_frame + 2;  // a whole frame from its start
            Log("[draws] Ctrl+Shift+L: logging every draw of frame %llu", (unsigned long long)g_drawLog.frame);
        }
        static bool yWasDown = false;
        bool tHit = devPressed('T', tWasDown), yHit = devPressed('Y', yWasDown);
        if (tHit || yHit) {
            if (g_test.running || g_test.armed) StopTest("cancelled");
            else {
                g_test = AutoTest{};
                g_test.phases = tHit ? kTest : kWobbleTest;
                g_test.count = tHit ? (int)(sizeof kTest / sizeof kTest[0]) : (int)(sizeof kWobbleTest / sizeof kWobbleTest[0]);
                g_test.captureFrames = tHit ? 2 : kWobbleFrames;
                g_test.armed = true;
                Log("[test] Ctrl+Shift+%c: %s armed; starts when Ctrl/Shift are released", tHit ? 'T' : 'Y',
                    tHit ? "camera test" : "tilt wobble test");
            }
        }
        AutoTestStep(focused, (GetAsyncKeyState(VK_CONTROL) & 0x8000) || (GetAsyncKeyState(VK_SHIFT) & 0x8000));
        if (devPressed('D', dWasDown)) {
            g_dlss.enabled = !g_dlss.enabled;
            g_dlss.resetNext = true;
            Log("[upscale] Ctrl+Shift+D: upscaler %s (frame %llu)", g_dlss.enabled ? "ON" : "OFF",
                (unsigned long long)g_frame);
        }
        if (devPressed('V', vWasDown)) {
            g_cfg.debugView = (g_cfg.debugView + 1) % 3;
            static const char* names[] = {"off", "motion vectors", "jitter"};
            Log("[debug] Ctrl+Shift+V: debug view %s (frame %llu)", names[g_cfg.debugView], (unsigned long long)g_frame);
        }
        if (devPressed('M', mWasDown)) {  // cycle: camera + objects -> camera only -> off
            if (g_cfg.motionVectors && g_cfg.objectMotion) g_cfg.objectMotion = 0;
            else if (g_cfg.motionVectors) g_cfg.motionVectors = 0;
            else g_cfg.motionVectors = g_cfg.objectMotion = 1;
            Log("[mv] Ctrl+Shift+M: motion vectors %s (frame %llu)",
                !g_cfg.motionVectors ? "OFF" : g_cfg.objectMotion ? "camera + objects" : "camera only",
                (unsigned long long)g_frame);
        }
#endif
#ifdef __APPLE__
        // macOS hotkeys (debug_keys=1, game in front): Ctrl+Shift+D upscaler on/off (compare with the game's
        // FSR 1), Ctrl+Shift+O info box, Ctrl+Shift+Up/Down sharpness +/- 0.1, Ctrl+Shift+S sharpening RCAS <->
        // CAS, Ctrl+Shift+A motion-adaptive sharpening on/off, Ctrl+Shift+R reactive mask (on -> on in red -> off),
        // Ctrl+Shift+F frame generation on/off (when the
        // game started with frame_gen=1), Ctrl+Shift+G write a generated frame and its neighbours to mvcap/ (90
        // frames later). Like the cfg_live reload, they last until tf3_dlss.cfg is saved again.
        if (g_cfg.debugKeys && g_frame % 2 == 0) {
            enum { kA = 0, kS = 1, kD = 2, kF = 3, kG = 5, kR = 15, kO = 31, kUp = 126, kDown = 125, kShiftL = 56,
                   kShiftR = 60, kCtrlL = 59, kCtrlR = 62 };
            bool ctrlShift = (mfx_key_down(kCtrlL) || mfx_key_down(kCtrlR)) &&
                             (mfx_key_down(kShiftL) || mfx_key_down(kShiftR)) && mfx_app_focused();
            static bool dWas = false, oWas = false, upWas = false, downWas = false, sWas = false, aWas = false,
                        fWas = false, gWas = false, rWas = false;
            static float motionWas = 0.4f;  // the motion_sharpness Ctrl+Shift+A turns back on
            auto edge = [&](int key, bool& was) {
                bool down = ctrlShift && mfx_key_down(key);
                bool e = down && !was;
                was = down;
                return e;
            };
            if (edge(kD, dWas)) {
                g_dlss.enabled = !g_dlss.enabled;
                g_dlss.resetNext = true;
                Log("[upscale] Ctrl+Shift+D: upscaler %s (frame %llu)", g_dlss.enabled ? "ON" : "OFF",
                    (unsigned long long)g_frame);
            }
            if (edge(kO, oWas)) {
                g_cfg.overlay = g_cfg.overlay ? 0 : 1;
                Log("[overlay] Ctrl+Shift+O: info box %s", g_cfg.overlay ? "ON" : "OFF");
            }
            bool up = edge(kUp, upWas), dn = edge(kDown, downWas);
            if (up || dn) {
                g_cfg.sharpness = std::min(1.f, std::max(0.f, g_cfg.sharpness + (up ? 0.1f : -0.1f)));
                Log("[sharpen] Ctrl+Shift+%s: sharpness %.1f", up ? "Up" : "Down", g_cfg.sharpness);
            }
            if (edge(kS, sWas)) {
                g_cfg.sharpenType = g_cfg.sharpenType == 1 ? 0 : 1;
                Log("[sharpen] Ctrl+Shift+S: %s (frame %llu)", g_cfg.sharpenType == 1 ? "CAS-style" : "RCAS",
                    (unsigned long long)g_frame);
            }
            if (edge(kA, aWas)) {
                if (g_cfg.motionSharpness != 0.f) {
                    motionWas = g_cfg.motionSharpness;
                    g_cfg.motionSharpness = 0.f;
                } else g_cfg.motionSharpness = motionWas;
                Log("[sharpen] Ctrl+Shift+A: motion-adaptive sharpening %+.1f (frame %llu)", g_cfg.motionSharpness,
                    (unsigned long long)g_frame);
            }
            if (edge(kF, fWas)) {
                if (!g_fg.swapchain)
                    Log("[fg] Ctrl+Shift+F: the swapchain wasn't set up for frame generation; set frame_gen=1 in "
                        "tf3_dlss.cfg and restart the game");
                else {
                    g_cfg.frameGen = g_cfg.frameGen ? 0 : 1;
                    Log("[fg] Ctrl+Shift+F: frame generation %s (frame %llu)", g_cfg.frameGen ? "ON" : "OFF",
                        (unsigned long long)g_frame);
                }
            }
            if (edge(kR, rWas)) {  // reactive mask: on -> on + red tint -> off -> on
                if (g_cfg.reactive && !g_cfg.reactiveDebug) g_cfg.reactiveDebug = 1;
                else if (g_cfg.reactive) g_cfg.reactive = 0, g_cfg.reactiveDebug = 0;
                else g_cfg.reactive = 1;
                Log("[reactive] Ctrl+Shift+R: reactive mask %s (frame %llu)",
                    !g_cfg.reactive ? "OFF" : g_cfg.reactiveDebug ? "ON, shown in red" : "ON", (unsigned long long)g_frame);
            }
            if (edge(kG, gWas) && g_cfg.frameGen) {  // time to start moving the camera first
                g_fg.dumpKeyFrame = g_frame + 90;
                Log("[fg] Ctrl+Shift+G: writing the frames around frame %llu's generated frame (in about 3 s)",
                    (unsigned long long)g_fg.dumpKeyFrame);
            }
        }
#endif
        if (CaptureRequestedAt(g_frame)) {
            g_cap.requested = 2;
            g_cap.phase = "frame " + std::to_string(g_frame);
        }
        if (g_cfg.cfgLive && g_frame % 60 == 0 && ConfigChanged()) {
            int wasEnabled = g_cfg.enabled;
            UpscalerPref wasPref = g_cfg.upscaler;
            if (LoadConfig()) {
                if (g_cfg.enabled != wasEnabled) {
                    g_dlss.enabled = g_cfg.enabled != 0;
                    g_dlss.resetNext = true;
                    Log("[cfg] upscaler %s", g_dlss.enabled ? "ON" : "OFF");
                }
                if (g_cfg.upscaler != wasPref) g_dlss.override_ = -1;  // the cfg wins over Ctrl+Shift+U
            }
        }
        // TF3-DLSS-DID-NOT-START.txt: gone once an upscaled frame is shown, else says why not.
        if (!g_statusCleared) {
            if (g_lastSwapFrame == g_frame) {
                ClearStatus();
                Log("[upscale] first upscaled frame shown (frame %llu, %s)", (unsigned long long)g_frame,
                    BackendName(g_dlss.backend));
            } else if (g_frame % 120 == 0) {
                std::string why;
                if (!g_dlss.enabled) why = "the upscaler is switched off (enabled=0 in tf3_dlss.cfg, or Ctrl+Shift+D with debug_keys=1)";
                else if (g_dlss.dev && g_dlss.backend == Backend::None && g_firstFsrFrame) why = NoUpscalerReason();
                else if (!g_firstFsrFrame)
                    why = "the game hasn't run its own FSR 1 upscaling pass yet, which is the pass the mod replaces. "
                          "Check that the game's resolution scale (graphics options) is below 100%.";
                else why = "an upscaler is selected but no frame was upscaled yet; see the log";
                if (why != g_statusReason) {
                    WriteStatus(why);
                    Log("[status] not upscaling yet: %s", why.c_str());
                }
            }
        }
        ProbeReport();
        CountFrameForFps();
        if (LogFrame(g_frame)) {
            Log("[frame] %llu: dlss %s, blit swapped %s", (unsigned long long)g_frame,
                g_lastDlssFrame == g_frame ? "ran" : "did not run", g_lastSwapFrame == g_frame ? "yes" : "no");
#ifdef __APPLE__
            if (g_dlss.backend == Backend::MetalFx && mfx_gpu_ms() > 0.f)
                Log("[mfx] frame %llu: MetalFX GPU %.2f ms (median of the last frames), %ux%u -> %ux%u, %.1f fps",
                    (unsigned long long)g_frame, mfx_gpu_ms(), g_dlss.inW, g_dlss.inH, g_dlss.outW, g_dlss.outH, g_fps);
            if (g_cfg.frameGen)
                Log("[fg] frame %llu: %llu frames generated, %llu not (%s), interpolation GPU %.2f ms (median)",
                    (unsigned long long)g_frame, (unsigned long long)g_fg.generated, (unsigned long long)g_fg.skipped,
                    g_fg.skipWhy.empty() ? "-" : g_fg.skipWhy.c_str(), mfx_fg_gpu_ms());
#endif
        }
        presentFrame = g_frame;
        upscaledNow = g_lastDlssFrame == g_frame && g_lastSwapFrame == g_frame;
        ++g_frame;
        UpdateHot();
    }
    {
        TLock lk(g_t);
        PerfPresent();  // perf=1: frame times, and the report every perf_seconds
    }
#ifdef __APPLE__
    return FgPresent(GetDevice(Key(queue)), queue, pInfo, presentFrame, upscaledNow);
#else
    (void)presentFrame;
    (void)upscaledNow;
    return GetDevice(Key(queue))->QueuePresentKHR(queue, pInfo);
#endif
}

// =============================================================================================
// Proc-address plumbing
// =============================================================================================
struct HookEntry {
    const char* name;
    PFN_vkVoidFunction fn;
    size_t offset;
};

static const HookEntry kHooks[] = {
#define X(name) {"vk" #name, (PFN_vkVoidFunction)&tf3_##name, offsetof(DeviceData, name)},
    DEVICE_HOOKS(X)
#undef X
};

static PFN_vkVoidFunction DeviceHook(const char* name, DeviceData* d) {
    for (const auto& h : kHooks)
        if (!std::strcmp(name, h.name)) {
            if (d && !*reinterpret_cast<void**>(reinterpret_cast<char*>(d) + h.offset)) return nullptr;
            return h.fn;
        }
    return nullptr;
}

TF3_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL tf3_GetDeviceProcAddr(VkDevice device, const char* pName);

TF3_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL tf3_GetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (!std::strcmp(pName, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)&tf3_GetInstanceProcAddr;
    if (!std::strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)&tf3_GetDeviceProcAddr;
    if (!std::strcmp(pName, "vkCreateInstance")) return (PFN_vkVoidFunction)&tf3_CreateInstance;
    if (!std::strcmp(pName, "vkDestroyInstance")) return (PFN_vkVoidFunction)&tf3_DestroyInstance;
    if (!std::strcmp(pName, "vkCreateDevice")) return (PFN_vkVoidFunction)&tf3_CreateDevice;
    if (PFN_vkVoidFunction f = DeviceHook(pName, nullptr)) return f;
    if (!instance) return nullptr;
    InstanceData* d = GetInstance(Key(instance));
    return d ? d->GetInstanceProcAddr(instance, pName) : nullptr;
}

TF3_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL tf3_GetDeviceProcAddr(VkDevice device, const char* pName) {
    if (!std::strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)&tf3_GetDeviceProcAddr;
    DeviceData* d = GetDevice(Key(device));
    if (!d) return nullptr;
    if (PFN_vkVoidFunction f = DeviceHook(pName, d)) return f;
    return d->GetDeviceProcAddr(device, pName);
}

// Log header: what's running where, for bug reports.
static void LogHeader() {
    Log("TF3-DLSS %s (layer %s, built %s %s)", TF3_DLSS_VERSION, LAYER_NAME, __DATE__, __TIME__);
#ifdef _WIN32
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::string ver = "?";
    DWORD dummy = 0, size = GetFileVersionInfoSizeW(exe, &dummy);
    if (size) {
        std::vector<char> buf(size);
        VS_FIXEDFILEINFO* fi = nullptr;
        UINT len = 0;
        if (GetFileVersionInfoW(exe, 0, size, buf.data()) &&
            VerQueryValueW(buf.data(), L"\\", (void**)&fi, &len) && fi && len) {
            char b[64];
            std::snprintf(b, sizeof b, "%u.%u.%u.%u", HIWORD(fi->dwFileVersionMS), LOWORD(fi->dwFileVersionMS),
                          HIWORD(fi->dwFileVersionLS), LOWORD(fi->dwFileVersionLS));
            ver = b;
        }
    }
    Log("game: %s (version %s)", Narrow(exe).c_str(), ver.c_str());
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    OSVERSIONINFOW os{sizeof(OSVERSIONINFOW)};
    if (HMODULE nt = GetModuleHandleW(L"ntdll.dll"))
        if (auto fn = (RtlGetVersionFn)(void*)GetProcAddress(nt, "RtlGetVersion")) fn(&os);
    Log("Windows %lu.%lu build %lu", (unsigned long)os.dwMajorVersion, (unsigned long)os.dwMinorVersion,
        (unsigned long)os.dwBuildNumber);
#elif defined(__APPLE__)
    char exe[4096] = {};
    uint32_t exeSize = sizeof exe;
    if (_NSGetExecutablePath(exe, &exeSize) != 0) exe[0] = 0;
    Log("game: %s", exe);
    char osVer[64] = {}, model[64] = {};
    size_t n = sizeof osVer;
    sysctlbyname("kern.osproductversion", osVer, &n, nullptr, 0);
    n = sizeof model;
    sysctlbyname("hw.model", model, &n, nullptr, 0);
    Log("macOS %s, %s", osVer, model);
#endif
    Log("mod folder: %s", Narrow(RootDir()).c_str());
    Log("log: %s", Narrow(g_logPath).c_str());
}

TF3_EXPORT VKAPI_ATTR VkResult VKAPI_CALL tf3_NegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* p) {
    if (!p || p->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
    // Installed as an implicit layer, the loader offers us to every Vulkan program. Outside the game,
    // fail the negotiation (before touching any file): the loader then leaves the layer out.
    if (!IsGameProcess()) return VK_ERROR_INITIALIZATION_FAILED;
    if (p->loaderLayerInterfaceVersion < 2) return VK_ERROR_INITIALIZATION_FAILED;
    p->pfnGetInstanceProcAddr = &tf3_GetInstanceProcAddr;
    p->pfnGetDeviceProcAddr = &tf3_GetDeviceProcAddr;
    p->pfnGetPhysicalDeviceProcAddr = nullptr;
    p->loaderLayerInterfaceVersion = 2;
    static bool first = true;
    if (first) {
        first = false;
        LogHeader();
        LoadConfig();
        WriteStatus("The game loaded the layer, but hasn't drawn its 3D scene through it yet (it was closed or "
                    "crashed while loading, or the game's Vulkan device couldn't be created).");
    }
    Log("[layer] negotiated with the Vulkan loader (interface v%u)", p->loaderLayerInterfaceVersion);
    return VK_SUCCESS;
}
