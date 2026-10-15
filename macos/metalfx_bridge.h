// MetalFX Temporal for the macOS build of the layer: a small C interface over the Objective-C API, so
// layer.cpp stays plain C++. All Metal objects cross this boundary as void* (id<...> bridged, not retained).
//
// Synchronisation with the game's Vulkan command buffer (MoltenVK) uses two MTLSharedEvents per frame slot,
// exported from VkEvents: the game's command buffer signals "in" after the upscaler's inputs are written and
// waits for "out" before anything reads the output. The MetalFX work runs on our own MTLCommandQueue, so the
// game's queue can't block it.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Creates the command queue on the game's MTLDevice. False if MetalFX Temporal isn't supported.
bool mfx_init(void* mtlDevice, char* err, int errLen);
void mfx_shutdown(void);

// (Re)creates the temporal scaler for these sizes. colorVkFormat: the VkFormat of the colour input
// (B10G11R11 float, RGBA16 float, RGBA8, BGRA8 or A2B10G10R10). Depth is R32 float, motion RG16 float,
// output RGBA8 unorm. reactiveMask: the scaler also takes a reactive mask (R32 float, render size; macOS 14.4+),
// which every MfxFrame must then provide.
bool mfx_configure(uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH, uint32_t colorVkFormat,
                   bool reactiveMask, char* err, int errLen);

// True if this macOS has MetalFX's reactive mask (14.4+).
bool mfx_reactive_supported(void);

// The largest upscale MetalFX Temporal accepts on this GPU, per axis (3 on the M4 with macOS 26/27).
float mfx_max_scale(void);

// Shared event helpers (the events come from vkExportMetalObjectsEXT).
uint64_t mfx_event_value(void* sharedEvent);
// Makes the value even ("not set" for MoltenVK's VkEvent), so the next vkCmdSetEvent / vkCmdWaitEvents
// encode value + 1. Returns the (even) value.
uint64_t mfx_event_unset(void* sharedEvent);

// Sets the event's value from the CPU (used to release the game's GPU work if MetalFX couldn't be encoded).
void mfx_event_set(void* sharedEvent, uint64_t value);

// Blocks until the MetalFX command buffer last committed for this slot has finished (or there is none).
void mfx_slot_wait(int slot);

struct MfxFrame {
    void* color;   // id<MTLTexture>
    void* depth;
    void* motion;
    void* output;
    void* reactive;          // reactive mask (R32 float, 0 = use the history, 1 = only this frame) or null
    void* eventIn;           // id<MTLSharedEvent>, signalled by the game's command buffer
    uint64_t waitValue;
    void* eventOut;          // id<MTLSharedEvent>, the game's command buffer waits for it
    uint64_t signalValue;
    float jitterX, jitterY;  // pixels
    float mvScaleX, mvScaleY;
    bool reset;
    bool depthReversed;
    uint32_t inW, inH;       // input content size
};
// Encodes wait(in) -> MetalFX -> signal(out) on our queue and commits it. If MetalFX can't run, it still
// signals "out" so the game never waits forever. False on error (err filled).
bool mfx_encode(int slot, const MfxFrame* f, char* err, int errLen);

// GPU time of the MetalFX command buffers, GPUStartTime to GPUEndTime. The GPU starts a command buffer only
// once the game's inputs are signalled, so the wait for them is not included (measured on the M4).
// When the game keeps the GPU busy, a command buffer's time now and then also covers some of the game's work, so
// the typical cost is the median.
// mfx_gpu_ms: median of the last 63 command buffers, in ms (0 until one finished).
float mfx_gpu_ms(void);
// Since the last call, for the perf=1 report: median, total and slowest ms, and how many command buffers.
void mfx_gpu_window(double* medianMs, double* sumMs, double* maxMs, uint32_t* count);

// Frame generation with MTLFXFrameInterpolator (macOS 26+). It runs at the game's present: from this frame's
// upscaled image (no UI), the last one, depth and motion it makes the frame halfway between them. The UI comes
// from a copy of the finished frame: with uiComposited, MetalFX separates the UI from the image below it.
bool mfx_fg_supported(void);
// (Re)creates the interpolator. Depth / motion inW x inH (R32 float / RG16 float), colour outW x outH RGBA8,
// UI and output outW x outH in the swapchain's format (BGRA8 or RGBA8 unorm, as a VkFormat).
bool mfx_fg_configure(uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH, uint32_t swapVkFormat, char* err,
                      int errLen);

struct MfxFgFrame {
    void* color;      // this frame without UI (the upscaler's output)
    void* prevColor;  // the last frame without UI
    void* depth;
    void* motion;
    void* ui;         // this frame as shown, UI included
    void* output;     // the frame in between
    void* eventIn;
    uint64_t waitValue;
    void* eventOut;
    uint64_t signalValue;
    float jitterX, jitterY;  // pixels
    float mvScaleX, mvScaleY;
    float deltaTime;         // seconds between the last frame and this one
    float nearPlane, farPlane, fovDegrees, aspect;
    bool reset;
    bool depthReversed;
};
// Encodes wait(in) -> interpolation -> signal(out) on our queue and commits it. If the interpolator can't
// run, it copies "ui" to "output" (the frame is shown twice) and still signals "out". False on error.
bool mfx_fg_encode(int slot, const MfxFgFrame* f, char* err, int errLen);
void mfx_fg_slot_wait(int slot);
// GPU time of the interpolation command buffers, like mfx_gpu_ms / mfx_gpu_window.
float mfx_fg_gpu_ms(void);
void mfx_fg_gpu_window(double* medianMs, double* sumMs, double* maxMs, uint32_t* count);

// perf=1: what limits the frame rate. GPU busy is the GPU driver's own statistic (IOAccelerator
// "PerformanceStatistics", "Device Utilization %", all apps together), read 4 times a second on a background queue.
// Thread CPU comes from the Mach thread list of this process.
void mfx_sys_start(void);
// Since the last call (or mfx_sys_start): average GPU busy % (-1 if unknown), the process's CPU % (100 = one core),
// the busiest thread's CPU % and the three busiest threads as text ("Main Thread 97%, ...").
void mfx_sys_window(double* gpuBusyPct, double* processCpuPct, double* topThreadPct, char* topThreads,
                    int topThreadsLen);

// Refresh clock of the main display (CVDisplayLink, started on first call): the time of the last refresh and the
// refresh period, both in ns of mach_absolute_time (clock_gettime_nsec_np(CLOCK_UPTIME_RAW)). False until the
// first refresh was seen (or if there is no display link).
bool mfx_vsync(uint64_t* lastVsyncNs, uint64_t* periodNs);

// Debug hotkeys on macOS (debug_keys=1): is this macOS virtual key held, and is our process the frontmost app.
bool mfx_key_down(int macKeyCode);
bool mfx_app_focused(void);

#ifdef __cplusplus
}
#endif
