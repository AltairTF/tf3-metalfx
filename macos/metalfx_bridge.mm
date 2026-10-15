// MetalFX Temporal bridge (see metalfx_bridge.h). Built with ARC.
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#include <unistd.h>
#include "metalfx_bridge.h"
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <vector>

namespace {
constexpr int kSlots = 16;
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLFXTemporalScaler> g_scaler = nil;
id<MTLCommandBuffer> g_slotCb[kSlots] = {};

// GPU timings, written by the command buffers' completion handlers (Metal's threads). Medians, not means:
// when the game keeps the GPU busy, the GPU interleaves its work with ours, and now and then a command buffer's
// start-to-end time covers some of the game's work too (seen in game: 3.4 ms typical, single frames up to 85 ms).
std::mutex g_timeLock;
constexpr int kRecent = 63;            // odd: one middle value
double g_recent[kRecent] = {};         // last command buffers, for the overlay
int g_recentN = 0, g_recentPos = 0;
std::vector<double> g_winTimes;        // since the last mfx_gpu_window call
constexpr size_t kWinCap = 16384;      // the median then uses the first ones (the sum and maximum stay exact)
double g_winSumMs = 0.0, g_winMaxMs = 0.0;
uint32_t g_winCount = 0;

void SetErr(char* err, int len, const char* msg) {
    if (err && len > 0) std::snprintf(err, (size_t)len, "%s", msg);
}

double Median(double* v, size_t n) {
    if (!n) return 0.0;
    std::nth_element(v, v + n / 2, v + n);
    double m = v[n / 2];
    if (n % 2 == 0) m = (m + *std::max_element(v, v + n / 2)) * 0.5;
    return m;
}

void AddGpuTime(double ms) {
    std::lock_guard<std::mutex> lk(g_timeLock);
    g_recent[g_recentPos] = ms;
    g_recentPos = (g_recentPos + 1) % kRecent;
    if (g_recentN < kRecent) ++g_recentN;
    if (g_winTimes.size() < kWinCap) g_winTimes.push_back(ms);
    g_winSumMs += ms;
    if (ms > g_winMaxMs) g_winMaxMs = ms;
    ++g_winCount;
}
}  // namespace

float mfx_gpu_ms(void) {
    double v[kRecent];
    int n;
    {
        std::lock_guard<std::mutex> lk(g_timeLock);
        n = g_recentN;
        std::copy(g_recent, g_recent + n, v);
    }
    return (float)Median(v, (size_t)n);
}

void mfx_gpu_window(double* medianMs, double* sumMs, double* maxMs, uint32_t* count) {
    std::vector<double> times;
    {
        std::lock_guard<std::mutex> lk(g_timeLock);
        times.swap(g_winTimes);
        if (sumMs) *sumMs = g_winSumMs;
        if (maxMs) *maxMs = g_winMaxMs;
        if (count) *count = g_winCount;
        g_winSumMs = g_winMaxMs = 0.0;
        g_winCount = 0;
        g_winTimes.reserve(1024);
    }
    if (medianMs) *medianMs = Median(times.data(), times.size());
}

bool mfx_init(void* mtlDevice, char* err, int errLen) {
    g_device = (__bridge id<MTLDevice>)mtlDevice;
    if (!g_device) {
        SetErr(err, errLen, "no MTLDevice from MoltenVK (VK_EXT_metal_objects missing?)");
        return false;
    }
    if (![MTLFXTemporalScalerDescriptor supportsDevice:g_device]) {
        SetErr(err, errLen, "MetalFX temporal scaling is not supported on this GPU");
        return false;
    }
    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        SetErr(err, errLen, "could not create a Metal command queue");
        return false;
    }
    g_queue.label = @"TF3 MetalFX";
    return true;
}

void mfx_shutdown(void) {
    for (int i = 0; i < kSlots; ++i) {
        if (g_slotCb[i]) [g_slotCb[i] waitUntilCompleted];
        g_slotCb[i] = nil;
    }
    g_scaler = nil;
    g_queue = nil;
    g_device = nil;
}

bool mfx_configure(uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH, uint32_t colorVkFormat, char* err,
                   int errLen) {
    if (!g_device) {
        SetErr(err, errLen, "MetalFX not initialised");
        return false;
    }
    MTLPixelFormat colorFmt;
    switch (colorVkFormat) {
        case 122: colorFmt = MTLPixelFormatRG11B10Float; break;   // VK_FORMAT_B10G11R11_UFLOAT_PACK32
        case 97: colorFmt = MTLPixelFormatRGBA16Float; break;     // VK_FORMAT_R16G16B16A16_SFLOAT
        case 37: colorFmt = MTLPixelFormatRGBA8Unorm; break;      // VK_FORMAT_R8G8B8A8_UNORM
        case 44: colorFmt = MTLPixelFormatBGRA8Unorm; break;      // VK_FORMAT_B8G8R8A8_UNORM
        case 64: colorFmt = MTLPixelFormatRGB10A2Unorm; break;    // VK_FORMAT_A2B10G10R10_UNORM_PACK32
        default:
            if (err && errLen > 0) std::snprintf(err, (size_t)errLen, "colour format VkFormat %u not supported", colorVkFormat);
            return false;
    }
    const MTLPixelFormat depthFmt = MTLPixelFormatR32Float, motionFmt = MTLPixelFormatRG16Float,
                         outFmt = MTLPixelFormatRGBA8Unorm;
    for (int i = 0; i < kSlots; ++i)
        if (g_slotCb[i]) [g_slotCb[i] waitUntilCompleted];
    MTLFXTemporalScalerDescriptor* desc = [MTLFXTemporalScalerDescriptor new];
    desc.colorTextureFormat = colorFmt;
    desc.depthTextureFormat = depthFmt;
    desc.motionTextureFormat = motionFmt;
    desc.outputTextureFormat = outFmt;
    desc.inputWidth = inW;
    desc.inputHeight = inH;
    desc.outputWidth = outW;
    desc.outputHeight = outH;
    desc.autoExposureEnabled = NO;  // the input is already tone-mapped
    g_scaler = [desc newTemporalScalerWithDevice:g_device];
    {
        std::lock_guard<std::mutex> lk(g_timeLock);
        g_recentN = g_recentPos = 0;  // new sizes: start the overlay's median again
    }
    if (!g_scaler) {
        SetErr(err, errLen, "newTemporalScalerWithDevice returned nil (unsupported sizes or formats)");
        return false;
    }
    return true;
}

uint64_t mfx_event_value(void* sharedEvent) {
    id<MTLSharedEvent> e = (__bridge id<MTLSharedEvent>)sharedEvent;
    return e ? e.signaledValue : 0;
}

uint64_t mfx_event_unset(void* sharedEvent) {
    id<MTLSharedEvent> e = (__bridge id<MTLSharedEvent>)sharedEvent;
    if (!e) return 0;
    uint64_t v = e.signaledValue;
    if (v & 1) {
        e.signaledValue = v + 1;
        v += 1;
    }
    return v;
}

void mfx_event_set(void* sharedEvent, uint64_t value) {
    id<MTLSharedEvent> e = (__bridge id<MTLSharedEvent>)sharedEvent;
    if (e && e.signaledValue < value) e.signaledValue = value;
}

bool mfx_key_down(int macKeyCode) {
    return CGEventSourceKeyState(kCGEventSourceStateCombinedSessionState, (CGKeyCode)macKeyCode);
}

bool mfx_app_focused(void) {
    NSRunningApplication* app = [NSWorkspace sharedWorkspace].frontmostApplication;
    return app && app.processIdentifier == getpid();
}

void mfx_slot_wait(int slot) {
    if (slot < 0 || slot >= kSlots) return;
    if (g_slotCb[slot]) [g_slotCb[slot] waitUntilCompleted];
    g_slotCb[slot] = nil;
}

bool mfx_encode(int slot, const MfxFrame* f, char* err, int errLen) {
    if (!g_queue || slot < 0 || slot >= kSlots || !f) {
        SetErr(err, errLen, "MetalFX not ready");
        return false;
    }
    id<MTLSharedEvent> evIn = (__bridge id<MTLSharedEvent>)f->eventIn;
    id<MTLSharedEvent> evOut = (__bridge id<MTLSharedEvent>)f->eventOut;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb || !evIn || !evOut) {
        SetErr(err, errLen, "no command buffer or events");
        return false;
    }
    cb.label = @"TF3 MetalFX temporal";
    [cb encodeWaitForEvent:evIn value:f->waitValue];
    bool ok = true;
    id<MTLTexture> color = (__bridge id<MTLTexture>)f->color, depth = (__bridge id<MTLTexture>)f->depth,
                   motion = (__bridge id<MTLTexture>)f->motion, output = (__bridge id<MTLTexture>)f->output;
    if (g_scaler && color && depth && motion && output) {
        g_scaler.colorTexture = color;
        g_scaler.depthTexture = depth;
        g_scaler.motionTexture = motion;
        g_scaler.outputTexture = output;
        g_scaler.inputContentWidth = f->inW;
        g_scaler.inputContentHeight = f->inH;
        g_scaler.jitterOffsetX = f->jitterX;
        g_scaler.jitterOffsetY = f->jitterY;
        g_scaler.motionVectorScaleX = f->mvScaleX;
        g_scaler.motionVectorScaleY = f->mvScaleY;
        g_scaler.reset = f->reset;
        g_scaler.depthReversed = f->depthReversed;
        [g_scaler encodeToCommandBuffer:cb];
    } else {
        SetErr(err, errLen, "MetalFX scaler or textures missing; output not written this frame");
        ok = false;
    }
    // Always signal, or the game's GPU work would wait forever.
    [cb encodeSignalEvent:evOut value:f->signalValue];
    if (ok)
        [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
            double ms = (done.GPUEndTime - done.GPUStartTime) * 1000.0;
            if (done.status == MTLCommandBufferStatusCompleted && ms > 0.0 && ms < 1000.0) AddGpuTime(ms);
        }];
    [cb commit];
    g_slotCb[slot] = cb;
    return ok;
}
