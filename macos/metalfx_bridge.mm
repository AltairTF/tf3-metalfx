// MetalFX Temporal bridge (see metalfx_bridge.h). Built with ARC.
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <CoreVideo/CoreVideo.h>
#include <mach/mach_time.h>
#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <sys/resource.h>
#include <unistd.h>
#include "metalfx_bridge.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr int kSlots = 16;
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLFXTemporalScaler> g_scaler = nil;
id<MTLCommandBuffer> g_slotCb[kSlots] = {};
// frame generation: the interpolator is id<MTLFXFrameInterpolator> (macOS 26+; the build targets 13)
constexpr int kFgSlots = 8;
id g_fg = nil;
id<MTLCommandBuffer> g_fgSlotCb[kFgSlots] = {};

// GPU timings, written by the command buffers' completion handlers (Metal's threads). Medians, not means:
// when the game keeps the GPU busy, the GPU interleaves its work with ours, and now and then a command buffer's
// start-to-end time covers some of the game's work too (seen in game: 3.4 ms typical, single frames up to 85 ms).
constexpr int kRecent = 63;        // odd: one middle value
constexpr size_t kWinCap = 16384;  // the window's median then uses the first ones (the sum and maximum stay exact)
struct GpuTimes {
    std::mutex lock;
    double recent[kRecent] = {};   // last command buffers, for the overlay
    int recentN = 0, recentPos = 0;
    std::vector<double> win;       // since the last Window() call
    double winSum = 0.0, winMax = 0.0;
    uint32_t winCount = 0;
};
GpuTimes g_scalerTimes, g_fgTimes;

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

void AddGpuTime(GpuTimes& t, double ms) {
    std::lock_guard<std::mutex> lk(t.lock);
    t.recent[t.recentPos] = ms;
    t.recentPos = (t.recentPos + 1) % kRecent;
    if (t.recentN < kRecent) ++t.recentN;
    if (t.win.size() < kWinCap) t.win.push_back(ms);
    t.winSum += ms;
    if (ms > t.winMax) t.winMax = ms;
    ++t.winCount;
}

void ResetRecent(GpuTimes& t) {
    std::lock_guard<std::mutex> lk(t.lock);
    t.recentN = t.recentPos = 0;
}

float RecentMedian(GpuTimes& t) {
    double v[kRecent];
    int n;
    {
        std::lock_guard<std::mutex> lk(t.lock);
        n = t.recentN;
        std::copy(t.recent, t.recent + n, v);
    }
    return (float)Median(v, (size_t)n);
}

void Window(GpuTimes& t, double* medianMs, double* sumMs, double* maxMs, uint32_t* count) {
    std::vector<double> times;
    {
        std::lock_guard<std::mutex> lk(t.lock);
        times.swap(t.win);
        if (sumMs) *sumMs = t.winSum;
        if (maxMs) *maxMs = t.winMax;
        if (count) *count = t.winCount;
        t.winSum = t.winMax = 0.0;
        t.winCount = 0;
        t.win.reserve(1024);
    }
    if (medianMs) *medianMs = Median(times.data(), times.size());
}

// Times a committed-to-be command buffer into t (only if it completes without error).
void TimeCommandBuffer(id<MTLCommandBuffer> cb, GpuTimes* t) {
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        double ms = (done.GPUEndTime - done.GPUStartTime) * 1000.0;
        if (done.status == MTLCommandBufferStatusCompleted && ms > 0.0 && ms < 1000.0) AddGpuTime(*t, ms);
    }];
}
}  // namespace

float mfx_gpu_ms(void) { return RecentMedian(g_scalerTimes); }

void mfx_gpu_window(double* medianMs, double* sumMs, double* maxMs, uint32_t* count) {
    Window(g_scalerTimes, medianMs, sumMs, maxMs, count);
}

float mfx_fg_gpu_ms(void) { return RecentMedian(g_fgTimes); }

void mfx_fg_gpu_window(double* medianMs, double* sumMs, double* maxMs, uint32_t* count) {
    Window(g_fgTimes, medianMs, sumMs, maxMs, count);
}

float mfx_max_scale(void) {
    static float cached = 0.f;
    if (cached > 0.f) return cached;
    id<MTLDevice> dev = g_device ? g_device : MTLCreateSystemDefaultDevice();
    if (!dev) return 2.f;
    if (@available(macOS 14.0, *)) cached = [MTLFXTemporalScalerDescriptor supportedInputContentMaxScaleForDevice:dev];
    if (cached < 1.f) cached = 2.f;  // macOS 13: no query; 2x is what MetalFX supported there
    return cached;
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
    for (int i = 0; i < kFgSlots; ++i) {
        if (g_fgSlotCb[i]) [g_fgSlotCb[i] waitUntilCompleted];
        g_fgSlotCb[i] = nil;
    }
    g_fg = nil;
    g_scaler = nil;
    g_queue = nil;
    g_device = nil;
}

bool mfx_reactive_supported(void) {
    if (@available(macOS 14.4, *)) return true;
    return false;
}

bool mfx_configure(uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH, uint32_t colorVkFormat,
                   bool reactiveMask, char* err, int errLen) {
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
    if (@available(macOS 14.4, *)) {
        if (reactiveMask) {
            desc.reactiveMaskTextureEnabled = YES;
            desc.reactiveMaskTextureFormat = MTLPixelFormatR32Float;
        }
    }
    g_scaler = [desc newTemporalScalerWithDevice:g_device];
    ResetRecent(g_scalerTimes);  // new sizes: start the overlay's median again
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

bool mfx_fg_supported(void) {
    if (!g_device) return false;
    if (@available(macOS 26.0, *)) return [MTLFXFrameInterpolatorDescriptor supportsDevice:g_device];
    return false;
}

bool mfx_fg_configure(uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH, uint32_t swapVkFormat, char* err,
                      int errLen) {
    if (!mfx_fg_supported()) {
        SetErr(err, errLen, "MetalFX frame interpolation needs macOS 26 and a GPU that supports it");
        return false;
    }
    MTLPixelFormat swapFmt;
    switch (swapVkFormat) {
        case 44: swapFmt = MTLPixelFormatBGRA8Unorm; break;  // VK_FORMAT_B8G8R8A8_UNORM
        case 37: swapFmt = MTLPixelFormatRGBA8Unorm; break;  // VK_FORMAT_R8G8B8A8_UNORM
        default:
            if (err && errLen > 0)
                std::snprintf(err, (size_t)errLen, "swapchain format VkFormat %u not supported", swapVkFormat);
            return false;
    }
    for (int i = 0; i < kFgSlots; ++i)
        if (g_fgSlotCb[i]) [g_fgSlotCb[i] waitUntilCompleted];
    g_fg = nil;
    ResetRecent(g_fgTimes);
    if (@available(macOS 26.0, *)) {
        MTLFXFrameInterpolatorDescriptor* desc = [MTLFXFrameInterpolatorDescriptor new];
        desc.colorTextureFormat = MTLPixelFormatRGBA8Unorm;  // the upscaler's output
        desc.outputTextureFormat = swapFmt;
        desc.uiTextureFormat = swapFmt;                      // a copy of the swapchain image
        desc.depthTextureFormat = MTLPixelFormatR32Float;
        desc.motionTextureFormat = MTLPixelFormatRG16Float;
        desc.inputWidth = inW;
        desc.inputHeight = inH;
        desc.outputWidth = outW;
        desc.outputHeight = outH;
        g_fg = [desc newFrameInterpolatorWithDevice:g_device];
    }
    if (!g_fg) {
        SetErr(err, errLen, "newFrameInterpolatorWithDevice returned nil (unsupported sizes or formats)");
        return false;
    }
    return true;
}

void mfx_fg_slot_wait(int slot) {
    if (slot < 0 || slot >= kFgSlots) return;
    if (g_fgSlotCb[slot]) [g_fgSlotCb[slot] waitUntilCompleted];
    g_fgSlotCb[slot] = nil;
}

bool mfx_fg_encode(int slot, const MfxFgFrame* f, char* err, int errLen) {
    if (!g_queue || slot < 0 || slot >= kFgSlots || !f) {
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
    cb.label = @"TF3 MetalFX frame interpolation";
    [cb encodeWaitForEvent:evIn value:f->waitValue];
    id<MTLTexture> color = (__bridge id<MTLTexture>)f->color, prev = (__bridge id<MTLTexture>)f->prevColor,
                   depth = (__bridge id<MTLTexture>)f->depth, motion = (__bridge id<MTLTexture>)f->motion,
                   ui = (__bridge id<MTLTexture>)f->ui, output = (__bridge id<MTLTexture>)f->output;
    bool ok = false;
    if (@available(macOS 26.0, *)) {
        id<MTLFXFrameInterpolator> fi = (id<MTLFXFrameInterpolator>)g_fg;
        if (fi && color && prev && depth && motion && ui && output) {
            fi.colorTexture = color;
            fi.prevColorTexture = prev;
            fi.depthTexture = depth;
            fi.motionTexture = motion;
            fi.uiTexture = ui;
            fi.uiTextureComposited = YES;  // ui = the finished frame; MetalFX takes the UI out of it
            fi.outputTexture = output;
            fi.motionVectorScaleX = f->mvScaleX;
            fi.motionVectorScaleY = f->mvScaleY;
            fi.jitterOffsetX = f->jitterX;
            fi.jitterOffsetY = f->jitterY;
            fi.deltaTime = f->deltaTime;
            fi.nearPlane = f->nearPlane;
            fi.farPlane = f->farPlane;
            fi.fieldOfView = f->fovDegrees;
            fi.aspectRatio = f->aspect;
            fi.shouldResetHistory = f->reset;
            fi.depthReversed = f->depthReversed;
            [fi encodeToCommandBuffer:cb];
            ok = true;
        }
    }
    if (!ok) {
        SetErr(err, errLen, "frame interpolator or textures missing; showing this frame twice");
        if (ui && output && ui.pixelFormat == output.pixelFormat && ui.width == output.width &&
            ui.height == output.height) {
            id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
            [be copyFromTexture:ui toTexture:output];
            [be endEncoding];
        }
    }
    // Always signal, or the game's GPU work would wait forever.
    [cb encodeSignalEvent:evOut value:f->signalValue];
    if (ok) TimeCommandBuffer(cb, &g_fgTimes);
    [cb commit];
    g_fgSlotCb[slot] = cb;
    return ok;
}

bool mfx_key_down(int macKeyCode) {
    return CGEventSourceKeyState(kCGEventSourceStateCombinedSessionState, (CGKeyCode)macKeyCode);
}

namespace {
std::atomic<uint64_t> g_vsyncNs{0}, g_vsyncPeriodNs{0};
uint64_t HostToNs(uint64_t host) {
    static mach_timebase_info_data_t tb = [] {
        mach_timebase_info_data_t t{};
        mach_timebase_info(&t);
        return t;
    }();
    return host * tb.numer / tb.denom;
}
CVReturn VsyncCallback(CVDisplayLinkRef, const CVTimeStamp* now, const CVTimeStamp*, CVOptionFlags, CVOptionFlags*,
                       void*) {
    if (now->flags & kCVTimeStampHostTimeValid) g_vsyncNs.store(HostToNs(now->hostTime), std::memory_order_release);
    if ((now->flags & kCVTimeStampVideoRefreshPeriodValid) && now->videoTimeScale > 0)
        g_vsyncPeriodNs.store((uint64_t)((double)now->videoRefreshPeriod * 1e9 / (double)now->videoTimeScale),
                              std::memory_order_release);
    return kCVReturnSuccess;
}
}  // namespace

bool mfx_vsync(uint64_t* lastVsyncNs, uint64_t* periodNs) {
    static CVDisplayLinkRef link = [] {
        CVDisplayLinkRef l = nullptr;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"  // CVDisplayLink: deprecated in macOS 15, still works
        if (CVDisplayLinkCreateWithCGDisplay(CGMainDisplayID(), &l) == kCVReturnSuccess && l) {
            CVDisplayLinkSetOutputCallback(l, VsyncCallback, nullptr);
            CVDisplayLinkStart(l);
        }
#pragma clang diagnostic pop
        return l;
    }();
    uint64_t v = g_vsyncNs.load(std::memory_order_acquire), p = g_vsyncPeriodNs.load(std::memory_order_acquire);
    if (!link || !v || p < 1000000) return false;
    *lastVsyncNs = v;
    *periodNs = p;
    return true;
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
        if (@available(macOS 14.4, *)) g_scaler.reactiveMaskTexture = (__bridge id<MTLTexture>)f->reactive;
        [g_scaler encodeToCommandBuffer:cb];
    } else {
        SetErr(err, errLen, "MetalFX scaler or textures missing; output not written this frame");
        ok = false;
    }
    // Always signal, or the game's GPU work would wait forever.
    [cb encodeSignalEvent:evOut value:f->signalValue];
    if (ok) TimeCommandBuffer(cb, &g_scalerTimes);
    [cb commit];
    g_slotCb[slot] = cb;
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// perf=1 system sampling
namespace {
dispatch_source_t g_gpuTimer = nil;
std::atomic<int64_t> g_gpuBusySum{0}, g_gpuBusyN{0};

int ReadGpuBusy() {
    io_iterator_t it = 0;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) != KERN_SUCCESS)
        return -1;
    int busy = -1;
    for (io_object_t svc; (svc = IOIteratorNext(it)) != 0; IOObjectRelease(svc)) {
        CFTypeRef st = IORegistryEntryCreateCFProperty(svc, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0);
        if (!st) continue;
        if (CFGetTypeID(st) == CFDictionaryGetTypeID()) {
            CFTypeRef n = CFDictionaryGetValue((CFDictionaryRef)st, CFSTR("Device Utilization %"));
            int v = 0;
            if (n && CFGetTypeID(n) == CFNumberGetTypeID() && CFNumberGetValue((CFNumberRef)n, kCFNumberIntType, &v))
                busy = std::max(busy, v);
        }
        CFRelease(st);
    }
    IOObjectRelease(it);
    return busy;
}

struct ThreadSample {
    double cpuSec;
    std::string name;
};
std::mutex g_sysLock;
std::unordered_map<uint64_t, double> g_threadCpu;  // thread id -> CPU seconds at the last window
double g_sysWallSec = 0.0, g_sysProcSec = 0.0;

double WallSec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double ProcessCpuSec() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e6;
}

std::unordered_map<uint64_t, ThreadSample> SampleThreads() {
    std::unordered_map<uint64_t, ThreadSample> out;
    thread_act_array_t list = nullptr;
    mach_msg_type_number_t count = 0;
    if (task_threads(mach_task_self(), &list, &count) != KERN_SUCCESS) return out;
    for (mach_msg_type_number_t i = 0; i < count; ++i) {
        thread_identifier_info_data_t id{};
        mach_msg_type_number_t n = THREAD_IDENTIFIER_INFO_COUNT;
        thread_extended_info_data_t ext{};
        mach_msg_type_number_t m = THREAD_EXTENDED_INFO_COUNT;
        if (thread_info(list[i], THREAD_IDENTIFIER_INFO, (thread_info_t)&id, &n) == KERN_SUCCESS &&
            thread_info(list[i], THREAD_EXTENDED_INFO, (thread_info_t)&ext, &m) == KERN_SUCCESS) {
            ext.pth_name[sizeof ext.pth_name - 1] = 0;
            out[id.thread_id] = {(ext.pth_user_time + ext.pth_system_time) / 1e9,
                                 ext.pth_name[0] ? ext.pth_name : "thread " + std::to_string(id.thread_id)};
        }
        mach_port_deallocate(mach_task_self(), list[i]);
    }
    vm_deallocate(mach_task_self(), (vm_address_t)list, count * sizeof(thread_act_t));
    return out;
}
}  // namespace

void mfx_sys_start(void) {
    std::lock_guard<std::mutex> lk(g_sysLock);
    g_threadCpu.clear();
    for (auto& [tid, t] : SampleThreads()) g_threadCpu[tid] = t.cpuSec;
    g_sysWallSec = WallSec();
    g_sysProcSec = ProcessCpuSec();
    g_gpuBusySum = 0;
    g_gpuBusyN = 0;
    if (g_gpuTimer) return;
    g_gpuTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
    dispatch_source_set_timer(g_gpuTimer, dispatch_time(DISPATCH_TIME_NOW, 0), 250 * NSEC_PER_MSEC, 50 * NSEC_PER_MSEC);
    dispatch_source_set_event_handler(g_gpuTimer, ^{
        int v = ReadGpuBusy();
        if (v < 0) return;
        g_gpuBusySum += v;
        g_gpuBusyN += 1;
    });
    dispatch_resume(g_gpuTimer);
}

void mfx_sys_window(double* gpuBusyPct, double* processCpuPct, double* topThreadPct, char* topThreads,
                    int topThreadsLen) {
    std::lock_guard<std::mutex> lk(g_sysLock);
    int64_t n = g_gpuBusyN.exchange(0), sum = g_gpuBusySum.exchange(0);
    *gpuBusyPct = n ? (double)sum / (double)n : -1.0;
    double wall = WallSec(), proc = ProcessCpuSec();
    double dt = std::max(wall - g_sysWallSec, 1e-3);
    *processCpuPct = 100.0 * (proc - g_sysProcSec) / dt;
    g_sysWallSec = wall;
    g_sysProcSec = proc;

    std::vector<std::pair<double, std::string>> busy;
    std::unordered_map<uint64_t, double> now;
    for (auto& [tid, t] : SampleThreads()) {
        now[tid] = t.cpuSec;
        auto prev = g_threadCpu.find(tid);
        double d = t.cpuSec - (prev != g_threadCpu.end() ? prev->second : t.cpuSec);
        busy.push_back({100.0 * d / dt, t.name});
    }
    g_threadCpu = std::move(now);
    std::sort(busy.begin(), busy.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    *topThreadPct = busy.empty() ? 0.0 : busy[0].first;
    std::string text;
    for (size_t i = 0; i < busy.size() && i < 3; ++i) {
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %.0f%%", i ? ", " : "", busy[i].second.c_str(), busy[i].first);
        text += b;
    }
    if (topThreads && topThreadsLen > 0) std::snprintf(topThreads, (size_t)topThreadsLen, "%s", text.c_str());
}
