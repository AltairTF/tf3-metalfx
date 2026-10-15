// fake_tf3: a small Vulkan program that renders frames shaped like Transport Fever 3's, so the
// TF3-DLSS layer can be tested without the game.
//
// Frame: scene (instanced meshes with mat4 instAttrModel, camera UBO at set 0 binding 10, D24S8
// depth) -> tonemap (its framebuffer holds the depth) -> FSR 1 EASU -> RCAS -> blit to the
// swapchain -> UI panel. The EASU and RCAS stand-ins draw magenta: if the layer's upscaled image
// isn't swapped into the blit, the screen is magenta.
//
// The scene is a flat checkerboard wall facing the camera and a cube sliding sideways in front of
// it. The camera stands still, then pans sideways, then up, so the motion vectors the layer
// records (TF3_DLSS_CAPTURE) can be checked against exact values.
//
// After the run it checks (exit code 0 = all passed, 1 = a check failed, 2 = it couldn't run):
//   * the layer log says the upscaled image was shown,
//   * the last frame has no magenta (the blit swap worked) and the UI panel is on top,
//   * every recorded frame's motion vectors, camera only and with the object pass, match the
//     analytic motion of the wall and the cube to within 0.02 px (this includes "zero while
//     still", which only holds if the jitter is taken out correctly),
//   * no validation errors (with --validation and the Khronos validation layer enabled),
// and reports its own CPU time per frame for recording, which the layer's hooks add to.
//
// Usage: fake_tf3 [--out DIR] [--frames N] [--width W --height H] [--scale S] [--tiles X Y]
//                 [--threads N] [--window] [--validation] [--no-checks]
// The scripts (harness/run.sh, scripts/run_harness.ps1) enable the layer and set its variables.

#ifdef _WIN32
#include <windows.h>
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

const uint32_t kSceneVert[] = {
#include "scene_vert.spv.inc"
};
const uint32_t kSceneFrag[] = {
#include "scene_frag.spv.inc"
};
const uint32_t kFullscreenVert[] = {
#include "fullscreen_vert.spv.inc"
};
const uint32_t kTonemapFrag[] = {
#include "tonemap_frag.spv.inc"
};
const uint32_t kEasuFrag[] = {
#include "easu_frag.spv.inc"
};
const uint32_t kRcasFrag[] = {
#include "rcas_frag.spv.inc"
};
const uint32_t kBlitFrag[] = {
#include "blit_frag.spv.inc"
};
const uint32_t kUiFrag[] = {
#include "ui_frag.spv.inc"
};

// ---------------------------------------------------------------------------------------------
// Options and scenario
// ---------------------------------------------------------------------------------------------
struct Options {
    fs::path out = ".";
    uint32_t frames = 300;
    uint32_t width = 2048, height = 1152;  // output (swapchain) size
    float scale = 0.5f;                    // render size = scale * output size
    uint32_t tilesX = 64, tilesY = 24;     // the wall is drawn as tilesX * tilesY separate draws
    uint32_t threads = 4;                  // threads recording the scene (TF3 records on several)
    bool window = false, validation = false, checks = true;
    uint32_t msaa = 1;                     // --msaa N: scene + tonemap passes multisampled (TF3's MSAA option)
} g_opt;

// Camera and cube positions per frame. Captures are made away from the changes of motion.
constexpr float kNear = 1.f, kFar = 5000.f, kFovY = 1.0471976f;  // 60 degrees
constexpr float kWallZ = -20.f;    // the wall faces the camera at this depth
constexpr float kCubeZ = -12.f;    // cube centre; its front face is at view depth 11
constexpr float kPanSpeed = 0.05f, kCubeSpeed = 0.04f;
float CamX(uint32_t f) { return kPanSpeed * (float)std::clamp<int>((int)f - 100, 0, 80); }   // pans right on 100..180
float CamY(uint32_t f) { return kPanSpeed * (float)std::clamp<int>((int)f - 180, 0, 80); }   // pans up on 180..260
float CubeX(uint32_t f) {  // slides back and forth between -5 and 5, turning at frames 250, 500, ...
    float t = std::fmod(kCubeSpeed * (float)f, 20.f);
    return t < 10.f ? -5.f + t : 15.f - t;
}
const char* kCaptureFrames = "60,140,220";  // still, panning right, panning up

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------
[[noreturn]] void Die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "fake_tf3: ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::exit(2);
}
#define VKC(x)                                                     \
    do {                                                           \
        VkResult r_ = (x);                                         \
        if (r_ != VK_SUCCESS) Die("%s failed (%d)", #x, (int)r_); \
    } while (0)

// Column-major 4x4 matrices, as GLSL.
struct Mat4 {
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};
Mat4 Mul(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    return r;
}
Mat4 Translate(float x, float y, float z) {
    Mat4 r;
    r.m[12] = x, r.m[13] = y, r.m[14] = z;
    return r;
}
Mat4 Scale(float x, float y, float z) {
    Mat4 r;
    r.m[0] = x, r.m[5] = y, r.m[10] = z;
    return r;
}
Mat4 Perspective(float fovY, float aspect, float n, float f) {  // GL-style: NDC z in -1..1
    Mat4 r;
    float t = 1.f / std::tan(fovY / 2);
    std::memset(r.m, 0, sizeof r.m);
    r.m[0] = t / aspect;
    r.m[5] = t;
    r.m[10] = (f + n) / (n - f);
    r.m[11] = -1;
    r.m[14] = 2 * f * n / (n - f);
    return r;
}
Mat4 Inverse(const Mat4& a) {  // Gauss-Jordan in double
    double m[4][8];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) m[r][c] = a.m[c * 4 + r], m[r][c + 4] = r == c;
    for (int c = 0; c < 4; ++c) {
        int p = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::fabs(m[r][c]) > std::fabs(m[p][c])) p = r;
        for (int k = 0; k < 8; ++k) std::swap(m[c][k], m[p][k]);
        double d = m[c][c];
        for (int k = 0; k < 8; ++k) m[c][k] /= d;
        for (int r = 0; r < 4; ++r)
            if (r != c) {
                double f = m[r][c];
                for (int k = 0; k < 8; ++k) m[r][k] -= f * m[c][k];
            }
    }
    Mat4 out;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) out.m[c * 4 + r] = (float)m[r][c + 4];
    return out;
}

// TF3's camera block (set 0, binding 10), std140; the layer reads these offsets from mapped memory.
struct ViewUbo {
    Mat4 projView;         // 0
    Mat4 projViewInverse;  // 64
    Mat4 proj;             // 128
    Mat4 projInverse;      // 192
    Mat4 view;             // 256
    Mat4 viewInverse;      // 320
    float camPos[4];       // 384
    float pad0[4];         // 400
    float nearZ, farZ;     // 416, 420
    float pad1;            // 424
    int32_t width, height; // 428, 432
    float pad2[5];         // 436..455
    float fov;             // 456
    float pad3;            // 460
};
static_assert(sizeof(ViewUbo) == 464, "camera block size");

float HalfToFloat(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? INFINITY : std::ldexp((float)(m | 1024), (int)e - 25);
    return s ? -v : v;
}

std::string ReadText(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::vector<uint32_t> ReadWords(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<uint32_t> w(b.size() / 4);
    if (!w.empty()) std::memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}
double JsonNumber(const std::string& j, const char* key) {
    std::string k = std::string("\"") + key + "\":";
    size_t p = j.find(k);
    return p == std::string::npos ? NAN : std::atof(j.c_str() + p + k.size());
}

// ---------------------------------------------------------------------------------------------
// Vulkan setup
// ---------------------------------------------------------------------------------------------
struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};
struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* ptr = nullptr;
};

struct App {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkDevice dev = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapFmt = VK_FORMAT_UNDEFINED;
    VkExtent2D out{}, render{};
    std::vector<VkImage> swapImages;
    std::vector<VkImageView> swapViews;
    VkFormat depthFmt = VK_FORMAT_D24_UNORM_S8_UINT;
#ifdef _WIN32
    HWND hwnd = nullptr;
#endif
} g;

int g_validationErrors = 0, g_validationWarnings = 0;
VKAPI_ATTR VkBool32 VKAPI_CALL OnValidation(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT,
                                            const VkDebugUtilsMessengerCallbackDataEXT* d, void*) {
    bool err = sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    int& n = err ? g_validationErrors : g_validationWarnings;
    if (n++ < 20) std::fprintf(stderr, "[validation %s] %s\n", err ? "ERROR" : "warning", d->pMessage);
    return VK_FALSE;
}

uint32_t MemType(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g.gpu, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    Die("no memory type for %x", want);
}

Image MakeImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) {
    Image im;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = samples;
    ci.usage = usage;
    VKC(vkCreateImage(g.dev, &ci, nullptr, &im.image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(g.dev, im.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = MemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VKC(vkAllocateMemory(g.dev, &ai, nullptr, &im.mem));
    VKC(vkBindImageMemory(g.dev, im.image, im.mem, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = {aspect, 0, 1, 0, 1};
    VKC(vkCreateImageView(g.dev, &vi, nullptr, &im.view));
    return im;
}

// Host-visible and persistently mapped, as the game keeps its per-frame buffers.
Buffer MakeBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer b;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    VKC(vkCreateBuffer(g.dev, &ci, nullptr, &b.buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g.dev, b.buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        MemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VKC(vkAllocateMemory(g.dev, &ai, nullptr, &b.mem));
    VKC(vkBindBufferMemory(g.dev, b.buf, b.mem, 0));
    VKC(vkMapMemory(g.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.ptr));
    return b;
}

VkShaderModule Shader(const uint32_t* code, size_t bytes) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes;
    ci.pCode = code;
    VkShaderModule m;
    VKC(vkCreateShaderModule(g.dev, &ci, nullptr, &m));
    return m;
}

bool HasExt(const std::vector<VkExtensionProperties>& list, const char* name) {
    for (auto& e : list)
        if (!std::strcmp(e.extensionName, name)) return true;
    return false;
}

#ifdef _WIN32
LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_CLOSE) return 0;  // runs to the end on its own
    return DefWindowProcW(h, m, w, l);
}
#endif

void CreateInstanceAndSurface() {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());
    std::vector<const char*> want = {VK_KHR_SURFACE_EXTENSION_NAME};
#ifdef _WIN32
    bool headless = !g_opt.window && HasExt(exts, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
    if (!headless) want.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
    bool headless = true;
    if (g_opt.window) std::fprintf(stderr, "fake_tf3: --window is Windows only; running headless\n");
#endif
    if (headless) {
        if (!HasExt(exts, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME)) Die("the Vulkan driver has no VK_EXT_headless_surface");
        want.push_back(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
    }
    bool debug = g_opt.validation && HasExt(exts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debug) want.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    // MoltenVK is a portability driver: the loader only lists it with this extension + flag.
    bool portability = HasExt(exts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    if (portability) want.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "fake_tf3";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = (uint32_t)want.size();
    ci.ppEnabledExtensionNames = want.data();
    if (portability) ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    VKC(vkCreateInstance(&ci, nullptr, &g.instance));

    if (debug) {
        VkDebugUtilsMessengerCreateInfoEXT mi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        mi.pfnUserCallback = OnValidation;
        auto create = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(g.instance, "vkCreateDebugUtilsMessengerEXT");
        if (create) VKC(create(g.instance, &mi, nullptr, &g.messenger));
    } else if (g_opt.validation) {
        std::fprintf(stderr, "fake_tf3: VK_EXT_debug_utils not available; validation messages can't be counted\n");
    }

    if (headless) {
        VkHeadlessSurfaceCreateInfoEXT si{VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT};
        auto create = (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(g.instance, "vkCreateHeadlessSurfaceEXT");
        VKC(create(g.instance, &si, nullptr, &g.surface));
    }
#ifdef _WIN32
    else {
        WNDCLASSW wc{};
        wc.lpfnWndProc = WndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"fake_tf3";
        RegisterClassW(&wc);
        RECT r{0, 0, (LONG)g_opt.width, (LONG)g_opt.height};
        AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
        g.hwnd = CreateWindowW(L"fake_tf3", L"fake_tf3 (TF3-DLSS test harness)", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                               CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, nullptr, nullptr,
                               wc.hInstance, nullptr);
        VkWin32SurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        si.hinstance = wc.hInstance;
        si.hwnd = g.hwnd;
        VKC(vkCreateWin32SurfaceKHR(g.instance, &si, nullptr, &g.surface));
    }
#endif
}

void CreateDevice() {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g.instance, &n, nullptr);
    if (!n) Die("no Vulkan device");
    std::vector<VkPhysicalDevice> gpus(n);
    vkEnumeratePhysicalDevices(g.instance, &n, gpus.data());
    // a discrete GPU if there is one (the game's choice too), else the first
    g.gpu = gpus[0];
    for (auto p : gpus) {
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            g.gpu = p;
            break;
        }
    }
    vkGetPhysicalDeviceProperties(g.gpu, &g.props);

    vkGetPhysicalDeviceQueueFamilyProperties(g.gpu, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(g.gpu, &n, fams.data());
    g.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < n && g.qfam == UINT32_MAX; ++i) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(g.gpu, i, g.surface, &present);
        if ((fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && present)
            g.qfam = i;
    }
    if (g.qfam == UINT32_MAX) Die("no graphics queue that can present");

    float prio = 1.f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = g.qfam;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    std::vector<const char*> exts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    {   // portability drivers (MoltenVK) require VK_KHR_portability_subset to be enabled when offered
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(g.gpu, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> dexts(n);
        vkEnumerateDeviceExtensionProperties(g.gpu, nullptr, &n, dexts.data());
        if (HasExt(dexts, "VK_KHR_portability_subset")) exts.push_back("VK_KHR_portability_subset");
    }
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qi;
    ci.enabledExtensionCount = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    VKC(vkCreateDevice(g.gpu, &ci, nullptr, &g.dev));
    vkGetDeviceQueue(g.dev, g.qfam, 0, &g.queue);

    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(g.gpu, VK_FORMAT_D24_UNORM_S8_UINT, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
        g.depthFmt = VK_FORMAT_D32_SFLOAT_S8_UINT;  // AMD has no D24S8
}

void CreateSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    VKC(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g.gpu, g.surface, &caps));
    g.out = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : VkExtent2D{g_opt.width, g_opt.height};
    g.render = {(uint32_t)std::lround(g.out.width * g_opt.scale), (uint32_t)std::lround(g.out.height * g_opt.scale)};
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) Die("swapchain images can't be read back");

    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g.gpu, g.surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(g.gpu, g.surface, &n, fmts.data());
    VkSurfaceFormatKHR sf = fmts[0];
    for (auto f : fmts)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) {
            sf = f;
            break;
        }
    g.swapFmt = sf.format;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g.gpu, g.surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(g.gpu, g.surface, &n, modes.data());
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;  // uncapped if possible, to measure frame time
    for (auto m : modes)
        if (m == VK_PRESENT_MODE_IMMEDIATE_KHR || (m == VK_PRESENT_MODE_MAILBOX_KHR && mode == VK_PRESENT_MODE_FIFO_KHR))
            mode = m;

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = g.surface;
    ci.minImageCount = std::max(caps.minImageCount, 3u);
    if (caps.maxImageCount) ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
    ci.imageFormat = sf.format;
    ci.imageColorSpace = sf.colorSpace;
    ci.imageExtent = g.out;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;
    VKC(vkCreateSwapchainKHR(g.dev, &ci, nullptr, &g.swapchain));
    vkGetSwapchainImagesKHR(g.dev, g.swapchain, &n, nullptr);
    g.swapImages.resize(n);
    vkGetSwapchainImagesKHR(g.dev, g.swapchain, &n, g.swapImages.data());
    for (VkImage img : g.swapImages) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = g.swapFmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView v;
        VKC(vkCreateImageView(g.dev, &vi, nullptr, &v));
        g.swapViews.push_back(v);
    }
}

// One colour attachment, optionally the depth buffer as well. Every pass is followed by reads of
// what it wrote (sampling, the next pass's depth test, a readback copy).
enum class Load { Clear, KeepDepth, KeepAll };  // what the pass starts from
// samples > 1: colour and depth are multisampled and the colour is resolved into a third attachment
// (single-sample, colorFinal), as TF3 does with MSAA on.
VkRenderPass MakeRenderPass(VkFormat color, VkImageLayout colorFinal, bool withDepth, Load load,
                            VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) {
    const bool ms = samples != VK_SAMPLE_COUNT_1_BIT;
    const VkImageLayout resolveFinal = colorFinal;
    if (ms) colorFinal = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;  // the MS image is only ever an attachment
    VkAttachmentDescription att[3]{};
    att[0].format = color;
    att[0].samples = samples;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = colorFinal;
    if (load == Load::KeepAll) {
        att[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        att[0].initialLayout = colorFinal;
    }
    att[1] = att[0];
    att[1].format = g.depthFmt;
    bool clearDepth = load == Load::Clear;
    att[1].loadOp = att[1].stencilLoadOp = clearDepth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    att[1].storeOp = att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[1].initialLayout = clearDepth ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    att[2] = att[0];  // resolve target
    att[2].samples = VK_SAMPLE_COUNT_1_BIT;
    att[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[2].finalLayout = resolveFinal;
    VkAttachmentReference rref{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference dref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount = 1;
    sp.pColorAttachments = &cref;
    sp.pDepthStencilAttachment = withDepth ? &dref : nullptr;
    if (ms) sp.pResolveAttachments = &rref;
    const VkPipelineStageFlags kAll = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                      VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    const VkAccessFlags kWrites = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    const VkAccessFlags kReads = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_SHADER_READ_BIT;
    VkSubpassDependency dep[2]{};
    dep[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dep[0].dstSubpass = 0;
    dep[0].srcStageMask = kAll | VK_PIPELINE_STAGE_TRANSFER_BIT;
    dep[0].dstStageMask = kAll;
    dep[0].srcAccessMask = kWrites | VK_ACCESS_TRANSFER_WRITE_BIT;
    dep[0].dstAccessMask = kWrites | kReads;
    dep[1].srcSubpass = 0;
    dep[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dep[1].srcStageMask = kAll;
    dep[1].dstStageMask = kAll | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dep[1].srcAccessMask = kWrites;
    dep[1].dstAccessMask = kWrites | kReads | VK_ACCESS_TRANSFER_READ_BIT;
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    if (ms && !withDepth) Die("multisampled passes have a depth buffer here");
    ci.attachmentCount = ms ? 3 : withDepth ? 2 : 1;
    ci.pAttachments = att;
    ci.subpassCount = 1;
    ci.pSubpasses = &sp;
    ci.dependencyCount = 2;
    ci.pDependencies = dep;
    VkRenderPass rp;
    VKC(vkCreateRenderPass(g.dev, &ci, nullptr, &rp));
    return rp;
}

VkFramebuffer MakeFramebuffer(VkRenderPass rp, std::vector<VkImageView> views, VkExtent2D size) {
    VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    ci.renderPass = rp;
    ci.attachmentCount = (uint32_t)views.size();
    ci.pAttachments = views.data();
    ci.width = size.width;
    ci.height = size.height;
    ci.layers = 1;
    VkFramebuffer fb;
    VKC(vkCreateFramebuffer(g.dev, &ci, nullptr, &fb));
    return fb;
}

struct PipeDesc {
    const uint32_t* vs;
    size_t vsBytes;
    const uint32_t* fs;
    size_t fsBytes;
    VkPipelineLayout layout;
    VkRenderPass rp;
    bool scene = false;  // vertex input + depth test
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
};
VkPipeline MakePipeline(const PipeDesc& p) {
    VkShaderModule vs = Shader(p.vs, p.vsBytes), fs = Shader(p.fs, p.fsBytes);
    VkPipelineShaderStageCreateInfo st[2]{};
    st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main"};
    st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main"};
    VkVertexInputBindingDescription vb[2] = {{0, 12, VK_VERTEX_INPUT_RATE_VERTEX}, {1, 64, VK_VERTEX_INPUT_RATE_INSTANCE}};
    VkVertexInputAttributeDescription va[5] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                                               {1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0},
                                               {2, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16},
                                               {3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 32},
                                               {4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 48}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    if (p.scene) {
        vi.vertexBindingDescriptionCount = 2;
        vi.pVertexBindingDescriptions = vb;
        vi.vertexAttributeDescriptionCount = 5;
        vi.pVertexAttributeDescriptions = va;
    }
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = p.samples;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = ds.depthWriteEnable = p.scene;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState cb{};
    cb.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = 1;
    cbs.pAttachments = &cb;
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dyn;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2;
    ci.pStages = st;
    ci.pVertexInputState = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cbs;
    ci.pDynamicState = &dy;
    ci.layout = p.layout;
    ci.renderPass = p.rp;
    VkPipeline pipe;
    VKC(vkCreateGraphicsPipelines(g.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
    vkDestroyShaderModule(g.dev, vs, nullptr);
    vkDestroyShaderModule(g.dev, fs, nullptr);
    return pipe;
}

VkDescriptorSetLayout SetLayout(std::vector<VkDescriptorSetLayoutBinding> b) {
    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = (uint32_t)b.size();
    ci.pBindings = b.data();
    VkDescriptorSetLayout l;
    VKC(vkCreateDescriptorSetLayout(g.dev, &ci, nullptr, &l));
    return l;
}
VkPipelineLayout PipeLayout(std::vector<VkDescriptorSetLayout> sets, uint32_t pushBytes = 0) {
    VkPushConstantRange pr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, pushBytes};
    VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    ci.setLayoutCount = (uint32_t)sets.size();
    ci.pSetLayouts = sets.data();
    ci.pushConstantRangeCount = pushBytes ? 1 : 0;
    ci.pPushConstantRanges = &pr;
    VkPipelineLayout l;
    VKC(vkCreatePipelineLayout(g.dev, &ci, nullptr, &l));
    return l;
}

// Threads that record parts of the scene; Run() returns when all have finished.
class Workers {
public:
    explicit Workers(uint32_t n) : n_(n) {
        for (uint32_t i = 0; i < n; ++i) threads_.emplace_back([this, i] { Loop(i); });
    }
    ~Workers() {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
        }
        go_.notify_all();
        for (auto& t : threads_) t.join();
    }
    void Run(const std::function<void(uint32_t)>& fn) {
        std::unique_lock<std::mutex> lk(m_);
        fn_ = &fn;
        left_ = n_;
        ++gen_;
        go_.notify_all();
        done_.wait(lk, [&] { return left_ == 0; });
    }

private:
    void Loop(uint32_t i) {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(uint32_t)>* fn;
            {
                std::unique_lock<std::mutex> lk(m_);
                go_.wait(lk, [&] { return quit_ || gen_ != seen; });
                if (quit_) return;
                seen = gen_;
                fn = fn_;
            }
            (*fn)(i);
            std::lock_guard<std::mutex> lk(m_);
            if (--left_ == 0) done_.notify_one();
        }
    }
    uint32_t n_;
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable go_, done_;
    const std::function<void(uint32_t)>* fn_ = nullptr;
    uint32_t left_ = 0;
    uint64_t gen_ = 0;
    bool quit_ = false;
};

// ---------------------------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------------------------
struct Check {
    std::string name;
    bool pass;
    std::string detail;
};
std::vector<Check> g_checks;
void AddCheck(const std::string& name, bool pass, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    g_checks.push_back({name, pass, buf});
}

// Compare one recorded frame's motion vectors with the exact motion of the wall and the cube.
// Every scene movement is a translation along x or y, so a surface point keeps its view depth z and
// moves on screen by (P00 * W/2 * dx / z, -P11 * H/2 * dy / z) render pixels (rows top-down), where
// dx, dy is how far the point moved relative to the camera. The vectors point to the previous
// position: previous - current. z comes from the recorded depth buffer.
void CheckCapture(const fs::path& json, const Mat4& proj) {
    std::string j = ReadText(json);
    std::string base = json.string();
    base.resize(base.size() - 5);
    uint32_t frame = (uint32_t)JsonNumber(j, "frame"), w = (uint32_t)JsonNumber(j, "file_w"),
             h = (uint32_t)JsonNumber(j, "file_h"), rw = (uint32_t)JsonNumber(j, "render_w"),
             rh = (uint32_t)JsonNumber(j, "render_h");
    auto mv = ReadWords(base + ".mv.bin"), mvf = ReadWords(base + ".mvfinal.bin"), dep = ReadWords(base + ".depth.bin");
    std::string name = "motion vectors, frame " + std::to_string(frame);
    if (!frame || mv.size() != (size_t)w * h || mvf.size() != mv.size() || dep.size() != mv.size()) {
        AddCheck(name, false, "recording incomplete (%s)", json.filename().string().c_str());
        return;
    }
    float camDx = CamX(frame) - CamX(frame - 1), camDy = CamY(frame) - CamY(frame - 1);
    float cubeDx = CubeX(frame) - CubeX(frame - 1);
    double sx = proj.m[0] * rw / 2.0, sy = proj.m[5] * rh / 2.0;
    const bool depthR32f = JsonNumber(j, "depth_is_r32f") == 1.0;  // newer layers record their own R32F copy
    std::vector<float> z(dep.size());
    for (size_t i = 0; i < dep.size(); ++i) {
        float d;
        if (g.depthFmt == VK_FORMAT_D24_UNORM_S8_UINT && !depthR32f) d = (float)(dep[i] & 0xFFFFFF) / 16777215.f;
        else std::memcpy(&d, &dep[i], 4);
        float zn = d * 2.f - 1.f;
        z[i] = 2.f * kNear * kFar / (kFar + kNear - zn * (kFar - kNear));
    }
    // Inside a surface only: all four neighbours (2 render pixels away) at the same depth.
    size_t wall = 0, cube = 0, bad = 0, badCam = 0;
    double worst = 0, worstCam = 0;
    float cubeLimit = (float)(-kWallZ - kCubeZ) / 2 + 0.5f;  // between the cube front (11) and the wall (20)
    for (uint32_t y = 1; y + 1 < h; ++y)
        for (uint32_t x = 1; x + 1 < w; ++x) {
            size_t i = (size_t)y * w + x;
            float zc = z[i];
            if (std::fabs(z[i - 1] - zc) > zc * 1e-4f || std::fabs(z[i + 1] - zc) > zc * 1e-4f ||
                std::fabs(z[i - w] - zc) > zc * 1e-4f || std::fabs(z[i + w] - zc) > zc * 1e-4f)
                continue;
            bool isCube = zc < cubeLimit;
            if (!isCube && std::fabs(zc + kWallZ) > 0.01f) continue;  // background (nothing drawn)
            (isCube ? cube : wall)++;
            double camX = sx * camDx / zc, camY = -sy * camDy / zc;
            double fullX = sx * (camDx - (isCube ? cubeDx : 0.f)) / zc;
            auto err = [](uint32_t packed, double ex, double ey) {
                double gx = HalfToFloat((uint16_t)(packed & 0xFFFF)), gy = HalfToFloat((uint16_t)(packed >> 16));
                return std::max(std::fabs(gx - ex), std::fabs(gy - ey));
            };
            double tol = 0.02 + 0.002 * std::fabs(fullX);
            double e = err(mvf[i], fullX, camY), ec = err(mv[i], camX, camY);
            worst = std::max(worst, e);
            worstCam = std::max(worstCam, ec);
            if (e > tol) ++bad;
            if (ec > tol) ++badCam;
        }
    size_t total = wall + cube;
    bool enough = wall > (size_t)w * h / 4 && cube > 200;
    bool ok = enough && bad <= total / 200 && badCam <= total / 200;
    AddCheck(name, ok,
             "camera moved (%.3f, %.3f), cube %.3f per frame; %zu wall + %zu cube pixels checked; off by more than "
             "0.02 px: %zu with objects (worst %.3f px), %zu camera only (worst %.3f px)%s",
             camDx, camDy, cubeDx, wall, cube, bad, worst, badCam, worstCam,
             enough ? "" : "; too few pixels of the wall or the cube were found");
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) Die("%s needs a value", a.c_str());
            return argv[++i];
        };
        if (a == "--out") g_opt.out = next();
        else if (a == "--frames") g_opt.frames = (uint32_t)std::atoi(next());
        else if (a == "--width") g_opt.width = (uint32_t)std::atoi(next());
        else if (a == "--height") g_opt.height = (uint32_t)std::atoi(next());
        else if (a == "--scale") g_opt.scale = (float)std::atof(next());
        else if (a == "--tiles") g_opt.tilesX = (uint32_t)std::atoi(next()), g_opt.tilesY = (uint32_t)std::atoi(next());
        else if (a == "--threads") g_opt.threads = std::max(1, std::atoi(next()));
        else if (a == "--window") g_opt.window = true;
        else if (a == "--validation") g_opt.validation = true;
        else if (a == "--msaa") g_opt.msaa = (uint32_t)std::max(1, std::atoi(next()));
        else if (a == "--no-checks") g_opt.checks = false;
        else Die("unknown option %s (see the top of fake_tf3.cpp)", a.c_str());
    }
    if (g_opt.frames < 300 && g_opt.checks) {
        std::fprintf(stderr, "fake_tf3: checks need 300 frames; running without them\n");
        g_opt.checks = false;
    }
    std::error_code ec;
    fs::create_directories(g_opt.out, ec);

    CreateInstanceAndSurface();
    CreateDevice();
    CreateSwapchain();
    std::printf("fake_tf3: %s, render %ux%u -> output %ux%u, %u wall draws + 1 cube on %u threads, %u frames\n",
                g.props.deviceName, g.render.width, g.render.height, g.out.width, g.out.height,
                g_opt.tilesX * g_opt.tilesY, g_opt.threads, g_opt.frames);

    // ---- images: TF3's formats (B10G11R11 HDR and tonemapped colour, D24S8 depth) ----
    const VkFormat kHdr = VK_FORMAT_B10G11R11_UFLOAT_PACK32, kPost = VK_FORMAT_R8G8B8A8_UNORM;
    const VkImageUsageFlags kTarget = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    Image hdr = MakeImage(g.render.width, g.render.height, kHdr, kTarget, VK_IMAGE_ASPECT_COLOR_BIT);
    Image ldr = MakeImage(g.render.width, g.render.height, kHdr, kTarget, VK_IMAGE_ASPECT_COLOR_BIT);
    const VkSampleCountFlagBits kSamples = (VkSampleCountFlagBits)g_opt.msaa;
    const bool msaa = g_opt.msaa > 1;
    Image depth = MakeImage(g.render.width, g.render.height, g.depthFmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                            VK_IMAGE_ASPECT_DEPTH_BIT, kSamples);  // the layer adds SAMPLED
    // --msaa: multisampled colour targets, resolved into hdr / ldr at the end of each pass
    Image hdrMs, ldrMs;
    if (msaa) {
        hdrMs = MakeImage(g.render.width, g.render.height, kHdr, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                          VK_IMAGE_ASPECT_COLOR_BIT, kSamples);
        ldrMs = MakeImage(g.render.width, g.render.height, kHdr, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                          VK_IMAGE_ASPECT_COLOR_BIT, kSamples);
    }
    Image easu = MakeImage(g.out.width, g.out.height, kPost, kTarget, VK_IMAGE_ASPECT_COLOR_BIT);
    Image rcas = MakeImage(g.out.width, g.out.height, kPost, kTarget, VK_IMAGE_ASPECT_COLOR_BIT);
    // (the depth view isn't used by the app; TF3's own depth view has both aspects)

    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkRenderPass rpScene = MakeRenderPass(kHdr, RO, true, Load::Clear, kSamples);
    VkRenderPass rpSceneMore = MakeRenderPass(kHdr, RO, true, Load::KeepAll, kSamples);  // the other threads' parts
    VkRenderPass rpTonemap = MakeRenderPass(kHdr, RO, true, Load::KeepDepth, kSamples);
    VkRenderPass rpPost = MakeRenderPass(kPost, RO, false, Load::Clear);
    VkRenderPass rpFinal = MakeRenderPass(g.swapFmt, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, false, Load::Clear);
    VkFramebuffer fbScene = msaa ? MakeFramebuffer(rpScene, {hdrMs.view, depth.view, hdr.view}, g.render)
                                 : MakeFramebuffer(rpScene, {hdr.view, depth.view}, g.render);
    VkFramebuffer fbTonemap = msaa ? MakeFramebuffer(rpTonemap, {ldrMs.view, depth.view, ldr.view}, g.render)
                                   : MakeFramebuffer(rpTonemap, {ldr.view, depth.view}, g.render);
    VkFramebuffer fbEasu = MakeFramebuffer(rpPost, {easu.view}, g.out);
    VkFramebuffer fbRcas = MakeFramebuffer(rpPost, {rcas.view}, g.out);
    std::vector<VkFramebuffer> fbFinal;
    for (VkImageView v : g.swapViews) fbFinal.push_back(MakeFramebuffer(rpFinal, {v}, g.out));

    // ---- descriptors ----
    auto bind = [](uint32_t b, VkDescriptorType t, VkShaderStageFlags s) {
        return VkDescriptorSetLayoutBinding{b, t, 1, s, nullptr};
    };
    VkDescriptorSetLayout dslView = SetLayout({bind(10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)});
    VkDescriptorSetLayout dslMat =
        SetLayout({bind(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_FRAGMENT_BIT)});
    VkDescriptorSetLayout dslTex =
        SetLayout({bind(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)});
    VkDescriptorSetLayout dslFsr = SetLayout({bind(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
                                              bind(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)});
    VkPipelineLayout plScene = PipeLayout({dslView, dslMat}), plTex = PipeLayout({dslTex}),
                     plTonemap = PipeLayout({dslView, dslTex}),
                     plFsr = PipeLayout({dslFsr}), plUi = PipeLayout({}, 32);

    constexpr uint32_t kInFlight = 2, kMaterials = 16, kMatStride = 256;
    VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8},
                                 {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2},
                                 {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8}};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 8;
    pci.poolSizeCount = 3;
    pci.pPoolSizes = ps;
    VkDescriptorPool pool;
    VKC(vkCreateDescriptorPool(g.dev, &pci, nullptr, &pool));
    auto alloc = [&](VkDescriptorSetLayout l) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &l;
        VkDescriptorSet s;
        VKC(vkAllocateDescriptorSets(g.dev, &ai, &s));
        return s;
    };
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler sampler;
    VKC(vkCreateSampler(g.dev, &sci, nullptr, &sampler));

    Buffer viewUbo[kInFlight], inst[kInFlight];
    uint32_t tiles = g_opt.tilesX * g_opt.tilesY;
    VkDescriptorSet viewSet[kInFlight];
    for (uint32_t i = 0; i < kInFlight; ++i) {
        viewUbo[i] = MakeBuffer(sizeof(ViewUbo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        inst[i] = MakeBuffer((VkDeviceSize)(tiles + 1) * 64, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        viewSet[i] = alloc(dslView);
    }
    Buffer mat = MakeBuffer(kMaterials * kMatStride, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    for (uint32_t m = 0; m < kMaterials; ++m) {
        float* t = (float*)((char*)mat.ptr + m * kMatStride);
        bool cube = m == kMaterials - 1;
        t[0] = cube ? 1.f : 0.45f + 0.05f * (m % 4);
        t[1] = cube ? 0.5f : 0.55f + 0.04f * (m % 3);
        t[2] = cube ? 0.1f : 0.6f;
        t[3] = 1.f;
    }
    Buffer fsrUbo = MakeBuffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    VkDescriptorSet matSet = alloc(dslMat), tonemapSet = alloc(dslTex), blitSet = alloc(dslTex),
                    easuSet = alloc(dslFsr), rcasSet = alloc(dslFsr);
    {
        std::vector<VkWriteDescriptorSet> w;
        VkDescriptorBufferInfo vbi[kInFlight];
        for (uint32_t i = 0; i < kInFlight; ++i) {
            vbi[i] = {viewUbo[i].buf, 0, sizeof(ViewUbo)};
            VkWriteDescriptorSet x{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            x.dstSet = viewSet[i];
            x.dstBinding = 10;
            x.descriptorCount = 1;
            x.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            x.pBufferInfo = &vbi[i];
            w.push_back(x);
        }
        VkDescriptorBufferInfo mbi{mat.buf, 0, 16}, fbi{fsrUbo.buf, 0, 256};
        VkDescriptorImageInfo ii[4] = {{sampler, hdr.view, RO}, {sampler, rcas.view, RO}, {sampler, ldr.view, RO},
                                       {sampler, easu.view, RO}};
        auto img = [&](VkDescriptorSet s, VkDescriptorImageInfo* info) {
            VkWriteDescriptorSet x{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            x.dstSet = s;
            x.descriptorCount = 1;
            x.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            x.pImageInfo = info;
            w.push_back(x);
        };
        auto ubo = [&](VkDescriptorSet s, uint32_t b, VkDescriptorType t, VkDescriptorBufferInfo* info) {
            VkWriteDescriptorSet x{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            x.dstSet = s;
            x.dstBinding = b;
            x.descriptorCount = 1;
            x.descriptorType = t;
            x.pBufferInfo = info;
            w.push_back(x);
        };
        ubo(matSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, &mbi);
        img(tonemapSet, &ii[0]);
        img(blitSet, &ii[1]);
        img(easuSet, &ii[2]);
        ubo(easuSet, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &fbi);
        img(rcasSet, &ii[3]);
        ubo(rcasSet, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &fbi);
        vkUpdateDescriptorSets(g.dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    // ---- pipelines ----
#define SPV(a) a, sizeof a
    VkPipeline pScene = MakePipeline({SPV(kSceneVert), SPV(kSceneFrag), plScene, rpScene, true, kSamples});
    VkPipeline pTonemap = MakePipeline({SPV(kFullscreenVert), SPV(kTonemapFrag), plTonemap, rpTonemap, false, kSamples});
    VkPipeline pEasu = MakePipeline({SPV(kFullscreenVert), SPV(kEasuFrag), plFsr, rpPost});
    VkPipeline pRcas = MakePipeline({SPV(kFullscreenVert), SPV(kRcasFrag), plFsr, rpPost});
    VkPipeline pBlit = MakePipeline({SPV(kFullscreenVert), SPV(kBlitFrag), plTex, rpFinal});
    VkPipeline pUi = MakePipeline({SPV(kFullscreenVert), SPV(kUiFrag), plUi, rpFinal});
#undef SPV

    // ---- meshes: a unit quad (wall tiles) and a 2x2x2 cube ----
    const float quadV[] = {-.5f, -.5f, 0, .5f, -.5f, 0, .5f, .5f, 0, -.5f, .5f, 0};
    const uint16_t quadI[] = {0, 1, 2, 0, 2, 3};
    float cubeV[24];
    for (int i = 0; i < 8; ++i) cubeV[i * 3] = i & 1 ? 1.f : -1.f, cubeV[i * 3 + 1] = i & 2 ? 1.f : -1.f,
                                cubeV[i * 3 + 2] = i & 4 ? 1.f : -1.f;
    const uint16_t cubeI[] = {0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
                              2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3};
    Buffer quadVb = MakeBuffer(sizeof quadV, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT),
           quadIb = MakeBuffer(sizeof quadI, VK_BUFFER_USAGE_INDEX_BUFFER_BIT),
           cubeVb = MakeBuffer(sizeof cubeV, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT),
           cubeIb = MakeBuffer(sizeof cubeI, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    std::memcpy(quadVb.ptr, quadV, sizeof quadV);
    std::memcpy(quadIb.ptr, quadI, sizeof quadI);
    std::memcpy(cubeVb.ptr, cubeV, sizeof cubeV);
    std::memcpy(cubeIb.ptr, cubeI, sizeof cubeI);
    // The wall: 80 x 30 units, wide enough for the pan.
    const float wallW = 80.f, wallH = 30.f, tileW = wallW / g_opt.tilesX, tileH = wallH / g_opt.tilesY;

    // ---- frame resources ----
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = g.qfam;
    // one command pool per recording thread; command buffer [t] = scene part t, [threads] = the rest
    const uint32_t nThreads = std::min(g_opt.threads, tiles);
    std::vector<std::array<VkCommandBuffer, kInFlight>> cmd(nThreads + 1);
    for (uint32_t t = 0; t <= nThreads; ++t) {
        VkCommandPool cmdPool;
        VKC(vkCreateCommandPool(g.dev, &cpi, nullptr, &cmdPool));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cmdPool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = kInFlight;
        VKC(vkAllocateCommandBuffers(g.dev, &cai, cmd[t].data()));
    }
    Workers workers(nThreads > 1 ? nThreads : 0);
    VkFence fence[kInFlight];
    VkSemaphore acquired[kInFlight];
    std::vector<VkSemaphore> rendered(g.swapImages.size());
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (uint32_t i = 0; i < kInFlight; ++i) {
        VKC(vkCreateFence(g.dev, &fci, nullptr, &fence[i]));
        VKC(vkCreateSemaphore(g.dev, &semi, nullptr, &acquired[i]));
    }
    for (auto& s : rendered) VKC(vkCreateSemaphore(g.dev, &semi, nullptr, &s));
    Buffer readback = MakeBuffer((VkDeviceSize)g.out.width * g.out.height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    const float aspect = (float)g.render.width / (float)g.render.height;
    const Mat4 proj = Perspective(kFovY, aspect, kNear, kFar);
    const float uiRect[8] = {0.02f, 0.85f, 0.30f, 0.97f, 0.9f, 0.9f, 0.9f, 1.f};  // light grey panel, bottom left
    double recordMs = 0, frameMs = 0;
    auto tStart = std::chrono::steady_clock::now(), tPrev = tStart;
    const uint32_t warmup = std::min<uint32_t>(30, g_opt.frames / 4);  // left out of the averages

    for (uint32_t f = 0; f < g_opt.frames; ++f) {
#ifdef _WIN32
        MSG msg;
        while (g.hwnd && PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
#endif
        uint32_t slot = f % kInFlight;
        VKC(vkWaitForFences(g.dev, 1, &fence[slot], VK_TRUE, UINT64_MAX));
        uint32_t img = 0;
        VKC(vkAcquireNextImageKHR(g.dev, g.swapchain, UINT64_MAX, acquired[slot], VK_NULL_HANDLE, &img));
        VKC(vkResetFences(g.dev, 1, &fence[slot]));

        // this frame's camera and instance data, written before recording (as the game does)
        ViewUbo v{};
        Mat4 view = Translate(-CamX(f), -CamY(f), 0.f);
        v.view = view;
        v.viewInverse = Inverse(view);
        v.proj = proj;
        v.projInverse = Inverse(proj);
        v.projView = Mul(proj, view);
        v.projViewInverse = Inverse(v.projView);
        v.camPos[0] = CamX(f), v.camPos[1] = CamY(f), v.camPos[2] = 0.f;
        v.nearZ = kNear, v.farZ = kFar;
        v.width = (int32_t)g.render.width, v.height = (int32_t)g.render.height;
        v.fov = kFovY;
        std::memcpy(viewUbo[slot].ptr, &v, sizeof v);
        Mat4* im = (Mat4*)inst[slot].ptr;
        for (uint32_t ty = 0; ty < g_opt.tilesY; ++ty)
            for (uint32_t tx = 0; tx < g_opt.tilesX; ++tx)
                im[ty * g_opt.tilesX + tx] = Mul(Translate(-wallW / 2 + (tx + .5f) * tileW, -wallH / 2 + (ty + .5f) * tileH, kWallZ),
                                                 Scale(tileW, tileH, 1.f));
        im[tiles] = Translate(CubeX(f), 0.f, kCubeZ);

        auto t0 = std::chrono::steady_clock::now();
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        auto begin = [&](VkCommandBuffer cb, VkRenderPass rp, VkFramebuffer fb, VkExtent2D size) {
            VkClearValue cv[2]{};
            cv[0].color = {{0.05f, 0.05f, 0.08f, 1.f}};
            cv[1].depthStencil = {1.f, 0};
            VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rbi.renderPass = rp;
            rbi.framebuffer = fb;
            rbi.renderArea = {{0, 0}, size};
            rbi.clearValueCount = 2;
            rbi.pClearValues = cv;
            vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
            VkViewport vp{0, 0, (float)size.width, (float)size.height, 0, 1};
            VkRect2D sc{{0, 0}, size};
            vkCmdSetViewport(cb, 0, 1, &vp);
            vkCmdSetScissor(cb, 0, 1, &sc);
        };
        // 1. scene, in nThreads parts recorded at the same time: one draw per wall tile, a material bind
        //    before each (as the game binds per object); the last part also draws the cube
        auto recordScene = [&](uint32_t part) {
            VkCommandBuffer cb = cmd[part][slot];
            VKC(vkBeginCommandBuffer(cb, &bi));
            begin(cb, part == 0 ? rpScene : rpSceneMore, fbScene, g.render);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pScene);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, plScene, 0, 1, &viewSet[slot], 0, nullptr);
            VkDeviceSize zero = 0;
            vkCmdBindVertexBuffers(cb, 0, 1, &quadVb.buf, &zero);
            vkCmdBindVertexBuffers(cb, 1, 1, &inst[slot].buf, &zero);
            vkCmdBindIndexBuffer(cb, quadIb.buf, 0, VK_INDEX_TYPE_UINT16);
            for (uint32_t t = tiles * part / nThreads; t < tiles * (part + 1) / nThreads; ++t) {
                uint32_t off = (t % (kMaterials - 1)) * kMatStride;
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, plScene, 1, 1, &matSet, 1, &off);
                vkCmdDrawIndexed(cb, 6, 1, 0, 0, t);
            }
            if (part + 1 == nThreads) {
                uint32_t cubeOff = (kMaterials - 1) * kMatStride;
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, plScene, 1, 1, &matSet, 1, &cubeOff);
                vkCmdBindVertexBuffers(cb, 0, 1, &cubeVb.buf, &zero);
                vkCmdBindIndexBuffer(cb, cubeIb.buf, 0, VK_INDEX_TYPE_UINT16);
                vkCmdDrawIndexed(cb, 36, 1, 0, 0, tiles);
            }
            vkCmdEndRenderPass(cb);
            VKC(vkEndCommandBuffer(cb));
        };
        if (nThreads > 1) workers.Run(recordScene);
        else recordScene(0);
        // 2-4: the rest of the frame, after the scene (the layer replays the moving objects then)
        VkCommandBuffer cb = cmd[nThreads][slot];
        VKC(vkBeginCommandBuffer(cb, &bi));
        // 2. tonemap (the depth buffer is attached here too, as in TF3)
        begin(cb, rpTonemap, fbTonemap, g.render);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pTonemap);
        VkDescriptorSet tonemapSets[] = {viewSet[slot], tonemapSet};
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, plTonemap, 0, 2, tonemapSets, 0, nullptr);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRenderPass(cb);
        // 3. FSR 1 EASU and RCAS stand-ins (magenta)
        begin(cb, rpPost, fbEasu, g.out);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pEasu);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, plFsr, 0, 1, &easuSet, 0, nullptr);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRenderPass(cb);
        begin(cb, rpPost, fbRcas, g.out);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pRcas);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, plFsr, 0, 1, &rcasSet, 0, nullptr);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRenderPass(cb);  // the layer records the upscaler here
        // 4. blit to the screen, then the UI on top
        begin(cb, rpFinal, fbFinal[img], g.out);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pBlit);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, plTex, 0, 1, &blitSet, 0, nullptr);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pUi);
        vkCmdPushConstants(cb, plUi, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof uiRect, uiRect);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRenderPass(cb);
        if (f + 1 == g_opt.frames) {  // read the last frame back
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = g.swapImages[img];
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &b);
            VkBufferImageCopy c{};
            c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.imageExtent = {g.out.width, g.out.height, 1};
            vkCmdCopyImageToBuffer(cb, g.swapImages[img], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buf, 1, &c);
            b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            b.dstAccessMask = 0;
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &b);
        }
        VKC(vkEndCommandBuffer(cb));
        auto t1 = std::chrono::steady_clock::now();
        if (f >= warmup) recordMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &acquired[slot];
        si.pWaitDstStageMask = &waitStage;
        std::vector<VkCommandBuffer> cbs;
        for (auto& c : cmd) cbs.push_back(c[slot]);
        si.commandBufferCount = (uint32_t)cbs.size();
        si.pCommandBuffers = cbs.data();
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &rendered[img];
        VKC(vkQueueSubmit(g.queue, 1, &si, fence[slot]));
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &rendered[img];
        pi.swapchainCount = 1;
        pi.pSwapchains = &g.swapchain;
        pi.pImageIndices = &img;
        VkResult pr = vkQueuePresentKHR(g.queue, &pi);
        if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) Die("vkQueuePresentKHR failed (%d)", (int)pr);
        auto now = std::chrono::steady_clock::now();
        if (f >= warmup) frameMs += std::chrono::duration<double, std::milli>(now - tPrev).count();
        tPrev = now;
    }
    VKC(vkDeviceWaitIdle(g.dev));
    uint32_t measured = g_opt.frames - warmup;
    recordMs /= measured;
    frameMs /= measured;

    // ---- the last frame ----
    const uint8_t* px = (const uint8_t*)readback.ptr;
    bool bgr = g.swapFmt == VK_FORMAT_B8G8R8A8_UNORM || g.swapFmt == VK_FORMAT_B8G8R8A8_SRGB;
    size_t npx = (size_t)g.out.width * g.out.height, magenta = 0;
    double sum = 0, sum2 = 0;
    {
        std::ofstream ppm(g_opt.out / "last_frame.ppm", std::ios::binary);
        ppm << "P6\n" << g.out.width << " " << g.out.height << "\n255\n";
        std::vector<uint8_t> row((size_t)g.out.width * 3);
        for (uint32_t y = 0; y < g.out.height; ++y) {
            for (uint32_t x = 0; x < g.out.width; ++x) {
                const uint8_t* p = px + ((size_t)y * g.out.width + x) * 4;
                uint8_t r = bgr ? p[2] : p[0], gg = p[1], b = bgr ? p[0] : p[2];
                row[x * 3] = r, row[x * 3 + 1] = gg, row[x * 3 + 2] = b;
                if (r > 200 && gg < 60 && b > 200) ++magenta;
                double l = 0.299 * r + 0.587 * gg + 0.114 * b;
                sum += l, sum2 += l * l;
            }
            ppm.write((const char*)row.data(), (std::streamsize)row.size());
        }
    }
    double mean = sum / npx, stddev = std::sqrt(std::max(0.0, sum2 / npx - mean * mean));
    auto pixel = [&](float u, float vv) {  // luminance at a uv position
        const uint8_t* p = px + ((size_t)(vv * g.out.height) * g.out.width + (size_t)(u * g.out.width)) * 4;
        return 0.299 * (bgr ? p[2] : p[0]) + 0.587 * p[1] + 0.114 * (bgr ? p[0] : p[2]);
    };
    double uiLum = pixel((uiRect[0] + uiRect[2]) / 2, (uiRect[1] + uiRect[3]) / 2);

    std::printf("fake_tf3: CPU time per frame: %.3f ms recording, %.3f ms in total (%.1f fps), over %u frames\n",
                recordMs, frameMs, 1000.0 / frameMs, measured);

    if (g_opt.checks) {
        std::string log = ReadText(g_opt.out / "tf3_dlss.log");
        bool layerRan = !log.empty();
        AddCheck("layer loaded", layerRan, layerRan ? "tf3_dlss.log written" : "no tf3_dlss.log in %s",
                 g_opt.out.string().c_str());
        bool shown = log.find("first upscaled frame shown") != std::string::npos;
        AddCheck("upscaled image shown", shown, shown ? "the log says the first upscaled frame was shown"
                                                      : "the log never says 'first upscaled frame shown'");
        AddCheck("no magenta (blit swap)", magenta < npx / 1000, "%.3f%% of the last frame is magenta (the game's FSR 1 image)",
                 100.0 * magenta / npx);
        AddCheck("image has content", stddev > 20, "brightness spread %.1f (a flat image means nothing was drawn)", stddev);
        AddCheck("UI on top", uiLum > 200, "UI panel brightness %.0f (should be about 230)", uiLum);
        std::vector<fs::path> caps;
        for (auto& e : fs::directory_iterator(g_opt.out / "mvcap", ec))
            if (e.path().extension() == ".json") caps.push_back(e.path());
        std::sort(caps.begin(), caps.end());
        AddCheck("motion vectors recorded", caps.size() >= 6, "%zu frames recorded (TF3_DLSS_CAPTURE=%s gives 6)",
                 caps.size(), kCaptureFrames);
        for (auto& c : caps) CheckCapture(c, proj);
        if (g_opt.validation)
            AddCheck("validation", g_validationErrors == 0, "%d errors, %d warnings", g_validationErrors,
                     g_validationWarnings);
    }

    // ---- report ----
    bool allPass = true;
    std::ofstream js(g_opt.out / "result.json");
    js << "{\n  \"gpu\": \"" << g.props.deviceName << "\",\n  \"render\": [" << g.render.width << ", " << g.render.height
       << "],\n  \"output\": [" << g.out.width << ", " << g.out.height << "],\n  \"draws\": " << tiles + 1
       << ",\n  \"frames\": " << g_opt.frames << ",\n  \"record_ms\": " << recordMs << ",\n  \"frame_ms\": " << frameMs
       << ",\n  \"checks\": [";
    for (size_t i = 0; i < g_checks.size(); ++i) {
        const Check& c = g_checks[i];
        allPass &= c.pass;
        std::printf("  %s  %s: %s\n", c.pass ? "PASS" : "FAIL", c.name.c_str(), c.detail.c_str());
        js << (i ? "," : "") << "\n    {\"name\": \"" << c.name << "\", \"pass\": " << (c.pass ? "true" : "false")
           << ", \"detail\": \"" << c.detail << "\"}";
    }
    js << "\n  ],\n  \"pass\": " << (allPass ? "true" : "false") << "\n}\n";
    if (g_opt.checks) std::printf("fake_tf3: %s\n", allPass ? "all checks passed" : "SOME CHECKS FAILED");

    // The app's own objects go with the process; destroying the device still runs the layer's teardown.
    // The messenger goes first, so validation doesn't list the objects left alive.
    if (g.messenger) {
        auto destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(g.instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy) destroy(g.instance, g.messenger, nullptr);
    }
    vkDestroyDevice(g.dev, nullptr);
    vkDestroySurfaceKHR(g.instance, g.surface, nullptr);
    vkDestroyInstance(g.instance, nullptr);
    return allPass ? 0 : 1;
}
