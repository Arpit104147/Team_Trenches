# XEFRAME — Custom GPU Upscaler & Frame Generator
## Master Build Prompt for Gemini CLI
### From-Scratch, Ubuntu-Native, Intel Iris Xe iGPU Optimized

---

> **HOW TO USE THIS PROMPT:**
> Paste this entire document as your first message to Gemini CLI.
> Gemini will act as your lead systems engineer and guide you through every phase.
> Each phase ends with a checkpoint. Confirm before proceeding.
> Do not skip phases. Architecture decisions in Phase 1 cascade into Phase 8.

---

## PREAMBLE — MISSION STATEMENT & ABSOLUTE CONSTRAINTS

You are the lead engineer for **XEFRAME**, a custom real-time video upscaling and frame generation application built entirely from scratch for Ubuntu Linux. This project must not use any existing upscaler, frame interpolation library, or third-party ML inference backend as a black box. No DLSS, no FSR, no XeSS, no RIFE, no DAIN, no ESRGAN, no NCNN, no TensorRT, no ONNX Runtime, no OpenVINO as a plug-in inference engine. Every algorithm — spatial reconstruction, temporal accumulation, motion estimation, frame synthesis — will be designed, implemented, and trained by us.

The target hardware is the **Intel Iris Xe iGPU (Xe-LP architecture, Gen12)**, specifically as found in 11th/12th gen Intel mobile processors (e.g., i5-1235U). This GPU has:
- No dedicated RT (Ray Tracing) cores
- No dedicated tensor/NPU accelerators accessible via standard compute APIs
- Unified memory shared with system RAM (DDR4/LPDDR4)
- Intel Xe-LP Execution Units (EUs) — 80 EUs on i5-1235U
- OpenCL 3.0, Vulkan 1.3, and OpenGL 4.6 support via Mesa i915/ANV drivers
- No CUDA. No ROCm. No Metal.

The primary compute API will be **Vulkan Compute** (SPIR-V shaders). OpenCL is a fallback path. The UI will be built in **GTK4 with libadwaita**. Training infrastructure will use **pure PyTorch with CPU/iGPU-friendly backends** — specifically Intel Extension for PyTorch (IPEX) with XPU backend, or fallback to CPU training with ONNX export for inference.

The final application must:
1. Accept any video source (game capture, video file, screen region)
2. Upscale from a lower resolution to a higher one (e.g., 720p → 1080p, 1080p → 1440p)
3. Generate intermediate frames (1x → 2x or 3x framerate)
4. Run in real-time at acceptable latency on the target iGPU
5. Expose a GTK4 GUI for profile management, source selection, and performance metrics
6. Be installable as a .deb package or via a single setup script on Ubuntu 22.04 LTS and 24.04 LTS

No phase of this project is negotiable in terms of originality. If you suggest using a pre-built inference engine or pre-trained model as a final artifact, I will redirect you. We train our own models, export them ourselves, and run inference ourselves via Vulkan compute shaders.

Now begin. Acknowledge this mission statement, then ask me to confirm Phase 1.

---

## PHASE 1 — SYSTEM ARCHITECTURE & REPOSITORY BOOTSTRAP

### 1.1 — Architecture Overview

Design and explain the complete system architecture for XEFRAME. Your explanation must cover:

**A. Pipeline Architecture**
Design a single-producer, multi-consumer pipeline with the following stages:
- **Stage 0 — Source Ingestion**: Captures frames from one of three sources: (a) a video file via libavformat/libavcodec (FFmpeg libraries, not the ffmpeg binary), (b) a screen region capture via PipeWire portal (for Wayland) or XShm (for X11), (c) a V4L2 device (webcam or capture card).
- **Stage 1 — Pre-processing**: Color space conversion (e.g., YUV420 → RGB), normalization to float16, downscaling to the model's expected input resolution, and frame queuing.
- **Stage 2 — Upscaling Inference**: Vulkan Compute dispatch of our custom upscaling model (a lightweight CNN whose architecture you will define in Phase 3). Input is the low-res frame. Output is a high-res reconstructed frame. This runs on the Xe-LP EUs.
- **Stage 3 — Frame Generation Inference**: Vulkan Compute dispatch of our custom frame interpolation model (a motion-aware synthesis network defined in Phase 4). Inputs are two consecutive upscaled frames (t and t+1). Output is one or more synthesized intermediate frames.
- **Stage 4 — Post-processing**: Sharpening pass, optional HDR tone-mapping, output format conversion.
- **Stage 5 — Display/Output**: Renders the final frame stream to a Vulkan swapchain window, or encodes to a video file, or pipes to a virtual V4L2 device.

Describe the inter-stage communication mechanism. Use lock-free ring buffers (implement using C++ `std::atomic` or a custom SPSC queue). Explain why you chose this over mutexed queues for this use case.

**B. Process & Thread Model**
Define the threading model. There must be:
- 1 dedicated capture thread (real-time priority, SCHED_FIFO optional)
- 1 Vulkan command recording thread per inference stage (stages 2 and 3 may overlap if GPU queue families allow)
- 1 UI thread (GTK main loop, must never block on GPU)
- 1 stats/monitoring thread (collects frame times, EU utilization via intel_gpu_top sysfs interface)
Define the synchronization points between threads using Vulkan semaphores and timeline semaphores. Explain the fence strategy.

**C. Memory Architecture**
For Intel Iris Xe unified memory:
- Explain the significance of `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` being available simultaneously on Xe-LP
- Design a memory allocation strategy using our own simple pool allocator on top of raw Vulkan memory (we will NOT use VMA — Vulkan Memory Allocator — since we want to understand and control every allocation)
- Explain the trade-off between persistent mapped host-visible memory (zero-copy for upload) versus staging buffers
- Define how model weights will be loaded: as `VkBuffer` objects in device-local memory, how they are uploaded at startup, and kept resident during inference

**D. Shader Compilation Strategy**
- All compute shaders are written in GLSL (version 450, with compute extension)
- Compile to SPIR-V at build time using `glslangValidator` or `glslc`
- Embed SPIR-V as C++ `uint32_t[]` arrays (via a Python `xxd`-style script) so the final binary is self-contained
- Describe the descriptor set layout strategy for binding model weight buffers and I/O image buffers to compute shaders

**E. Model Weight Format**
Design a custom binary weight format for XEFRAME:
- Magic bytes: `XEFM` (4 bytes)
- Version: uint16
- Layer count: uint16
- Per-layer header: name (32 bytes, null-padded), shape as uint32[4] (N, C, H, W), dtype (uint8: 0=float32, 1=float16), byte offset into data section (uint64)
- Data section: raw contiguous tensor bytes, float16 preferred for inference
- A Python utility (`xeframe_export.py`) converts PyTorch `.pth` checkpoints to `.xefm` files
- A C++ loader (`XeFrameWeightLoader`) reads `.xefm` and uploads each layer's data into the corresponding `VkBuffer`

### 1.2 — Repository Structure

Create the following directory tree. Explain every directory's purpose:

```
xeframe/
├── CMakeLists.txt                   # Top-level CMake (min version 3.25)
├── cmake/
│   └── FindVulkan.cmake             # Custom Vulkan detection
├── src/
│   ├── main.cpp                     # Entry point
│   ├── core/
│   │   ├── pipeline.cpp/h           # Master pipeline orchestrator
│   │   ├── ringbuffer.h             # Lock-free SPSC ring buffer template
│   │   ├── frame.h                  # Frame struct (metadata + VkBuffer handle)
│   │   └── stats.cpp/h              # Performance monitoring thread
│   ├── capture/
│   │   ├── capture_base.h           # Abstract capture interface
│   │   ├── capture_file.cpp/h       # FFmpeg-based file capture
│   │   ├── capture_pipewire.cpp/h   # PipeWire portal capture (Wayland)
│   │   └── capture_xshm.cpp/h      # XShm capture (X11 fallback)
│   ├── vulkan/
│   │   ├── vk_context.cpp/h         # Instance, device, queue selection
│   │   ├── vk_memory.cpp/h          # Custom pool allocator
│   │   ├── vk_pipeline.cpp/h        # Compute pipeline creation
│   │   ├── vk_descriptor.cpp/h      # Descriptor set management
│   │   └── vk_swapchain.cpp/h       # Display swapchain
│   ├── inference/
│   │   ├── weight_loader.cpp/h      # .xefm loader + VkBuffer uploader
│   │   ├── upscaler.cpp/h           # Upscaling inference dispatcher
│   │   └── framegen.cpp/h           # Frame generation inference dispatcher
│   ├── shaders/
│   │   ├── preprocess.comp          # Pre-processing GLSL compute shader
│   │   ├── upscale_conv.comp        # Convolution layer shader
│   │   ├── upscale_pixelshuffle.comp# Pixel shuffle / sub-pixel conv shader
│   │   ├── framegen_flow.comp       # Optical flow estimation shader
│   │   ├── framegen_warp.comp       # Backward warping shader
│   │   ├── framegen_blend.comp      # Frame blending/synthesis shader
│   │   └── postprocess.comp         # Sharpening + tonemap shader
│   ├── ui/
│   │   ├── app.cpp/h                # GTK4/libadwaita app entry
│   │   ├── main_window.cpp/h        # Main window widget
│   │   ├── profile_panel.cpp/h      # Profile management panel
│   │   └── overlay.cpp/h            # On-screen metrics overlay
│   └── export/
│       └── virtual_v4l2.cpp/h       # V4L2 loopback output
├── training/
│   ├── upscaler/
│   │   ├── model.py                 # Upscaler PyTorch model definition
│   │   ├── dataset.py               # Dataset loader
│   │   ├── train.py                 # Training script
│   │   ├── loss.py                  # Custom loss functions
│   │   └── validate.py              # PSNR/SSIM validation
│   ├── framegen/
│   │   ├── model.py                 # Frame generation model
│   │   ├── dataset.py               # Video dataset loader
│   │   ├── train.py                 # Training script
│   │   └── loss.py                  # Perceptual + flow loss
│   ├── shared/
│   │   ├── ipex_setup.py            # Intel Extension for PyTorch XPU setup
│   │   └── export.py                # PyTorch → .xefm export utility
│   └── data/
│       ├── download_div2k.sh        # DIV2K dataset download script
│       └── download_vimeo90k.sh     # Vimeo-90K triplet dataset download
├── tools/
│   ├── spv_embed.py                 # SPIR-V → C++ array embedder
│   ├── benchmark.cpp                # Standalone inference benchmarker
│   └── weight_inspect.py            # .xefm file inspector CLI
├── packaging/
│   ├── debian/                      # Debian package control files
│   └── xeframe.desktop              # .desktop entry
└── tests/
    ├── test_ringbuffer.cpp
    ├── test_weight_loader.cpp
    └── test_pipeline.cpp
```

### 1.3 — Build System

Write the complete `CMakeLists.txt`. It must:
- Detect Vulkan SDK (prefer system-installed via apt: `libvulkan-dev`)
- Find and link: `libavformat`, `libavcodec`, `libavutil`, `libswscale` (FFmpeg dev libs), `gtk4`, `libadwaita-1`, `libpipewire-0.3`
- Define a custom target `compile_shaders` that runs `glslc` on all `.comp` files and produces `.spv` output in `build/shaders/`
- Run `spv_embed.py` to embed all `.spv` files as `const uint32_t spv_<name>[]` into `src/shaders/embedded_shaders.h`
- Set C++20 standard, enable `-O3 -march=native` for Release, `-g -fsanitize=address,undefined` for Debug
- Add CTest entries for the test binaries

### 1.4 — Development Environment Bootstrap

Write a complete bash script `setup_dev_env.sh` that:
- Installs all required apt packages on Ubuntu 22.04 and 24.04 (auto-detects via `lsb_release`)
- Installs the Vulkan SDK tools (`vulkan-tools`, `glslang-tools`, `spirv-tools`)
- Installs Python 3.11+ and pip packages: `torch`, `intel-extension-for-pytorch`, `torchvision`, `numpy`, `pillow`, `scipy`, `tqdm`, `tensorboard`, `onnx`
- Sets up a Python venv at `training/.venv`
- Clones no external C++ dependency (all are apt-installable)
- Creates the initial CMake build directory and runs the first configure step
- Prints a success summary

**CHECKPOINT 1**: After you present the complete architecture diagram (text-based), repository tree, CMakeLists.txt, and setup_dev_env.sh, pause and wait for my confirmation before proceeding to Phase 2.

---

## PHASE 2 — VULKAN FOUNDATION & COMPUTE INFRASTRUCTURE

### 2.1 — Vulkan Context Initialization

Implement `vk_context.cpp/h` with full detail:

**Instance Creation**:
- Enable validation layers in Debug builds (`VK_LAYER_KHRONOS_validation`)
- Required instance extensions: `VK_KHR_surface`, `VK_KHR_xcb_surface` (X11), `VK_KHR_wayland_surface`, `VK_EXT_debug_utils`
- Set up `VkDebugUtilsMessengerEXT` with a callback that logs to `stderr` with color-coded severity

**Physical Device Selection**:
- Enumerate all physical devices
- Score them: prefer `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU` (Xe-LP will appear as this)
- Verify required features: `shaderStorageImageExtendedFormats`, `shaderFloat16` (via `VkPhysicalDeviceFloat16Int8FeaturesKHR`), `shaderInt8`
- Query and store: `maxComputeWorkGroupSize`, `maxComputeWorkGroupInvocations`, `subgroupSize` (Intel Xe-LP typically has subgroup size 8 or 16 — this matters for shader optimization)
- Print a formatted device report: EU count approximation from `maxComputeWorkGroupCount`, driver version, API version

**Queue Family Selection**:
- Select one queue family that supports both `VK_QUEUE_COMPUTE_BIT` and `VK_QUEUE_GRAPHICS_BIT` (Intel Xe-LP exposes a unified queue)
- Also select a transfer-only queue if available (for async staging uploads)
- Explain why Intel's unified queue model differs from discrete GPUs with separate async compute queues

**Logical Device Creation**:
- Enable device extensions: `VK_KHR_swapchain`, `VK_KHR_shader_float16_int8`, `VK_KHR_16bit_storage`, `VK_EXT_subgroup_size_control`
- Enable features: `shaderFloat16`, `storageInputOutput16`, `shaderInt8`

**Command Pool & Buffer Management**:
- Create one command pool per thread that submits commands
- Implement a simple `CommandBufferPool` that recycles command buffers: keeps a free list of `VkCommandBuffer` objects, allocates new ones when the free list is empty, returns them on `reset()`

### 2.2 — Custom Memory Allocator

Implement `vk_memory.cpp/h`:

**Pool Design**:
- One pool per memory type index (query `vkGetPhysicalDeviceMemoryProperties`)
- Each pool starts by allocating a 256 MB `VkDeviceMemory` block (configurable)
- Sub-allocates using a free-list allocator with 256-byte alignment (for `nonCoherentAtomSize`)
- Thread-safe via `std::mutex` (note: this is the allocator, not the hot path — mutex is acceptable here)

**Allocation API**:
```cpp
struct XfAllocation {
    VkDeviceMemory memory;
    VkDeviceSize   offset;
    VkDeviceSize   size;
    void*          mapped_ptr; // non-null if HOST_VISIBLE
};
XfAllocation xf_alloc(VkMemoryRequirements req, VkMemoryPropertyFlags flags);
void         xf_free(XfAllocation alloc);
```

**Buffer & Image Helpers**:
```cpp
struct XfBuffer {
    VkBuffer    handle;
    XfAllocation alloc;
    VkDeviceSize size;
};
XfBuffer xf_create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_flags);
void     xf_destroy_buffer(XfBuffer& buf);

struct XfImage {
    VkImage     handle;
    VkImageView view;
    XfAllocation alloc;
    VkFormat    format;
    uint32_t    width, height;
};
XfImage xf_create_image(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage);
void    xf_destroy_image(XfImage& img);
```

**Staging Upload Pattern**:
Implement `xf_upload_buffer(XfBuffer dst, const void* data, size_t size)`:
- Creates a temporary host-visible staging buffer
- Maps, memcpy, unmaps
- Records a `vkCmdCopyBuffer` into a one-time-submit command buffer
- Submits and waits (via `vkWaitForFences`)
- Destroys the staging buffer
- On Xe-LP, note that device-local+host-visible is available, so explain when staging is skipped

### 2.3 — Compute Pipeline Infrastructure

Implement `vk_pipeline.cpp/h`:

**Descriptor Set Layout Factory**:
Define a fluent builder:
```cpp
DescriptorSetLayoutBuilder builder;
builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  1, VK_SHADER_STAGE_COMPUTE_BIT) // weights
       .add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   1, VK_SHADER_STAGE_COMPUTE_BIT) // input
       .add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   1, VK_SHADER_STAGE_COMPUTE_BIT) // output
       .add_binding(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  1, VK_SHADER_STAGE_COMPUTE_BIT); // push params
VkDescriptorSetLayout layout = builder.build(device);
```

**Pipeline Cache**:
- Use `VkPipelineCache` with a file-backed cache at `~/.cache/xeframe/pipeline_cache.bin`
- On startup: load cache from file if exists
- On shutdown: save cache to file
- This avoids SPIR-V recompilation on subsequent launches — critical for startup time on Xe-LP

**Compute Pipeline Creation**:
```cpp
struct ComputePipelineSpec {
    const uint32_t* spir_v;
    size_t          spir_v_word_count;
    VkDescriptorSetLayout desc_layout;
    uint32_t        push_constant_size;
    std::string     entry_point;
};
VkPipeline create_compute_pipeline(const ComputePipelineSpec& spec, VkPipelineLayout& layout_out);
```

**Subgroup Size Hint for Xe-LP**:
Intel Xe-LP's EUs prefer work in multiples of 8 or 16 invocations. Use `VK_EXT_subgroup_size_control` to request a specific subgroup size:
```cpp
VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_info{};
subgroup_info.requiredSubgroupSize = 16; // Xe-LP sweet spot
```
Explain why this matters for the convolutional kernel dispatch and how to determine the correct value at runtime.

### 2.4 — Descriptor Set Management

Implement a `DescriptorCache` that:
- Maintains a `VkDescriptorPool` with pre-allocated capacity for all expected descriptor sets
- Allocates descriptor sets per inference pass
- Provides `update_storage_buffer(set, binding, buffer, offset, range)` and `update_storage_image(set, binding, view, layout)` helpers
- Handles descriptor set reuse between frames (double-buffered: set A is written while set B is bound)

**CHECKPOINT 2**: Present all code implementations for Phase 2 (headers + cpp stubs with full logic). Pause for my confirmation.

---

## PHASE 3 — UPSCALING MODEL: ARCHITECTURE, TRAINING, VULKAN INFERENCE

### 3.1 — Model Architecture: XeScaler-Tiny

Design a custom CNN upscaler called **XeScaler-Tiny** optimized for real-time inference on 80 EU Iris Xe. The design constraints are:
- Must run a 720p → 1080p upscale in under 8ms on Xe-LP (1.5x spatial scale)
- No attention mechanisms (too slow on non-NPU hardware)
- No batch norm at inference (bake into weights after training)
- Float16 weights and activations at inference
- All convolutions must map efficiently to Vulkan compute dispatches

**Architecture Specification**:

```
Input: (1, 3, H, W) — RGB float16, normalized [0,1]

Layer 0: Conv2D(3, 32, kernel=3, padding=1, bias=True) → ReLU
Layer 1: Conv2D(32, 32, kernel=3, padding=1, bias=True) → ReLU
Layer 2: Conv2D(32, 32, kernel=3, padding=1, bias=True) → ReLU
Layer 3: Conv2D(32, 32, kernel=3, padding=1, bias=True) → ReLU
[Residual skip: input_after_layer0 + output_of_layer3 → layer 4 input]
Layer 4: Conv2D(32, 64, kernel=3, padding=1, bias=True) → ReLU
Layer 5: Conv2D(64, 64, kernel=3, padding=1, bias=True) → ReLU
Layer 6: Conv2D(64, 3 * scale^2, kernel=3, padding=1, bias=True)
         ↓ PixelShuffle(upscale_factor=scale)
Output: (1, 3, H*scale, W*scale) — RGB float16, clamped [0,1]
```

For scale=2 (1080p from 540p) or scale=1.5 (use scale=2 then downscale, or tile with overlap).

Explain why PixelShuffle (sub-pixel convolution) is preferred over transposed convolution or bilinear upsampling + conv for this use case. Relate it to checkerboard artifact avoidance and the efficiency of keeping computation at low resolution.

**Batch Normalization Strategy**:
- Train with BatchNorm (helps convergence)
- At export time, fold BatchNorm parameters into preceding Conv2D weights and biases (standard bn_folding technique)
- Implement `fold_bn(model)` in `export.py`
- Result: inference model has no BN layers — only Conv2D + ReLU

### 3.2 — Upscaler Training

Write `training/upscaler/train.py` with full implementation:

**Dataset** (`dataset.py`):
- Use DIV2K (800 train, 100 val high-resolution images)
- Random crop 256×256 HR patches → generate LR by bicubic downsampling (×2 or ×3 or ×4 with random choice per batch for scale generalization)
- Augmentation: random horizontal flip, random 90° rotation, random color jitter (±0.1 brightness, ±0.05 hue)
- Do NOT use JPEG compression augmentation in base training — add it as a separate fine-tuning pass
- Normalize to [0,1] float32 during training (cast to float16 at export)
- DataLoader with num_workers=4, pin_memory=True

**Loss Function** (`loss.py`):
Design a compound loss:
```
L_total = λ1 * L_pixel + λ2 * L_perceptual + λ3 * L_edge
```
- `L_pixel`: L1 loss (more robust to outliers than MSE for super-resolution)
- `L_perceptual`: VGG-style feature matching — but since we avoid heavy dependencies, implement a lightweight alternative: use a fixed 5-layer pretrained VGG-11 (torchvision.models.vgg11, frozen) and compute MSE between feature maps at layers relu1_1 and relu2_1. Cache these features to avoid recomputing. λ2=0.1
- `L_edge`: Sobel-filtered difference between prediction and ground truth. Compute Sobel X and Y gradients of both, compute L1 loss on gradient maps. λ3=0.05
- Explain the motivation for each loss component in the context of super-resolution for video games (sharp edges matter more than perfect color fidelity)

**Training Loop**:
- Optimizer: AdamW (lr=2e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR over 200 epochs with warm restart every 50 epochs
- Gradient clipping: max_norm=1.0
- Mixed precision: `torch.cuda.amp.GradScaler` or `torch.amp.GradScaler('cpu')` for CPU training
- Intel XPU path: detect `torch.xpu.is_available()`, move model and data to `device='xpu'` if true
- Checkpoint every 10 epochs to `checkpoints/upscaler_epoch_{N}.pth`
- TensorBoard logging: loss curves, PSNR, sample output images (LR input, HR ground truth, model output)
- Early stopping: if validation PSNR doesn't improve by >0.1 dB for 20 consecutive epochs, reduce LR by ×0.5

**Validation** (`validate.py`):
- Compute PSNR and SSIM on DIV2K validation set at full image size (tile with overlap if needed to fit in memory)
- Compute inference time per image on current device
- Report: mean PSNR, mean SSIM, P95 inference latency

**Expected Training Results** (to verify your implementation is correct):
- After 50 epochs: PSNR > 28 dB on DIV2K ×2 val set
- After 200 epochs: PSNR > 31.5 dB
- Model size: ~1.2 MB float32 weights

### 3.3 — Export to .xefm Format

Write `training/shared/export.py`:

**Steps**:
1. Load checkpoint: `model.load_state_dict(torch.load('checkpoints/upscaler_best.pth')['model'])`
2. Call `fold_bn(model)` — implement this: for each Sequential block, detect Conv+BN pairs, fold BN into Conv
3. Convert all weights to float16: `model.half()`
4. Trace the model with `torch.jit.trace` on a dummy input to validate the graph
5. Extract all named parameters: for each `(name, tensor)` in `model.named_parameters()`:
   - Compute shape (as 4 uint32s, padding unused dims with 1)
   - Convert to `numpy.ndarray`, dtype=float16, contiguous C order
6. Write the `.xefm` binary file per the format spec in Phase 1.1.E
7. Print a summary: layer count, total parameter count, file size

**Implement `weight_inspect.py`** in `tools/`:
- CLI: `python weight_inspect.py model.xefm`
- Prints: magic bytes, version, layer table with names, shapes, dtypes, offsets, byte sizes
- Optional `--dump-layer <name>` flag: prints first 16 float values of that layer's data

### 3.4 — Vulkan Inference: Upscaler

Now implement the Vulkan compute shaders and C++ dispatcher for XeScaler-Tiny inference.

**Shader Strategy**:
Each Conv2D layer is dispatched as a single compute shader invocation. We do NOT fuse all layers into one mega-shader — instead, we pipeline them with intermediate `VkBuffer` objects storing activation tensors. This gives us:
- Cleaner code
- Debuggable intermediate activations
- Ability to profile individual layers with `VK_EXT_calibrated_timestamps`

**`upscale_conv.comp` — 3×3 Convolution Compute Shader**:

Write the complete GLSL shader. It must:
- Accept: input activation buffer, weight buffer, bias buffer, output buffer
- Push constants: `in_channels`, `out_channels`, `in_height`, `in_width`, `kernel_size` (3), `padding` (1)
- Layout: each thread handles one output element `(out_channel, out_h, out_w)`
- Work group size: `local_size_x=8, local_size_y=8, local_size_z=1` — these 8×8 spatial tiles map well to Xe-LP's EU workload
- Use `float16_t` (GLSL) via `#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require`
- Implement the convolution loop with padding (clamp-to-edge border mode)
- After accumulation, apply ReLU: `output = max(output, float16_t(0.0))`
- Explain the memory access pattern and why `local_size_z=out_channels` was NOT chosen (exceeds `maxComputeWorkGroupSize.z` for large channel counts)

**`upscale_pixelshuffle.comp` — Pixel Shuffle Shader**:

Write the complete GLSL shader:
- Input: buffer of shape `(1, C*r^2, H, W)` where r is upscale factor
- Output: buffer of shape `(1, C, H*r, W*r)`
- For each output pixel `(c, oh, ow)`: compute source channel index `c_in = c * r^2 + (oh % r) * r + (ow % r)`, source spatial `(oh // r, ow // r)`
- Clamp output to [0, 1]

**`upscaler.cpp/h` — C++ Dispatcher**:

Implement `XeUpscaler` class:
```cpp
class XeUpscaler {
public:
    XeUpscaler(VkDevice device, XeFrameWeightLoader& weights, uint32_t scale_factor);
    void init(uint32_t max_input_width, uint32_t max_input_height);
    void dispatch(VkCommandBuffer cmd, XfImage input, XfImage output);
    void destroy();
private:
    // Per-layer: VkPipeline, VkPipelineLayout, VkDescriptorSet, XfBuffer (activations)
    std::vector<LayerDispatch> layers_;
    XfBuffer weights_buf_; // contiguous weights for all layers
    uint32_t scale_;
};
```

The `dispatch()` method must:
- Insert `vkCmdPipelineBarrier` between each layer dispatch (compute→compute, `VK_ACCESS_SHADER_WRITE_BIT` → `VK_ACCESS_SHADER_READ_BIT`)
- Use push constants to pass per-layer shape parameters
- Execute PixelShuffle dispatch last
- Insert a final barrier before returning (compute→transfer or compute→graphics depending on next stage)

Explain the dispatch grid calculation:
- For a 720p input (1280×720), `out_channels=32`, work group 8×8: dispatch `(1280/8, 720/8, 32/1)` = `(160, 90, 32)` groups — is this within `maxComputeWorkGroupCount`? Verify and handle overflow.

**CHECKPOINT 3**: Present model architecture diagram, full training code, export utility, all shaders, and C++ dispatcher. Pause for confirmation.

---

## PHASE 4 — FRAME GENERATION MODEL: MOTION ESTIMATION & FRAME SYNTHESIS

### 4.1 — Frame Generation Architecture: XeFlow-Mini

Design **XeFlow-Mini**, a lightweight optical flow + frame synthesis network. This is architecturally more complex than the upscaler. The frame generation pipeline has three sub-networks:

**Sub-network A — FlowNet-Micro (Motion Estimation)**:
Estimate bidirectional optical flow between frame I_t and I_{t+1}.
```
Input: concat(I_t, I_{t+1}) → shape (1, 6, H, W)

Encoder:
  E0: Conv2D(6,  32, k=3, s=1, p=1) → LeakyReLU(0.1)
  E1: Conv2D(32, 64, k=3, s=2, p=1) → LeakyReLU(0.1)  [H/2, W/2]
  E2: Conv2D(64, 64, k=3, s=2, p=1) → LeakyReLU(0.1)  [H/4, W/4]
  E3: Conv2D(64, 96, k=3, s=2, p=1) → LeakyReLU(0.1)  [H/8, W/8]

Decoder (flow prediction at each scale):
  D3: Conv2D(96, 64, k=3, p=1) → LeakyReLU → Conv2D(64, 4, k=3, p=1)
      [4 channels: flow_t→t+1 (2ch) + flow_t+1→t (2ch)] at H/8, W/8
  D2: Upsample(×2) + concat(E2) → Conv2D(68, 64, k=3, p=1) → LeakyReLU → Conv2D(64, 4, k=3, p=1) at H/4
  D1: Upsample(×2) + concat(E1) → Conv2D(68, 32, k=3, p=1) → LeakyReLU → Conv2D(32, 4, k=3, p=1) at H/2
  D0: Upsample(×2) + concat(E0) → Conv2D(38, 32, k=3, p=1) → LeakyReLU → Conv2D(32, 4, k=3, p=1) at H
      Final flow at full resolution.
```

Explain why a coarse-to-fine (encoder-decoder) structure is used for optical flow rather than a single-scale approach. Relate it to the aperture problem and large displacement handling.

**Sub-network B — WarpNet (Frame Warping)**:
Given flow fields F_{t→t+1} and F_{t+1→t}, and a time parameter `τ ∈ (0,1)`:
1. Scale flows: `F_t_to_τ = -τ * F_{t→t+1}`, `F_{t+1}_to_τ = (1-τ) * F_{t+1→t}`
2. Backward warp I_t using F_t_to_τ → Ĩ_t
3. Backward warp I_{t+1} using F_{t+1}_to_τ → Ĩ_{t+1}
4. Blend: I_synth = (1-τ) * Ĩ_t + τ * Ĩ_{t+1}
   This is the basic linear blend — Sub-network C refines it.

Backward warping implementation: for each output pixel (x,y), sample the source image at (x + flow_x, y + flow_y) using bilinear interpolation.

**Sub-network C — RefineNet (Occlusion-Aware Blending)**:
```
Input: concat(Ĩ_t, Ĩ_{t+1}, I_t, I_{t+1}, F_t_to_τ, F_{t+1}_to_τ) → shape (1, 16, H, W)

R0: Conv2D(16, 32, k=3, p=1) → ReLU
R1: Conv2D(32, 32, k=3, p=1) → ReLU
R2: Conv2D(32, 16, k=3, p=1) → ReLU
R3: Conv2D(16,  3, k=3, p=1) → Sigmoid [output residual, range 0-1]

Output: I_final = I_synth + RefineNet_output (clamped to [0,1])
```

Explain the role of RefineNet: it learns to correct artifacts from flow inaccuracies near object boundaries, fast motion, and occlusion regions. The concatenation of original frames helps it "fill in" occluded regions that warping cannot reconstruct.

### 4.2 — Frame Generation Training

**Dataset** (`training/framegen/dataset.py`):
- Use Vimeo-90K Triplet dataset (51,312 training triplets of 3 consecutive frames at 448×256)
- Each sample: (frame_0, frame_1, frame_2) — train to predict frame_1 from frame_0 and frame_2
- Random crop 256×256, random horizontal flip, random temporal reversal (frame order: sometimes swap frame_0 and frame_2 and negate flow — this teaches bidirectional consistency)
- Normalize to [0,1]

**Loss Function** (`training/framegen/loss.py`):
```
L_total = λ1 * L_reconstruction + λ2 * L_perceptual + λ3 * L_flow_smooth + λ4 * L_temporal
```
- `L_reconstruction`: L1 loss between predicted I_mid and ground truth frame_1
- `L_perceptual`: Same VGG-11 feature loss as upscaler training, but applied to the predicted middle frame
- `L_flow_smooth`: Total variation regularization on flow fields — penalizes abrupt flow discontinuities:
  `TV(F) = mean(|F[x+1,y] - F[x,y]| + |F[x,y+1] - F[x,y]|)` — encourages smooth flow in textureless regions
- `L_temporal`: For stability across synthesized frames, penalize the difference between I_synth at τ=0.25 and a naive linear blend of frame_0 and frame_1 — ensures temporal coherence at sub-integer intervals. λ4=0.02

**Training Loop** (`training/framegen/train.py`):
- Train FlowNet-Micro and RefineNet jointly (end-to-end)
- Optimizer: AdamW, lr=1e-4 for FlowNet, lr=2e-4 for RefineNet (use param groups)
- Warmup: linear lr warmup over 5 epochs
- Total: 300 epochs, cosine schedule after warmup
- Checkpoint: save FlowNet and RefineNet separately (`flownet_epoch_{N}.pth`, `refinenet_epoch_{N}.pth`)
- Val metric: PSNR on Vimeo-90K test set

**Expected results**:
- After 100 epochs: PSNR > 33 dB on Vimeo-90K test
- After 300 epochs: PSNR > 35 dB

### 4.3 — Vulkan Inference: Frame Generation

Write all shaders and the C++ dispatcher:

**`framegen_flow.comp` — FlowNet Inference**:
- Structure similar to `upscale_conv.comp` but handles the encoder-decoder architecture
- Requires intermediate feature maps at each scale — allocate `XfBuffer` objects per encoder level
- Upsample (bilinear) shader needed: write `framegen_upsample.comp` that performs bilinear upsampling of a buffer by ×2 in both spatial dimensions

**`framegen_warp.comp` — Backward Warping Shader**:
Write complete GLSL:
- Input: source image buffer (RGB float16, H×W), flow buffer (float16×2, H×W), output buffer
- Push constant: `τ` (float), image dimensions
- Each thread handles one output pixel (oh, ow):
  - `sample_x = oh + flow_u * (-τ)` (for t→τ direction), bilinear clamp
  - `sample_y = ow + flow_v * (-τ)`
  - Bilinear sample from source: implement in GLSL with bounds clamping
  - Write RGB to output

Detail the bilinear interpolation implementation in GLSL — do not use `texture()` sampler (we use storage buffers, not sampler2D, for maximum control and float16 precision):
```glsl
float16_t bilinear_sample(readonly buffer float16_t[] src, float x, float y, int W, int H) {
    // Floor, frac, 4-sample weighted average
}
```

**`framegen_blend.comp` — Synthesis Blending Shader**:
- Takes warped_t, warped_t1, refine_output, τ
- Computes: `(1-τ)*warped_t + τ*warped_t1 + refine_output`, clamped to [0,1]
- One thread per pixel, trivially parallel

**`framegen.cpp/h` — XeFrameGen Class**:
```cpp
class XeFrameGen {
public:
    XeFrameGen(VkDevice device, XeFrameWeightLoader& weights);
    void init(uint32_t width, uint32_t height);
    // Synthesize frame at time τ between frame_a and frame_b
    void synthesize(VkCommandBuffer cmd, XfImage frame_a, XfImage frame_b,
                    float tau, XfImage output);
    void destroy();
};
```

Detail the full command buffer recording sequence for one `synthesize()` call:
1. Dispatch FlowNet encoder passes (E0→E3), inserting compute barriers between each
2. Dispatch FlowNet decoder passes (D3→D0), inserting barriers
3. Dispatch WarpNet (two calls: one for frame_a warped to τ, one for frame_b warped to τ)
4. Dispatch RefineNet (R0→R3)
5. Dispatch blend shader (final output)
Total expected GPU time on Xe-LP: 3–6ms for 720p interpolation.

**CHECKPOINT 4**: Present full model architecture (both sub-networks), complete training scripts, all shaders, C++ dispatcher. Pause.

---

## PHASE 5 — CAPTURE, PIPELINE ORCHESTRATION & SOURCE MANAGEMENT

### 5.1 — Abstract Capture Interface

Define `capture_base.h`:
```cpp
struct CaptureFrame {
    uint8_t*     data;        // RGB or YUV data
    size_t       data_size;
    uint32_t     width, height;
    uint64_t     timestamp_ns; // monotonic clock
    PixelFormat  format;       // RGB24, YUVI420, NV12
};

class CaptureBase {
public:
    virtual ~CaptureBase() = default;
    virtual bool open(const CaptureConfig& config) = 0;
    virtual bool read_frame(CaptureFrame& out) = 0; // blocking, returns false on EOF/error
    virtual void close() = 0;
    virtual std::string source_name() const = 0;
    virtual uint32_t native_fps() const = 0;
};
```

### 5.2 — File Capture (FFmpeg)

Implement `capture_file.cpp` using `libavformat` and `libavcodec` (not the `ffmpeg` binary):
- `avformat_open_input` → `avformat_find_stream_info` → find video stream → `avcodec_find_decoder` → `avcodec_open2`
- `av_read_frame` → `avcodec_send_packet` / `avcodec_receive_frame`
- Convert decoded frame to RGB using `sws_scale` (libswscale) with `SWS_LANCZOS` for quality
- Return `CaptureFrame` with pixel data copied to a pooled buffer (pre-allocated, avoid malloc per frame)
- Support seeking: implement `seek_to_second(double t)` using `av_seek_frame`

### 5.3 — PipeWire Portal Capture (Wayland)

Implement `capture_pipewire.cpp`:
- Use `libpipewire` and the XDG Desktop Portal (`org.freedesktop.portal.ScreenCast`) via D-Bus
- Walk through the complete sequence: D-Bus call to `CreateSession` → `SelectSources` (monitor or window) → `Start` → get PipeWire remote → `pw_stream_new` → `on_process` callback
- In `on_process`: `pw_stream_dequeue_buffer` → copy `spa_data` to `CaptureFrame` → `pw_stream_queue_buffer`
- Handle format negotiation: prefer `SPA_VIDEO_FORMAT_BGRx` or `SPA_VIDEO_FORMAT_RGBA`, convert to RGB24

### 5.4 — Pipeline Orchestrator

Implement `core/pipeline.cpp`:

The pipeline runs as a state machine with states: `IDLE → STARTING → RUNNING → PAUSING → STOPPED → ERROR`.

**Ring Buffer Architecture**:
```cpp
template<typename T, size_t N>
class SPSCRingBuffer {
    // Lock-free: one producer, one consumer
    // Uses std::atomic<size_t> head and tail (cache-line padded to avoid false sharing)
    // T must be trivially copyable or have explicit move semantics
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    T data_[N];
public:
    bool push(const T& item);        // producer side
    bool pop(T& item);               // consumer side
    size_t size() const;
};
```

Define the concrete ring buffer instances:
- `capture_to_preprocess`: holds `CaptureFrame`, capacity 4
- `preprocess_to_upscale`: holds `XfImage` handles (GPU-resident), capacity 2
- `upscale_to_framegen`: holds `XfImage` pairs (frame N and N+1), capacity 2
- `framegen_to_display`: holds `XfImage` handle (final frame), capacity 4

**Frame Pool**:
Pre-allocate a pool of `XfImage` objects at startup. The pool holds 16 images at the output resolution. Images are checked out by pipeline stages and returned to the pool after the display stage is done with them. Implement `FramePool` with `acquire()` and `release(const XfImage&)` methods.

**Thread Entry Functions**:
Write the main loop body for each thread:
1. `capture_thread_fn()`: calls `source->read_frame()`, converts color space on CPU (small helper function), pushes to `capture_to_preprocess`
2. `preprocess_thread_fn()`: pops from `capture_to_preprocess`, allocates GPU image from pool, records and submits `preprocess.comp` command buffer, pushes GPU image to `preprocess_to_upscale`
3. `upscale_thread_fn()`: buffers pairs of frames, calls `XeUpscaler::dispatch()`, pushes result to `upscale_to_framegen`
4. `framegen_thread_fn()`: pops frame pairs, calls `XeFrameGen::synthesize()` for τ=0.5 (and τ=0.25, τ=0.75 if 3× framerate mode), pushes synthesized frames to `framegen_to_display`
5. `display_thread_fn()`: pops final frames, presents to swapchain or encodes to output

Implement graceful shutdown: on `SIGINT`, set an atomic `shutdown_flag`, each thread exits its loop, joins cleanly.

**CHECKPOINT 5**: Present ring buffer implementation, pipeline state machine, all thread entry functions, frame pool. Pause.

---

## PHASE 6 — GTK4 USER INTERFACE

### 6.1 — Application Structure

The UI is built with GTK4 + libadwaita following the GNOME HIG (Human Interface Guidelines). The app ID is `io.github.xeframe`.

Implement `ui/app.cpp` as a `GtkApplication` subclass (using GObject C API or a thin C++ wrapper):
- Application ID: `io.github.xeframe`
- Flags: `G_APPLICATION_DEFAULT_FLAGS`
- Connect `activate` signal to `on_activate()`
- In `on_activate()`: create the `MainWindow` and present it

### 6.2 — Main Window

Implement `ui/main_window.cpp`:

**Layout** (AdwApplicationWindow):
```
AdwApplicationWindow
└── AdwToolbarView
    ├── AdwHeaderBar (top)
    │   ├── [Left] menu button (hamburger) → popover with: About, Preferences, Quit
    │   └── [Right] power button (start/stop pipeline)
    └── GtkBox (vertical, main content)
        ├── AdwPreferencesGroup: "Source"
        │   ├── AdwComboRow: "Source Type" [File | Screen Capture | Camera]
        │   ├── AdwActionRow: "File Path" (shows only when File selected)
        │   └── AdwActionRow: "Capture Region" (shows only when Screen selected)
        ├── AdwPreferencesGroup: "Processing"
        │   ├── AdwComboRow: "Upscale Factor" [Off | ×1.5 | ×2 | ×3]
        │   ├── AdwComboRow: "Frame Generation" [Off | ×2 | ×3]
        │   └── AdwSwitchRow: "Post-processing Sharpen"
        ├── AdwPreferencesGroup: "Output"
        │   ├── AdwComboRow: "Output" [Preview Window | Virtual Camera | Encode to File]
        │   └── AdwActionRow: "Output Path"
        └── [Metrics Area] — GtkGrid showing live stats
            ├── GPU EU Utilization: ██████░░ 74%
            ├── Input FPS: 60
            ├── Output FPS: 120 (with frame gen ×2)
            ├── Pipeline Latency: 12ms
            └── VRAM (Shared): 312 MB / 8192 MB
```

The metrics area updates every 500ms via `g_timeout_add(500, update_metrics_cb, window)`.

### 6.3 — Profile System

Implement a profile system: users can save/load named processing configurations.

- Profile file format: JSON at `~/.config/xeframe/profiles/`
- Fields: `name`, `source_type`, `upscale_factor`, `framegen_multiplier`, `sharpen_enabled`, `output_type`
- Implement `ProfileManager`: load all profiles from disk at startup, save on change, export/import
- UI: `AdwPreferencesGroup` "Profiles" with `AdwActionRow` entries, add/remove buttons, apply button

### 6.4 — On-Screen Overlay

Implement `ui/overlay.cpp`:
A transparent always-on-top window that shows minimal metrics as a corner overlay (inspired by MangoHUD):
- Created as `GtkWindow` with `gtk_window_set_decorated(FALSE)`, `gtk_widget_set_opacity(0.7)`
- Position: top-right corner of the target display
- Content: "XEFRAME | FPS IN: 60 → OUT: 120 | LAT: 12ms | EU: 74%"
- Toggle via keyboard shortcut (configurable, default: `Ctrl+Shift+O`)
- Rendered using `GtkDrawingArea` with Cairo for custom styling (dark background, monospace font)

**CHECKPOINT 6**: Present all UI code (GtkApplication setup, MainWindow layout, ProfileManager, overlay). Pause.

---

## PHASE 7 — POST-PROCESSING, OUTPUT & PACKAGING

### 7.1 — Post-Processing Shaders

**`postprocess.comp` — Adaptive Sharpening + Tone Mapping**:

**Sharpening** (unsharp mask variant):
```glsl
// At each pixel, compute blurred value via 3×3 box blur, then:
// sharpened = original + strength * (original - blurred)
// strength: push constant float, 0=off, 1=strong
```

**Tone Mapping** (optional HDR→SDR):
- Implement the `Reinhard` operator: `rgb_out = rgb_in / (1 + rgb_in)`
- Also implement `ACES` approximate: `rgb_out = (rgb_in*(2.51*rgb_in+0.03))/(rgb_in*(2.43*rgb_in+0.59)+0.14)`
- Selectable via push constant enum

### 7.2 — Virtual V4L2 Camera Output

Implement `export/virtual_v4l2.cpp`:
- Requires `v4l2loopback` kernel module (`sudo modprobe v4l2loopback`)
- Open `/dev/video{N}` (detect the loopback device)
- Use `ioctl(VIDIOC_S_FMT)` to set output format (YUYV or MJPEG)
- Each frame: convert RGB final frame to YUYV (CPU-side, fast enough at 1080p60), `write()` to fd
- The virtual camera appears as a real webcam to any application (OBS, Discord, etc.)

### 7.3 — Debian Package

Write `packaging/debian/control`:
```
Package: xeframe
Version: 0.1.0
Architecture: amd64
Depends: libvulkan1, libgtk-4-1, libadwaita-1-0, libavcodec60 | libavcodec59,
         libavformat60 | libavformat59, libswscale7 | libswscale6,
         libpipewire-0.3-0, v4l2loopback-dkms
Recommends: vulkan-tools, mesa-vulkan-drivers, intel-media-va-driver
Description: Real-time video upscaler and frame generator for Intel Iris Xe iGPUs
 XEFRAME provides custom-trained CNN-based upscaling and frame interpolation
 optimized for Intel Xe-LP integrated graphics on Ubuntu Linux.
```

Write `packaging/debian/rules` (CMake-based build).

Write an `install.sh` one-step installer:
- Builds from source
- Installs to `/usr/local/bin/xeframe`
- Installs `.desktop` entry and application icon
- Optionally downloads pre-trained `.xefm` model weights from a configurable URL

**CHECKPOINT 7**: Present all post-processing shaders, V4L2 output, and packaging files. Pause.

---

## PHASE 8 — PERFORMANCE PROFILING, OPTIMIZATION & DEPLOYMENT

### 8.1 — Intel Xe-LP Specific Optimizations

Detail each of the following optimizations and implement them:

**A. Subgroup Operations**:
Xe-LP supports subgroup operations (`GL_KHR_shader_subgroup`). In the convolution shader's inner accumulation loop, use subgroup reductions where channels map to subgroup lanes:
```glsl
// Sum partial dot products across subgroup lanes
float16_t partial = ...; // each lane computes one input channel contribution
float16_t total = subgroupAdd(partial); // hardware-accelerated reduction
```
Explain when this is beneficial vs. when the overhead of subgroup coordination negates the gain.

**B. Workgroup Shared Memory (LDS)**:
For the 3×3 convolution shader, tile the input spatially into workgroup shared memory (local data share) to reduce redundant global memory reads:
- Load a 10×10 patch of input into `shared float16_t tile[10][10][CHANNELS_PER_WG]`
- All 8×8 threads in the workgroup read their 3×3 neighborhood from shared memory instead of global memory
- Each border thread loads an extra row/column of padding
Calculate the shared memory requirement and verify it fits within Xe-LP's 64KB local memory per workgroup.

**C. Float16 Pipeline**:
Verify that all activations and weights are float16 throughout the entire Vulkan inference pipeline. Identify any operations that must remain float32 (e.g., accumulation in the convolution loop should use `float32_t` accumulators and cast to `float16_t` only for storage — explain this precision decision).

**D. EU Occupancy**:
On Xe-LP, each EU can hold 7 hardware threads. With 80 EUs, that's 560 concurrent threads. Our workgroup size is 64 (8×8). Calculate: for a 720p upscale (1280×720 output), we dispatch `160 × 90 = 14,400` workgroups × 64 threads = 921,600 threads. This is GPU-saturating — the driver will schedule in waves. Explain how to measure actual EU occupancy using `intel_gpu_top` and `VK_EXT_calibrated_timestamps`.

**E. Pipeline Overlap (Async Stages)**:
Design the command buffer submission strategy for Stage 2 (upscaling) and Stage 3 (frame generation) to overlap on the GPU's timeline:
- While frame N+1's upscale is running, frame N's frame generation can run (they operate on different frame data)
- Use Vulkan timeline semaphores to express this dependency:
  ```
  upscale(N)   ─── signal sem[N]=1 ───► framegen(N) [waits sem[N]>=1 and sem[N+1]>=1]
  upscale(N+1) ─── signal sem[N+1]=1 ─►
  ```
- Submit both upscale commands to the queue before waiting for either, allowing the GPU scheduler to interleave them

### 8.2 — Profiling Infrastructure

Implement GPU-side profiling using `VK_EXT_calibrated_timestamps`:
- At the start and end of each major compute dispatch, insert `vkCmdWriteTimestamp2` calls
- At frame end, read timestamps back via a host-visible buffer
- Compute per-stage GPU time in microseconds
- Feed into the `stats.cpp` monitoring system, which averages over 60 frames and exposes via a `Stats` struct accessible from the UI

Implement `tools/benchmark.cpp`:
- A standalone command-line benchmark that:
  - Loads models from `.xefm` files
  - Runs the full upscale+framegen pipeline on a synthetic input for 1000 iterations
  - Reports: mean GPU time per stage, P99 latency, estimated max throughput FPS at each resolution
  - Compares float16 vs float32 inference time (runs both if float32 fallback is compiled in)

### 8.3 — Regression Test Suite

Write `tests/test_pipeline.cpp`:
- A self-contained test that:
  1. Initializes the Vulkan context
  2. Loads test weights (minimal random weights, not trained — just for structural testing)
  3. Runs the pipeline on a 128×128 synthetic frame for 10 iterations
  4. Verifies output image dimensions are correct
  5. Verifies no Vulkan validation layer errors were reported (hook the debug callback)
  6. Measures and reports average dispatch time

Write `tests/test_ringbuffer.cpp`:
- Multi-threaded stress test of `SPSCRingBuffer<int, 1024>`
- Producer thread pushes 1,000,000 incrementing integers
- Consumer thread pops them and verifies order and completeness
- Times the operation: must complete in < 100ms on any modern machine

### 8.4 — First-Run Tuning Wizard

On first launch, run a brief tuning routine:
1. Dispatch a synthetic 1920×1080 workload with the upscale shader for 100 iterations
2. Measure average frame time
3. Based on the result, recommend a preset: "Performance" (720p→1080p, ×2 framegen), "Quality" (1080p→1440p, no framegen), "Balanced"
4. Save the recommendation to the user's profile config as the default
5. This entire routine runs in a background thread with a GTK progress bar in the UI

**CHECKPOINT 8**: Present all optimization implementations, profiling code, benchmark tool, and tuning wizard. Pause.

---

## PHASE 9 — DOCUMENTATION, CHANGELOG & MAINTENANCE ROADMAP

### 9.1 — Man Page

Write a complete `man 1 xeframe` manual page in troff format:
- NAME, SYNOPSIS, DESCRIPTION, OPTIONS (all CLI flags), FILES (config paths, model weight paths, log location), NOTES (Wayland vs X11 behavior, V4L2 module requirement), BUGS, SEE ALSO, AUTHOR

### 9.2 — Technical Architecture Document

Write `docs/ARCHITECTURE.md` (full, detailed) covering:
- System overview diagram (ASCII art)
- Pipeline data flow with buffer sizes and frame timing budget
- Vulkan object lifecycle (when each VkImage, VkBuffer, VkPipeline is created and destroyed)
- Model architecture summary tables (layer counts, parameter counts, FLOPs at 720p)
- Performance budget: 720p→1080p upscale in 4ms, frame generation at 720p in 5ms, display overhead 2ms → total 11ms = 90 FPS theoretical maximum output rate
- Known limitations and workarounds (Wayland portals requiring user permission, V4L2 loopback module requirement, memory pressure on 8GB shared RAM systems)

### 9.3 — Contributor Guide

Write `CONTRIBUTING.md`:
- How to add a new capture backend
- How to add a new post-processing shader
- How to train a new model variant and export it
- Code style: clang-format config (based on LLVM style), clang-tidy enabled, no raw new/delete (use RAII wrappers), no exceptions (use `std::expected` or error codes), no RTTI

### 9.4 — Version Roadmap

Define a concrete roadmap:

**v0.1.0 — Foundation** (current build target):
- File capture, screen capture (X11 only)
- Upscaling only (frame generation disabled)
- Preview window output
- Basic GTK4 UI with no profiles

**v0.2.0 — Frame Generation**:
- Full frame generation pipeline enabled
- Wayland/PipeWire screen capture
- V4L2 virtual camera output
- Profile system

**v0.3.0 — Quality**:
- XeScaler-Medium (larger model, better quality, optional for higher-end hardware)
- Per-game artifact correction fine-tuning (load game-specific LoRA-style weight deltas — design the weight delta format)
- HDR pass-through support

**v0.4.0 — Multi-GPU & Community**:
- AMD iGPU support via ROCm/OpenCL path (the Vulkan shaders are already vendor-agnostic — explain what needs to change)
- Model weight sharing: community `.xefm` model repository format and download manager in UI

---

## PHASE 10 — FINAL INTEGRATION & FIRST WORKING BUILD

### 10.1 — Integration Checklist

Walk me through the complete first build in the following order:

1. Run `setup_dev_env.sh` — verify all dependencies installed
2. `mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug`
3. Verify shader compilation target runs and embeds SPIR-V
4. `make -j$(nproc)` — fix any compilation errors (list the most likely ones and their fixes)
5. `./xeframe --version` — should print version and Vulkan device info
6. `./xeframe --benchmark --input /path/to/test_720p.mp4 --upscale 2` — runs headless benchmark
7. `./xeframe --gui` — opens GTK4 window
8. Load a test video, enable upscaling, hit play — verify frames appear in preview

### 10.2 — Debugging Guide

Document the most common failure modes and debugging steps:

**Vulkan Validation Errors**:
- Set `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` before running
- Common on Xe-LP: `VK_ERROR_OUT_OF_DEVICE_MEMORY` (shared memory exhausted — reduce intermediate buffer sizes)
- Descriptor set mismatch: verify binding indices in shaders match descriptor set layout exactly

**Shader Compilation Failures**:
- Float16 extension not recognized: verify `glslc` version ≥ 2021, add `--target-env=vulkan1.3`
- Subgroup size request rejected: query `VkPhysicalDeviceSubgroupSizeControlProperties` and respect `minSubgroupSize/maxSubgroupSize`

**Pipeline Stalls**:
- If the display thread is starving: the framegen stage is taking too long — enable the "Performance" preset which skips frame generation
- If the capture thread is dropping frames: check PipeWire buffer size, increase to 8 frames via `pw_stream_connect` parameters

**GTK4 Crashes**:
- All GTK calls must happen on the main thread — use `g_idle_add()` to post UI updates from worker threads

### 10.3 — First Training Run

After the application builds and runs (even without trained models — it will output garbage frames):

1. `cd training && source .venv/bin/activate`
2. `bash data/download_div2k.sh` — downloads ~8GB, verify checksums
3. `cd upscaler && python train.py --epochs 10 --device cpu` — do a 10-epoch smoke test on CPU (should take ~2 hours on i5-1235U, produces a rough but functional model)
4. Check TensorBoard: `tensorboard --logdir runs/` — view loss curves
5. `python validate.py --checkpoint checkpoints/upscaler_epoch_10.pth` — check PSNR > 25 dB (smoke test threshold)
6. `python ../shared/export.py --model upscaler --checkpoint checkpoints/upscaler_epoch_10.pth --output ../../models/upscaler_smoke.xefm`
7. Run the app with this model: `./xeframe --upscale-model models/upscaler_smoke.xefm --input test.mp4 --preview`
8. Verify the output is upscaled (it will look soft — full quality requires all 200 epochs)

For Intel XPU training (if `torch.xpu.is_available()` returns True):
9. `python train.py --epochs 200 --device xpu` — full training, ~12–18 hours on Iris Xe

### 10.4 — Release Build & Distribution

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
make install
# Or build .deb:
cd ..
dpkg-buildpackage -us -uc -b
# Output: xeframe_0.1.0_amd64.deb
```

Test the .deb on a clean Ubuntu 24.04 VM to verify all dependencies resolve correctly.

---

## APPENDIX A — KEY TECHNICAL DECISIONS REFERENCE

Keep this section as a quick-reference for decisions made during the design phases:

| Decision | Choice | Reason |
|---|---|---|
| Compute API | Vulkan Compute (SPIR-V) | Universal on Xe-LP via ANV/Mesa, full control |
| Fallback API | OpenCL 3.0 | Available on all Intel iGPUs if Vulkan fails |
| Memory | Custom pool allocator | No VMA dependency, understand every byte |
| Model format | .xefm custom binary | No ONNX/flatbuffers dependency |
| Training backend | PyTorch + IPEX XPU | Native Intel GPU training on Xe-LP |
| UI toolkit | GTK4 + libadwaita | Native GNOME, Wayland-first |
| Screen capture | PipeWire (primary), XShm (fallback) | Wayland compatibility |
| Upscaler arch | Custom CNN + PixelShuffle | Fast, no RT cores needed |
| Frame gen arch | Custom FlowNet + WarpNet + RefineNet | No pre-built RIFE/DAIN usage |
| Float precision | Float16 inference, Float32 accumulation | Speed + accuracy balance |
| Inter-thread comms | Lock-free SPSC ring buffers | Real-time performance |
| Sharpening | Unsharp mask compute shader | Simple, effective, controllable |

---

## APPENDIX B — QUICK COMMAND REFERENCE

```bash
# Build (debug)
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc)

# Run with validation
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./xeframe --gui

# Benchmark
./xeframe --benchmark --input test.mp4 --upscale 2 --framegen 2

# Train upscaler (CPU)
cd training && source .venv/bin/activate
cd upscaler && python train.py --epochs 200 --device cpu

# Train upscaler (Intel XPU, if available)
cd upscaler && python train.py --epochs 200 --device xpu

# Export model
python ../shared/export.py --model upscaler --checkpoint checkpoints/upscaler_best.pth \
  --output ../../models/upscaler_v1.xefm

# Inspect model weights
python tools/weight_inspect.py models/upscaler_v1.xefm

# Build .deb package
dpkg-buildpackage -us -uc -b

# Install
sudo dpkg -i ../xeframe_0.1.0_amd64.deb
```

---

## FINAL INSTRUCTION TO GEMINI CLI

You have now received the complete specification for XEFRAME. Here is how to proceed:

1. **Acknowledge** this full specification. Summarize it back in your own words in 10–15 bullet points to confirm understanding.
2. **Ask clarifying questions** (maximum 5) about any ambiguous technical choices before writing a single line of code.
3. **Begin Phase 1**. For each phase, produce:
   - Complete, compilable C++ code (no pseudocode for critical paths)
   - Complete, runnable Python code (no placeholder `# TODO` comments in training scripts)
   - Complete GLSL shaders (no abbreviated kernels)
   - All configuration and build files in full
4. **At each CHECKPOINT**, stop and wait for my explicit "continue" before proceeding to the next phase.
5. If at any point you are about to use a pre-built upscaling library, pre-trained model, or external inference engine, **stop** and design the equivalent from scratch instead. This is non-negotiable.
6. **Optimize for Intel Xe-LP throughout**. Whenever a design choice arises (shader layout, memory pattern, dispatch grid), choose the option that works best for 80-EU Xe-LP with unified DDR4 memory, not for discrete AMD/NVIDIA GPUs.
7. **Track progress**: at the end of each phase, output a progress summary: `[Phase N/10 complete — N files written — estimated lines of code: XXXX]`

Begin now.
