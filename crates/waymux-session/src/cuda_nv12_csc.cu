// BT.709 limited-range ARGB(block-linear cuArray) -> NV12, GPU-side.
// Replicates crates/waymux-session/src/vulkan_compute.glsl byte-for-byte
// (same integer fixed-point coefficients + 2x2-average chroma).
// Source texture is DRM ARGB8888 = byte order B,G,R,A, so uchar4.x=B .y=G .z=R.
//
// Compile to PTX with (run on any GPU VM with the CUDA toolkit):
//   nvcc -ptx -arch=sm_80 cuda_nv12_csc.cu -o cuda_nv12_csc.ptx
// arch=sm_80 (NOT sm_89) on purpose: the driver JITs PTX at runtime via
// cuModuleLoadData, and JIT only targets GPUs with compute capability >= the
// PTX `.target`. sm_80 covers the whole NVENC failover ladder — A6000 (Ampere
// 8.6), A100 (8.0), L40 (Ada 8.9), Hopper (9.0). This kernel is a plain integer
// CSC with no compute_89-gated features, so the lower virtual arch is
// performance-neutral (JIT still emits native SASS for the real GPU).
// No CUDA toolkit is needed at runtime.

extern "C" __device__ __forceinline__ int clampb(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

extern "C" __global__ void argb_to_nv12_bt709(
    cudaTextureObject_t src, unsigned char* dst, int width, int height, int pitch)
{
    int bx = blockIdx.x * blockDim.x + threadIdx.x;  // 2x2-block coordinates
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int x = bx * 2;
    int y = by * 2;
    if (x >= width || y >= height) return;

    unsigned char* yplane  = dst;
    unsigned char* uvplane = dst + (size_t)pitch * height;

    // Per-pixel Y for the up-to-4 pixels in the block (edge-guarded).
    int sumr = 0, sumg = 0, sumb = 0, n = 0;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int px = x + dx, py = y + dy;
            if (px >= width || py >= height) continue;
            uchar4 p = tex2D<uchar4>(src, px, py);
            int r = p.z, g = p.y, b = p.x;
            int yv = (((47 * r + 157 * g + 16 * b + 128) >> 8) + 16);
            yplane[(size_t)py * pitch + px] = (unsigned char)clampb(yv);
            sumr += r; sumg += g; sumb += b; ++n;
        }
    }
    // Chroma from the 2x2 average (n is 1,2, or 4 at edges).
    int ar = (sumr + n/2) / n, ag = (sumg + n/2) / n, ab = (sumb + n/2) / n;
    int cb = (((-26 * ar - 87 * ag + 112 * ab + 128) >> 8) + 128);
    int cr = (((112 * ar - 102 * ag - 10 * ab + 128) >> 8) + 128);
    unsigned char* uvrow = uvplane + (size_t)by * pitch + (size_t)bx * 2;
    uvrow[0] = (unsigned char)clampb(cb);
    uvrow[1] = (unsigned char)clampb(cr);
}
