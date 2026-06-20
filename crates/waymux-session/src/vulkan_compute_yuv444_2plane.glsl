#version 450
// BGRA -> 2-plane YUV 4:4:4 (G8_B8R8_2PLANE_444_UNORM), BT.709 limited range.
//
// Sister of `vulkan_compute.glsl` (NV12 / 4:2:0). Difference: chroma is
// kept at FULL resolution — one chroma sample per source pixel, no 2×2
// averaging. The picture layout is 2-plane (Y plane + interleaved UV
// plane) like NV12, just with U/V each storing full-res data.
//
// Why 2-plane 4:4:4 and not 3-plane? NVIDIA's Vulkan driver (560) only
// reports `G8_B8R8_2PLANE_444_UNORM` as a supported encode-src format
// for the H.264 Hi444PP profile. 3-plane (`G8_B8_R8_3PLANE_444_UNORM`)
// is rejected. This shader emits the 2-plane layout the driver wants.
//
// Bindings:
//   layout(set=0, binding=0): sampler2D u_src — BGRA8 input
//   layout(set=0, binding=1, r8) image2D u_y — Y plane (full-res, R8)
//   layout(set=0, binding=2, rg8) image2D u_uv — UV plane (full-res, R8G8)
//
// Workgroup tiling: 16×16, one invocation per source pixel.
// Dispatch (width/16, height/16, 1).

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_src;
layout(set = 0, binding = 1, r8) uniform writeonly image2D u_y;
layout(set = 0, binding = 2, rg8) uniform writeonly image2D u_uv;

float clamp_byte(int v) {
    return float(clamp(v, 0, 255)) / 255.0;
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(u_y);
    if (px.x >= size.x || px.y >= size.y) {
        return;
    }

    vec4 p = texelFetch(u_src, px, 0);
    int r = int(p.r * 255.0 + 0.5);
    int g = int(p.g * 255.0 + 0.5);
    int b = int(p.b * 255.0 + 0.5);

    // BT.709 limited-range, integer fixed point.
    int y  = ((  47 * r + 157 * g +  16 * b + 128) >> 8) + 16;
    int cb = (( -26 * r -  87 * g + 112 * b + 128) >> 8) + 128;
    int cr = (( 112 * r - 102 * g -  10 * b + 128) >> 8) + 128;

    imageStore(u_y, px, vec4(clamp_byte(y), 0.0, 0.0, 1.0));
    imageStore(u_uv, px, vec4(clamp_byte(cb), clamp_byte(cr), 0.0, 1.0));
}
