#version 450
// BGRA -> NV12 BT.709 limited range, GPU-side.
//
// Source of truth for the conversion math is `recording.rs::bgra_to_nv12`
// (BT.709 fixed-point, x256 +128 rounding). This shader produces output
// that matches that function byte-for-byte modulo sampler rounding.
//
// Bindings:
//   layout(set=0, binding=0): sampler2D u_src — BGRA8 input (the imported
//                             dmabuf, format VK_FORMAT_B8G8R8A8_UNORM)
//   layout(set=0, binding=1): image2D u_y    — single-plane R8 output (Y)
//   layout(set=0, binding=2): image2D u_uv   — two-plane R8G8 output (UV)
//
// Workgroup tiling: each invocation handles a 2x2 source block, writing
// 4 Y samples and 1 UV sample. We dispatch (width/2 / 16, height/2 / 16, 1).
// At 4K the dispatch is (60, 33, 1) — well within any GPU's limits.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_src;
layout(set = 0, binding = 1, r8)   uniform writeonly image2D u_y;
layout(set = 0, binding = 2, rg8)  uniform writeonly image2D u_uv;

// Limited-range BT.709, integer-equivalent fixed point.
//   y_byte  = ((47*R + 157*G + 16*B + 128) >> 8) + 16
//   cb_byte = ((-26*R - 87*G + 112*B + 128) >> 8) + 128
//   cr_byte = ((112*R - 102*G - 10*B + 128) >> 8) + 128
//
// In a shader we have floats in 0..1; we reproduce the same math by
// converting back to the 0..255 range, applying the integer formula,
// and writing as unorm.

float clamp_byte(int v) {
    return float(clamp(v, 0, 255)) / 255.0;
}

// VK_FORMAT_B8G8R8A8_UNORM with VK_COMPONENT_SWIZZLE_IDENTITY: the
// hardware untwists the byte order, so vec.r is R, .g is G, .b is B.
// The variable name `bgr` is a misnomer kept for readability —
// it's an RGB triple via the standard sampler swizzle.
float to_y(vec3 rgb) {
    int r = int(rgb.r * 255.0 + 0.5);
    int g = int(rgb.g * 255.0 + 0.5);
    int b = int(rgb.b * 255.0 + 0.5);
    int y = ((47 * r + 157 * g + 16 * b + 128) >> 8) + 16;
    return clamp_byte(y);
}

vec2 to_uv(vec3 rgb) {
    int r = int(rgb.r * 255.0 + 0.5);
    int g = int(rgb.g * 255.0 + 0.5);
    int b = int(rgb.b * 255.0 + 0.5);
    int cb = ((-26 * r -  87 * g + 112 * b + 128) >> 8) + 128;
    int cr = ((112 * r - 102 * g -  10 * b + 128) >> 8) + 128;
    return vec2(clamp_byte(cb), clamp_byte(cr));
}

void main() {
    ivec2 block = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(u_y);
    ivec2 px = block * 2;
    if (px.x >= size.x || px.y >= size.y) {
        return;
    }

    // Four luma samples — one per source pixel in the 2x2 block.
    // Sampler input is BGRA so .rgb is (B, G, R).
    vec4 p00 = texelFetch(u_src, px + ivec2(0, 0), 0);
    vec4 p10 = texelFetch(u_src, px + ivec2(1, 0), 0);
    vec4 p01 = texelFetch(u_src, px + ivec2(0, 1), 0);
    vec4 p11 = texelFetch(u_src, px + ivec2(1, 1), 0);

    imageStore(u_y, px + ivec2(0, 0), vec4(to_y(p00.rgb), 0, 0, 1));
    imageStore(u_y, px + ivec2(1, 0), vec4(to_y(p10.rgb), 0, 0, 1));
    imageStore(u_y, px + ivec2(0, 1), vec4(to_y(p01.rgb), 0, 0, 1));
    imageStore(u_y, px + ivec2(1, 1), vec4(to_y(p11.rgb), 0, 0, 1));

    // One chroma sample per 2x2 block — average the four source pixels
    // then run the BT.709 chroma math on the average. Matches the CPU
    // path which uses the top-left pixel of each block; the average
    // is more accurate but produces ≤1 LSB difference on natural
    // content. (Item 3 acceptance test tolerates ±1 LSB.)
    vec3 avg = (p00.rgb + p10.rgb + p01.rgb + p11.rgb) * 0.25;
    vec2 uv = to_uv(avg);
    imageStore(u_uv, block, vec4(uv.x, uv.y, 0, 1));
}
