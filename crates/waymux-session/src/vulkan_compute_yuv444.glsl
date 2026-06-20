#version 450
// BGRA -> YUV 4:4:4 (G8_B8_R8_3PLANE_444_UNORM), BT.709 limited range.
//
// Sister shader to vulkan_compute.glsl (BGRA -> NV12 4:2:0). The
// conversion math is identical (BT.709 limited-range integer fixed
// point); only the chroma sampling is different — Hi444PP / RangeExt
// keep U and V at full resolution, one chroma sample per source pixel.
//
// Bindings:
//   layout(set=0, binding=0): sampler2D u_src — BGRA8 input
//   layout(set=0, binding=1, r8) image2D u_y — Y plane (full-res, R8)
//   layout(set=0, binding=2, r8) image2D u_u — U plane (full-res, R8)
//   layout(set=0, binding=3, r8) image2D u_v — V plane (full-res, R8)
//
// Workgroup tiling: each invocation handles ONE source pixel and writes
// one Y, one U, one V sample. Dispatch (width/16, height/16, 1). At 4K
// that's (240, 135, 1).

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_src;
layout(set = 0, binding = 1, r8) uniform writeonly image2D u_y;
layout(set = 0, binding = 2, r8) uniform writeonly image2D u_u;
layout(set = 0, binding = 3, r8) uniform writeonly image2D u_v;

// Limited-range BT.709, integer-equivalent fixed point.
//   y_byte  = ((47*R + 157*G + 16*B + 128) >> 8) + 16
//   cb_byte = ((-26*R - 87*G + 112*B + 128) >> 8) + 128
//   cr_byte = ((112*R - 102*G - 10*B + 128) >> 8) + 128

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

    int y  = ((  47 * r + 157 * g +  16 * b + 128) >> 8) + 16;
    int cb = (( -26 * r -  87 * g + 112 * b + 128) >> 8) + 128;
    int cr = (( 112 * r - 102 * g -  10 * b + 128) >> 8) + 128;

    imageStore(u_y, px, vec4(clamp_byte(y),  0.0, 0.0, 1.0));
    imageStore(u_u, px, vec4(clamp_byte(cb), 0.0, 0.0, 1.0));
    imageStore(u_v, px, vec4(clamp_byte(cr), 0.0, 0.0, 1.0));
}
