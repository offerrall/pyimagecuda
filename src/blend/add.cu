#include <stdint.h>
#include <cuda_runtime.h>

extern "C" {

__global__ void blend_add_kernel(
    float4* __restrict__ base,
    const float4* __restrict__ overlay,
    const uint32_t base_width,
    const uint32_t base_height,
    const uint32_t overlay_width,
    const uint32_t overlay_height,
    const int32_t pos_x,
    const int32_t pos_y,
    const float opacity
) {
    const uint32_t ox = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t oy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ox >= overlay_width || oy >= overlay_height) return;
    
    const int32_t bx = pos_x + (int32_t)ox;
    const int32_t by = pos_y + (int32_t)oy;
    
    if (bx < 0 || by < 0 || bx >= (int32_t)base_width || by >= (int32_t)base_height) return;
    
    const uint32_t overlay_idx = oy * overlay_width + ox;
    const uint32_t base_idx = (uint32_t)by * base_width + (uint32_t)bx;
    
    const float4 dst = base[base_idx];
    const float4 src = overlay[overlay_idx];
    
    const float blend_r = fminf(dst.x + src.x, 1.0f);
    const float blend_g = fminf(dst.y + src.y, 1.0f);
    const float blend_b = fminf(dst.z + src.z, 1.0f);

    const float src_alpha = src.w * opacity;
    const float dst_alpha = dst.w;

    const float out_alpha = src_alpha + dst_alpha * (1.0f - src_alpha);

    float out_r, out_g, out_b;
    
    if (out_alpha > 1e-6f) {
        const float inv_out_alpha = 1.0f / out_alpha;
        
        const float blend_factor = src_alpha;
        const float base_factor = dst_alpha * (1.0f - src_alpha);
        
        out_r = (blend_r * blend_factor + dst.x * base_factor) * inv_out_alpha;
        out_g = (blend_g * blend_factor + dst.y * base_factor) * inv_out_alpha;
        out_b = (blend_b * blend_factor + dst.z * base_factor) * inv_out_alpha;
    } else {
        out_r = out_g = out_b = 0.0f;
    }
    
    base[base_idx] = make_float4(out_r, out_g, out_b, out_alpha);
}

}