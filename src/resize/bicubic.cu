#include <stdint.h>
#include <cuda_runtime.h>

extern "C" {

__device__ __forceinline__ float cubic_weight(const float x) {
    const float a = -0.5f;
    const float abs_x = fabsf(x);
    
    if (abs_x <= 1.0f) {
        return ((a + 2.0f) * abs_x - (a + 3.0f)) * abs_x * abs_x + 1.0f;
    } else if (abs_x < 2.0f) {
        return ((a * abs_x - 5.0f * a) * abs_x + 8.0f * a) * abs_x - 4.0f * a;
    }
    return 0.0f;
}

__device__ __forceinline__ float4 sample_bicubic(
    const float4* __restrict__ src,
    const float sx,
    const float sy,
    const uint32_t src_width,
    const uint32_t src_height
) {
    const int x0 = (int)floorf(sx);
    const int y0 = (int)floorf(sy);
    
    const float fx = sx - (float)x0;
    const float fy = sy - (float)y0;
    
    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    
    for (int dy = -1; dy <= 2; ++dy) {
        const int sy_idx = y0 + dy;
        if (sy_idx < 0 || sy_idx >= (int)src_height) continue;
        
        const float wy = cubic_weight((float)dy - fy);
        
        for (int dx = -1; dx <= 2; ++dx) {
            const int sx_idx = x0 + dx;
            if (sx_idx < 0 || sx_idx >= (int)src_width) continue;
            
            const float wx = cubic_weight((float)dx - fx);
            const float w = wx * wy;
            
            const float4 pixel = src[sy_idx * src_width + sx_idx];
            result.x += pixel.x * w;
            result.y += pixel.y * w;
            result.z += pixel.z * w;
            result.w += pixel.w * w;
            
            weight_sum += w;
        }
    }
    
    if (weight_sum > 0.0f) {
        const float inv_weight = 1.0f / weight_sum;
        result.x *= inv_weight;
        result.y *= inv_weight;
        result.z *= inv_weight;
        result.w *= inv_weight;
    }
    
    return result;
}

__global__ void resize_bicubic_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const uint32_t src_width,
    const uint32_t src_height,
    const uint32_t dst_width,
    const uint32_t dst_height
) {
    const uint32_t dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    const float scale_x = (float)src_width / (float)dst_width;
    const float scale_y = (float)src_height / (float)dst_height;
    
    const float src_x = ((float)dst_x + 0.5f) * scale_x - 0.5f;
    const float src_y = ((float)dst_y + 0.5f) * scale_y - 0.5f;
    
    const float4 result = sample_bicubic(src, src_x, src_y, src_width, src_height);
    
    dst[dst_y * dst_width + dst_x] = result;
}

}