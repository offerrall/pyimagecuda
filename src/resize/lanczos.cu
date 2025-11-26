#include <stdint.h>
#include <cuda_runtime.h>

extern "C" {

__device__ __forceinline__ float sinc(const float x) {
    if (fabsf(x) < 1e-5f) return 1.0f;
    const float pi_x = 3.14159265358979323846f * x;
    return sinf(pi_x) / pi_x;
}

__device__ __forceinline__ float lanczos_weight(const float x, const int a) {
    const float abs_x = fabsf(x);
    if (abs_x < (float)a) {
        return sinc(x) * sinc(x / (float)a);
    }
    return 0.0f;
}

__device__ __forceinline__ float4 sample_lanczos(
    const float4* __restrict__ src,
    const float sx,
    const float sy,
    const uint32_t src_width,
    const uint32_t src_height
) {
    const int a = 3;
    const int x_center = (int)floorf(sx + 0.5f);
    const int y_center = (int)floorf(sy + 0.5f);
    
    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    
    for (int dy = -a + 1; dy <= a; ++dy) {
        const int sy_idx = y_center + dy;
        if (sy_idx < 0 || sy_idx >= (int)src_height) continue;
        
        const float wy = lanczos_weight(sy - (float)sy_idx, a);
        
        for (int dx = -a + 1; dx <= a; ++dx) {
            const int sx_idx = x_center + dx;
            if (sx_idx < 0 || sx_idx >= (int)src_width) continue;
            
            const float wx = lanczos_weight(sx - (float)sx_idx, a);
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
    
    result.x = fmaxf(0.0f, fminf(1.0f, result.x));
    result.y = fmaxf(0.0f, fminf(1.0f, result.y));
    result.z = fmaxf(0.0f, fminf(1.0f, result.z));
    result.w = fmaxf(0.0f, fminf(1.0f, result.w));
    
    return result;
}

__global__ void resize_lanczos_kernel(
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
    
    const float4 result = sample_lanczos(src, src_x, src_y, src_width, src_height);
    
    dst[dst_y * dst_width + dst_x] = result;
}

}