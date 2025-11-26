#include <stdint.h>
#include <cuda_runtime.h>

extern "C" {

__global__ void resize_nearest_kernel(
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
    
    const float src_x_f = ((float)dst_x + 0.5f) * scale_x - 0.5f;
    const float src_y_f = ((float)dst_y + 0.5f) * scale_y - 0.5f;
    
    const uint32_t src_x = min((uint32_t)floorf(src_x_f + 0.5f), src_width - 1);
    const uint32_t src_y = min((uint32_t)floorf(src_y_f + 0.5f), src_height - 1);
    
    const uint32_t src_idx = src_y * src_width + src_x;
    const uint32_t dst_idx = dst_y * dst_width + dst_x;
    
    dst[dst_idx] = src[src_idx];
}

}