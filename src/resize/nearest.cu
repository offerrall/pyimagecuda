#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h"

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
    
    const float src_x = ((float)dst_x + 0.5f) * scale_x - 0.5f;
    const float src_y = ((float)dst_y + 0.5f) * scale_y - 0.5f;
    
    dst[dst_y * dst_width + dst_x] = sample_nearest(src, src_x, src_y, src_width, src_height);
}

}