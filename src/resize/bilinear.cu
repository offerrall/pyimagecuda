#include <stdint.h>
#include <cuda_runtime.h>

extern "C" {

__device__ __forceinline__ float4 lerp_float4(const float4 a, const float4 b, const float t) {
    return make_float4(
        fmaf(b.x - a.x, t, a.x),
        fmaf(b.y - a.y, t, a.y),
        fmaf(b.z - a.z, t, a.z),
        fmaf(b.w - a.w, t, a.w)
    );
}

__global__ void resize_bilinear_kernel(
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
    
    const float src_x = fmaxf(0.0f, fminf(
        ((float)dst_x + 0.5f) * scale_x - 0.5f,
        (float)(src_width - 1)
    ));
    const float src_y = fmaxf(0.0f, fminf(
        ((float)dst_y + 0.5f) * scale_y - 0.5f,
        (float)(src_height - 1)
    ));
    
    const uint32_t x0 = (uint32_t)src_x;
    const uint32_t y0 = (uint32_t)src_y;
    const uint32_t x1 = min(x0 + 1, src_width - 1);
    const uint32_t y1 = min(y0 + 1, src_height - 1);
    
    const float fx = src_x - (float)x0;
    const float fy = src_y - (float)y0;
    
    const float4 p00 = src[y0 * src_width + x0];
    const float4 p10 = src[y0 * src_width + x1];
    const float4 p01 = src[y1 * src_width + x0];
    const float4 p11 = src[y1 * src_width + x1];
    
    const float4 top = lerp_float4(p00, p10, fx);
    const float4 bottom = lerp_float4(p01, p11, fx);
    const float4 result = lerp_float4(top, bottom, fy);
    
    dst[dst_y * dst_width + dst_x] = result;
}

}