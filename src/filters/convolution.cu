#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__device__ float4 get_px(const float4* buffer, int x, int y, uint32_t w, uint32_t h) {
    x = max(0, min(x, (int)w - 1));
    y = max(0, min(y, (int)h - 1));
    return buffer[y * w + x];
}

__global__ void sobel_kernel(const float4* src, float4* dst, uint32_t width, uint32_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float4 gx = make_float4(0,0,0,0);
    float4 gy = make_float4(0,0,0,0);

    for(int j=-1; j<=1; j++) {
        for(int i=-1; i<=1; i++) {
            float4 p = get_px(src, x + i, y + j, width, height);

            float wx = 0;
            if (i == -1) wx = -1; else if (i == 1) wx = 1;
            if (j == 0) wx *= 2;

            float wy = 0;
            if (j == -1) wy = -1; else if (j == 1) wy = 1;
            if (i == 0) wy *= 2;

            gx.x += p.x * wx; gx.y += p.y * wx; gx.z += p.z * wx;
            gy.x += p.x * wy; gy.y += p.y * wy; gy.z += p.z * wy;
        }
    }

    float4 res;
    res.x = sqrtf(gx.x*gx.x + gy.x*gy.x);
    res.y = sqrtf(gx.y*gx.y + gy.y*gy.y);
    res.z = sqrtf(gx.z*gx.z + gy.z*gy.z);
    res.w = 1.0f; // Opaco

    dst[y * width + x] = res;
}

__global__ void emboss_kernel(const float4* src, float4* dst, uint32_t width, uint32_t height, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Emboss
    // -2 -1  0
    // -1  1  1
    //  0  1  2
    
    float4 sum = make_float4(0,0,0,0);

    sum = get_px(src, x-1, y-1, width, height);
    sum.x *= -2.0f; sum.y *= -2.0f; sum.z *= -2.0f;
    
    float4 p1 = get_px(src, x, y-1, width, height);
    sum.x -= p1.x; sum.y -= p1.y; sum.z -= p1.z;

    float4 p2 = get_px(src, x-1, y, width, height);
    sum.x -= p2.x; sum.y -= p2.y; sum.z -= p2.z;
    
    float4 center = get_px(src, x, y, width, height);
    sum.x += center.x; sum.y += center.y; sum.z += center.z;

    float4 p3 = get_px(src, x+1, y, width, height);
    sum.x += p3.x; sum.y += p3.y; sum.z += p3.z;

    float4 p4 = get_px(src, x, y+1, width, height);
    sum.x += p4.x; sum.y += p4.y; sum.z += p4.z;

    float4 p5 = get_px(src, x+1, y+1, width, height);
    sum.x += p5.x * 2.0f; sum.y += p5.y * 2.0f; sum.z += p5.z * 2.0f;

    sum.x *= strength;
    sum.y *= strength;
    sum.z *= strength;

    float4 res;
    res.x = sum.x + 0.5f;
    res.y = sum.y + 0.5f;
    res.z = sum.z + 0.5f;
    res.w = 1.0f;

    dst[y * width + x] = res;
}


PyObject* py_filter_sobel_f32(PyObject* self, PyObject* args) {
    PyObject *src, *dst; uint32_t w, h;
    if (!PyArg_ParseTuple(args, "OOII", &src, &dst, &w, &h)) return NULL;
    
    BufferContext* s_ctx = get_buffer_context(src);
    BufferContext* d_ctx = get_buffer_context(dst);
    if (!s_ctx || !d_ctx) return NULL;

    dim3 block(16, 16);
    dim3 grid((w + 15)/16, (h + 15)/16);
    
    sobel_kernel<<<grid, block>>>((float4*)s_ctx->ptr, (float4*)d_ctx->ptr, w, h);
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

PyObject* py_filter_emboss_f32(PyObject* self, PyObject* args) {
    PyObject *src, *dst; uint32_t w, h; float str;
    if (!PyArg_ParseTuple(args, "OOIIf", &src, &dst, &w, &h, &str)) return NULL;
    
    BufferContext* s_ctx = get_buffer_context(src);
    BufferContext* d_ctx = get_buffer_context(dst);
    if (!s_ctx || !d_ctx) return NULL;

    dim3 block(16, 16);
    dim3 grid((w + 15)/16, (h + 15)/16);
    
    emboss_kernel<<<grid, block>>>((float4*)s_ctx->ptr, (float4*)d_ctx->ptr, w, h, str);
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

}