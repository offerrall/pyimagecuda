#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void sharpen_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const uint32_t width,
    const uint32_t height,
    const float strength
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const float4 center = src[y * width + x];
    
    const float center_weight = 1.0f + 4.0f * strength;
    const float neighbor_weight = -strength;
    
    float4 result = make_float4(
        center.x * center_weight,
        center.y * center_weight,
        center.z * center_weight,
        center.w
    );

    if (y > 0) {
        const float4 p = src[(y - 1) * width + x];
        result.x += p.x * neighbor_weight;
        result.y += p.y * neighbor_weight;
        result.z += p.z * neighbor_weight;
    }

    if (y < height - 1) {
        const float4 p = src[(y + 1) * width + x];
        result.x += p.x * neighbor_weight;
        result.y += p.y * neighbor_weight;
        result.z += p.z * neighbor_weight;
    }

    if (x > 0) {
        const float4 p = src[y * width + (x - 1)];
        result.x += p.x * neighbor_weight;
        result.y += p.y * neighbor_weight;
        result.z += p.z * neighbor_weight;
    }

    if (x < width - 1) {
        const float4 p = src[y * width + (x + 1)];
        result.x += p.x * neighbor_weight;
        result.y += p.y * neighbor_weight;
        result.z += p.z * neighbor_weight;
    }

    dst[y * width + x] = make_float4(
        fmaxf(0.0f, fminf(1.0f, result.x)),
        fmaxf(0.0f, fminf(1.0f, result.y)),
        fmaxf(0.0f, fminf(1.0f, result.z)),
        result.w
    );
}

PyObject* py_sharpen_f32(PyObject* self, PyObject* args) {
    PyObject* src_capsule;
    PyObject* dst_capsule;
    uint32_t width, height;
    float strength;
    
    if (!PyArg_ParseTuple(args, "OOIIf", 
                          &src_capsule, &dst_capsule,
                          &width, &height,
                          &strength)) {
        return NULL;
    }

    if (validate_f32_buffer(src_capsule, "Source") < 0) return NULL;
    if (validate_f32_buffer(dst_capsule, "Destination") < 0) return NULL;

    if (validate_dimensions(width, height) < 0) return NULL;
    
    if (strength < 0.0f) {
        PyErr_SetString(PyExc_ValueError, "Strength must be non-negative");
        return NULL;
    }

    BufferContext* src_ctx = get_buffer_context(src_capsule);
    if (src_ctx == NULL) return NULL;
    
    BufferContext* dst_ctx = get_buffer_context(dst_capsule);
    if (dst_ctx == NULL) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    const float4* src_ptr = (const float4*)src_ctx->ptr;
    float4* dst_ptr = (float4*)dst_ctx->ptr;

    sharpen_kernel<<<gridSize, blockSize>>>(
        src_ptr, dst_ptr,
        width, height,
        strength
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}