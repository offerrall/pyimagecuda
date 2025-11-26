#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__device__ inline unsigned char float_to_uchar_safe(float v) {
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    
    return (unsigned char)(v * 255.0f + 0.5f);
}

__global__ void convert_f32_to_u8_kernel(uchar4* dst, const float4* src, 
                                          uint32_t width, uint32_t height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    uint32_t idx = y * width + x;
    float4 pixel = src[idx];
    
    dst[idx] = make_uchar4(
        float_to_uchar_safe(pixel.x),
        float_to_uchar_safe(pixel.y),
        float_to_uchar_safe(pixel.z),
        float_to_uchar_safe(pixel.w)
    );
}

__global__ void convert_u8_to_f32_kernel(float4* dst, const uchar4* src,
                                          uint32_t width, uint32_t height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    uint32_t idx = y * width + x;
    uchar4 pixel = src[idx];
    
    dst[idx] = make_float4(
        pixel.x / 255.0f,
        pixel.y / 255.0f,
        pixel.z / 255.0f,
        pixel.w / 255.0f
    );
}

PyObject* py_convert_f32_to_u8(PyObject* self, PyObject* args) {
    PyObject *dst_capsule, *src_capsule;
    uint32_t width, height;
    
    if (!PyArg_ParseTuple(args, "OOII", &dst_capsule, &src_capsule, &width, &height)) {
        return NULL;
    }
    
    int dst_is_float32 = is_float32_buffer(dst_capsule);
    if (dst_is_float32 < 0) return NULL;
    if (dst_is_float32) {
        PyErr_SetString(PyExc_TypeError, "Destination buffer must be U8, not F32");
        return NULL;
    }
    
    int src_is_float32 = is_float32_buffer(src_capsule);
    if (src_is_float32 < 0) return NULL;
    if (!src_is_float32) {
        PyErr_SetString(PyExc_TypeError, "Source buffer must be F32, not U8");
        return NULL;
    }
    
    BufferContext* dst_ctx = get_buffer_context(dst_capsule);
    if (dst_ctx == NULL) return NULL;
    
    BufferContext* src_ctx = get_buffer_context(src_capsule);
    if (src_ctx == NULL) return NULL;
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    convert_f32_to_u8_kernel<<<gridSize, blockSize>>>(
        (uchar4*)dst_ctx->ptr,
        (const float4*)src_ctx->ptr,
        width, height
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

PyObject* py_convert_u8_to_f32(PyObject* self, PyObject* args) {
    PyObject *dst_capsule, *src_capsule;
    uint32_t width, height;
    
    if (!PyArg_ParseTuple(args, "OOII", &dst_capsule, &src_capsule, &width, &height)) {
        return NULL;
    }
    
    int dst_is_float32 = is_float32_buffer(dst_capsule);
    if (dst_is_float32 < 0) return NULL;
    if (!dst_is_float32) {
        PyErr_SetString(PyExc_TypeError, "Destination buffer must be F32, not U8");
        return NULL;
    }
    
    int src_is_float32 = is_float32_buffer(src_capsule);
    if (src_is_float32 < 0) return NULL;
    if (src_is_float32) {
        PyErr_SetString(PyExc_TypeError, "Source buffer must be U8, not F32");
        return NULL;
    }
    
    BufferContext* dst_ctx = get_buffer_context(dst_capsule);
    if (dst_ctx == NULL) return NULL;
    
    BufferContext* src_ctx = get_buffer_context(src_capsule);
    if (src_ctx == NULL) return NULL;
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    convert_u8_to_f32_kernel<<<gridSize, blockSize>>>(
        (float4*)dst_ctx->ptr,
        (const uchar4*)src_ctx->ptr,
        width, height
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}