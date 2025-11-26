#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void resize_nearest_kernel(
    const float4* src, float4* dst,
    uint32_t src_width, uint32_t src_height,
    uint32_t dst_width, uint32_t dst_height
);

__global__ void resize_bilinear_kernel(
    const float4* src, float4* dst,
    uint32_t src_width, uint32_t src_height,
    uint32_t dst_width, uint32_t dst_height
);

__global__ void resize_bicubic_kernel(
    const float4* src, float4* dst,
    uint32_t src_width, uint32_t src_height,
    uint32_t dst_width, uint32_t dst_height
);

__global__ void resize_lanczos_kernel(
    const float4* src, float4* dst,
    uint32_t src_width, uint32_t src_height,
    uint32_t dst_width, uint32_t dst_height
);

enum ResizeMethod {
    RESIZE_NEAREST = 0,
    RESIZE_BILINEAR = 1,
    RESIZE_BICUBIC = 2,
    RESIZE_LANCZOS = 3
};

PyObject* py_resize_f32(PyObject* self, PyObject* args) {
    PyObject* src_capsule;
    PyObject* dst_capsule;
    uint32_t src_width, src_height;
    uint32_t dst_width, dst_height;
    int method;
    
    if (!PyArg_ParseTuple(args, "OOIIIIi", 
                          &src_capsule, &dst_capsule,
                          &src_width, &src_height,
                          &dst_width, &dst_height,
                          &method)) {
        return NULL;
    }

    if (method < RESIZE_NEAREST || method > RESIZE_LANCZOS) {
        PyErr_SetString(PyExc_ValueError, 
            "Invalid resize method. Use 0 (nearest), 1 (bilinear), 2 (bicubic), or 3 (lanczos)");
        return NULL;
    }

    if (validate_f32_buffer(src_capsule, "Source") < 0) return NULL;
    if (validate_f32_buffer(dst_capsule, "Destination") < 0) return NULL;

    if (validate_dimensions(src_width, src_height) < 0) return NULL;
    if (validate_dimensions(dst_width, dst_height) < 0) return NULL;

    BufferContext* src_ctx = get_buffer_context(src_capsule);
    if (src_ctx == NULL) return NULL;
    
    BufferContext* dst_ctx = get_buffer_context(dst_capsule);
    if (dst_ctx == NULL) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (dst_width + blockSize.x - 1) / blockSize.x,
        (dst_height + blockSize.y - 1) / blockSize.y
    );
    
    const float4* src_ptr = (const float4*)src_ctx->ptr;
    float4* dst_ptr = (float4*)dst_ctx->ptr;

    switch(method) {
        case RESIZE_NEAREST:
            resize_nearest_kernel<<<gridSize, blockSize>>>(
                src_ptr, dst_ptr,
                src_width, src_height,
                dst_width, dst_height
            );
            break;
            
        case RESIZE_BILINEAR:
            resize_bilinear_kernel<<<gridSize, blockSize>>>(
                src_ptr, dst_ptr,
                src_width, src_height,
                dst_width, dst_height
            );
            break;
            
        case RESIZE_BICUBIC:
            resize_bicubic_kernel<<<gridSize, blockSize>>>(
                src_ptr, dst_ptr,
                src_width, src_height,
                dst_width, dst_height
            );
            break;
            
        case RESIZE_LANCZOS:
            resize_lanczos_kernel<<<gridSize, blockSize>>>(
                src_ptr, dst_ptr,
                src_width, src_height,
                dst_width, dst_height
            );
            break;
            
        default:
            PyErr_SetString(PyExc_ValueError, "Unknown resize method");
            return NULL;
    }
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}