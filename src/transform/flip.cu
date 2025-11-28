#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h"

extern "C" {

enum FlipMode {
    FLIP_HORIZONTAL = 0,
    FLIP_VERTICAL = 1,
    FLIP_BOTH = 2
};

__global__ void flip_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const uint32_t width,
    const uint32_t height,
    const int mode
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    uint32_t src_x = x;
    uint32_t src_y = y;
    
    if (mode == FLIP_HORIZONTAL) {
        src_x = width - 1 - x;
    } else if (mode == FLIP_VERTICAL) {
        src_y = height - 1 - y;
    } else if (mode == FLIP_BOTH) {
        src_x = width - 1 - x;
        src_y = height - 1 - y;
    }
    
    const uint32_t src_idx = src_y * width + src_x;
    const uint32_t dst_idx = y * width + x;
    
    dst[dst_idx] = src[src_idx];
}

PyObject* py_flip_f32(PyObject* self, PyObject* args) {
    PyObject* src_capsule;
    PyObject* dst_capsule;
    uint32_t width, height;
    int mode;
    
    if (!PyArg_ParseTuple(args, "OOIIi", 
                          &src_capsule, &dst_capsule, 
                          &width, &height, 
                          &mode)) {
        return NULL;
    }

    if (mode < 0 || mode > 2) {
        PyErr_SetString(PyExc_ValueError, "Invalid flip mode. 0=H, 1=V, 2=Both");
        return NULL;
    }

    if (validate_f32_buffer(src_capsule, "Source") < 0) return NULL;
    if (validate_f32_buffer(dst_capsule, "Destination") < 0) return NULL;
    
    if (validate_dimensions(width, height) < 0) return NULL;
    
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

    flip_kernel<<<gridSize, blockSize>>>(
        src_ptr, dst_ptr, width, height, mode
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}