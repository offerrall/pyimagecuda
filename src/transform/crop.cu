#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h"

extern "C" {

__global__ void crop_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const uint32_t src_width,
    const uint32_t dst_width,
    const int src_start_x,
    const int src_start_y,
    const int dst_start_x,
    const int dst_start_y,
    const int copy_width,
    const int copy_height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= copy_width || y >= copy_height) return;
    
    const int sx = src_start_x + x;
    const int sy = src_start_y + y;
    
    const int dx = dst_start_x + x;
    const int dy = dst_start_y + y;
    
    const uint32_t src_idx = sy * src_width + sx;
    const uint32_t dst_idx = dy * dst_width + dx;
    
    dst[dst_idx] = src[src_idx];
}

PyObject* py_crop_f32(PyObject* self, PyObject* args) {
    PyObject* src_capsule;
    PyObject* dst_capsule;
    uint32_t src_width, dst_width;
    int src_x, src_y, dst_x, dst_y;
    int copy_w, copy_h;
    
    if (!PyArg_ParseTuple(args, "OOIIiiiiii", 
                          &src_capsule, &dst_capsule, 
                          &src_width, &dst_width,
                          &src_x, &src_y,
                          &dst_x, &dst_y,
                          &copy_w, &copy_h)) {
        return NULL;
    }

    if (validate_f32_buffer(src_capsule, "Source") < 0) return NULL;
    if (validate_f32_buffer(dst_capsule, "Destination") < 0) return NULL;
    
    BufferContext* src_ctx = get_buffer_context(src_capsule);
    if (src_ctx == NULL) return NULL;
    
    BufferContext* dst_ctx = get_buffer_context(dst_capsule);
    if (dst_ctx == NULL) return NULL;

    if (copy_w <= 0 || copy_h <= 0) {
        Py_RETURN_NONE;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (copy_w + blockSize.x - 1) / blockSize.x,
        (copy_h + blockSize.y - 1) / blockSize.y
    );
    
    const float4* src_ptr = (const float4*)src_ctx->ptr;
    float4* dst_ptr = (float4*)dst_ctx->ptr;

    crop_kernel<<<gridSize, blockSize>>>(
        src_ptr, dst_ptr,
        src_width, dst_width,
        src_x, src_y,
        dst_x, dst_y,
        copy_w, copy_h
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}