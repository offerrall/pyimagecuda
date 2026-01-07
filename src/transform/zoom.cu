#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h"

extern "C" {

__global__ void zoom_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const uint32_t src_width,
    const uint32_t src_height,
    const uint32_t dst_width,
    const uint32_t dst_height,
    const float zoom_factor,
    const float center_x,
    const float center_y,
    const int interp_method
) {
    const uint32_t dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    const float viewport_w = (float)dst_width / zoom_factor;
    const float viewport_h = (float)dst_height / zoom_factor;
    
    const float viewport_x = center_x - viewport_w * 0.5f;
    const float viewport_y = center_y - viewport_h * 0.5f;
    
    const float src_x = viewport_x + ((float)dst_x / (float)dst_width) * viewport_w;
    const float src_y = viewport_y + ((float)dst_y / (float)dst_height) * viewport_h;
    
    float4 result;
    
    if (src_x < 0.0f || src_x >= (float)src_width || 
        src_y < 0.0f || src_y >= (float)src_height) {
        // Return transparent black for out-of-bounds
        result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        switch (interp_method) {
            case 0:
                result = sample_nearest(src, src_x, src_y, src_width, src_height);
                break;
            case 1:
                result = sample_bilinear(src, src_x, src_y, src_width, src_height);
                break;
            case 2:
                result = sample_bicubic(src, src_x, src_y, src_width, src_height);
                break;
            case 3:
                result = sample_lanczos(src, src_x, src_y, src_width, src_height);
                break;
            default:
                result = sample_bilinear(src, src_x, src_y, src_width, src_height);
                break;
        }
    }
    
    dst[dst_y * dst_width + dst_x] = result;
}

PyObject* py_zoom_f32(PyObject* self, PyObject* args) {
    PyObject* src_capsule;
    PyObject* dst_capsule;
    uint32_t src_width, src_height;
    uint32_t dst_width, dst_height;
    float zoom_factor, center_x, center_y;
    int interp_method;
    
    if (!PyArg_ParseTuple(args, "OOIIIIfffi",
                          &src_capsule, &dst_capsule,
                          &src_width, &src_height,
                          &dst_width, &dst_height,
                          &zoom_factor, &center_x, &center_y,
                          &interp_method)) {
        return NULL;
    }
    
    if (zoom_factor <= 0.0f) {
        PyErr_SetString(PyExc_ValueError, "Zoom factor must be positive");
        return NULL;
    }
    
    if (interp_method < 0 || interp_method > 3) {
        PyErr_SetString(PyExc_ValueError, 
            "Invalid interpolation method (0=nearest, 1=bilinear, 2=bicubic, 3=lanczos)");
        return NULL;
    }
    
    if (validate_f32_buffer(src_capsule, "Source") < 0) return NULL;
    if (validate_f32_buffer(dst_capsule, "Destination") < 0) return NULL;
    
    BufferContext* src_ctx = get_buffer_context(src_capsule);
    BufferContext* dst_ctx = get_buffer_context(dst_capsule);
    if (!src_ctx || !dst_ctx) return NULL;
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (dst_width + blockSize.x - 1) / blockSize.x,
        (dst_height + blockSize.y - 1) / blockSize.y
    );
    
    zoom_kernel<<<gridSize, blockSize>>>(
        (const float4*)src_ctx->ptr,
        (float4*)dst_ctx->ptr,
        src_width, src_height,
        dst_width, dst_height,
        zoom_factor, center_x, center_y,
        interp_method
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}