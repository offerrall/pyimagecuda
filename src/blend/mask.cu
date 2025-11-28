#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h"

extern "C" {

enum MaskMode {
    MASK_ALPHA = 0,
    MASK_LUMINANCE = 1
};

__global__ void blend_mask_kernel(
    float4* __restrict__ base,
    const float4* __restrict__ mask,
    const uint32_t base_width,
    const uint32_t base_height,
    const uint32_t mask_width,
    const uint32_t mask_height,
    const int32_t pos_x,
    const int32_t pos_y,
    const int mode
) {
    const uint32_t ox = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= mask_width || oy >= mask_height) return;

    const int32_t bx = pos_x + (int32_t)ox;
    const int32_t by = pos_y + (int32_t)oy;

    if (bx < 0 || by < 0 || bx >= (int32_t)base_width || by >= (int32_t)base_height) return;
    
    const uint32_t mask_idx = oy * mask_width + ox;
    const uint32_t base_idx = (uint32_t)by * base_width + (uint32_t)bx;
    
    float4 pixel = base[base_idx];
    const float4 mask_pixel = mask[mask_idx];
    
    float mask_val;
    
    if (mode == MASK_LUMINANCE) {
        mask_val = mask_pixel.x * 0.299f + mask_pixel.y * 0.587f + mask_pixel.z * 0.114f;
    } else {
        mask_val = mask_pixel.w;
    }
    
    pixel.w *= mask_val;
    
    base[base_idx] = pixel;
}

PyObject* py_blend_mask_f32(PyObject* self, PyObject* args) {
    PyObject* base_capsule;
    PyObject* mask_capsule;
    uint32_t base_width, base_height;
    uint32_t mask_width, mask_height;
    int pos_x, pos_y;
    int mode;
    
    if (!PyArg_ParseTuple(args, "OOIIIIiii", 
                          &base_capsule, &mask_capsule,
                          &base_width, &base_height,
                          &mask_width, &mask_height,
                          &pos_x, &pos_y, &mode)) {
        return NULL;
    }
    
    if (validate_f32_buffer(base_capsule, "Base") < 0) return NULL;
    if (validate_f32_buffer(mask_capsule, "Mask") < 0) return NULL;
    
    BufferContext* base_ctx = get_buffer_context(base_capsule);
    BufferContext* mask_ctx = get_buffer_context(mask_capsule);
    if (!base_ctx || !mask_ctx) return NULL;
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mask_width + blockSize.x - 1) / blockSize.x,
        (mask_height + blockSize.y - 1) / blockSize.y
    );
    
    float4* base_ptr = (float4*)base_ctx->ptr;
    const float4* mask_ptr = (const float4*)mask_ctx->ptr;
    
    blend_mask_kernel<<<gridSize, blockSize>>>(
        base_ptr, mask_ptr, 
        base_width, base_height, 
        mask_width, mask_height, 
        pos_x, pos_y, mode
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}