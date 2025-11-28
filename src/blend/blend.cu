#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void blend_normal_kernel(float4* base, const float4* overlay, uint32_t base_width, uint32_t base_height, uint32_t overlay_width, uint32_t overlay_height, int32_t pos_x, int32_t pos_y, float opacity);
__global__ void blend_multiply_kernel(float4* base, const float4* overlay, uint32_t base_width, uint32_t base_height, uint32_t overlay_width, uint32_t overlay_height, int32_t pos_x, int32_t pos_y, float opacity);
__global__ void blend_screen_kernel(float4* base, const float4* overlay, uint32_t base_width, uint32_t base_height, uint32_t overlay_width, uint32_t overlay_height, int32_t pos_x, int32_t pos_y, float opacity);
__global__ void blend_add_kernel(float4* base, const float4* overlay, uint32_t base_width, uint32_t base_height, uint32_t overlay_width, uint32_t overlay_height, int32_t pos_x, int32_t pos_y, float opacity);
__global__ void blend_overlay_kernel(float4* base, const float4* overlay, uint32_t base_width, uint32_t base_height, uint32_t overlay_width, uint32_t overlay_height, int32_t pos_x, int32_t pos_y, float opacity);
__global__ void blend_soft_light_kernel(float4* base, const float4* overlay, uint32_t base_width, uint32_t base_height, uint32_t overlay_width, uint32_t overlay_height, int32_t pos_x, int32_t pos_y, float opacity);
__global__ void blend_hard_light_kernel(float4* base, const float4* overlay, uint32_t base_width, uint32_t base_height, uint32_t overlay_width, uint32_t overlay_height, int32_t pos_x, int32_t pos_y, float opacity);

enum BlendMode {
    BLEND_NORMAL = 0,
    BLEND_MULTIPLY = 1,
    BLEND_SCREEN = 2,
    BLEND_ADD = 3,
    BLEND_OVERLAY = 4,
    BLEND_SOFT_LIGHT = 5,
    BLEND_HARD_LIGHT = 6
};

PyObject* py_blend_f32(PyObject* self, PyObject* args) {
    PyObject* base_capsule;
    PyObject* overlay_capsule;
    uint32_t base_width, base_height;
    uint32_t overlay_width, overlay_height;
    int32_t pos_x, pos_y;
    int mode;
    float opacity;
    
    if (!PyArg_ParseTuple(args, "OOIIIIiiif", 
                          &base_capsule, &overlay_capsule,
                          &base_width, &base_height,
                          &overlay_width, &overlay_height,
                          &pos_x, &pos_y, &mode, &opacity)) {
        return NULL;
    }
    
    if (mode < BLEND_NORMAL || mode > BLEND_HARD_LIGHT) {
        PyErr_SetString(PyExc_ValueError, "Invalid blend mode");
        return NULL;
    }
    
    if (validate_f32_buffer(base_capsule, "Base") < 0) return NULL;
    if (validate_f32_buffer(overlay_capsule, "Overlay") < 0) return NULL;
    if (validate_dimensions(base_width, base_height) < 0) return NULL;
    if (validate_dimensions(overlay_width, overlay_height) < 0) return NULL;
    
    BufferContext* base_ctx = get_buffer_context(base_capsule);
    BufferContext* overlay_ctx = get_buffer_context(overlay_capsule);
    if (!base_ctx || !overlay_ctx) return NULL;
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (overlay_width + blockSize.x - 1) / blockSize.x,
        (overlay_height + blockSize.y - 1) / blockSize.y
    );
    
    float4* base_ptr = (float4*)base_ctx->ptr;
    const float4* overlay_ptr = (const float4*)overlay_ctx->ptr;
    
    switch(mode) {
        case BLEND_NORMAL:
            blend_normal_kernel<<<gridSize, blockSize>>>(base_ptr, overlay_ptr, base_width, base_height, overlay_width, overlay_height, pos_x, pos_y, opacity);
            break;
        case BLEND_MULTIPLY:
            blend_multiply_kernel<<<gridSize, blockSize>>>(base_ptr, overlay_ptr, base_width, base_height, overlay_width, overlay_height, pos_x, pos_y, opacity);
            break;
        case BLEND_SCREEN:
            blend_screen_kernel<<<gridSize, blockSize>>>(base_ptr, overlay_ptr, base_width, base_height, overlay_width, overlay_height, pos_x, pos_y, opacity);
            break;
        case BLEND_ADD:
            blend_add_kernel<<<gridSize, blockSize>>>(base_ptr, overlay_ptr, base_width, base_height, overlay_width, overlay_height, pos_x, pos_y, opacity);
            break;
        case BLEND_OVERLAY:
            blend_overlay_kernel<<<gridSize, blockSize>>>(base_ptr, overlay_ptr, base_width, base_height, overlay_width, overlay_height, pos_x, pos_y, opacity);
            break;
        case BLEND_SOFT_LIGHT:
            blend_soft_light_kernel<<<gridSize, blockSize>>>(base_ptr, overlay_ptr, base_width, base_height, overlay_width, overlay_height, pos_x, pos_y, opacity);
            break;
        case BLEND_HARD_LIGHT:
            blend_hard_light_kernel<<<gridSize, blockSize>>>(base_ptr, overlay_ptr, base_width, base_height, overlay_width, overlay_height, pos_x, pos_y, opacity);
            break;
    }
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}