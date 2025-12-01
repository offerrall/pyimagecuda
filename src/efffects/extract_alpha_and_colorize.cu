#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void extract_alpha_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const uint32_t width,
    const uint32_t height
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint32_t idx = y * width + x;
    const float alpha = src[idx].w;
    
    dst[idx] = make_float4(alpha, alpha, alpha, alpha);
}

__global__ void colorize_alpha_mask_kernel(
    float4* __restrict__ image,
    const uint32_t width,
    const uint32_t height,
    const float r,
    const float g,
    const float b,
    const float strength
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint32_t idx = y * width + x;
    const float4 pixel = image[idx];
    
    const float blurred_alpha = pixel.w;
    const float final_alpha = blurred_alpha * strength;
    
    image[idx] = make_float4(
        r * final_alpha,
        g * final_alpha,
        b * final_alpha,
        final_alpha
    );
}

PyObject* py_extract_alpha_f32(PyObject* self, PyObject* args) {
    PyObject* src_capsule;
    PyObject* dst_capsule;
    uint32_t width, height;
    
    if (!PyArg_ParseTuple(args, "OOII", 
                          &src_capsule, &dst_capsule,
                          &width, &height)) {
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

    extract_alpha_kernel<<<gridSize, blockSize>>>(
        src_ptr, dst_ptr, width, height
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

PyObject* py_colorize_alpha_mask_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    PyObject* color_obj;
    
    if (!PyArg_ParseTuple(args, "OIIO", 
                          &capsule,
                          &width, &height,
                          &color_obj)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    if (validate_dimensions(width, height) < 0) return NULL;

    BufferContext* ctx = get_buffer_context(capsule);
    if (ctx == NULL) return NULL;
    
    float rgba[4];
    if (parse_rgba_color(color_obj, rgba) < 0) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    float4* ptr = (float4*)ctx->ptr;

    colorize_alpha_mask_kernel<<<gridSize, blockSize>>>(
        ptr, width, height,
        rgba[0], rgba[1], rgba[2], rgba[3]
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}