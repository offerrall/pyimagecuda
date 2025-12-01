#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void vignette_kernel(
    float4* __restrict__ buffer,
    const uint32_t width,
    const uint32_t height,
    const float radius,
    const float softness,
    const float r, const float g, const float b, const float a
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const float u = ((float)x / (float)(width - 1)) * 2.0f - 1.0f;
    const float v = ((float)y / (float)(height - 1)) * 2.0f - 1.0f;

    const float dist = sqrtf(u * u + v * v);

    const float start = radius - (softness * 0.5f);
    const float end = radius + (softness * 0.5f);

    float vignette_strength = 0.0f;
    if (dist < start) {
        vignette_strength = 0.0f;
    } else if (dist > end) {
        vignette_strength = 1.0f;
    } else {
        float t = (dist - start) / (end - start);
        vignette_strength = t * t * (3.0f - 2.0f * t);
    }

    const uint32_t idx = y * width + x;
    float4 pixel = buffer[idx];
    
    float mix_factor = vignette_strength * a; 

    pixel.x = pixel.x * (1.0f - mix_factor) + r * mix_factor;
    pixel.y = pixel.y * (1.0f - mix_factor) + g * mix_factor;
    pixel.z = pixel.z * (1.0f - mix_factor) + b * mix_factor;
    
    buffer[idx] = pixel;
}

PyObject* py_effect_vignette_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    PyObject* color_obj;
    uint32_t width, height;
    float radius, softness;

    if (!PyArg_ParseTuple(args, "OIIffO", 
                          &capsule, &width, &height, 
                          &radius, &softness, &color_obj)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    if (validate_dimensions(width, height) < 0) return NULL;
    
    float rgba[4];
    if (parse_rgba_color(color_obj, rgba) < 0) return NULL;
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    vignette_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, width, height, 
        radius, softness, 
        rgba[0], rgba[1], rgba[2], rgba[3]
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}