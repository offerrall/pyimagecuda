#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h"

extern "C" {

__global__ void adjust_vibrance_kernel(
    float4* __restrict__ image,
    const uint32_t width,
    const uint32_t height,
    const float amount
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint32_t idx = y * width + x;
    float4 pixel = image[idx];

    float h, s, v;
    rgb_to_hsv(pixel.x, pixel.y, pixel.z, &h, &s, &v);

    if (s < 0.001f) return;

    const float sat_boost = amount * (1.0f - s) * s;

    float skin_protection = 1.0f;
    if ((h >= 0.0f && h <= 50.0f) || (h >= 330.0f && h <= 360.0f)) {
        skin_protection = 0.3f;
    }

    s = fmaxf(0.0f, fminf(1.0f, s + sat_boost * skin_protection));

    float r, g, b;
    hsv_to_rgb(h, s, v, &r, &g, &b);
    
    image[idx] = make_float4(r, g, b, pixel.w);
}

PyObject* py_adjust_vibrance_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    float amount;
    
    if (!PyArg_ParseTuple(args, "OIIf", 
                          &capsule, 
                          &width, &height, 
                          &amount)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Image") < 0) return NULL;
    if (validate_dimensions(width, height) < 0) return NULL;
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (ctx == NULL) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    float4* ptr = (float4*)ctx->ptr;

    adjust_vibrance_kernel<<<gridSize, blockSize>>>(
        ptr, width, height, amount
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}