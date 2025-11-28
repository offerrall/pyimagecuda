#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h"

extern "C" {

__global__ void adjust_gamma_kernel(
    float4* __restrict__ image,
    const uint32_t width,
    const uint32_t height,
    const float inv_gamma
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint32_t idx = y * width + x;
    float4 pixel = image[idx];
    
    pixel.x = powf(fmaxf(0.0f, pixel.x), inv_gamma);
    pixel.y = powf(fmaxf(0.0f, pixel.y), inv_gamma);
    pixel.z = powf(fmaxf(0.0f, pixel.z), inv_gamma);
    
    image[idx] = pixel;
}

PyObject* py_adjust_gamma_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    float gamma;
    
    if (!PyArg_ParseTuple(args, "OIIf", 
                          &capsule, 
                          &width, &height, 
                          &gamma)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Image") < 0) return NULL;
    if (validate_dimensions(width, height) < 0) return NULL;

    if (gamma <= 0.001f) {
        PyErr_SetString(PyExc_ValueError, "Gamma must be > 0.001");
        return NULL;
    }
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (ctx == NULL) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    float4* ptr = (float4*)ctx->ptr;

    float inv_gamma = 1.0f / gamma;

    adjust_gamma_kernel<<<gridSize, blockSize>>>(
        ptr, width, height, inv_gamma
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}