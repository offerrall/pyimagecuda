#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h"

extern "C" {

__global__ void adjust_contrast_kernel(
    float4* __restrict__ image,
    const uint32_t width,
    const uint32_t height,
    const float factor
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint32_t idx = y * width + x;
    float4 pixel = image[idx];
   
    pixel.x = (pixel.x - 0.5f) * factor + 0.5f;
    pixel.y = (pixel.y - 0.5f) * factor + 0.5f;
    pixel.z = (pixel.z - 0.5f) * factor + 0.5f;
    
    pixel.x = fmaxf(0.0f, fminf(1.0f, pixel.x));
    pixel.y = fmaxf(0.0f, fminf(1.0f, pixel.y));
    pixel.z = fmaxf(0.0f, fminf(1.0f, pixel.z));
    
    image[idx] = pixel;
}

PyObject* py_adjust_contrast_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    float factor;
    
    if (!PyArg_ParseTuple(args, "OIIf", 
                          &capsule, 
                          &width, &height, 
                          &factor)) {
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

    adjust_contrast_kernel<<<gridSize, blockSize>>>(
        ptr, width, height, factor
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}