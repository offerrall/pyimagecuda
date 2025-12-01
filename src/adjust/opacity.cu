#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void opacity_kernel(float4* buffer, uint32_t width, uint32_t height, float factor) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint32_t idx = y * width + x;
    
    buffer[idx].w *= factor;
}

PyObject* py_adjust_opacity_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    float factor;

    if (!PyArg_ParseTuple(args, "OIIf", &capsule, &width, &height, &factor)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    opacity_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, 
        width, height, factor
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}