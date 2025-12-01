#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void sepia_kernel(
    float4* buffer,
    const uint32_t width,
    const uint32_t height,
    const float intensity
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const uint32_t idx = y * width + x;

    const float4 in_px = buffer[idx];

    float r = in_px.x * 0.393f + in_px.y * 0.769f + in_px.z * 0.189f;
    float g = in_px.x * 0.349f + in_px.y * 0.686f + in_px.z * 0.168f;
    float b = in_px.x * 0.272f + in_px.y * 0.534f + in_px.z * 0.131f;

    float4 out_px;
    out_px.x = in_px.x * (1.0f - intensity) + r * intensity;
    out_px.y = in_px.y * (1.0f - intensity) + g * intensity;
    out_px.z = in_px.z * (1.0f - intensity) + b * intensity;
    out_px.w = in_px.w; 

    buffer[idx] = out_px;
}

PyObject* py_sepia_f32(PyObject* self, PyObject* args) {
    PyObject *capsule;
    uint32_t width, height;
    float intensity;

    if (!PyArg_ParseTuple(args, "OIIf", &capsule, &width, &height, &intensity)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    sepia_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, 
        width, height, intensity
    );
    
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

}