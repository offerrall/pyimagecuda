#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void invert_kernel(float4* buffer, uint32_t width, uint32_t height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint32_t idx = y * width + x;
    float4 p = buffer[idx];

    p.x = 1.0f - p.x;
    p.y = 1.0f - p.y;
    p.z = 1.0f - p.z;

    buffer[idx] = p;
}

__global__ void threshold_kernel(float4* buffer, uint32_t width, uint32_t height, float thresh) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint32_t idx = y * width + x;
    float4 p = buffer[idx];

    float luma = p.x * 0.299f + p.y * 0.587f + p.z * 0.114f;

    float val = (luma >= thresh) ? 1.0f : 0.0f;

    p.x = val;
    p.y = val;
    p.z = val;

    buffer[idx] = p;
}

__global__ void solarize_kernel(float4* buffer, uint32_t width, uint32_t height, float thresh) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint32_t idx = y * width + x;
    float4 p = buffer[idx];

    if (p.x > thresh) p.x = 1.0f - p.x;
    if (p.y > thresh) p.y = 1.0f - p.y;
    if (p.z > thresh) p.z = 1.0f - p.z;

    buffer[idx] = p;
}


PyObject* py_invert_f32(PyObject* self, PyObject* args) {
    PyObject* capsule; uint32_t width, height;
    if (!PyArg_ParseTuple(args, "OII", &capsule, &width, &height)) return NULL;
    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    invert_kernel<<<gridSize, blockSize>>>((float4*)ctx->ptr, width, height);
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

PyObject* py_threshold_f32(PyObject* self, PyObject* args) {
    PyObject* capsule; uint32_t width, height; float thresh;

    if (!PyArg_ParseTuple(args, "OIIf", &capsule, &width, &height, &thresh)) return NULL;
    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    threshold_kernel<<<gridSize, blockSize>>>((float4*)ctx->ptr, width, height, thresh);
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

PyObject* py_solarize_f32(PyObject* self, PyObject* args) {
    PyObject* capsule; uint32_t width, height; float thresh;
    if (!PyArg_ParseTuple(args, "OIIf", &capsule, &width, &height, &thresh)) return NULL;
    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    solarize_kernel<<<gridSize, blockSize>>>((float4*)ctx->ptr, width, height, thresh);
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

}