#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h"

extern "C" {

__constant__ float3 LUMA_VEC = {0.299f, 0.587f, 0.114f};

__global__ void adjust_saturation_kernel(
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
    
    float gray = pixel.x * LUMA_VEC.x + pixel.y * LUMA_VEC.y + pixel.z * LUMA_VEC.z;
    
    pixel.x = gray + (pixel.x - gray) * factor;
    pixel.y = gray + (pixel.y - gray) * factor;
    pixel.z = gray + (pixel.z - gray) * factor;

    pixel.x = fmaxf(0.0f, fminf(1.0f, pixel.x));
    pixel.y = fmaxf(0.0f, fminf(1.0f, pixel.y));
    pixel.z = fmaxf(0.0f, fminf(1.0f, pixel.z));
    
    image[idx] = pixel;
}

PyObject* py_adjust_saturation_f32(PyObject* self, PyObject* args) {
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

    adjust_saturation_kernel<<<gridSize, blockSize>>>(
        ptr, width, height, factor
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}