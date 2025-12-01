#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include "../common.h"

extern "C" {


__device__ float random_hash(float x, float y, float seed) {

    float dot = x * 12.9898f + y * 78.233f + seed;
    float sin_val = sinf(dot) * 43758.5453f;

    return sin_val - floorf(sin_val);
}

__global__ void noise_kernel(
    float4* buffer, uint32_t w, uint32_t h,
    float seed, int monochrome
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    float fx = (float)x;
    float fy = (float)y;

    float r = random_hash(fx, fy, seed);
    float g, b;

    if (monochrome) {
        g = r;
        b = r;
    } else {
        g = random_hash(fx, fy, seed + 1.0f);
        b = random_hash(fx, fy, seed + 2.0f);
    }


    buffer[y * w + x] = make_float4(r, g, b, 1.0f);
}

PyObject* py_fill_noise_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    float seed;
    int monochrome;

    if (!PyArg_ParseTuple(args, "OIIfi", 
                          &capsule, &width, &height, 
                          &seed, &monochrome)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15)/16, (height + 15)/16);

    noise_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, width, height, 
        seed, monochrome
    );
    
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

}