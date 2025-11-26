#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__device__ __forceinline__ float corner_distance(
    const float x,
    const float y,
    const float corner_x,
    const float corner_y,
    const float radius
) {
    const float dx = x - corner_x;
    const float dy = y - corner_y;
    return sqrtf(dx * dx + dy * dy) - radius;
}

__device__ __forceinline__ float smoothstep_aa(const float distance) {

    const float aa_width = 1.0f;
    const float t = fminf(fmaxf((-distance) / aa_width, 0.0f), 1.0f);

    return t * t * (3.0f - 2.0f * t);
}

__global__ void rounded_corners_kernel(
    float4* __restrict__ image,
    const uint32_t width,
    const uint32_t height,
    const float radius
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const float fx = (float)x + 0.5f;
    const float fy = (float)y + 0.5f;

    const float left = radius;
    const float right = (float)width - radius;
    const float top = radius;
    const float bottom = (float)height - radius;
    
    float mask = 1.0f;

    if (fx < left && fy < top) {
        const float dist = corner_distance(fx, fy, left, top, radius);
        mask = smoothstep_aa(dist);
    }
    else if (fx > right && fy < top) {
        const float dist = corner_distance(fx, fy, right, top, radius);
        mask = smoothstep_aa(dist);
    }
    else if (fx < left && fy > bottom) {
        const float dist = corner_distance(fx, fy, left, bottom, radius);
        mask = smoothstep_aa(dist);
    }
    else if (fx > right && fy > bottom) {
        const float dist = corner_distance(fx, fy, right, bottom, radius);
        mask = smoothstep_aa(dist);
    }

    const uint32_t idx = y * width + x;
    float4 pixel = image[idx];
    pixel.w *= mask;
    image[idx] = pixel;
}

PyObject* py_rounded_corners_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    float radius;
    
    if (!PyArg_ParseTuple(args, "OIIf", 
                          &capsule,
                          &width, &height,
                          &radius)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;

    if (validate_dimensions(width, height) < 0) return NULL;
    
    if (radius < 0.0f) {
        PyErr_SetString(PyExc_ValueError, "Radius must be non-negative");
        return NULL;
    }
    
    const float max_radius = fminf((float)width, (float)height) / 2.0f;
    if (radius > max_radius) {
        PyErr_Format(PyExc_ValueError, 
            "Radius %.1f exceeds maximum %.1f (half of smallest dimension)",
            radius, max_radius);
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

    rounded_corners_kernel<<<gridSize, blockSize>>>(
        ptr, width, height, radius
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}