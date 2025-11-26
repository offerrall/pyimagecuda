#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__host__ void compute_gaussian_kernel(float* kernel, int radius, float sigma) {
    float sum = 0.0f;
    const float sigma2 = 2.0f * sigma * sigma;
    
    for (int i = 0; i <= radius; i++) {
        float x = (float)i;
        kernel[i] = expf(-(x * x) / sigma2);
        sum += (i == 0) ? kernel[i] : 2.0f * kernel[i];
    }

    for (int i = 0; i <= radius; i++) {
        kernel[i] /= sum;
    }
}

__global__ void gaussian_blur_horizontal_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const uint32_t width,
    const uint32_t height,
    const int radius,
    const float* __restrict__ kernel
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float4 center = src[y * width + x];
    sum.x = center.x * kernel[0];
    sum.y = center.y * kernel[0];
    sum.z = center.z * kernel[0];
    sum.w = center.w * kernel[0];

    for (int i = 1; i <= radius; i++) {
        const float weight = kernel[i];

        const int x_left = max(0, (int)x - i);
        const float4 left = src[y * width + x_left];

        const int x_right = min((int)width - 1, (int)x + i);
        const float4 right = src[y * width + x_right];
 
        sum.x += (left.x + right.x) * weight;
        sum.y += (left.y + right.y) * weight;
        sum.z += (left.z + right.z) * weight;
        sum.w += (left.w + right.w) * weight;
    }
    
    dst[y * width + x] = sum;
}

__global__ void gaussian_blur_vertical_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const uint32_t width,
    const uint32_t height,
    const int radius,
    const float* __restrict__ kernel
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float4 center = src[y * width + x];
    sum.x = center.x * kernel[0];
    sum.y = center.y * kernel[0];
    sum.z = center.z * kernel[0];
    sum.w = center.w * kernel[0];

    for (int i = 1; i <= radius; i++) {
        const float weight = kernel[i];

        const int y_top = max(0, (int)y - i);
        const float4 top = src[y_top * width + x];

        const int y_bottom = min((int)height - 1, (int)y + i);
        const float4 bottom = src[y_bottom * width + x];

        sum.x += (top.x + bottom.x) * weight;
        sum.y += (top.y + bottom.y) * weight;
        sum.z += (top.z + bottom.z) * weight;
        sum.w += (top.w + bottom.w) * weight;
    }
    
    dst[y * width + x] = sum;
}

PyObject* py_gaussian_blur_separable_f32(PyObject* self, PyObject* args) {
    PyObject* src_capsule;
    PyObject* temp_capsule;
    PyObject* dst_capsule;
    uint32_t width, height;
    int radius;
    float sigma;
    
    if (!PyArg_ParseTuple(args, "OOOIIif", 
                          &src_capsule, &temp_capsule, &dst_capsule,
                          &width, &height, &radius, &sigma)) {
        return NULL;
    }

    if (validate_f32_buffer(src_capsule, "Source") < 0) return NULL;
    if (validate_f32_buffer(temp_capsule, "Temporary") < 0) return NULL;
    if (validate_f32_buffer(dst_capsule, "Destination") < 0) return NULL;

    if (validate_dimensions(width, height) < 0) return NULL;
    
    if (radius < 0) {
        PyErr_SetString(PyExc_ValueError, "Radius must be non-negative");
        return NULL;
    }
    
    if (radius > 100) {
        PyErr_SetString(PyExc_ValueError, "Radius too large (max 100)");
        return NULL;
    }
    
    if (sigma <= 0.0f) {
        PyErr_SetString(PyExc_ValueError, "Sigma must be positive");
        return NULL;
    }

    BufferContext* src_ctx = get_buffer_context(src_capsule);
    if (src_ctx == NULL) return NULL;
    
    BufferContext* temp_ctx = get_buffer_context(temp_capsule);
    if (temp_ctx == NULL) return NULL;
    
    BufferContext* dst_ctx = get_buffer_context(dst_capsule);
    if (dst_ctx == NULL) return NULL;

    float* h_kernel = (float*)malloc((radius + 1) * sizeof(float));
    if (h_kernel == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate kernel");
        return NULL;
    }
    compute_gaussian_kernel(h_kernel, radius, sigma);

    float* d_kernel = NULL;
    cudaError_t err = cudaMalloc(&d_kernel, (radius + 1) * sizeof(float));
    if (err != cudaSuccess) {
        free(h_kernel);
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate GPU kernel");
        return NULL;
    }
    
    err = cudaMemcpy(d_kernel, h_kernel, (radius + 1) * sizeof(float), cudaMemcpyHostToDevice);
    free(h_kernel);
    if (err != cudaSuccess) {
        cudaFree(d_kernel);
        PyErr_SetString(PyExc_RuntimeError, "Failed to copy kernel to GPU");
        return NULL;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    const float4* src_ptr = (const float4*)src_ctx->ptr;
    float4* temp_ptr = (float4*)temp_ctx->ptr;
    float4* dst_ptr = (float4*)dst_ctx->ptr;

    gaussian_blur_horizontal_kernel<<<gridSize, blockSize>>>(
        src_ptr, temp_ptr,
        width, height,
        radius, d_kernel
    );
    
    if (check_cuda_launch() < 0) {
        cudaFree(d_kernel);
        return NULL;
    }

    gaussian_blur_vertical_kernel<<<gridSize, blockSize>>>(
        temp_ptr, dst_ptr,
        width, height,
        radius, d_kernel
    );
    
    if (check_cuda_launch() < 0) {
        cudaFree(d_kernel);
        return NULL;
    }
    
    cudaFree(d_kernel);
    
    Py_RETURN_NONE;
}

}