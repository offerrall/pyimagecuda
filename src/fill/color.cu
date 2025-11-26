#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void fill_color_kernel(float4* buffer, uint32_t width, uint32_t height, 
                                   float r, float g, float b, float a) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    buffer[y * width + x] = make_float4(r, g, b, a);
}

PyObject* py_fill_color_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    PyObject* color_obj;
    uint32_t width, height;
    
    if (!PyArg_ParseTuple(args, "OOII", &capsule, &color_obj, &width, &height)) {
        return NULL;
    }
    
    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    if (validate_dimensions(width, height) < 0) return NULL;
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (ctx == NULL) return NULL;
    
    float rgba[4];
    if (parse_rgba_color(color_obj, rgba) < 0) return NULL;
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    fill_color_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, width, height,
        rgba[0], rgba[1], rgba[2], rgba[3]
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}