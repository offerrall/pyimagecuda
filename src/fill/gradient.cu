#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void fill_gradient_kernel(float4* buffer, uint32_t width, uint32_t height,
                                      float r1, float g1, float b1, float a1,
                                      float r2, float g2, float b2, float a2,
                                      int direction, int seamless) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float factor = 0.0f;
    
    float nx = (width > 1) ? ((float)x / (float)(width - 1) - 0.5f) : 0.0f;
    float ny = (height > 1) ? ((float)y / (float)(height - 1) - 0.5f) : 0.0f;
    
    switch(direction) {
        case 0:
            factor = (width > 1) ? (float)x / (float)(width - 1) : 0.5f;
            break;
            
        case 1:
            factor = (height > 1) ? (float)y / (float)(height - 1) : 0.5f;
            break;
            
        case 2:
            if (width > 1 && height > 1) {
                float u = (float)x / (float)(width - 1);
                float v = (float)y / (float)(height - 1);
                factor = (u + v) * 0.5f;
            } else {
                factor = 0.5f;
            }
            break;
            
        case 3:
            factor = sqrtf(nx * nx + ny * ny) * 1.414213562f;
            factor = fminf(1.0f, factor);
            break;
            
        default:
            factor = 0.5f;
            break;
    }
    
    if (seamless) {
        factor = (factor < 0.5f) ? (factor * 2.0f) : (2.0f * (1.0f - factor));
    }
    
    factor = fminf(fmaxf(factor, 0.0f), 1.0f);
    
    float r = r1 + (r2 - r1) * factor;
    float g = g1 + (g2 - g1) * factor;
    float b = b1 + (b2 - b1) * factor;
    float a = a1 + (a2 - a1) * factor;
    
    uint32_t idx = y * width + x;
    buffer[idx] = make_float4(r, g, b, a);
}

PyObject* py_fill_gradient_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    PyObject* color1_obj;
    PyObject* color2_obj;
    uint32_t width, height;
    int direction;
    int seamless;
    
    if (!PyArg_ParseTuple(args, "OOOIIip", &capsule, &color1_obj, &color2_obj, 
                          &width, &height, &direction, &seamless)) {
        return NULL;
    }
    
    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    if (validate_dimensions(width, height) < 0) return NULL;
    
    if (direction < 0 || direction > 3) {
        PyErr_SetString(PyExc_ValueError, 
            "Direction must be 0 (horizontal), 1 (vertical), 2 (diagonal), or 3 (radial)");
        return NULL;
    }
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (ctx == NULL) return NULL;
    
    float rgba1[4], rgba2[4];
    if (parse_rgba_color(color1_obj, rgba1) < 0) return NULL;
    if (parse_rgba_color(color2_obj, rgba2) < 0) return NULL;
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    fill_gradient_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, width, height,
        rgba1[0], rgba1[1], rgba1[2], rgba1[3],
        rgba2[0], rgba2[1], rgba2[2], rgba2[3],
        direction, seamless
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}