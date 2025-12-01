#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void checkerboard_kernel(
    float4* buffer, uint32_t w, uint32_t h, 
    int size, int off_x, int off_y,
    float4 c1, float4 c2
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int sx = (int)x + off_x;
    int sy = (int)y + off_y;

    int grid_x = sx / size;
    int grid_y = sy / size;

    if (sx < 0) grid_x--;
    if (sy < 0) grid_y--;

    bool is_even = (grid_x + grid_y) % 2 == 0;

    buffer[y * w + x] = is_even ? c1 : c2;
}

PyObject* py_fill_checkerboard_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    int size, off_x, off_y;
    PyObject *o1, *o2;

    if (!PyArg_ParseTuple(args, "OIIiiiOO", 
                          &capsule, &width, &height, 
                          &size, &off_x, &off_y, 
                          &o1, &o2)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    float rgba1[4], rgba2[4];
    if (parse_rgba_color(o1, rgba1) < 0) return NULL;
    if (parse_rgba_color(o2, rgba2) < 0) return NULL;

    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15)/16, (height + 15)/16);

    checkerboard_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, width, height, 
        size, off_x, off_y,
        make_float4(rgba1[0], rgba1[1], rgba1[2], rgba1[3]),
        make_float4(rgba2[0], rgba2[1], rgba2[2], rgba2[3])
    );
    
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

}