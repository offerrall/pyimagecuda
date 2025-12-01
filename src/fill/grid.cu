#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h" 

extern "C" {

__global__ void grid_kernel(
    float4* buffer, uint32_t w, uint32_t h, 
    int spacing, int line_w, 
    int off_x, int off_y,
    float4 line_col, float4 bg_col
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int sx = (int)x + off_x;
    int sy = (int)y + off_y;

    int mod_x = sx % spacing;
    if (mod_x < 0) mod_x += spacing;

    int mod_y = sy % spacing;
    if (mod_y < 0) mod_y += spacing;

    bool is_line = (mod_x < line_w) || (mod_y < line_w);

    buffer[y * w + x] = is_line ? line_col : bg_col;
}

PyObject* py_fill_grid_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    int spacing, line_w, off_x, off_y;
    PyObject *o_line, *o_bg;

    if (!PyArg_ParseTuple(args, "OIIiiiiOO", 
                          &capsule, &width, &height, 
                          &spacing, &line_w, &off_x, &off_y, 
                          &o_line, &o_bg)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;

    float col_l[4], col_bg[4];
    if (parse_rgba_color(o_line, col_l) < 0) return NULL;
    if (parse_rgba_color(o_bg, col_bg) < 0) return NULL;

    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15)/16, (height + 15)/16);

    grid_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, width, height, 
        spacing, line_w, off_x, off_y,
        make_float4(col_l[0], col_l[1], col_l[2], col_l[3]),
        make_float4(col_bg[0], col_bg[1], col_bg[2], col_bg[3])
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}