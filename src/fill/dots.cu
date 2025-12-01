#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include "../common.h"

extern "C" {

__global__ void dots_kernel(
    float4* buffer, uint32_t w, uint32_t h,
    int spacing, float radius, 
    int off_x, int off_y, float softness,
    float4 dot_col, float4 bg_col
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int sx = (int)x - off_x;
    int sy = (int)y - off_y;

    float mx = fmodf((float)sx, (float)spacing);
    if (mx < 0) mx += spacing;
    
    float my = fmodf((float)sy, (float)spacing);
    if (my < 0) my += spacing;

    float half_space = (float)spacing * 0.5f;
    float dx = mx - half_space;
    float dy = my - half_space;

    float dist = sqrtf(dx*dx + dy*dy);
    
    float edge = (softness > 0.0f) ? (softness * radius) : 0.5f;
    float t = 1.0f - smoothstep(radius - edge, radius + edge, dist);

    float4 out;
    out.x = bg_col.x * (1.0f - t) + dot_col.x * t;
    out.y = bg_col.y * (1.0f - t) + dot_col.y * t;
    out.z = bg_col.z * (1.0f - t) + dot_col.z * t;
    out.w = bg_col.w * (1.0f - t) + dot_col.w * t;

    buffer[y * w + x] = out;
}

PyObject* py_fill_dots_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    int spacing, off_x, off_y;
    float radius, softness;
    PyObject *odot, *obg;

    if (!PyArg_ParseTuple(args, "OIIifiifOO", 
                          &capsule, &width, &height, 
                          &spacing, &radius, 
                          &off_x, &off_y, &softness, 
                          &odot, &obg)) {
        return NULL;
    }
    
    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    float cdot[4], cbg[4];
    if (parse_rgba_color(odot, cdot) < 0) return NULL;
    if (parse_rgba_color(obg, cbg) < 0) return NULL;

    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15)/16, (height + 15)/16);

    dots_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, width, height, 
        spacing, radius, 
        off_x, off_y, softness,
        make_float4(cdot[0],cdot[1],cdot[2],cdot[3]), 
        make_float4(cbg[0],cbg[1],cbg[2],cbg[3])
    );
    
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

}