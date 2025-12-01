#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

__global__ void fill_circle_kernel(
    float4* buffer, 
    uint32_t w, 
    uint32_t h,
    float cx, 
    float cy, 
    float r, 
    float softness,
    float4 shape_col, 
    float4 bg_col
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    float dx = (float)x - cx;
    float dy = (float)y - cy;
    float dist = sqrtf(dx*dx + dy*dy);


    float edge_width = (softness > 0.0f) ? (softness * r) : 0.5f;
    
    float t = 1.0f - smoothstep(r - edge_width, r + edge_width, dist);

    float4 out_px;
    out_px.x = bg_col.x * (1.0f - t) + shape_col.x * t;
    out_px.y = bg_col.y * (1.0f - t) + shape_col.y * t;
    out_px.z = bg_col.z * (1.0f - t) + shape_col.z * t;
    out_px.w = bg_col.w * (1.0f - t) + shape_col.w * t;

    buffer[y * w + x] = out_px;
}

PyObject* py_fill_circle_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    float softness;
    PyObject *col_obj, *bg_obj;

    if (!PyArg_ParseTuple(args, "OIIfOO", 
                          &capsule, 
                          &width, &height, 
                          &softness, 
                          &col_obj, &bg_obj)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;

    float s_rgba[4], b_rgba[4];
    if (parse_rgba_color(col_obj, s_rgba) < 0) return NULL;
    if (parse_rgba_color(bg_obj, b_rgba) < 0) return NULL;

    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    float cx = (float)width / 2.0f;
    float cy = (float)height / 2.0f;
    float radius = fminf((float)width, (float)height) / 2.0f;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15)/16, (height + 15)/16);

    fill_circle_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, 
        width, height, 
        cx, cy, radius, softness,
        make_float4(s_rgba[0], s_rgba[1], s_rgba[2], s_rgba[3]),
        make_float4(b_rgba[0], b_rgba[1], b_rgba[2], b_rgba[3])
    );
    
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

}