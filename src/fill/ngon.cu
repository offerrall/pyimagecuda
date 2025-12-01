#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include "../common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C" {

__global__ void ngon_kernel(
    float4* buffer, uint32_t w, uint32_t h,
    float cx, float cy, float radius, 
    int sides, float rotation_rad, float softness,
    float4 shape_col, float4 bg_col
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    float px = (float)x - cx;
    float py = (float)y - cy;

    float cos_r = cosf(-rotation_rad);
    float sin_r = sinf(-rotation_rad);
    float rx = px * cos_r - py * sin_r;
    float ry = px * sin_r + py * cos_r;

    float angle = atan2f(rx, ry);
    float slice = (M_PI * 2.0f) / (float)sides;
    
    float sector_angle = floorf(0.5f + angle / slice) * slice;
    float angle_local = angle - sector_angle;
    
    float len = hypotf(rx, ry);

    float dist = cosf(angle_local) * len;

    float apothem = radius * cosf(M_PI / (float)sides);

    float sdf = dist - apothem;

    float edge_width = (softness > 0.0f) ? (softness * radius) : 0.5f;
    
    float t = 1.0f - smoothstep(-edge_width, edge_width, sdf);

    float4 out_px;
    out_px.x = bg_col.x * (1.0f - t) + shape_col.x * t;
    out_px.y = bg_col.y * (1.0f - t) + shape_col.y * t;
    out_px.z = bg_col.z * (1.0f - t) + shape_col.z * t;
    out_px.w = bg_col.w * (1.0f - t) + shape_col.w * t;

    buffer[y * w + x] = out_px;
}

PyObject* py_fill_ngon_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    int sides; float rotation_deg, softness;
    PyObject *col_obj, *bg_obj;

    if (!PyArg_ParseTuple(args, "OIIiffOO", 
                          &capsule, &width, &height, 
                          &sides, &rotation_deg, &softness, 
                          &col_obj, &bg_obj)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    float shape_rgba[4], bg_rgba[4];
    if (parse_rgba_color(col_obj, shape_rgba) < 0) return NULL;
    if (parse_rgba_color(bg_obj, bg_rgba) < 0) return NULL;

    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;
    
    float cx = (float)width / 2.0f;
    float cy = (float)height / 2.0f;
    float radius = fminf((float)width, (float)height) / 2.0f;
    
    float rotation_rad = rotation_deg * (M_PI / 180.0f);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15)/16, (height + 15)/16);

    ngon_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, width, height, 
        cx, cy, radius, 
        sides, rotation_rad, softness,
        make_float4(shape_rgba[0], shape_rgba[1], shape_rgba[2], shape_rgba[3]),
        make_float4(bg_rgba[0], bg_rgba[1], bg_rgba[2], bg_rgba[3])
    );
    
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

}