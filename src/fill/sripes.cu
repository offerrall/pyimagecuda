#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include "../common.h" 

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C" {

__global__ void stripes_kernel(
    float4* buffer, uint32_t w, uint32_t h,
    float cos_a, float sin_a, 
    float spacing, float stripe_width, float offset,
    float4 c1, float4 c2
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    float pos = (float)x * cos_a + (float)y * sin_a;

    pos += offset;

    float t = fmodf(pos, spacing);
    if (t < 0.0f) t += spacing;

    float aa = 1.0f;

    float edge_up = smoothstep(0.0f, aa, t);
    if (t > spacing - aa) {
        edge_up = smoothstep(spacing, spacing + aa, t + aa); 
    }

    float edge_down = 1.0f - smoothstep(stripe_width - aa, stripe_width + aa, t);
    float mask = fminf(edge_up, edge_down);

    float4 out;
    out.x = c2.x * (1.0f - mask) + c1.x * mask;
    out.y = c2.y * (1.0f - mask) + c1.y * mask;
    out.z = c2.z * (1.0f - mask) + c1.z * mask;
    out.w = c2.w * (1.0f - mask) + c1.w * mask;

    buffer[y * w + x] = out;
}

PyObject* py_fill_stripes_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    float angle; 
    int spacing_i, width_i, offset_i;
    PyObject *o1, *o2;

    if (!PyArg_ParseTuple(args, "OIIfiiiOO", 
                          &capsule, &width, &height, 
                          &angle, &spacing_i, &width_i, &offset_i, 
                          &o1, &o2)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    float c1[4], c2[4];
    if (parse_rgba_color(o1, c1) < 0) return NULL;
    if (parse_rgba_color(o2, c2) < 0) return NULL;

    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    float rads = angle * (M_PI / 180.0f);
    float cos_a = cosf(rads);
    float sin_a = sinf(rads);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15)/16, (height + 15)/16);

    stripes_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, width, height, 
        cos_a, sin_a, 
        (float)spacing_i, (float)width_i, (float)offset_i,
        make_float4(c1[0],c1[1],c1[2],c1[3]), 
        make_float4(c2[0],c2[1],c2[2],c2[3])
    );
    
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

}