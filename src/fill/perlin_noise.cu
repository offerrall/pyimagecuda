#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include "../common.h"

extern "C" {

__device__ float hash(float x, float y, float seed) {
    float n = sinf(x * 12.9898f + y * 78.233f + seed) * 43758.5453f;
    return n - floorf(n);
}

__device__ float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ float lerp_val(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ float grad(float hash_val, float x, float y) {
    float angle = hash_val * 6.283185f; // 2*PI
    float gx = cosf(angle);
    float gy = sinf(angle);
    return gx * x + gy * y;
}

__device__ float perlin_single(float x, float y, float seed) {
    float X = floorf(x);
    float Y = floorf(y);
    float rx = x - X;
    float ry = y - Y;
    
    float aa = hash(X, Y, seed);
    float ab = hash(X, Y + 1.0f, seed);
    float ba = hash(X + 1.0f, Y, seed);
    float bb = hash(X + 1.0f, Y + 1.0f, seed);
    
    float g_aa = grad(aa, rx, ry);
    float g_ab = grad(ab, rx, ry - 1.0f);
    float g_ba = grad(ba, rx - 1.0f, ry);
    float g_bb = grad(bb, rx - 1.0f, ry - 1.0f);
    
    float u = fade(rx);
    float v = fade(ry);
    
    float x1 = lerp_val(g_aa, g_ba, u);
    float x2 = lerp_val(g_ab, g_bb, u);
    
    return lerp_val(x1, x2, v);
}

__global__ void perlin_kernel(
    float4* buffer, uint32_t w, uint32_t h,
    float scale, float seed,
    int octaves, float persistence, float lacunarity,
    float off_x, float off_y,
    float4 c1, float4 c2
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    float u = ((float)x + off_x) / scale;
    float v = ((float)y + off_y) / scale;

    float total = 0.0f;
    float frequency = 1.0f;
    float amplitude = 1.0f;
    float max_value = 0.0f; 

    for(int i = 0; i < octaves; i++) {
        float octave_seed = seed + (float)i * 13.5f; 
        total += perlin_single(u * frequency, v * frequency, octave_seed) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    float val = total / max_value;
    val = val * 0.5f + 0.5f;
    val = fmaxf(0.0f, fminf(1.0f, val));

    float4 out;
    out.x = c1.x * (1.0f - val) + c2.x * val;
    out.y = c1.y * (1.0f - val) + c2.y * val;
    out.z = c1.z * (1.0f - val) + c2.z * val;
    out.w = c1.w * (1.0f - val) + c2.w * val;

    buffer[y * w + x] = out;
}

PyObject* py_fill_perlin_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    float scale, seed, persistence, lacunarity, off_x, off_y;
    int octaves;
    PyObject *o1, *o2;

    if (!PyArg_ParseTuple(args, "OIIffiffffOO", 
                          &capsule, &width, &height, 
                          &scale, &seed, 
                          &octaves, &persistence, &lacunarity, 
                          &off_x, &off_y, 
                          &o1, &o2)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Buffer") < 0) return NULL;
    
    float c1[4], c2[4];
    if (parse_rgba_color(o1, c1) < 0) return NULL;
    if (parse_rgba_color(o2, c2) < 0) return NULL;

    BufferContext* ctx = get_buffer_context(capsule);
    if (!ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15)/16, (height + 15)/16);

    perlin_kernel<<<gridSize, blockSize>>>(
        (float4*)ctx->ptr, width, height, 
        scale, seed, 
        octaves, persistence, lacunarity, 
        off_x, off_y,
        make_float4(c1[0],c1[1],c1[2],c1[3]), 
        make_float4(c2[0],c2[1],c2[2],c2[3])
    );
    
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

}