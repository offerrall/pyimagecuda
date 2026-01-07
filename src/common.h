#ifndef PYIMAGECUDA_BUFFER_COMMON_H
#define PYIMAGECUDA_BUFFER_COMMON_H

#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <string.h>

#define BUFFER_TYPE_FLOAT32 "pyimagecuda.Buffer.F32"
#define BUFFER_TYPE_UINT8   "pyimagecuda.Buffer.U8"

#define MAX_DIMENSION 32768


typedef struct {
    void* ptr;
    int freed;
} BufferContext;


static inline int validate_dimensions(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        PyErr_SetString(PyExc_ValueError, "Width and height must be positive");
        return -1;
    }
    
    if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
        PyErr_Format(PyExc_ValueError, "Dimensions too large (max %d)", MAX_DIMENSION);
        return -1;
    }
    
    return 0;
}

static inline BufferContext* get_buffer_context(PyObject* capsule) {
    const char* name = PyCapsule_GetName(capsule);
    BufferContext* ctx = (BufferContext*)PyCapsule_GetPointer(capsule, name);
    
    if (ctx == NULL || ctx->freed || ctx->ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Buffer has been freed or is invalid");
        return NULL;
    }
    
    return ctx;
}

static inline int parse_rgba_color(PyObject* color_obj, float rgba[4]) {
    for (int i = 0; i < 4; i++) {
        PyObject* item = PySequence_GetItem(color_obj, i);
        if (item == NULL) return -1;
        
        rgba[i] = (float)PyFloat_AsDouble(item);
        Py_DECREF(item);
        
        if (PyErr_Occurred()) return -1;
    }
    
    return 0;
}

static inline int is_float32_buffer(PyObject* capsule) {
    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "Expected a Buffer capsule");
        return -1;
    }
    
    const char* name = PyCapsule_GetName(capsule);
    if (name == NULL) {
        PyErr_SetString(PyExc_TypeError, "Invalid buffer capsule");
        return -1;
    }
    
    if (strcmp(name, BUFFER_TYPE_FLOAT32) == 0) {
        return 1;  // Is float32
    } else if (strcmp(name, BUFFER_TYPE_UINT8) == 0) {
        return 0;  // Is uint8
    } else {
        PyErr_SetString(PyExc_TypeError, "Unknown buffer type");
        return -1;
    }
}

static inline int validate_f32_buffer(PyObject* capsule, const char* buffer_name) {
    int is_f32 = is_float32_buffer(capsule);
    if (is_f32 < 0) return -1;  // Error already set
    
    if (is_f32 == 0) {
        PyErr_Format(PyExc_TypeError, "%s buffer must be F32, not U8", buffer_name);
        return -1;
    }
    
    return 0;
}

static inline int check_cuda_launch() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, "Kernel launch failed: %s", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static __device__ __forceinline__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f); 
    return t * t * (3.0f - 2.0f * t);
}

static __device__ __forceinline__ float4 sample_nearest(
    const float4* __restrict__ src,
    const float sx,
    const float sy,
    const uint32_t src_width,
    const uint32_t src_height
) {
    const int x = (int)floorf(sx + 0.5f);
    const int y = (int)floorf(sy + 0.5f);
    
    const int cx = max(0, min(x, (int)src_width - 1));
    const int cy = max(0, min(y, (int)src_height - 1));
    
    return src[cy * src_width + cx];
}

static __device__ __forceinline__ float4 sample_bilinear(
    const float4* __restrict__ src,
    const float sx,
    const float sy,
    const uint32_t src_width,
    const uint32_t src_height
) {
    const float src_x = fmaxf(0.0f, fminf(sx, (float)(src_width - 1)));
    const float src_y = fmaxf(0.0f, fminf(sy, (float)(src_height - 1)));
    
    const uint32_t x0 = (uint32_t)src_x;
    const uint32_t y0 = (uint32_t)src_y;
    const uint32_t x1 = min(x0 + 1, src_width - 1);
    const uint32_t y1 = min(y0 + 1, src_height - 1);
    
    const float fx = src_x - (float)x0;
    const float fy = src_y - (float)y0;
    
    const float4 p00 = src[y0 * src_width + x0];
    const float4 p10 = src[y0 * src_width + x1];
    const float4 p01 = src[y1 * src_width + x0];
    const float4 p11 = src[y1 * src_width + x1];
    
    const float4 top = make_float4(
        fmaf(p10.x - p00.x, fx, p00.x),
        fmaf(p10.y - p00.y, fx, p00.y),
        fmaf(p10.z - p00.z, fx, p00.z),
        fmaf(p10.w - p00.w, fx, p00.w)
    );
    
    const float4 bottom = make_float4(
        fmaf(p11.x - p01.x, fx, p01.x),
        fmaf(p11.y - p01.y, fx, p01.y),
        fmaf(p11.z - p01.z, fx, p01.z),
        fmaf(p11.w - p01.w, fx, p01.w)
    );
    
    return make_float4(
        fmaf(bottom.x - top.x, fy, top.x),
        fmaf(bottom.y - top.y, fy, top.y),
        fmaf(bottom.z - top.z, fy, top.z),
        fmaf(bottom.w - top.w, fy, top.w)
    );
}

static __device__ __forceinline__ float cubic_weight(const float x) {
    const float a = -0.5f;
    const float abs_x = fabsf(x);
    
    if (abs_x <= 1.0f) {
        return ((a + 2.0f) * abs_x - (a + 3.0f)) * abs_x * abs_x + 1.0f;
    } else if (abs_x < 2.0f) {
        return ((a * abs_x - 5.0f * a) * abs_x + 8.0f * a) * abs_x - 4.0f * a;
    }
    return 0.0f;
}

static __device__ __forceinline__ float4 sample_bicubic(
    const float4* __restrict__ src,
    const float sx,
    const float sy,
    const uint32_t src_width,
    const uint32_t src_height
) {
    const int x0 = (int)floorf(sx);
    const int y0 = (int)floorf(sy);
    
    const float fx = sx - (float)x0;
    const float fy = sy - (float)y0;
    
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    float sum_w = 0.0f;
    float weight_sum = 0.0f;
    
    #pragma unroll
    for (int dy = -1; dy <= 2; ++dy) {
        const int sy_idx = y0 + dy;
        if (sy_idx < 0 || sy_idx >= (int)src_height) continue;
        
        const float wy = cubic_weight((float)dy - fy);
        const int row_offset = sy_idx * src_width;
        
        #pragma unroll
        for (int dx = -1; dx <= 2; ++dx) {
            const int sx_idx = x0 + dx;
            if (sx_idx < 0 || sx_idx >= (int)src_width) continue;
            
            const float wx = cubic_weight((float)dx - fx);
            const float w = wx * wy;
            
            const float4 pixel = src[row_offset + sx_idx];
            sum_x = fmaf(pixel.x, w, sum_x);
            sum_y = fmaf(pixel.y, w, sum_y);
            sum_z = fmaf(pixel.z, w, sum_z);
            sum_w = fmaf(pixel.w, w, sum_w);
            
            weight_sum += w;
        }
    }
    
    if (weight_sum > 0.0f) {
        const float inv_weight = 1.0f / weight_sum;
        sum_x *= inv_weight;
        sum_y *= inv_weight;
        sum_z *= inv_weight;
        sum_w *= inv_weight;
    }
    
    return make_float4(sum_x, sum_y, sum_z, sum_w);
}

static __device__ __forceinline__ float sinc(const float x) {
    if (fabsf(x) < 1e-5f) return 1.0f;
    const float pi_x = 3.14159265358979323846f * x;
    return sinf(pi_x) / pi_x;
}

static __device__ __forceinline__ float lanczos_weight(const float x) {
    const float abs_x = fabsf(x);
    if (abs_x >= 3.0f) return 0.0f;
    
    const float inv_a = 0.333333333333f;
    return sinc(abs_x) * sinc(abs_x * inv_a);
}

static __device__ __forceinline__ float4 sample_lanczos(
    const float4* __restrict__ src,
    const float sx,
    const float sy,
    const uint32_t src_width,
    const uint32_t src_height
) {
    const int x_center = (int)floorf(sx + 0.5f);
    const int y_center = (int)floorf(sy + 0.5f);
    
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    float sum_w = 0.0f;
    float weight_sum = 0.0f;
    
    #pragma unroll
    for (int dy = -2; dy <= 3; ++dy) {
        const int sy_idx = y_center + dy;
        if (sy_idx < 0 || sy_idx >= (int)src_height) continue;
        
        const float wy = lanczos_weight(sy - (float)sy_idx);
        const int row_offset = sy_idx * src_width;
        
        #pragma unroll
        for (int dx = -2; dx <= 3; ++dx) {
            const int sx_idx = x_center + dx;
            if (sx_idx < 0 || sx_idx >= (int)src_width) continue;
            
            const float wx = lanczos_weight(sx - (float)sx_idx);
            const float w = wx * wy;
            
            const float4 pixel = src[row_offset + sx_idx];
            sum_x = fmaf(pixel.x, w, sum_x);
            sum_y = fmaf(pixel.y, w, sum_y);
            sum_z = fmaf(pixel.z, w, sum_z);
            sum_w = fmaf(pixel.w, w, sum_w);
            
            weight_sum += w;
        }
    }
    
    if (weight_sum > 0.0f) {
        const float inv_weight = 1.0f / weight_sum;
        sum_x = fmaxf(0.0f, fminf(1.0f, sum_x * inv_weight));
        sum_y = fmaxf(0.0f, fminf(1.0f, sum_y * inv_weight));
        sum_z = fmaxf(0.0f, fminf(1.0f, sum_z * inv_weight));
        sum_w = fmaxf(0.0f, fminf(1.0f, sum_w * inv_weight));
    }
    
    return make_float4(sum_x, sum_y, sum_z, sum_w);
}

#endif 