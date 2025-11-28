#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h"

extern "C" {

enum RotateMode {
    ROTATE_90_CW = 0,
    ROTATE_180 = 1,
    ROTATE_270_CW = 2
};

__global__ void rotate_fixed_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const uint32_t src_width,
    const uint32_t src_height,
    const uint32_t dst_width,
    const uint32_t dst_height,
    const int mode,
    const int offset_x,
    const int offset_y
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height) return;
    
    const int rx = x - offset_x;
    const int ry = y - offset_y;
    
    uint32_t rot_w, rot_h;
    if (mode == ROTATE_180) {
        rot_w = src_width;
        rot_h = src_height;
    } else {
        rot_w = src_height;
        rot_h = src_width;
    }
    
    if (rx < 0 || ry < 0 || rx >= rot_w || ry >= rot_h) {
        dst[y * dst_width + x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    
    uint32_t src_x, src_y;
    
    if (mode == ROTATE_90_CW) {
        src_x = ry;
        src_y = src_height - 1 - rx;
    } else if (mode == ROTATE_180) {
        src_x = src_width - 1 - rx;
        src_y = src_height - 1 - ry;
    } else { 
        src_x = src_width - 1 - ry;
        src_y = rx;
    }
    
    const uint32_t src_idx = src_y * src_width + src_x;
    const uint32_t dst_idx = y * dst_width + x;
    
    dst[dst_idx] = src[src_idx];
}

PyObject* py_rotate_fixed_f32(PyObject* self, PyObject* args) {
    PyObject* src_capsule;
    PyObject* dst_capsule;
    uint32_t src_width, src_height;
    uint32_t dst_width, dst_height;
    int mode;
    int offset_x, offset_y;
    
    if (!PyArg_ParseTuple(args, "OOIIIIiii", 
                          &src_capsule, &dst_capsule, 
                          &src_width, &src_height, 
                          &dst_width, &dst_height,
                          &mode, &offset_x, &offset_y)) {
        return NULL;
    }

    if (mode < 0 || mode > 2) {
        PyErr_SetString(PyExc_ValueError, "Invalid rotation mode (must be 0, 1, or 2)");
        return NULL;
    }

    if (validate_f32_buffer(src_capsule, "Source") < 0) return NULL;
    if (validate_f32_buffer(dst_capsule, "Destination") < 0) return NULL;
    
    BufferContext* src_ctx = get_buffer_context(src_capsule);
    BufferContext* dst_ctx = get_buffer_context(dst_capsule);
    if (!src_ctx || !dst_ctx) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (dst_width + blockSize.x - 1) / blockSize.x,
        (dst_height + blockSize.y - 1) / blockSize.y
    );
    
    rotate_fixed_kernel<<<gridSize, blockSize>>>(
        (const float4*)src_ctx->ptr, (float4*)dst_ctx->ptr, 
        src_width, src_height,
        dst_width, dst_height,
        mode, offset_x, offset_y
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

__device__ __forceinline__ float4 sample_bilinear(
    const float4* src,
    int width,
    int height,
    float u,
    float v
) {
    int x0 = (int)floorf(u);
    int y0 = (int)floorf(v);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wx = u - (float)x0;
    float wy = v - (float)y0;

    x0 = max(0, min(x0, width - 1));
    y0 = max(0, min(y0, height - 1));
    x1 = max(0, min(x1, width - 1));
    y1 = max(0, min(y1, height - 1));

    float4 p00 = src[y0 * width + x0];
    float4 p10 = src[y0 * width + x1];
    float4 p01 = src[y1 * width + x0];
    float4 p11 = src[y1 * width + x1];

    float4 top = make_float4(
        p00.x * (1 - wx) + p10.x * wx,
        p00.y * (1 - wx) + p10.y * wx,
        p00.z * (1 - wx) + p10.z * wx,
        p00.w * (1 - wx) + p10.w * wx
    );

    float4 bottom = make_float4(
        p01.x * (1 - wx) + p11.x * wx,
        p01.y * (1 - wx) + p11.y * wx,
        p01.z * (1 - wx) + p11.z * wx,
        p01.w * (1 - wx) + p11.w * wx
    );

    return make_float4(
        top.x * (1 - wy) + bottom.x * wy,
        top.y * (1 - wy) + bottom.y * wy,
        top.z * (1 - wy) + bottom.z * wy,
        top.w * (1 - wy) + bottom.w * wy
    );
}

__global__ void rotate_arbitrary_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    const uint32_t src_width,
    const uint32_t src_height,
    const uint32_t dst_width,
    const uint32_t dst_height,
    const float sin_theta,
    const float cos_theta,
    const float src_cx,
    const float src_cy,
    const float dst_cx,
    const float dst_cy
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) return;

    float dx = (float)x - dst_cx;
    float dy = (float)y - dst_cy;

    float sx = dx * cos_theta + dy * sin_theta;
    float sy = -dx * sin_theta + dy * cos_theta;

    sx += src_cx;
    sy += src_cy;

    if (sx > -0.5f && sx < (float)src_width - 0.5f && 
        sy > -0.5f && sy < (float)src_height - 0.5f) {
        
        dst[y * dst_width + x] = sample_bilinear(src, src_width, src_height, sx, sy);
    } else {
        dst[y * dst_width + x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

PyObject* py_rotate_arbitrary_f32(PyObject* self, PyObject* args) {
    PyObject* src_capsule;
    PyObject* dst_capsule;
    uint32_t src_width, src_height;
    uint32_t dst_width, dst_height;
    float angle_deg;
    
    if (!PyArg_ParseTuple(args, "OOIIIIf", 
                          &src_capsule, &dst_capsule, 
                          &src_width, &src_height, 
                          &dst_width, &dst_height,
                          &angle_deg)) {
        return NULL;
    }

    if (validate_f32_buffer(src_capsule, "Source") < 0) return NULL;
    if (validate_f32_buffer(dst_capsule, "Destination") < 0) return NULL;
    
    BufferContext* src_ctx = get_buffer_context(src_capsule);
    BufferContext* dst_ctx = get_buffer_context(dst_capsule);
    if (!src_ctx || !dst_ctx) return NULL;

    float angle_rad = angle_deg * (3.14159265359f / 180.0f);
    float sin_t = sinf(angle_rad);
    float cos_t = cosf(angle_rad);

    float src_cx = (float)src_width / 2.0f - 0.5f;
    float src_cy = (float)src_height / 2.0f - 0.5f;
    float dst_cx = (float)dst_width / 2.0f - 0.5f;
    float dst_cy = (float)dst_height / 2.0f - 0.5f;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (dst_width + blockSize.x - 1) / blockSize.x,
        (dst_height + blockSize.y - 1) / blockSize.y
    );

    rotate_arbitrary_kernel<<<gridSize, blockSize>>>(
        (const float4*)src_ctx->ptr,
        (float4*)dst_ctx->ptr,
        src_width, src_height,
        dst_width, dst_height,
        sin_t, cos_t,
        src_cx, src_cy,
        dst_cx, dst_cy
    );

    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}