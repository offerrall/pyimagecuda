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

enum InterpolationMethod {
    INTERP_NEAREST = 0,
    INTERP_BILINEAR = 1,
    INTERP_BICUBIC = 2,
    INTERP_LANCZOS = 3
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
    const float dst_cy,
    const int interp_method
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
        
        float4 result;
        
        switch (interp_method) {
            case INTERP_NEAREST:
                result = sample_nearest(src, sx, sy, src_width, src_height);
                break;
            case INTERP_BILINEAR:
                result = sample_bilinear(src, sx, sy, src_width, src_height);
                break;
            case INTERP_BICUBIC:
                result = sample_bicubic(src, sx, sy, src_width, src_height);
                break;
            case INTERP_LANCZOS:
                result = sample_lanczos(src, sx, sy, src_width, src_height);
                break;
            default:
                result = sample_bilinear(src, sx, sy, src_width, src_height);
                break;
        }
        
        dst[y * dst_width + x] = result;
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
    int interp_method;
    
    if (!PyArg_ParseTuple(args, "OOIIIIfi", 
                          &src_capsule, &dst_capsule, 
                          &src_width, &src_height, 
                          &dst_width, &dst_height,
                          &angle_deg, &interp_method)) {
        return NULL;
    }

    if (interp_method < INTERP_NEAREST || interp_method > INTERP_LANCZOS) {
        PyErr_SetString(PyExc_ValueError, 
            "Invalid interpolation method (0=nearest, 1=bilinear, 2=bicubic, 3=lanczos)");
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
        dst_cx, dst_cy,
        interp_method
    );

    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}