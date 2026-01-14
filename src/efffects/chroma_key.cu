#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../common.h"

extern "C" {

__device__ __forceinline__ float color_distance_hsv(
    const float h1, const float s1, const float v1,
    const float h2, const float s2, const float v2
) {
    float hue_diff = fabsf(h1 - h2);
    if (hue_diff > 180.0f) {
        hue_diff = 360.0f - hue_diff;
    }
    hue_diff /= 180.0f;
    
    const float sat_diff = fabsf(s1 - s2);
    const float val_diff = fabsf(v1 - v2);
    
    return sqrtf(
        hue_diff * hue_diff * 2.0f +
        sat_diff * sat_diff * 1.0f +
        val_diff * val_diff * 0.5f
    );
}

__device__ __forceinline__ void suppress_spill(
    float* r, float* g, float* b,
    const float kr, const float kg, const float kb,
    const float amount
) {
    const float spill_r = fmaxf(0.0f, *r - kr * amount);
    const float spill_g = fmaxf(0.0f, *g - kg * amount);
    const float spill_b = fmaxf(0.0f, *b - kb * amount);
    
    const float original_luma = *r * 0.299f + *g * 0.587f + *b * 0.114f;
    const float spill_luma = spill_r * 0.299f + spill_g * 0.587f + spill_b * 0.114f;
    
    if (spill_luma > 0.001f) {
        const float luma_ratio = original_luma / spill_luma;
        *r = fminf(1.0f, spill_r * luma_ratio);
        *g = fminf(1.0f, spill_g * luma_ratio);
        *b = fminf(1.0f, spill_b * luma_ratio);
    }
}

__global__ void chroma_key_kernel(
    float4* __restrict__ image,
    const uint32_t width,
    const uint32_t height,
    const float key_r,
    const float key_g,
    const float key_b,
    const float threshold,
    const float smoothness,
    const float spill_suppression
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint32_t idx = y * width + x;
    float4 pixel = image[idx];
    
    float h, s, v;
    rgb_to_hsv(pixel.x, pixel.y, pixel.z, &h, &s, &v);
    
    float kh, ks, kv;
    rgb_to_hsv(key_r, key_g, key_b, &kh, &ks, &kv);
    
    const float dist = color_distance_hsv(h, s, v, kh, ks, kv);
    
    float alpha = 1.0f;
    
    if (dist < threshold) {
        if (smoothness > 0.0001f) {
            const float edge_start = fmaxf(0.0f, threshold - smoothness);
            const float edge_end = threshold;
            alpha = smoothstep(edge_start, edge_end, dist);
        } else {
            alpha = 0.0f;
        }
    }
    
    if (spill_suppression > 0.001f && alpha > 0.1f && alpha < 0.9f) {
        const float spill_amount = (1.0f - alpha) * spill_suppression;
        suppress_spill(&pixel.x, &pixel.y, &pixel.z, key_r, key_g, key_b, spill_amount);
    }
    
    pixel.w *= alpha;
    
    image[idx] = pixel;
}

PyObject* py_chroma_key_f32(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    PyObject* key_color_obj;
    float threshold, smoothness, spill_suppression;
    
    if (!PyArg_ParseTuple(args, "OIIOfff",
                          &capsule,
                          &width, &height,
                          &key_color_obj,
                          &threshold, &smoothness,
                          &spill_suppression)) {
        return NULL;
    }

    if (validate_f32_buffer(capsule, "Image") < 0) return NULL;
    if (validate_dimensions(width, height) < 0) return NULL;

    if (!PyTuple_Check(key_color_obj) && !PyList_Check(key_color_obj)) {
        PyErr_SetString(PyExc_TypeError, "key_color must be a tuple or list");
        return NULL;
    }
    
    Py_ssize_t size = PySequence_Size(key_color_obj);
    if (size != 3) {
        PyErr_Format(PyExc_ValueError, "key_color must have 3 elements (RGB), got %zd", size);
        return NULL;
    }
    
    float key_r = 0.0f, key_g = 0.0f, key_b = 0.0f;
    
    PyObject* item0 = PySequence_GetItem(key_color_obj, 0);
    PyObject* item1 = PySequence_GetItem(key_color_obj, 1);
    PyObject* item2 = PySequence_GetItem(key_color_obj, 2);
    
    if (item0 && item1 && item2) {
        key_r = (float)PyFloat_AsDouble(item0);
        key_g = (float)PyFloat_AsDouble(item1);
        key_b = (float)PyFloat_AsDouble(item2);
    }
    
    Py_XDECREF(item0);
    Py_XDECREF(item1);
    Py_XDECREF(item2);
    
    if (PyErr_Occurred()) {
        return NULL;
    }

    if (threshold < 0.0f || threshold > 1.0f) {
        PyErr_SetString(PyExc_ValueError, "threshold must be in range [0.0, 1.0]");
        return NULL;
    }
    
    if (smoothness < 0.0f || smoothness > 1.0f) {
        PyErr_SetString(PyExc_ValueError, "smoothness must be in range [0.0, 1.0]");
        return NULL;
    }
    
    if (spill_suppression < 0.0f || spill_suppression > 1.0f) {
        PyErr_SetString(PyExc_ValueError, "spill_suppression must be in range [0.0, 1.0]");
        return NULL;
    }
    
    BufferContext* ctx = get_buffer_context(capsule);
    if (ctx == NULL) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    float4* ptr = (float4*)ctx->ptr;

    chroma_key_kernel<<<gridSize, blockSize>>>(
        ptr, width, height,
        key_r, key_g, key_b,
        threshold, smoothness, spill_suppression
    );
    
    if (check_cuda_launch() < 0) return NULL;
    
    Py_RETURN_NONE;
}

}