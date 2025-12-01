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
    
    return 0;  // Success
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

#endif