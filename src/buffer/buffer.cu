#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" {

static void buffer_capsule_destructor(PyObject* capsule) {
    const char* name = PyCapsule_GetName(capsule);
    if (name == NULL) return;
    
    BufferContext* ctx = (BufferContext*)PyCapsule_GetPointer(capsule, name);
    if (ctx == NULL) return;

    if (!ctx->freed && ctx->ptr != NULL) {
        cudaError_t err = cudaFree(ctx->ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaFree failed in destructor: %s\n",
                    cudaGetErrorString(err));
        }
    }

    free(ctx);
}

PyObject* py_create_buffer_f32(PyObject* self, PyObject* args) {
    uint32_t width, height;
    
    if (!PyArg_ParseTuple(args, "II", &width, &height)) {
        return NULL;
    }
    
    if (validate_dimensions(width, height) < 0) {
        return NULL;
    }
    
    size_t total_size = (size_t)width * height * sizeof(float4);
    
    float4* buffer;
    cudaError_t error = cudaMalloc(&buffer, total_size);
    
    if (error != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, "CUDA malloc failed: %s", 
                    cudaGetErrorString(error));
        return NULL;
    }
    
    BufferContext* ctx = (BufferContext*)malloc(sizeof(BufferContext));
    if (ctx == NULL) {
        cudaFree(buffer);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate buffer context");
        return NULL;
    }
    
    ctx->ptr = buffer;
    ctx->freed = 0;
    
    PyObject* capsule = PyCapsule_New(ctx, BUFFER_TYPE_FLOAT32, buffer_capsule_destructor);
    
    if (capsule == NULL) {
        cudaFree(buffer);
        free(ctx);
        return NULL;
    }
    
    return capsule;
}

PyObject* py_create_buffer_u8(PyObject* self, PyObject* args) {
    uint32_t width, height;
    
    if (!PyArg_ParseTuple(args, "II", &width, &height)) {
        return NULL;
    }
    
    if (validate_dimensions(width, height) < 0) {
        return NULL;
    }
    
    size_t total_size = (size_t)width * height * sizeof(uchar4);
    
    uchar4* buffer;
    cudaError_t error = cudaMalloc(&buffer, total_size);
    
    if (error != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, "CUDA malloc failed: %s", 
                    cudaGetErrorString(error));
        return NULL;
    }
    
    BufferContext* ctx = (BufferContext*)malloc(sizeof(BufferContext));
    if (ctx == NULL) {
        cudaFree(buffer);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate buffer context");
        return NULL;
    }
    
    ctx->ptr = buffer;
    ctx->freed = 0;
    
    PyObject* capsule = PyCapsule_New(ctx, BUFFER_TYPE_UINT8, buffer_capsule_destructor);
    
    if (capsule == NULL) {
        cudaFree(buffer);
        free(ctx);
        return NULL;
    }
    
    return capsule;
}

PyObject* py_free_buffer(PyObject* self, PyObject* args) {
    PyObject* capsule;
    
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    
    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "Expected a Buffer capsule");
        return NULL;
    }
    
    const char* name = PyCapsule_GetName(capsule);
    
    if (name == NULL || 
        (strcmp(name, BUFFER_TYPE_FLOAT32) != 0 && 
         strcmp(name, BUFFER_TYPE_UINT8) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Expected a Buffer capsule");
        return NULL;
    }
    
    BufferContext* ctx = (BufferContext*)PyCapsule_GetPointer(capsule, name);
    
    if (ctx == NULL || ctx->freed) {
        Py_RETURN_NONE;
    }
    
    if (ctx->ptr != NULL) {
        cudaError_t err = cudaFree(ctx->ptr);
        if (err != cudaSuccess) {
            PyErr_Format(PyExc_RuntimeError, "cudaFree failed: %s", 
                        cudaGetErrorString(err));
            return NULL;
        }
    }
    
    ctx->freed = 1;
    
    Py_RETURN_NONE;
}

PyObject* py_upload_to_buffer(PyObject* self, PyObject* args) {
    PyObject* capsule;
    PyObject* bytes_obj;
    uint32_t width, height;
    
    if (!PyArg_ParseTuple(args, "OOII", &capsule, &bytes_obj, &width, &height)) {
        return NULL;
    }
    
    if (validate_dimensions(width, height) < 0) {
        return NULL;
    }
    
    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "Expected a Buffer capsule");
        return NULL;
    }
    
    const char* name = PyCapsule_GetName(capsule);
    if (name == NULL || 
        (strcmp(name, BUFFER_TYPE_FLOAT32) != 0 && 
         strcmp(name, BUFFER_TYPE_UINT8) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Expected a Buffer capsule");
        return NULL;
    }
    
    BufferContext* ctx = (BufferContext*)PyCapsule_GetPointer(capsule, name);
    if (ctx == NULL || ctx->freed || ctx->ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Buffer has been freed");
        return NULL;
    }
    
    Py_buffer py_buf;
    if (PyObject_GetBuffer(bytes_obj, &py_buf, PyBUF_SIMPLE) < 0) {
        return NULL;
    }
    
    size_t expected_size;
    if (strcmp(name, BUFFER_TYPE_FLOAT32) == 0) {
        expected_size = (size_t)width * height * sizeof(float4);
    } else {
        expected_size = (size_t)width * height * sizeof(uchar4);
    }
    
    if ((size_t)py_buf.len != expected_size) {
        PyBuffer_Release(&py_buf);
        PyErr_Format(PyExc_ValueError, "Buffer size mismatch: expected %zu bytes, got %zd bytes", expected_size, py_buf.len);
        return NULL;
    }
    
    cudaError_t error = cudaMemcpy(ctx->ptr, py_buf.buf, py_buf.len, cudaMemcpyHostToDevice);
    PyBuffer_Release(&py_buf);
    
    if (error != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, "CUDA upload failed: %s", cudaGetErrorString(error));
        return NULL;
    }
    
    Py_RETURN_NONE;
}

PyObject* py_download_from_buffer(PyObject* self, PyObject* args) {
    PyObject* capsule;
    uint32_t width, height;
    
    if (!PyArg_ParseTuple(args, "OII", &capsule, &width, &height)) {
        return NULL;
    }
    
    if (validate_dimensions(width, height) < 0) {
        return NULL;
    }
    
    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "Expected a Buffer capsule");
        return NULL;
    }
    
    const char* name = PyCapsule_GetName(capsule);
    if (name == NULL || 
        (strcmp(name, BUFFER_TYPE_FLOAT32) != 0 && 
         strcmp(name, BUFFER_TYPE_UINT8) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Expected a Buffer capsule");
        return NULL;
    }
    
    BufferContext* ctx = (BufferContext*)PyCapsule_GetPointer(capsule, name);
    if (ctx == NULL || ctx->freed || ctx->ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Buffer has been freed");
        return NULL;
    }
    
    size_t data_size;
    if (strcmp(name, BUFFER_TYPE_FLOAT32) == 0) {
        data_size = (size_t)width * height * sizeof(float4);
    } else {
        data_size = (size_t)width * height * sizeof(uchar4);
    }
    
    PyObject* bytes_obj = PyBytes_FromStringAndSize(NULL, data_size);
    if (!bytes_obj) {
        return NULL;
    }
    
    char* bytes_buffer = PyBytes_AsString(bytes_obj);
    if (!bytes_buffer) {
        Py_DECREF(bytes_obj);
        return NULL;
    }
    
    cudaError_t error = cudaMemcpy(bytes_buffer, ctx->ptr, data_size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        Py_DECREF(bytes_obj);
        PyErr_Format(PyExc_RuntimeError, "CUDA download failed: %s", cudaGetErrorString(error));
        return NULL;
    }
    
    return bytes_obj;
}

PyObject* py_copy_buffer(PyObject* self, PyObject* args) {
    PyObject *dst_capsule, *src_capsule;
    uint32_t width, height;

    if (!PyArg_ParseTuple(args, "OOII", &dst_capsule, &src_capsule, &width, &height)) {
        return NULL;
    }

    if (validate_dimensions(width, height) < 0) {
        return NULL;
    }

    if (!PyCapsule_CheckExact(dst_capsule)) {
        PyErr_SetString(PyExc_TypeError, "Expected dst to be a Buffer capsule");
        return NULL;
    }

    if (!PyCapsule_CheckExact(src_capsule)) {
        PyErr_SetString(PyExc_TypeError, "Expected src to be a Buffer capsule");
        return NULL;
    }
    
    const char* dst_name = PyCapsule_GetName(dst_capsule);
    const char* src_name = PyCapsule_GetName(src_capsule);
    
    if (dst_name == NULL || src_name == NULL ||
        (strcmp(dst_name, BUFFER_TYPE_FLOAT32) != 0 && strcmp(dst_name, BUFFER_TYPE_UINT8) != 0) ||
        (strcmp(src_name, BUFFER_TYPE_FLOAT32) != 0 && strcmp(src_name, BUFFER_TYPE_UINT8) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Expected Buffer capsules");
        return NULL;
    }
    
    if (strcmp(dst_name, src_name) != 0) {
        PyErr_SetString(PyExc_TypeError, "Source and destination must be same buffer type");
        return NULL;
    }
    
    BufferContext* dst_ctx = (BufferContext*)PyCapsule_GetPointer(dst_capsule, dst_name);
    BufferContext* src_ctx = (BufferContext*)PyCapsule_GetPointer(src_capsule, src_name);
    
    if (dst_ctx == NULL || dst_ctx->freed || dst_ctx->ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Destination buffer has been freed");
        return NULL;
    }
    
    if (src_ctx == NULL || src_ctx->freed || src_ctx->ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Source buffer has been freed");
        return NULL;
    }

    size_t bytes;
    if (strcmp(dst_name, BUFFER_TYPE_FLOAT32) == 0) {
        bytes = (size_t)width * height * sizeof(float4);
    } else {
        bytes = (size_t)width * height * sizeof(uchar4);
    }

    cudaError_t err = cudaMemcpy(dst_ctx->ptr, src_ctx->ptr, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, "CUDA copy failed: %s", cudaGetErrorString(err));
        return NULL;
    }

    Py_RETURN_NONE;
}

PyObject* py_cuda_sync(PyObject* self, PyObject* args) {
    cudaError_t err = cudaDeviceSynchronize();
    
    if (err != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, "CUDA synchronize failed: %s", 
                    cudaGetErrorString(err));
        return NULL;
    }
    
    Py_RETURN_NONE;
}

}