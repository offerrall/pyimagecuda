#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "common.h"

typedef unsigned int GLuint;
typedef unsigned int GLenum;

struct cudaGraphicsResource;

extern "C" cudaError_t CUDARTAPI cudaGraphicsGLRegisterBuffer(
    struct cudaGraphicsResource **resource,
    GLuint buffer,
    unsigned int flags
);

extern "C" cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(
    struct cudaGraphicsResource *resource
);

extern "C" cudaError_t CUDARTAPI cudaGraphicsMapResources(
    int count,
    struct cudaGraphicsResource **resources,
    cudaStream_t stream
);

extern "C" cudaError_t CUDARTAPI cudaGraphicsUnmapResources(
    int count,
    struct cudaGraphicsResource **resources,
    cudaStream_t stream
);

extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(
    void **devPtr,
    size_t *size,
    struct cudaGraphicsResource *resource
);

#define cudaGraphicsRegisterFlagsWriteDiscard 2

extern "C" {

typedef struct {
    cudaGraphicsResource* resource;
    unsigned int pbo_id;
    int freed;
} GLResourceContext;

static void gl_resource_destructor(PyObject* capsule) {
    const char* name = PyCapsule_GetName(capsule);
    if (name == NULL) return;
    
    GLResourceContext* ctx = (GLResourceContext*)PyCapsule_GetPointer(capsule, name);
    if (ctx == NULL) return;

    if (!ctx->freed && ctx->resource != NULL) {
        cudaError_t err = cudaGraphicsUnregisterResource(ctx->resource);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsUnregisterResource failed: %s\n",
                    cudaGetErrorString(err));
        }
    }

    free(ctx);
}

PyObject* py_register_gl_pbo(PyObject* self, PyObject* args) {
    unsigned int pbo_id;
    
    if (!PyArg_ParseTuple(args, "I", &pbo_id)) {
        return NULL;
    }
    
    cudaGraphicsResource* resource;
    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &resource,
        pbo_id,
        cudaGraphicsRegisterFlagsWriteDiscard
    );
    
    if (err != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError,
                    "cudaGraphicsGLRegisterBuffer failed: %s",
                    cudaGetErrorString(err));
        return NULL;
    }
    
    GLResourceContext* ctx = (GLResourceContext*)malloc(sizeof(GLResourceContext));
    if (ctx == NULL) {
        cudaGraphicsUnregisterResource(resource);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate GL resource context");
        return NULL;
    }
    
    ctx->resource = resource;
    ctx->pbo_id = pbo_id;
    ctx->freed = 0;
    
    PyObject* capsule = PyCapsule_New(ctx, "cuda_gl_resource", gl_resource_destructor);
    
    if (capsule == NULL) {
        cudaGraphicsUnregisterResource(resource);
        free(ctx);
        return NULL;
    }
    
    return capsule;
}

PyObject* py_unregister_gl_resource(PyObject* self, PyObject* args) {
    PyObject* capsule;
    
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    
    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "Expected a GL resource capsule");
        return NULL;
    }
    
    GLResourceContext* ctx = (GLResourceContext*)PyCapsule_GetPointer(
        capsule, "cuda_gl_resource"
    );
    
    if (ctx == NULL || ctx->freed) {
        Py_RETURN_NONE;
    }
    
    if (ctx->resource != NULL) {
        cudaError_t err = cudaGraphicsUnregisterResource(ctx->resource);
        if (err != cudaSuccess) {
            PyErr_Format(PyExc_RuntimeError,
                        "cudaGraphicsUnregisterResource failed: %s",
                        cudaGetErrorString(err));
            return NULL;
        }
    }
    
    ctx->freed = 1;
    
    Py_RETURN_NONE;
}

PyObject* py_copy_to_gl_pbo(PyObject* self, PyObject* args) {
    PyObject* buffer_capsule;
    PyObject* resource_capsule;
    uint32_t width, height;
    
    if (!PyArg_ParseTuple(args, "OOII", 
                         &buffer_capsule, &resource_capsule, 
                         &width, &height)) {
        return NULL;
    }
    
    if (validate_dimensions(width, height) < 0) {
        return NULL;
    }

    if (!PyCapsule_CheckExact(buffer_capsule)) {
        PyErr_SetString(PyExc_TypeError, "Expected a Buffer capsule");
        return NULL;
    }
    
    const char* name = PyCapsule_GetName(buffer_capsule);
    if (name == NULL || strcmp(name, BUFFER_TYPE_UINT8) != 0) {
        PyErr_SetString(PyExc_TypeError, "Expected U8 Buffer (not F32)");
        return NULL;
    }
    
    BufferContext* buf_ctx = (BufferContext*)PyCapsule_GetPointer(buffer_capsule, name);
    if (buf_ctx == NULL || buf_ctx->freed || buf_ctx->ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Buffer has been freed");
        return NULL;
    }
    
    if (!PyCapsule_CheckExact(resource_capsule)) {
        PyErr_SetString(PyExc_TypeError, "Expected a GL resource capsule");
        return NULL;
    }
    
    GLResourceContext* gl_ctx = (GLResourceContext*)PyCapsule_GetPointer(
        resource_capsule, "cuda_gl_resource"
    );
    
    if (gl_ctx == NULL || gl_ctx->freed || gl_ctx->resource == NULL) {
        PyErr_SetString(PyExc_ValueError, "GL resource has been freed");
        return NULL;
    }

    cudaError_t err = cudaGraphicsMapResources(1, &gl_ctx->resource, 0);
    if (err != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError,
                    "cudaGraphicsMapResources failed: %s",
                    cudaGetErrorString(err));
        return NULL;
    }
    
    void* pbo_ptr;
    size_t pbo_size;
    err = cudaGraphicsResourceGetMappedPointer(&pbo_ptr, &pbo_size, gl_ctx->resource);
    if (err != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &gl_ctx->resource, 0);
        PyErr_Format(PyExc_RuntimeError,
                    "cudaGraphicsResourceGetMappedPointer failed: %s",
                    cudaGetErrorString(err));
        return NULL;
    }

    size_t copy_size = (size_t)width * height * sizeof(uchar4);
    
    if (copy_size > pbo_size) {
        cudaGraphicsUnmapResources(1, &gl_ctx->resource, 0);
        PyErr_Format(PyExc_ValueError,
                    "Buffer size mismatch: need %zu bytes, PBO has %zu bytes",
                    copy_size, pbo_size);
        return NULL;
    }
    
    err = cudaMemcpy(pbo_ptr, buf_ctx->ptr, copy_size, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &gl_ctx->resource, 0);
    
    if (err != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError,
                    "cudaMemcpy (GPU->GPU) failed: %s",
                    cudaGetErrorString(err));
        return NULL;
    }
    
    Py_RETURN_NONE;
}

}