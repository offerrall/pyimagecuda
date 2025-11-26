#pragma once
#include <Python.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// blend
// blend.cu
PyObject* py_blend_f32(PyObject* self, PyObject* args);

// buffer
// buffer.cu
PyObject* py_copy_buffer(PyObject* self, PyObject* args);
PyObject* py_create_buffer_f32(PyObject* self, PyObject* args);
PyObject* py_create_buffer_u8(PyObject* self, PyObject* args);
PyObject* py_cuda_sync(PyObject* self, PyObject* args);
PyObject* py_download_from_buffer(PyObject* self, PyObject* args);
PyObject* py_free_buffer(PyObject* self, PyObject* args);
PyObject* py_upload_to_buffer(PyObject* self, PyObject* args);

// convert.cu
PyObject* py_convert_f32_to_u8(PyObject* self, PyObject* args);
PyObject* py_convert_u8_to_f32(PyObject* self, PyObject* args);

// efffects
// drop_shadow.cu
PyObject* py_colorize_shadow_f32(PyObject* self, PyObject* args);
PyObject* py_extract_alpha_f32(PyObject* self, PyObject* args);

// rounded_corners.cu
PyObject* py_rounded_corners_f32(PyObject* self, PyObject* args);

// fill
// color.cu
PyObject* py_fill_color_f32(PyObject* self, PyObject* args);

// gradient.cu
PyObject* py_fill_gradient_f32(PyObject* self, PyObject* args);

// filters
// gaussian_blur_separable.cu
PyObject* py_gaussian_blur_separable_f32(PyObject* self, PyObject* args);

// sharpen.cu
PyObject* py_sharpen_f32(PyObject* self, PyObject* args);

// resize
// resize.cu
PyObject* py_resize_f32(PyObject* self, PyObject* args);


#ifdef __cplusplus
}
#endif
