#include <Python.h>
#include "include.h"

static PyMethodDef all_methods[] = {
    // adjust
    // brightness.cu
    {"adjust_brightness_f32", py_adjust_brightness_f32, METH_VARARGS, "Adjust Brightness F32"},

    // contrast.cu
    {"adjust_contrast_f32", py_adjust_contrast_f32, METH_VARARGS, "Adjust Contrast F32"},

    // gamma.cu
    {"adjust_gamma_f32", py_adjust_gamma_f32, METH_VARARGS, "Adjust Gamma F32"},

    // saturation.cu
    {"adjust_saturation_f32", py_adjust_saturation_f32, METH_VARARGS, "Adjust Saturation F32"},


    // blend
    // blend.cu
    {"blend_f32", py_blend_f32, METH_VARARGS, "Blend F32"},

    // mask.cu
    {"blend_mask_f32", py_blend_mask_f32, METH_VARARGS, "Blend Mask F32"},


    // buffer
    // buffer.cu
    {"copy_buffer", py_copy_buffer, METH_VARARGS, "Copy Buffer"},
    {"create_buffer_f32", py_create_buffer_f32, METH_VARARGS, "Create Buffer F32"},
    {"create_buffer_u8", py_create_buffer_u8, METH_VARARGS, "Create Buffer U8"},
    {"cuda_sync", py_cuda_sync, METH_VARARGS, "Cuda Sync"},
    {"download_from_buffer", py_download_from_buffer, METH_VARARGS, "Download From Buffer"},
    {"free_buffer", py_free_buffer, METH_VARARGS, "Free Buffer"},
    {"upload_to_buffer", py_upload_to_buffer, METH_VARARGS, "Upload To Buffer"},

    // convert.cu
    {"convert_f32_to_u8", py_convert_f32_to_u8, METH_VARARGS, "Convert F32 To U8"},
    {"convert_u8_to_f32", py_convert_u8_to_f32, METH_VARARGS, "Convert U8 To F32"},


    // efffects
    // drop_shadow.cu
    {"colorize_shadow_f32", py_colorize_shadow_f32, METH_VARARGS, "Colorize Shadow F32"},
    {"extract_alpha_f32", py_extract_alpha_f32, METH_VARARGS, "Extract Alpha F32"},

    // rounded_corners.cu
    {"rounded_corners_f32", py_rounded_corners_f32, METH_VARARGS, "Rounded Corners F32"},


    // fill
    // color.cu
    {"fill_color_f32", py_fill_color_f32, METH_VARARGS, "Fill Color F32"},

    // gradient.cu
    {"fill_gradient_f32", py_fill_gradient_f32, METH_VARARGS, "Fill Gradient F32"},


    // filters
    // gaussian_blur_separable.cu
    {"gaussian_blur_separable_f32", py_gaussian_blur_separable_f32, METH_VARARGS, "Gaussian Blur Separable F32"},

    // sharpen.cu
    {"sharpen_f32", py_sharpen_f32, METH_VARARGS, "Sharpen F32"},


    // resize
    // resize.cu
    {"resize_f32", py_resize_f32, METH_VARARGS, "Resize F32"},


    // transform
    // crop.cu
    {"crop_f32", py_crop_f32, METH_VARARGS, "Crop F32"},

    // flip.cu
    {"flip_f32", py_flip_f32, METH_VARARGS, "Flip F32"},

    // rotate.cu
    {"rotate_arbitrary_f32", py_rotate_arbitrary_f32, METH_VARARGS, "Rotate Arbitrary F32"},
    {"rotate_fixed_f32", py_rotate_fixed_f32, METH_VARARGS, "Rotate Fixed F32"},


    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "pyimagecuda_internal",
    "CUDA image processing internal module",
    -1,
    all_methods
};

PyMODINIT_FUNC PyInit_pyimagecuda_internal(void) {
    return PyModule_Create(&module);
}
