#include <Python.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include "common.h"

extern "C" {


__global__ void detect_edges_kernel(
    const float4* __restrict__ src,
    int2* __restrict__ seed_points,
    const uint32_t src_width,
    const uint32_t src_height,
    const uint32_t dst_width,
    const uint32_t dst_height,
    const int offset_x,
    const int offset_y
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height) return;
    
    seed_points[y * dst_width + x] = make_int2(-1, -1);

    const int sx = (int)x - offset_x;
    const int sy = (int)y - offset_y;
    
    float center_alpha = 0.0f;
    if (sx >= 0 && sx < src_width && sy >= 0 && sy < src_height) {
        center_alpha = src[sy * src_width + sx].w;
    }

    if (center_alpha <= 0.0f) return;

    bool is_edge = false;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = sx + dx;
            int ny = sy + dy;
            
            float neighbor_alpha = 0.0f;
            if (nx >= 0 && nx < src_width && ny >= 0 && ny < src_height) {
                neighbor_alpha = src[ny * src_width + nx].w;
            }
            
            if (neighbor_alpha <= 0.0f) {
                is_edge = true;
                break;
            }
        }
        if (is_edge) break;
    }
    
    if (is_edge) {
        seed_points[y * dst_width + x] = make_int2(x, y);
    }
}

__global__ void jump_flood_pass_kernel(
    const int2* __restrict__ seed_in,
    int2* __restrict__ seed_out,
    const uint32_t width,
    const uint32_t height,
    const int step_size
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint32_t idx = y * width + x;
    int2 best_seed = seed_in[idx];
    
    float best_dist = (best_seed.x >= 0) ? 
        hypotf((float)x - best_seed.x, (float)y - best_seed.y) : 1e10f;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            const int nx = (int)x + dx * step_size;
            const int ny = (int)y + dy * step_size;
            
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;
            
            const int2 candidate = seed_in[ny * width + nx];
            if (candidate.x < 0) continue;
            
            const float dist = hypotf((float)x - candidate.x, 
                                     (float)y - candidate.y);
            
            if (dist < best_dist) {
                best_dist = dist;
                best_seed = candidate;
            }
        }
    }
    
    seed_out[idx] = best_seed;
}

__global__ void generate_stroke_composite_kernel(
    const float4* __restrict__ src_image,
    const int2* __restrict__ nearest_edge,
    float4* __restrict__ output,
    const uint32_t src_width,
    const uint32_t src_height,
    const uint32_t dst_width,
    const uint32_t dst_height,
    const int offset_x,
    const int offset_y,
    const float stroke_width,
    const float r, const float g, const float b, const float a,
    const int position_mode
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height) return;
    
    const int sx = (int)x - offset_x;
    const int sy = (int)y - offset_y;
    
    float4 src_px = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (sx >= 0 && sx < src_width && sy >= 0 && sy < src_height) {
        src_px = src_image[sy * src_width + sx];
    }
    
    const int2 edge = nearest_edge[y * dst_width + x];
    float stroke_factor = 0.0f;

    if (edge.x >= 0) {
        const float dist = hypotf((float)x - edge.x, (float)y - edge.y);
        
        if (position_mode == 0) {
            if (dist <= stroke_width) {
                stroke_factor = 1.0f;
                float edge_dist = stroke_width - dist;
                if (edge_dist < 1.0f) stroke_factor = edge_dist;
            }
        } else {
            if (dist <= stroke_width) {
                stroke_factor = 1.0f;
                float edge_dist = stroke_width - dist;
                if (edge_dist < 1.0f) stroke_factor = edge_dist;
            }
        }
    }
    
    float4 final_premul;
    
    if (position_mode == 0) { 
        float3 src_premul = make_float3(src_px.x * src_px.w, src_px.y * src_px.w, src_px.z * src_px.w);
        float stroke_alpha_final = a * stroke_factor;
        float3 stroke_premul = make_float3(r * stroke_alpha_final, g * stroke_alpha_final, b * stroke_alpha_final);
        
        float inv_src_a = 1.0f - src_px.w;
        final_premul.x = src_premul.x + stroke_premul.x * inv_src_a;
        final_premul.y = src_premul.y + stroke_premul.y * inv_src_a;
        final_premul.z = src_premul.z + stroke_premul.z * inv_src_a;
        final_premul.w = src_px.w + stroke_alpha_final * inv_src_a;
        
    } else {
        float3 src_rgb = make_float3(src_px.x, src_px.y, src_px.z);
        float3 stroke_rgb = make_float3(r, g, b);

        float mix_val = a * stroke_factor; 
        
        float3 mixed_rgb;
        mixed_rgb.x = stroke_rgb.x * mix_val + src_rgb.x * (1.0f - mix_val);
        mixed_rgb.y = stroke_rgb.y * mix_val + src_rgb.y * (1.0f - mix_val);
        mixed_rgb.z = stroke_rgb.z * mix_val + src_rgb.z * (1.0f - mix_val);

        final_premul.x = mixed_rgb.x * src_px.w;
        final_premul.y = mixed_rgb.y * src_px.w;
        final_premul.z = mixed_rgb.z * src_px.w;
        final_premul.w = src_px.w;
    }
    
    float4 result;
    if (final_premul.w > 0.0001f) {
        result.x = final_premul.x / final_premul.w;
        result.y = final_premul.y / final_premul.w;
        result.z = final_premul.z / final_premul.w;
        result.w = final_premul.w;
    } else {
        result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    output[y * dst_width + x] = result;
}

PyObject* py_compute_distance_field_f32(PyObject* self, PyObject* args) {
    PyObject* src_capsule; PyObject* dst_capsule;
    uint32_t src_width, src_height, dst_width, dst_height;
    int offset_x, offset_y;
    
    if (!PyArg_ParseTuple(args, "OOIIIIii", &src_capsule, &dst_capsule,
                          &src_width, &src_height, &dst_width, &dst_height, &offset_x, &offset_y)) return NULL;

    if (validate_f32_buffer(src_capsule, "Source") < 0 || validate_f32_buffer(dst_capsule, "Destination") < 0) return NULL;

    BufferContext* src_ctx = get_buffer_context(src_capsule);
    BufferContext* dst_ctx = get_buffer_context(dst_capsule);
    if (!src_ctx || !dst_ctx) return NULL;

    int2* d_seeds_a = NULL; int2* d_seeds_b = NULL;
    cudaMalloc(&d_seeds_a, dst_width * dst_height * sizeof(int2));
    cudaMalloc(&d_seeds_b, dst_width * dst_height * sizeof(int2));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y);
    
    detect_edges_kernel<<<gridSize, blockSize>>>((const float4*)src_ctx->ptr, d_seeds_a, src_width, src_height, dst_width, dst_height, offset_x, offset_y);
    
    int max_steps = 0; int temp = (dst_width > dst_height) ? dst_width : dst_height;
    while (temp > 1) { temp >>= 1; max_steps++; }
    
    for (int i = 0; i < max_steps; i++) {
        int step = 1 << (max_steps - i - 1);
        jump_flood_pass_kernel<<<gridSize, blockSize>>>(d_seeds_a, d_seeds_b, dst_width, dst_height, step);
        int2* tmp = d_seeds_a; d_seeds_a = d_seeds_b; d_seeds_b = tmp;
    }
    jump_flood_pass_kernel<<<gridSize, blockSize>>>(d_seeds_a, d_seeds_b, dst_width, dst_height, 1);
    
    cudaMemcpy(dst_ctx->ptr, d_seeds_b, dst_width * dst_height * sizeof(int2), cudaMemcpyDeviceToDevice);
    cudaFree(d_seeds_a); cudaFree(d_seeds_b);
    
    Py_RETURN_NONE;
}

PyObject* py_generate_stroke_composite_f32(PyObject* self, PyObject* args) {
    PyObject *src_capsule, *distance_capsule, *output_capsule, *color_obj;
    uint32_t src_width, src_height, dst_width, dst_height;
    int offset_x, offset_y, position_mode;
    float stroke_width;
    
    if (!PyArg_ParseTuple(args, "OOOIIIIiifOi", &src_capsule, &distance_capsule, &output_capsule,
                          &src_width, &src_height, &dst_width, &dst_height, &offset_x, &offset_y,
                          &stroke_width, &color_obj, &position_mode)) return NULL;

    if (validate_f32_buffer(src_capsule, "Source") < 0 || validate_f32_buffer(distance_capsule, "Distance") < 0 || validate_f32_buffer(output_capsule, "Output") < 0) return NULL;

    BufferContext* src_ctx = get_buffer_context(src_capsule);
    BufferContext* dst_ctx = get_buffer_context(distance_capsule);
    BufferContext* out_ctx = get_buffer_context(output_capsule);

    float rgba[4];
    if (parse_rgba_color(color_obj, rgba) < 0) return NULL;

    dim3 blockSize(16, 16);
    dim3 gridSize((dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y);
    
    generate_stroke_composite_kernel<<<gridSize, blockSize>>>(
        (const float4*)src_ctx->ptr, (const int2*)dst_ctx->ptr, (float4*)out_ctx->ptr,
        src_width, src_height, dst_width, dst_height, offset_x, offset_y,
        stroke_width, rgba[0], rgba[1], rgba[2], rgba[3], position_mode
    );
    
    if (check_cuda_launch() < 0) return NULL;
    Py_RETURN_NONE;
}

}