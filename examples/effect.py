from pyimagecuda import Image, save, Fill, Blend, Effect


# Create large transparent canvas
canvas = Image(1920, 1080)
Fill.color(canvas, rgba=(0.0, 0.0, 0.0, 0.0))

# Create smaller square with gradient
square_size = 600
square = Image(square_size, square_size)
Fill.gradient(square,
              rgba1=(0.2, 0.5, 0.9, 1.0),  # Blue
              rgba2=(0.9, 0.3, 0.6, 1.0),  # Pink
              direction='diagonal')

# Round the corners
Effect.rounded_corners(square, radius=60.0)

# Blend square onto canvas FIRST (centered)
pos_x = (canvas.width - square_size) // 2
pos_y = (canvas.height - square_size) // 2
Blend.normal(canvas, square, pos_x=pos_x, pos_y=pos_y)

# Apply drop shadow with auto-allocated buffers
canvas_with_shadow = Effect.drop_shadow(canvas,
                                        offset_x=20,
                                        offset_y=20,
                                        blur=40,
                                        color=(0.0, 0.0, 0.0, 0.7))

save(canvas_with_shadow, "./effect_showcase.png")


# --- VERSION WITH CACHED BUFFERS FOR PERFORMANCE ---

# Pre-allocate reusable buffers
dst_buffer = Image(canvas.width, canvas.height)
shadow_buffer = Image(canvas.width, canvas.height)
temp_buffer = Image(canvas.width, canvas.height)

# Reset canvas for second version
Fill.color(canvas, rgba=(0.0, 0.0, 0.0, 0.0))
Blend.normal(canvas, square, pos_x=pos_x, pos_y=pos_y)

# Apply drop shadow with cached buffers (no allocation)
Effect.drop_shadow(canvas,
                   offset_x=20,
                   offset_y=20,
                   blur=40,
                   color=(0.0, 0.0, 0.0, 0.7),
                   dst_buffer=dst_buffer,
                   shadow_buffer=shadow_buffer,
                   temp_buffer=temp_buffer)

save(dst_buffer, "./effect_showcase_cached.png")


# Cleanup
canvas.free()
square.free()
canvas_with_shadow.free()
dst_buffer.free()
shadow_buffer.free()
temp_buffer.free()