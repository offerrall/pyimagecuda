from pyimagecuda import Image, load, save, Filter


# Load image
image = load("examples/test.jpg")

# Gaussian blur - auto-allocated buffers
blurred = Filter.gaussian_blur(image, radius=10)
save(blurred, "./blurred_auto.jpg")

# Gaussian blur - with custom sigma
blurred_custom = Filter.gaussian_blur(image, radius=15, sigma=5.0)
save(blurred_custom, "./blurred_custom_sigma.jpg")

# Sharpen filter
sharpened = Filter.sharpen(image, strength=1.5)
save(sharpened, "./sharpened.jpg")

# Reuse buffers for performance
dst_buffer = Image(image.width, image.height)
temp_buffer = Image(image.width, image.height)

Filter.gaussian_blur(image, radius=20, dst_buffer=dst_buffer, temp_buffer=temp_buffer)
save(dst_buffer, "./blurred_reused.jpg")

# Chain filters using buffer reuse
Filter.gaussian_blur(image, radius=5, dst_buffer=dst_buffer, temp_buffer=temp_buffer)
Filter.sharpen(dst_buffer, strength=2.0, dst_buffer=dst_buffer)
save(dst_buffer, "./blur_then_sharpen.jpg")


image.free()
blurred.free()
blurred_custom.free()
sharpened.free()
dst_buffer.free()
temp_buffer.free()