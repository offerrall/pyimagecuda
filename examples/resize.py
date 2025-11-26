from pyimagecuda import Image, load, save, Resize


target_resolution = (1024, 768)

image = load("examples/test.jpg")
cache_result = Image(*target_resolution)


# No dst buffer provided, a new Image will be created and returned
resized_image = Resize.bilinear(image, *target_resolution)
save(resized_image, "./resized_test.jpg")

# Reuse existing dst buffer
Resize.bilinear(image, *target_resolution, dst_buffer=cache_result)
save(cache_result, "./resized_test_cached.jpg")


image.free()
resized_image.free()
cache_result.free()