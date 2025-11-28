from pyimagecuda import load, Resize, save

img = load("photo.jpg")

# High-quality upscale
resized = Resize.bicubic(img, width=1920)

save(resized, 'output.jpg')