from pyimagecuda import load, Resize, save

img = load("photo.jpg")

# Maximum quality upscale
resized = Resize.lanczos(img, width=3840)

save(resized, 'output.jpg')