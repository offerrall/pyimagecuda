from pyimagecuda import load, Resize, save

img = load("photo.jpg")

# Fast general-purpose resize
resized = Resize.bilinear(img, width=800)

save(resized, 'output.jpg')