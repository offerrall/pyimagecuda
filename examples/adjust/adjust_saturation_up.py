from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.saturation(img, 1.8)

save(img, 'output.jpg')