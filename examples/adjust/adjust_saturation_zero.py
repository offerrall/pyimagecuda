from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.saturation(img, 0.0)

save(img, 'output.jpg')