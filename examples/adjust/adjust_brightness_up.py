from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.brightness(img, 0.2)

save(img, 'output.jpg')