from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.gamma(img, 1.5)

save(img, 'output.jpg')