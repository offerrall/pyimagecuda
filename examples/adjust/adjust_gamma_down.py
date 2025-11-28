from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.gamma(img, 0.6)

save(img, 'output.jpg')