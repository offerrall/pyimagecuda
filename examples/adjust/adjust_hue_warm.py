from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.hue(img, 15)

save(img, 'adjust_hue_warm.png')