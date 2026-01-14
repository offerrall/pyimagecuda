from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.hue(img, -30)

save(img, 'adjust_hue_cool.png')