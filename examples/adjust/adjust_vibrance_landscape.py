from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.vibrance(img, 0.8)

save(img, 'adjust_vibrance_landscape.png')