from pyimagecuda import load, Adjust, save

img = load("portrait.jpg")

Adjust.vibrance(img, 0.5)

save(img, 'adjust_vibrance_portrait.png')