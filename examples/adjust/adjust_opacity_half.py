from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.opacity(img, 0.5)

save(img, 'adjust_opacity_half.png')