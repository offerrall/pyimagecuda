from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.opacity(img, 0.8)

save(img, 'adjust_opacity_subtle.png')