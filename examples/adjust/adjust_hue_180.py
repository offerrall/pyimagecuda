from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

# Shift to opposite colors on the color wheel
Adjust.hue(img, 180)

save(img, 'adjust_hue_180.png')