from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.sepia(img, intensity=0.5)

save(img, 'filter_sepia_subtle.png')