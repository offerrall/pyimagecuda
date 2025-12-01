from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.sepia(img, intensity=1.0)

save(img, 'filter_sepia.png')