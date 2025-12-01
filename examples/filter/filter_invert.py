from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.invert(img)

save(img, 'filter_invert.png')