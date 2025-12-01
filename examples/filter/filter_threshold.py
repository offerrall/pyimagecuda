from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.threshold(img, value=0.5)

save(img, 'filter_threshold.png')