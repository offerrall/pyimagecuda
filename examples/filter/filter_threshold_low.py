from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.threshold(img, value=0.3)

save(img, 'filter_threshold_low.png')