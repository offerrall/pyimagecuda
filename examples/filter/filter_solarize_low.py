from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.solarize(img, threshold=0.3)

save(img, 'filter_solarize_low.png')