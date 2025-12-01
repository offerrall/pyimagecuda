from pyimagecuda import load, Filter, save

img = load("photo.jpg")

edges = Filter.sobel(img)

save(edges, 'filter_sobel.png')