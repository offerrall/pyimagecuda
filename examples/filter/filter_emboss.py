from pyimagecuda import load, Filter, save

img = load("photo.jpg")

embossed = Filter.emboss(img, strength=1.0)

save(embossed, 'filter_emboss.png')