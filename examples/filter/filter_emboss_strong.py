from pyimagecuda import load, Filter, save

img = load("photo.jpg")

embossed = Filter.emboss(img, strength=2.0)

save(embossed, 'filter_emboss_strong.png')