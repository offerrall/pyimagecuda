from pyimagecuda import load, Filter, save

img = load("photo.jpg")

sharpened = Filter.sharpen(img, strength=2.0)

save(sharpened, 'output.jpg')