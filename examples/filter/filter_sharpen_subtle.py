from pyimagecuda import load, Filter, save

img = load("photo.jpg")

sharpened = Filter.sharpen(img, strength=0.5)

save(sharpened, 'output.jpg')