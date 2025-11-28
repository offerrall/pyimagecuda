from pyimagecuda import load, Filter, save

img = load("photo.jpg")

blurred = Filter.gaussian_blur(img, radius=3)

save(blurred, 'output.jpg')