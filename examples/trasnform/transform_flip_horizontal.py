from pyimagecuda import load, Transform, save

img = load("photo.jpg")

flipped = Transform.flip(img, direction='horizontal')

save(flipped, 'output.jpg')