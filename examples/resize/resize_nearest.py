from pyimagecuda import load, Resize, save

img = load("photo.jpg")

scaled = Resize.nearest(img, width=128, height=128)

save(scaled, 'output.png')