from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.color(img, (1.0, 0.0, 0.0, 1.0))
save(img, 'output.png')