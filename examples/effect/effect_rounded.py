from pyimagecuda import Image, Fill, Effect, save

img = Image(512, 512)
Fill.gradient(img, (1, 0, 0, 1), (0, 0, 1, 1), 'radial')

Effect.rounded_corners(img, radius=50)

save(img, 'output.png')