from pyimagecuda import Image, Fill, Effect, save

img = Image(512, 512)
c1 = (1, 0.8, 0.6, 1)
c2 = (0.2, 0.4, 0.8, 1)
Fill.gradient(img, c1, c2, 'horizontal')

Effect.vignette(
    img,
    radius=0.9,
    softness=1.0,
    color=(0, 0, 0, 0.9)
)

save(img, 'output.png')