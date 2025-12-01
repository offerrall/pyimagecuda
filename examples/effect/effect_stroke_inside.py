from pyimagecuda import Image, Fill, Effect, save

img = Image(512, 512)
c1 = (0.2, 0.8, 0.3, 1)
c2 = (0.1, 0.3, 0.8, 1)
Fill.gradient(img, c1, c2, 'horizontal')
Effect.rounded_corners(img, 80)

stroked = Effect.stroke(
    img,
    width=30,
    color=(1, 1, 1, 1),
    position='inside'
)

save(stroked, 'effect_stroke_inside.png')