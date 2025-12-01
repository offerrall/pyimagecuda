from pyimagecuda import Image, Fill, Effect, save

img = Image(512, 512)
c1 = (1, 0, 0, 1)
c2 = (0, 0, 1, 1)

Fill.gradient(img, c1, c2, 'radial')
Effect.rounded_corners(img, 80)

stroked = Effect.stroke(
    img,
    width=15,
    color=(0, 0, 0, 1)
)

save(stroked, 'effect_stroke_outside.png')