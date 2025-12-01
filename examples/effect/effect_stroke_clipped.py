from pyimagecuda import Image, Fill, Effect, save

img = Image(512, 512)
Fill.color(img, (1, 0.5, 0, 1))
Effect.rounded_corners(img, 100)

stroked = Effect.stroke(
    img,
    width=20,
    color=(0, 0, 0, 1),
    expand=False
)

save(stroked, 'effect_stroke_clipped.png')