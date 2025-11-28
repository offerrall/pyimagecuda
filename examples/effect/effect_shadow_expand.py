from pyimagecuda import Image, Fill, Effect, save

img = Image(400, 300)
Fill.gradient(img, (1, 0, 0, 1), (0, 0, 1, 1), 'radial')
Effect.rounded_corners(img, 30)

shadowed = Effect.drop_shadow(
    img,
    offset_x=10,
    offset_y=10,
    blur=30,
    color=(0, 0, 0, 0.8)
)

save(shadowed, 'output.png')