from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.grid(
    img,
    spacing=50,
    line_width=2,
    color=(1.0, 1.0, 1.0, 1.0),
    bg_color=(0.2, 0.2, 0.2, 1.0)
)
save(img, 'fill_grid.png')