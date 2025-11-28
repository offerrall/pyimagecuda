from pyimagecuda import Image, Fill, Blend, save

base = Image(800, 600)
overlay = Image(200, 200)

Fill.color(base, (0.2, 0.2, 0.3, 1.0))
Fill.color(overlay, (1.0, 0.0, 0.0, 1.0))

# Center with offset
Blend.normal(base, overlay, anchor='center', offset_x=50, offset_y=-30)
save(base, 'output.png')