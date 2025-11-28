from pyimagecuda import Image, Fill, Blend, save

base = Image(800, 600)
overlay = Image(200, 200)

Fill.color(base, (0.2, 0.2, 0.3, 1.0))
Fill.color(overlay, (1.0, 0.0, 0.0, 1.0))

Blend.normal(base, overlay, anchor='center')
save(base, 'output.png')