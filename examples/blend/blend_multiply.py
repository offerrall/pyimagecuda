from pyimagecuda import Image, Fill, Blend, save

base = Image(800, 600)
overlay = Image(400, 300)

c1 = (0.2, 0.2, 0.3, 1.0)
c2 = (0.4, 0.3, 0.5, 1.0)
c3 = (1.0, 0.5, 0.0, 0.8)
c4 = (1.0, 0.0, 0.5, 0.8)

Fill.gradient(base, c1, c2, 'radial')
Fill.gradient(overlay, c3, c4, 'horizontal')

Blend.multiply(base, overlay, anchor='center')
save(base, 'output.png')