from pyimagecuda import Image, Fill, Blend, save


BASE_W, BASE_H = 800, 600
MASK_HOLE_W, MASK_HOLE_H = 200, 100

c1 = (0.2, 0.2, 0.3, 1.0)
c2 = (0.4, 0.3, 0.5, 1.0)
white = (1.0, 1.0, 1.0, 1.0)
black = (0.0, 0.0, 0.0, 1.0)

with Image(BASE_W, BASE_H) as base:
    Fill.gradient(base, c1, c2, 'radial')

    with Image(BASE_W, BASE_H) as mask:
        Fill.color(mask, white) 

        with Image(400, 300) as hole_shape:
            Fill.color(hole_shape, black)
            Blend.normal(mask,
                         hole_shape,
                         anchor='center')

        Blend.mask(base, mask)

    save(base, "output.png")