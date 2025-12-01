from pyimagecuda import Image, Fill, Effect, Blend, Transform, save

with Image(1024, 1024) as bg:
    Fill.color(bg, (0, 1, 0.8, 1))
    with Image(512, 512) as card:
        Fill.gradient(card, (1, 0, 0, 1), (0, 0, 1, 1), 'radial')
        Effect.rounded_corners(card, 50)

        with Effect.stroke(card, 10, (1, 1, 1, 1)) as stroked:
            with Effect.drop_shadow(stroked, blur=50, color=(0, 0, 0, 1)) as shadowed:
                with Transform.rotate(shadowed, 45) as rotated:
                    Blend.normal(bg, rotated, anchor='center')

    save(bg, 'output.png')