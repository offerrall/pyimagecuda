from pyimagecuda import Image, Fill, Effect, Blend, save

with Image(1024, 1024) as bg:
    with Image(512, 512) as card:
        Fill.color(bg, (1, 1, 1, 1))
        Fill.gradient(card, (1, 0, 0, 1), (0, 0, 1, 1), 'radial')
        Effect.rounded_corners(card, 50)

        with Effect.drop_shadow(card, blur=50, color=(0, 0, 0, 1)) as shadowed:
            Blend.normal(bg, shadowed, anchor='center')

        save(bg, 'output.png')