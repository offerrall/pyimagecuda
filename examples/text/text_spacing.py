from pyimagecuda import Text, save

text_img = Text.create(
    "Line 1\nLine 2\nLine 3",
    size=30,
    spacing=50,
    bg_color=(0.8, 0.8, 0.8, 1.0)
)

save(text_img, 'text_spacing.png')