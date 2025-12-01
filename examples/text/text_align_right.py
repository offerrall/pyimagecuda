from pyimagecuda import Text, save

text_img = Text.create(
    "Line One\nLine Two\nLine Three",
    size=30,
    align='right',
    bg_color=(0.9, 0.9, 0.9, 1.0)
)

save(text_img, 'text_align_right.png')