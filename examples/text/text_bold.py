from pyimagecuda import Text, save

text_img = Text.create(
    "Bold Title",
    font="Arial Bold",
    size=48,
    color=(0.2, 0.2, 0.2, 1.0)
)

save(text_img, 'text_bold.png')