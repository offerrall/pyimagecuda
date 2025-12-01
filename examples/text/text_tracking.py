from pyimagecuda import Text, save

text_img = Text.create(
    "CINEMATIC",
    font="Arial Bold",
    size=40,
    letter_spacing=10.0,
    color=(1.0, 1.0, 1.0, 1.0),
    bg_color=(0.0, 0.0, 0.0, 1.0)
)

save(text_img, 'text_tracking.png')