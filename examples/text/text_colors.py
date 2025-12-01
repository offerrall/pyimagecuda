from pyimagecuda import Text, save

text_img = Text.create(
    "ALERT\nSYSTEM",
    font="Arial Bold",
    size=50,
    color=(1.0, 0.2, 0.2, 1.0),
    bg_color=(0.0, 0.0, 0.3, 1.0),
    align='centre'
)

save(text_img, 'text_colors.png')