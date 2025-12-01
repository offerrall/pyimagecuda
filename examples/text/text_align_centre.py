from pyimagecuda import Text, save

text_img = Text.create(
    "Left Line\nCenter Line\nRight Line",
    size=30,
    align='centre',
    bg_color=(0.9, 0.9, 0.9, 1.0)
)

save(text_img, 'text_align_centre.png')