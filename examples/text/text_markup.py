from pyimagecuda import Text, save

text_img = Text.create(
    'Normal <b>Bold</b> <i>Italic</i>\n'
    '<span foreground="orange">Orange</span> '
    'and <sub>subscript</sub>',
    size=40,
    align='centre'
)

save(text_img, 'text_markup.png')