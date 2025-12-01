from pyimagecuda import Text, save

text_img = Text.create("Hello World", size=60)

save(text_img, 'text_basic.png')