from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.gradient(
    img,
    rgba1=(1.0, 0.0, 0.0, 1.0),  # Red
    rgba2=(0.0, 0.0, 1.0, 1.0),  # Blue
    direction='horizontal'
)
save(img, 'output.png')