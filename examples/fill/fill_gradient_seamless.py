from pyimagecuda import Image, Fill, save

texture = Image(512, 512)
Fill.gradient(
    texture,
    rgba1=(0.8, 0.6, 0.4, 1.0),
    rgba2=(0.4, 0.3, 0.2, 1.0),
    direction='diagonal',
    seamless=True
)
save(texture, 'output.png')