from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.noise(
    img,
    seed=42.0,
    monochrome=True
)
save(img, 'fill_noise_mono.png')