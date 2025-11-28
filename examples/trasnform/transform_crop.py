from pyimagecuda import load, Transform, save

img = load("photo.jpg")  # 1920×1080

# Crop 800×600 region starting at (200, 100)
cropped = Transform.crop(
    img,
    x=200,
    y=100,
    width=800,
    height=600
)

save(cropped, 'output.jpg')