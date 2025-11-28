from pyimagecuda import load, Transform, save

img = load("photo.jpg")

# Calculate center crop coordinates
crop_w, crop_h = 512, 512
x = (img.width - crop_w) // 2
y = (img.height - crop_h) // 2

cropped = Transform.crop(img, x, y, crop_w, crop_h)

save(cropped, 'output.jpg')