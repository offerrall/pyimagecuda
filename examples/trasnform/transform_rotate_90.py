from pyimagecuda import load, Transform, save

img = load("photo.jpg")

# 90°, 180°, 270° use fast fixed rotation
rotated = Transform.rotate(img, angle=90)

save(rotated, 'output.jpg')