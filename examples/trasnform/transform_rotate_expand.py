from pyimagecuda import load, Transform, save

img = load("photo.jpg")

# Canvas expands to fit rotated image
rotated = Transform.rotate(img, angle=45, expand=True)

save(rotated, 'output.png')