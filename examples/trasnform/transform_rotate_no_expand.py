from pyimagecuda import load, Transform, save

img = load("photo.jpg")

# Canvas stays same size, corners are clipped
rotated = Transform.rotate(img, angle=45, expand=False)

save(rotated, 'output.png')