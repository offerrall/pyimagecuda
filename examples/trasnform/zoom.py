from pyimagecuda import load, Transform, save

img = load("photo.jpg")

# Zoom 5× into center (creates new buffer)
zoomed = Transform.zoom(img, zoom_factor=5.0)

save(zoomed, 'zoomed_5x.jpg')