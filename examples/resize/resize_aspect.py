from pyimagecuda import load, Resize, save

img = load("photo.jpg")  # 1920×1080

# Scale to width 800
resized = Resize.lanczos(img, width=800)
print(resized.width, resized.height)
# Result: 800×450

save(resized, 'output.png')