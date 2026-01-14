from pyimagecuda import Image, load, Effect, save

# Load an image with a solid color background
img = load('chroma_key.jpg')

# Remove green background
Effect.chroma_key(
    img,
    key_color=(0, 1, 0),  # Green background
    threshold=0.7,
    smoothness=0.1,
    spill_suppression=0.5
)

save(img, 'chroma_key_out.png')