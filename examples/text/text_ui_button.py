from pyimagecuda import Text, Image, Fill, Blend, Effect, save

# Create text
text_img = Text.create(
    "START GAME",
    font="Arial Bold",
    size=30,
    color=(1.0, 1.0, 1.0, 1.0)
)

# Create button background
pad_w, pad_h = 60, 30
button = Image(text_img.width + pad_w, 
               text_img.height + pad_h)

# Style button
Fill.color(button, (0.2, 0.6, 1.0, 1.0))
Effect.rounded_corners(button, 15)

# Composite text onto button
Blend.normal(button, text_img, anchor='center')

# Add drop shadow
final = Effect.drop_shadow(
    button,
    blur=10,
    offset_y=5,
    color=(0.0, 0.0, 0.0, 0.5)
)

save(final, 'text_ui_button.png')

# Cleanup
text_img.free()
button.free()
final.free()