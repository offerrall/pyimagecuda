from pyimagecuda import Image, load, save, Fill, Blend


# Create background with gradient
background = Image(1920, 1080)
Fill.gradient(background, 
              rgba1=(0.2, 0.3, 0.5, 1.0),  # Dark blue
              rgba2=(0.8, 0.4, 0.6, 1.0),  # Pink
              direction='diagonal')
save(background, "./background_gradient.jpg")

# Load overlay image
overlay = load("examples/test.jpg")

# Blend overlay onto gradient background
Blend.normal(background, overlay, pos_x=100, pos_y=50, opacity=0.8)
save(background, "./blended_on_gradient.jpg")

# Create solid color background
background.free()
background = Image(1920, 1080)
Fill.color(background, rgba=(0.1, 0.1, 0.15, 1.0))  # Dark gray

# Multiply blend
Blend.multiply(background, overlay, pos_x=200, pos_y=100, opacity=0.9)
save(background, "./blended_multiply.jpg")

# Radial gradient background
background.free()
background = Image(1920, 1080)
Fill.gradient(background,
              rgba1=(1.0, 1.0, 1.0, 1.0),  # White center
              rgba2=(0.0, 0.0, 0.0, 1.0),  # Black edges
              direction='radial')

# Screen blend
Blend.screen(background, overlay, pos_x=400, pos_y=300, opacity=0.7)
save(background, "./blended_screen.jpg")


background.free()
overlay.free()