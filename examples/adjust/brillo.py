from pyimagecuda import load, save, Adjust

img = load("photo.jpg")

# 1. Brillo (Suma luz)
Adjust.brightness(img, 0.2)
save(img, "1_bright.jpg")

img2 = load("photo.jpg")

# 2. Contraste (Estira los colores desde el gris medio)
Adjust.contrast(img2, 1.5)  # +50% contraste
save(img2, "2_high_contrast.jpg")

img3 = load("photo.jpg")

Adjust.saturation(img3, 1.5)
save(img3, "3_high_saturation.jpg")

img4 = load("photo.jpg")
Adjust.saturation(img4, 0.0)
save(img4, "4_grayscale.jpg")

# gamma correction
img5 = load("photo.jpg")
Adjust.gamma(img5, 1.5)  # Aplicar corrección gamma
save(img5, "5_gamma_corrected.jpg")