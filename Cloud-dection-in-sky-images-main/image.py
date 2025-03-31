import numpy as np
import cv2
import matplotlib.pyplot as plt

def unwrap_fisheye(img, output_size):
    """
    Déroule une image fisheye en une projection rectangulaire.

    Paramètres :
    - img : Image d'entrée (fisheye).
    - output_size : Taille de sortie (largeur, hauteur).

    Retourne :
    - L'image déroulée.
    """
    H, W = img.shape[:2]
    center_x, center_y = W // 2, H // 2
    max_radius = min(center_x, center_y)  # Rayon maximal depuis le centre

    # Création d'une image de sortie vide (3 canaux couleur)
    output_img = np.zeros((output_size[1], output_size[0], 3), dtype=img.dtype)

    for y in range(output_size[1]):
        for x in range(output_size[0]):
            # Coordonnées normalisées (-1 à 1)
            norm_x = (x - output_size[0] / 2) / (output_size[0] / 2)
            norm_y = (y - output_size[1] / 2) / (output_size[1] / 2)

            # Coordonnées polaires
            r_output = np.sqrt(norm_x**2 + norm_y**2)
            theta = np.arctan2(norm_y, norm_x)

            # Correction du rayon pour éviter les pixels noirs
            r_input = r_output * max_radius
            if r_input > max_radius:  # Vérifier si on dépasse l’image d'origine
                continue

            # Coordonnées cartésiennes dans l'image originale
            input_x = int(center_x + r_input * np.cos(theta))
            input_y = int(center_y + r_input * np.sin(theta))

            # Vérifier les bornes (évite les erreurs d'accès mémoire)
            if 0 <= input_x < W and 0 <= input_y < H:
                output_img[y, x] = img[input_y, input_x]

    return output_img



# Charger l'image depuis un fichier .npy
file_path = "data/sky_images_cloudy.npy"  # Remplace par ton chemin
img = np.load(file_path, allow_pickle=True)
print("Type de données :", img.dtype)
print("Forme de l'image :", img.shape)
print("Valeurs min/max :", img.min(), img.max())

plt.figure(figsize=(6,6))
plt.imshow(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Image d'origine")
plt.show()


# Vérifier la forme et sélectionner la première image si nécessaire
if len(img.shape) == 4:  # Format (N, H, W, C) -> Plusieurs images
    img = img[0]  # Prendre la première image

# Taille de sortie (ajuster selon les besoins)
output_size = (512, 512)

# Transformer l'image fisheye en image déroulée
unwrapped_img = unwrap_fisheye(img, output_size)

# Afficher l'image transformée
plt.figure(figsize=(6, 6))
plt.imshow(unwrapped_img)  # Plus besoin de conversion BGR -> RGB
plt.axis("off")
plt.title("Image Unwrapped")
plt.show()

# Sauvegarder l'image transformée
cv2.imwrite("data/datacleaned/unwrapped_output.jpg", unwrapped_img)
