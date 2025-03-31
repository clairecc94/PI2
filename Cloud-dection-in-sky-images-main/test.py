import numpy as np
import cv2
import os

# Charger le fichier .npy contenant plusieurs images
images = np.load(r'data\clear_sky_library\csl_sun_center.npy')  # Notez le 'r' pour raw string

# Dossier de sortie
output_dir = r'data\csl_sun_center'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parcourir toutes les images du tableau
for i, img in enumerate(images):
    # Vérifier que les valeurs sont dans la plage correcte pour une image
    if img.max() > 1.0 and img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    elif img.max() <= 1.0 and img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # ✅ Vérifier que l'image est bien en RGB et non BGR
    if img.shape[-1] == 3:  # Vérifie qu'on a bien une image couleur
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV utilise BGR

    # 📷 Sauvegarde en haute qualité JPEG (100%)
    output_path = os.path.join(output_dir, f'image_{i}.jpg')
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    print(f"✅ Image {i} sauvegardée avec qualité maximale : {output_path}")

print("🎉 Toutes les images ont été converties avec succès en JPEG haute qualité !")
