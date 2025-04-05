import os
import numpy as np
import cv2
from PIL import Image

base_dir = 'data'
output_dir = 'data_npy/clear_sky_library'

def improve_image_brightness(img, gamma=1.2):
    """
    Améliore la luminosité de l'image avec correction gamma
    
    Parameters:
    -----------
    img : numpy.ndarray
        Image à améliorer
    gamma : float
        Facteur de correction gamma (>1 éclaircit, <1 assombrit)
        
    Returns:
    --------
    numpy.ndarray
        Image avec luminosité améliorée
    """
    # Normaliser l'image entre 0 et 1
    if img.dtype != np.float64:
        normalized = img.astype(np.float64) / 255.0
    else:
        normalized = img.copy()
    
    # Appliquer la correction gamma
    corrected = np.power(normalized, 1.0/gamma)
    
    # Reconvertir à la plage originale
    if img.dtype != np.float64:
        corrected = (corrected * 255).astype(img.dtype)
    
    return corrected

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Recherche des dossiers d'images déjà unwrappées
image_dirs = [d for d in os.listdir(base_dir) if d.startswith("csl_") and d.endswith("_unwrapped")]

for image_dir in image_dirs:
    image_path = os.path.join(base_dir, image_dir)
    npy_filename = os.path.join(output_dir, image_dir.replace("_unwrapped", ".npy"))

    print(f"📂 Traitement du dossier : {image_path}")

    image_files = sorted([f for f in os.listdir(image_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

    images_list = []
    
    for img_file in image_files:
        img = cv2.imread(os.path.join(image_path, img_file), cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ Erreur lors du chargement de {img_file}, ignoré.")
            continue
            
        # Améliorer la luminosité de l'image
        brightened_img = improve_image_brightness(img, gamma=1.2)
        
        # Redimensionner l'image
        brightened_img = cv2.resize(brightened_img, (64, 64))
        
        images_list.append(brightened_img)

    if images_list:
        images_array = np.array(images_list, dtype=np.uint8)
        np.save(npy_filename, images_array)
        print(f"✅ Enregistré : {npy_filename} avec {len(images_list)} images améliorées.")
    else:
        print(f"⚠️ Aucun fichier valide trouvé dans {image_path}.")

print("🎉 Conversion terminée ! Toutes les images sont bien améliorées en luminosité et enregistrées en 64x64.")