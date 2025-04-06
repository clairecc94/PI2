import os
import numpy as np
import cv2

base_dir = 'data'
output_dir = 'data_npy\clear_sky_library'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#image_dirs = [d for d in os.listdir(base_dir) if d.startswith("sky_images_") and d.endswith("_unwrapped")]
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
        img = cv2.resize(img, (64, 64))  # 🔥 Corrige la taille des images
        images_list.append(img)

    if images_list:
        images_array = np.array(images_list, dtype=np.uint8)
        np.save(npy_filename, images_array)
        print(f"✅ Enregistré : {npy_filename} avec {len(images_list)} images.")
    else:
        print(f"⚠️ Aucun fichier valide trouvé dans {image_path}.")

print("🎉 Conversion terminée ! Toutes les images sont bien enregistrées en 64x64.")
