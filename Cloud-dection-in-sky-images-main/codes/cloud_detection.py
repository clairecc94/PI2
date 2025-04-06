# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 16:55:00 2019
@author: ynie
"""
"""
import numpy as np
import os
from sun_position_identification import *

def cloud_detection(time, image, csl_time=None):
    
    
    Take inputs of sky image and its assoicated time
    identify the cloud pixels in the sky image
    return the cloud cover (defined as the fraction of cloud pixels within a sky image)
    and a binary cloud mask
    
    ### Load clear sky library
    if csl_time == None:
        proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        clear_sky_library_path = os.path.join(proj_path, 'data','clear_sky_library')
        csl_time = np.load(os.path.join(clear_sky_library_path,'csl_times.npy'),allow_pickle=True)
        csl_image = np.load(os.path.join(clear_sky_library_path,'csl_images.npy'),allow_pickle=True)
        csl_sun_center = np.load(os.path.join(clear_sky_library_path,'csl_sun_center.npy'),allow_pickle=True)
        csl_sun_center_x = csl_sun_center[:,0]
        csl_sun_center_y = csl_sun_center[:,1]
    
    ### Sun position in the original image
    sun_center_x, sun_center_y, sun_mask = sun_position(time)
    
    ### Match the image in CSL based on similar sun position
    dist_sun_center = np.sqrt((csl_sun_center_x-sun_center_x)**2+(csl_sun_center_y-sun_center_y)**2)
    match_csl_image = csl_image[np.argmin(dist_sun_center)]
    
    ### Modified threshold method to detect cloud
    NRBR_orig = np.divide((image[:,:,0].astype(int)-image[:,:,2].astype(int)),(image[:,:,0].astype(int)+image[:,:,2].astype(int)))
    NRBR_cs = np.divide((match_csl_image[:,:,0].astype(int)-match_csl_image[:,:,2].astype(int)),(match_csl_image[:,:,0].astype(int)+match_csl_image[:,:,2].astype(int)))
    d_NRBR = np.abs(NRBR_orig-NRBR_cs)
    cloud = np.zeros((64,64),dtype=int)

    for i in range(64):
        for j in range(64):
            if (i-29)**2+(j-30)**2<=29**2:
                if (d_NRBR[i,j] >= 0.175):
                    cloud[i,j] = 1
                    
    unique, counts = np.unique(cloud, return_counts=True)
    try:
        cloud_cover = dict(zip(unique, counts))[1]/int(pi*29**2)
    except:
        cloud_cover = 0
    
    if (cloud_cover>=0.045) and (cloud_cover<0.35):
        cloud = np.zeros((64,64),dtype=int)
        for i in range(64):
            for j in range(64):
                if (i-29)**2+(j-30)**2<=29**2:
                    if (i-sun_center_x)**2+(j-sun_center_y)**2>=7**2:
                        if NRBR_orig[i,j] <= 0.05:
                            cloud[i,j] = 1
    
    if (cloud_cover>=0.35):
        cloud = np.zeros((64,64),dtype=int)
        for i in range(64):
            for j in range(64):
                if (i-29)**2+(j-30)**2<=29**2:
                    if NRBR_orig[i,j] <= 0.05:
                        cloud[i,j] = 1
    
    unique, counts = np.unique(cloud, return_counts=True)
    try:
        cloud_cover = dict(zip(unique, counts))[1]/int(pi*29**2)
    except:
        cloud_cover = 0
    
    ### Cloud mask
    cloud_mask = np.zeros((64,64,3),dtype=np.uint8)
    cloud_mask[:,:,1] = 255 * cloud
        
    return cloud_cover, cloud_mask, sun_mask
    """
import numpy as np
import os
import cv2
from math import pi
from sun_position_identification import *
"""
def adapt_sun_position(time, unwrapped_shape, original_params=None):
    
    Calcule la position du soleil dans l'image unwrapped
    
    Parameters:
    -----------
    time : datetime.datetime
        Horodatage de l'image
    unwrapped_shape : tuple
        Dimensions de l'image unwrapped (hauteur, largeur)
    original_params : dict, optional
        Paramètres originaux de l'image (rayon, centre_x, centre_y)
        
    Returns:
    --------
    new_x, new_y : int, int
        Coordonnées du soleil dans l'image unwrapped
    sun_mask : numpy.ndarray
        Masque binaire indiquant la position du soleil
   
    # Paramètres par défaut de l'image originale (comme dans sun_position)
    if original_params is None:
        original_params = {
            'delta': 14.036,  # différence entre nord géologique et nord de l'image
            'r': 29,          # rayon de l'image originale
            'origin_x': 29,   # coordonnées du centre de l'image originale
            'origin_y': 30
        }
    
    # Calcul de la position du soleil dans l'image originale
    azimuth, zenith = solar_angle(time)
    
    # Conversion en coordonnées polaires (comme dans sun_position)
    rho = zenith/90 * original_params['r']  # distance du centre (rayon)
    theta = azimuth - original_params['delta'] + 90  # angle en degrés
    
    # Normaliser rho et theta pour l'unwrapping
    # rho normalisé (0 au centre, 1 au bord du cercle)
    rho_norm = rho / original_params['r']
    if rho_norm > 0.707:  # limite du unwrap (comme dans votre fonction unwarp)
        return None, None, None  # Le soleil est hors du champ de vision après unwrapping
    
    # Conversion vers les coordonnées de l'image unwrapped
    height, width = unwrapped_shape[0], unwrapped_shape[1]
    
    # Calculer la position dans l'image unwrapped
    width_to_height = 3  # comme dans votre fonction unwarp
    
    # Convertir en coordonnées normalisées -1 à 1
    X_res0 = rho_norm * np.cos(np.radians(theta))
    Y_res0 = rho_norm * np.sin(np.radians(theta))
    
    # Ajuster à la taille de l'image unwrapped
    new_x = int((1 + X_res0 / width_to_height) * width / 2)
    new_y = int((1 + Y_res0 / width_to_height) * height / 2)
    
    # Créer un masque pour le soleil dans l'image unwrapped
    sun_radius_unwrapped = max(2, int(width/100))  # rayon du soleil dans l'image unwrapped
    sun_mask = np.zeros((*unwrapped_shape[:2], 3), dtype=np.uint8)
    
    # Créer un masque circulaire pour le soleil
    for i in range(max(0, new_y - sun_radius_unwrapped*2), min(unwrapped_shape[0], new_y + sun_radius_unwrapped*2)):
        for j in range(max(0, new_x - sun_radius_unwrapped*2), min(unwrapped_shape[1], new_x + sun_radius_unwrapped*2)):
            if (j - new_x)**2 + (i - new_y)**2 <= sun_radius_unwrapped**2:
                sun_mask[i, j, 0] = 255  # Marquer le soleil en rouge
    
    return new_x, new_y, sun_mask
"""
def create_mask_for_unwrapped(unwrapped_shape, margin=0.05):
    """
    Crée un masque pour exclure les bords et zones hors de l'image unwrapped
    
    Parameters:
    -----------
    unwrapped_shape : tuple
        Dimensions de l'image unwrapped (hauteur, largeur, ...)
    margin : float
        Marge en pourcentage de la taille de l'image à exclure
        
    Returns:
    --------
    mask : numpy.ndarray
        Masque binaire (1 pour les pixels valides, 0 sinon)
    """
    height, width = unwrapped_shape[0], unwrapped_shape[1]
    margin_h = int(height * margin)
    margin_w = int(width * margin)
    
    # Créer un masque plein de 1
    mask = np.ones((height, width), dtype=np.uint8)
    
    # Mettre à 0 les marges
    mask[:margin_h, :] = 0
    mask[-margin_h:, :] = 0
    mask[:, :margin_w] = 0
    mask[:, -margin_w:] = 0
    
    return mask

def map_original_point_to_unwrapped(x, y, unwrapped_shape, original_params):
    """
    Transforme les coordonnées d'un point de l'image originale vers l'image unwrapped
    
    Parameters:
    -----------
    x, y : int, int
        Coordonnées du point dans l'image originale
    unwrapped_shape : tuple
        Dimensions de l'image unwrapped (hauteur, largeur)
    original_params : dict
        Paramètres de l'image originale
        
    Returns:
    --------
    new_x, new_y : int, int
        Coordonnées du point dans l'image unwrapped
    """
    # Calculer coordonnées polaires par rapport au centre de l'image originale
    dx = x - original_params['origin_x']
    dy = y - original_params['origin_y']
    rho = np.sqrt(dx**2 + dy**2) / original_params['r']  # Distance normalisée
    theta = np.arctan2(dy, dx)  # Angle en radians
    
    if rho > 0.707:  # Limite du unwrap
        return None, None
    
    # Convertir en coordonnées de l'image unwrapped
    height, width = unwrapped_shape[0], unwrapped_shape[1]
    width_to_height = 3
    
    # Convertir en coordonnées normalisées -1 à 1
    X_res0 = rho * np.cos(theta)
    Y_res0 = rho * np.sin(theta)
    
    # Ajuster à la taille de l'image unwrapped
    new_x = int((1 + X_res0 / width_to_height) * width / 2)
    new_y = int((1 + Y_res0 / width_to_height) * height / 2)
    
    return new_x, new_y

def load_matching_csl_unwrapped(time):
    """
    Charge l'image de ciel clair unwrapped correspondante basée sur la position du soleil
    
    Parameters:
    -----------
    time : datetime.datetime
        Horodatage de l'image
    data_dir : str
        Répertoire contenant les données
        
    Returns:
    --------
    unwrapped_csl_image : numpy.ndarray
        Image de ciel clair unwrapped correspondante
    """
    proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csl_path = os.path.join(proj_path, 'data_npy','clear_sky_library')
    # Chemin vers les données de ciel clair unwrapped
    
    
    # Charger les données
    csl_times = np.load(os.path.join(csl_path, 'csl_times.npy'), allow_pickle=True)
    csl_sun_center = np.load(os.path.join(csl_path, 'csl_sun_center.npy'), allow_pickle=True)
    
    # Calculer la position du soleil dans l'image originale
    sun_x, sun_y, _ = sun_position(time)
    
    # Trouver l'image avec la position du soleil la plus proche
    dist_sun_center = np.sqrt(
        (csl_sun_center[:, 0] - sun_x)**2 + 
        (csl_sun_center[:, 1] - sun_y)**2
    )
    
    # Obtenir l'index de l'image de ciel clair correspondante
    matched_idx = np.argmin(dist_sun_center)
    
    # Charger l'image unwrapped directement (puisqu'elles sont déjà unwrapped)
    matched_csl_image_name = f"csl_{matched_idx}.npy"
    matched_csl_image_path = os.path.join(csl_path, matched_csl_image_name)
    
    # Vérifier si le fichier existe
    if os.path.exists(matched_csl_image_path):
        unwrapped_csl_image = np.load(matched_csl_image_path)
    else:
        # Si le fichier spécifique n'existe pas, utiliser le format général des images CSL
        unwrapped_csl_images = np.load(os.path.join(csl_path, 'csl_images.npy'), allow_pickle=True)
        unwrapped_csl_image = unwrapped_csl_images[matched_idx]
    
    return unwrapped_csl_image


def cloud_detection_unwrapped(time, unwrapped_image, data_dir='data_npy'):
    """
    Détecte les nuages dans une image unwrapped en excluant les pixels noirs.
    
    Parameters:
    -----------
    time : datetime.datetime
        Horodatage de l'image
    unwrapped_image : numpy.ndarray
        Image unwrapped
    data_dir : str
        Répertoire contenant les données
        
    Returns:
    --------
    cloud_cover : float
        Fraction de pixels de nuage dans l'image
    cloud_mask : numpy.ndarray
        Masque binaire indiquant les pixels de nuage
    sun_mask : numpy.ndarray
        Masque binaire indiquant la position du soleil
    """
    import numpy as np
    import cv2
    
    # Ensure necessary functions are available or defined
    # If these functions aren't defined elsewhere in your code, you'll need to implement them
    
    original_params = {
        'delta': 14.036,
        'r': 29,
        'origin_x': 29,
        'origin_y': 30
    }
    
    try:
        sun_x, sun_y, sun_mask = detect_sun_in_unwrapped(unwrapped_image, time, original_params)
    except Exception as e:
        print(f"Sun detection failed: {e}")
        sun_x, sun_y, sun_mask = None, None, None
    
    try:
        valid_mask = create_mask_for_unwrapped(unwrapped_image.shape)
    except Exception as e:
        print(f"Mask creation failed: {e}")
        # Fallback to a simple mask if function is not available
        valid_mask = np.ones((unwrapped_image.shape[0], unwrapped_image.shape[1]), dtype=int)
    
    try:
        unwrapped_csl_image = load_matching_csl_unwrapped(time)
        if unwrapped_csl_image.shape != unwrapped_image.shape:
            unwrapped_csl_image = cv2.resize(unwrapped_csl_image, (unwrapped_image.shape[1], unwrapped_image.shape[0]))
    except Exception as e:
        print(f"Loading CSL image failed: {e}")
        # Fallback to a simple reference image if function is not available
        unwrapped_csl_image = np.zeros_like(unwrapped_image)
        unwrapped_csl_image[:,:,0] = np.ones_like(unwrapped_image[:,:,0]) * 180  # Higher blue for clear sky
        unwrapped_csl_image[:,:,2] = np.ones_like(unwrapped_image[:,:,2]) * 90   # Lower red for clear sky
    
    # Calculate NRBR for original image using the safer np.divide approach
    NRBR_orig = np.divide(
        (unwrapped_image[:,:,2].astype(int) - unwrapped_image[:,:,0].astype(int)),
        (unwrapped_image[:,:,2].astype(int) + unwrapped_image[:,:,0].astype(int)),
        out=np.zeros_like(unwrapped_image[:,:,0], dtype=float),
        where=(unwrapped_image[:,:,2].astype(int) + unwrapped_image[:,:,0].astype(int)) != 0
    )
    
    # Calculate NRBR for clear sky reference
    NRBR_cs = np.divide(
        (unwrapped_csl_image[:,:,2].astype(int) - unwrapped_csl_image[:,:,0].astype(int)),
        (unwrapped_csl_image[:,:,2].astype(int) + unwrapped_csl_image[:,:,0].astype(int)),
        out=np.zeros_like(unwrapped_csl_image[:,:,0], dtype=float),
        where=(unwrapped_csl_image[:,:,2].astype(int) + unwrapped_csl_image[:,:,0].astype(int)) != 0
    )
    
    # Ensure both arrays have the same shape before subtraction
    # They should be 2D arrays with shapes (height, width)
    if NRBR_orig.shape != NRBR_cs.shape:
        print(f"Shape mismatch: NRBR_orig {NRBR_orig.shape}, NRBR_cs {NRBR_cs.shape}")
        # Resize if necessary
        NRBR_cs = cv2.resize(NRBR_cs, (NRBR_orig.shape[1], NRBR_orig.shape[0]))
    
    # Calculate absolute difference
    d_NRBR = np.abs(NRBR_orig - NRBR_cs)
    
    height, width = unwrapped_image.shape[:2]
    cloud = np.zeros((height, width), dtype=int)

    # Threshold to exclude black pixels
    black_threshold = 30
    is_black_pixel = np.all(unwrapped_image <= black_threshold, axis=-1)

    # First pass: use d_NRBR for cloud detection
    for i in range(height):
        for j in range(width):
            if valid_mask[i, j] == 1 and not is_black_pixel[i, j]:
                if d_NRBR[i, j] >= 0.175:
                    cloud[i, j] = 1
    
    valid_area = np.sum(valid_mask)
    cloud_pixels = np.sum(cloud)
    
    cloud_cover = cloud_pixels / valid_area if valid_area > 0 else 0

    # Second pass: refine detection for medium cloud cover
    if (cloud_cover >= 0.045) and (cloud_cover < 0.35):
        cloud = np.zeros((height, width), dtype=int)
        sun_radius_unwrapped = max(2, int(width / 100))
        for i in range(height):
            for j in range(width):
                if valid_mask[i, j] == 1 and not is_black_pixel[i, j]:
                    if sun_x is not None and sun_y is not None:
                        if (j - sun_x) ** 2 + (i - sun_y) ** 2 >= (sun_radius_unwrapped * 3.5) ** 2:
                            if NRBR_orig[i, j] <= 0.05:
                                cloud[i, j] = 1
                    else:
                        if NRBR_orig[i, j] <= 0.05:
                            cloud[i, j] = 1

    # Third pass: refine detection for high cloud cover
    if cloud_cover >= 0.35:
        cloud = np.zeros((height, width), dtype=int)
        for i in range(height):
            for j in range(width):
                if valid_mask[i, j] == 1 and not is_black_pixel[i, j]:
                    if NRBR_orig[i, j] <= 0.05:
                        cloud[i, j] = 1
    
    # Recalculate cloud cover after refinement
    cloud_pixels = np.sum(cloud)
    cloud_cover = cloud_pixels / valid_area if valid_area > 0 else 0
    
    # Create color mask for visualization
    cloud_mask = np.zeros((height, width, 3), dtype=np.uint8)
    cloud_mask[:, :, 1] = 255 * cloud  

    return cloud_cover, cloud_mask, sun_mask


# Fonction pour visualiser les résultats
def visualize_cloud_detection(unwrapped_image, cloud_mask, sun_mask, cloud_cover):
    """
    Visualise les résultats de la détection de nuages
    
    Parameters:
    -----------
    unwrapped_image : numpy.ndarray
        Image unwrapped
    cloud_mask : numpy.ndarray
        Masque binaire indiquant les pixels de nuage
    sun_mask : numpy.ndarray
        Masque binaire indiquant la position du soleil
    cloud_cover : float
        Fraction de pixels de nuage dans l'image
    """
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Créer une version RGB de l'image originale si elle est en BGR
    if unwrapped_image.shape[2] == 3:
        rgb_image = cv2.cvtColor(unwrapped_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = unwrapped_image
    
    # Créer une image composite pour la visualisation
    vis_image = rgb_image.copy()
    
    # Superposer le masque de nuage (vert semi-transparent)
    cloud_indices = cloud_mask[:,:,1] > 0
    vis_image[cloud_indices, 1] = 255  # Vert pour les nuages
    
    # Superposer le masque du soleil (rouge)
    if sun_mask is not None:
        sun_indices = sun_mask[:,:,0] > 0
        vis_image[sun_indices, 0] = 255  # Rouge pour le soleil
    
    # Afficher les résultats
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(rgb_image)
    plt.title("Image Unwrapped Originale")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(vis_image)
    plt.title(f"Nuages (vert) et Soleil (rouge)\nCouverture nuageuse: {cloud_cover:.2%}")
    plt.axis('off')
    
    plt.subplot(133)
    # Afficher uniquement les masques
    combined_mask = np.zeros_like(cloud_mask)
    combined_mask[cloud_indices, 1] = 255  # Vert pour les nuages
    if sun_mask is not None:
        combined_mask[sun_indices, 0] = 255  # Rouge pour le soleil
    plt.imshow(combined_mask)
    plt.title("Masques de nuage et du soleil")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()