# -*- coding: utf-8 -*-
"""
Created on Aug 6 16:41:00 2019
Revised version on Feb 12 17:26:00 2020
@author: ynie
"""

import numpy as np
from math import *
import calendar

def doy_tod_conv(date_and_time,longitude,time_zone_center_longitude):
    """
    Takes a single datetime.datetime as input. 
    Returns two values: 1st being day of year
    and 2nd being time of day solely in seconds-24 hr clock.
    """
    # Time correction. The center of PST is -120 W, while the logitude of location of interest is about 2 degree west of PST center
    pst_center_longitude = time_zone_center_longitude # minus sign indicate west longitude
    loc_longitude = longitude # minus sign indicate west longitude
    correction = np.abs(60/15*(loc_longitude - pst_center_longitude))
    min_correction = int(correction) # local time delay in minutes from the PST
    sec_correction = int((correction - min_correction)*60)  # Local time delay in seconds from the PST
    if date_and_time.minute<=min_correction:
        date_and_time= date_and_time.replace(hour = date_and_time.hour-1, minute=60+date_and_time.minute-min_correction-1, second=60-sec_correction)
    else:
        date_and_time = date_and_time.replace(minute=date_and_time.minute-min_correction-1, second=60-sec_correction)
    
    time_of_day=date_and_time.hour * 3600 + date_and_time.minute * 60 + date_and_time.second
    
    # Following piece of code calculates day of year
    months=[31,28,31,30,31,30,31,31,30,31,30,31] # days in each month
    if (date_and_time.year % 4 == 0) and (date_and_time.year % 100 != 0 or date_and_time.year % 400 ==0 ) == True:
        months[1]=29 # Modification for leap year
    day_of_year=sum(months[:date_and_time.month-1])+date_and_time.day
    
    # Fix for daylight savings (NOTE: This doesn't work for 1st hour of each day in DST period.
    # which day of year is the 2nd Sunday of March in that year
    dst_start_day = sum(months[:2]) + calendar.monthcalendar(date_and_time.year,date_and_time.month)[1][6] 
    # which day of year is the 1st Sunday of Nov in that year 
    dst_end_day = sum(months[:10]) + calendar.monthcalendar(date_and_time.year,date_and_time.month)[0][6]
    if day_of_year >= dst_start_day and day_of_year < dst_end_day:
        time_of_day=time_of_day-3600
    
    return day_of_year, time_of_day

def solar_angle(times, latitude=37.424107, longitude=-122.174199, time_zone_center_longitude=-120):
    
    """
    Calculate the solar angles (Azimuth, Zenith) for a specific location
    Input: time stamp in datetime.datetime format,
    latitude and longitude of the location of interest in degree
    time_zone_center_longitude (for local time correction): the longitude in degree for the time zone center (e.g., for pst time zone, it is -120)
    """

    day_of_year, time_of_day=doy_tod_conv(times,longitude,time_zone_center_longitude)
    latitude=radians(latitude) # Latitudinal co-ordinate of Stanford

    # Calculating parameters dependent on time, day and location, refer to the textbook by DaRosa
    alpha=2*pi*(time_of_day-43200)/86400 # Hour angle in radians
    delta=radians(23.44*sin(radians((360/365.25)*(day_of_year-80)))); # Solar declination angle
    chi=acos(sin(delta)*sin(latitude)+cos(delta)*cos(latitude)*cos(alpha))# Zenith angle of sun
    tan_xi=sin(alpha)/(sin(latitude)*cos(alpha)-cos(latitude)*tan(delta)) # tan(Azimuth angle of sun,xi)
    if alpha>0 and tan_xi>0:
        xi=pi+atan(tan_xi)
    elif alpha>0 and tan_xi<0:
        xi=2*pi+atan(tan_xi)
    elif alpha<0 and tan_xi>0:
        xi=atan(tan_xi)
    else:
        xi=pi+atan(tan_xi)
    
    return degrees(xi), degrees(chi)

def sun_position(time):
    """
    Take the time stamp of the sky image
    return the position of the sun (x, y), in Cartesian coordinates, and a binary sun mask
    For explanation of the method, refer to Figure 7 of our paper https://doi.org/10.1063/5.0014016
    or Figure 4 in README of this repository
    """
    
    # default parameters
    delta = 14.036  # the difference between geological north and sky image north
    r = 29 # radius of sky image (the circle) 
    origin_x = 29 # Cartesian coordinates of the sky image center x=29
    origin_y = 30 # Cartesian coordinates of the sky image center y=30

    # calculate
    azimuth, zenith = solar_angle(time)
    rho = zenith/90*r # polar coordinate length dimension
    theta = azimuth-delta+90 # polar coordinate degree dimension
    sun_center_x = round(origin_x-rho*sin(radians(theta)))
    sun_center_y = round(origin_y+rho*cos(radians(theta)))
    
    sun_mask = np.zeros((64,64,3),dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            if (i-sun_center_x)**2+(j-sun_center_y)**2<=2**2:
                sun_mask[:,:,0][i,j]=255

    return sun_center_x, sun_center_y, sun_mask
import numpy as np
from math import sin, cos, radians, degrees, sqrt, atan2
import cv2

def adapt_sun_position(time, unwrapped_shape, original_params=None):
    """
    Calcule la position du soleil dans l'image unwrapped en respectant la transformation utilisée dans `unwarp`.
    Parameters:
    -----------
    time : datetime.datetime
        Horodatage de l'image.
    unwrapped_shape : tuple
        Dimensions de l'image unwrapped (hauteur, largeur).
    original_params : dict, optional
        Paramètres de l'image originale (rayon, centre_x, centre_y).
    Returns:
    --------
    new_x, new_y : int, int
        Coordonnées du soleil dans l'image unwrapped.
    sun_mask : numpy.ndarray
        Masque binaire indiquant la position du soleil.
    """
    # Paramètres par défaut de l'image originale
    if original_params is None:
        original_params = {
            'delta': 14.036,  # Différence entre nord géologique et nord image
            'r': 29,          # Rayon de l'image originale
            'origin_x': 29,   # Centre X de l'image originale
            'origin_y': 30    # Centre Y de l'image originale
        }
    
    # Calcul de la position du soleil dans l'image originale
    azimuth, zenith = solar_angle(time)
    
    # Conversion en coordonnées polaires
    s_sun = (zenith / 90) * original_params['r']  # Distance sur l'image originale
    theta = radians(azimuth - original_params['delta'] + 90)  # Angle ajusté en radians
    
    # Vérifier si le soleil est hors du champ de vision après unwrapping
    if s_sun > 0.707 * original_params['r']:
        return None, None, None  
    
    # Appliquer la même transformation que dans `unwarp`
    rho_sun = 2 * s_sun / (2 * sqrt(original_params['r'] ** 2 - s_sun ** 2) - original_params['r'])
    X_res0 = rho_sun * cos(theta)
    Y_res0 = rho_sun * sin(theta)
    
    # Dimensions de l'image unwrapped
    height, width = unwrapped_shape[:2]
    width_to_height = 3.0
    
    # Conversion aux coordonnées de l'image unwrapped
    new_x = int((1 + X_res0 / width_to_height) * width / 2)
    new_y = int((1 + Y_res0 / width_to_height) * height / 2)
    
    # Vérifier si le soleil est hors de l'image unwrapped
    if not (0 <= new_x < width and 0 <= new_y < height):
        return None, None, None
    
    # Créer un masque circulaire pour le soleil
    sun_radius_unwrapped = max(2, int(width / 100))  # Ajustable selon la résolution
    sun_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(max(0, new_y - sun_radius_unwrapped), min(height, new_y + sun_radius_unwrapped)):
        for j in range(max(0, new_x - sun_radius_unwrapped), min(width, new_x + sun_radius_unwrapped)):
            if (j - new_x) ** 2 + (i - new_y) ** 2 <= sun_radius_unwrapped ** 2:
                sun_mask[i, j, 0] = 255  # Marquer en rouge
    
    return new_x, new_y, sun_mask

def detect_sun_in_unwrapped(unwrapped_img, time, original_params=None, search_radius=100):
    """
    Détecte la position du soleil dans une image unwrapped
    
    Parameters:
    -----------
    unwrapped_img : numpy.ndarray
        Image unwrapped
    time : datetime.datetime
        Horodatage de l'image
    original_params : dict, optional
        Paramètres originaux de l'image
    search_radius : int, optional
        Rayon de recherche autour de la position calculée
        
    Returns:
    --------
    position : tuple
        (x, y) - coordonnées du soleil dans l'image unwrapped
    sun_mask : numpy.ndarray
        Masque binaire indiquant la position du soleil
    """
    # Obtenir les dimensions de l'image unwrapped
    unwrapped_shape = unwrapped_img.shape
    height, width = unwrapped_shape[:2]
    
    # Calculer la position du soleil dans l'image unwrapped (position théorique)
    sun_x, sun_y, sun_mask_initial = adapt_sun_position(time, unwrapped_shape, original_params)
    
    # Si le soleil est hors du champ de vision après unwrapping
    if sun_x is None or sun_y is None:
        # Dans ce cas, chercher directement le point le plus lumineux dans toute l'image
        if len(unwrapped_img.shape) == 3:
            gray_img = cv2.cvtColor(unwrapped_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = unwrapped_img.copy()
            
        # Appliquer un flou pour réduire le bruit
        blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        # Trouver le point le plus lumineux
        _, max_val, _, max_loc = cv2.minMaxLoc(blur_img)
        sun_x, sun_y = max_loc
        
        # Créer un masque pour cette position
        sun_radius = max(2, int(width / 100))
        sun_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(max(0, sun_y - sun_radius), min(height, sun_y + sun_radius)):
            for j in range(max(0, sun_x - sun_radius), min(width, sun_x + sun_radius)):
                if (j - sun_x) ** 2 + (i - sun_y) ** 2 <= sun_radius ** 2:
                    sun_mask[i, j, 0] = 255
                    
        return sun_x, sun_y, sun_mask
    
    # Définir la région d'intérêt (ROI) autour de la position théorique
    x_min = max(0, sun_x - search_radius)
    x_max = min(width, sun_x + search_radius)
    y_min = max(0, sun_y - search_radius)
    y_max = min(height, sun_y + search_radius)
    
    # Extraire la région d'intérêt
    if len(unwrapped_img.shape) == 3:
        roi = unwrapped_img[y_min:y_max, x_min:x_max].copy()
        # Convertir en niveaux de gris
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi = unwrapped_img[y_min:y_max, x_min:x_max].copy()
        roi_gray = roi.copy()
    
    # Si la ROI est vide, utiliser la position théorique
    if roi.size == 0:
        return sun_x, sun_y, sun_mask_initial
    
    # Appliquer un flou gaussien pour réduire le bruit
    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    
    # Trouver le point le plus lumineux dans la région d'intérêt
    _, _, _, max_loc = cv2.minMaxLoc(roi_blur)
    
    # Convertir les coordonnées locales en coordonnées globales
    refined_x = max_loc[0] + x_min
    refined_y = max_loc[1] + y_min
    
    # Affiner la position en analysant une petite région autour du point le plus lumineux
    # Créer une petite ROI autour du point le plus lumineux
    fine_radius = 10
    fine_x_min = max(0, refined_x - fine_radius)
    fine_x_max = min(width, refined_x + fine_radius)
    fine_y_min = max(0, refined_y - fine_radius)
    fine_y_max = min(height, refined_y + fine_radius)
    
    fine_roi = unwrapped_img[fine_y_min:fine_y_max, fine_x_min:fine_x_max].copy()
    
    if fine_roi.size > 0:
        if len(fine_roi.shape) == 3:
            fine_roi_gray = cv2.cvtColor(fine_roi, cv2.COLOR_BGR2GRAY)
        else:
            fine_roi_gray = fine_roi.copy()
        
        # Trouver les pixels lumineux (avec un seuil élevé)
        max_val = np.max(fine_roi_gray)
        threshold = max_val * 0.95  # Seuil à 95% de la valeur maximale
        bright_pixels = fine_roi_gray >= threshold
        
        y_coords, x_coords = np.where(bright_pixels)
        
        if len(y_coords) > 0:
            # Calculer le centre des pixels lumineux
            refined_x = int(np.mean(x_coords)) + fine_x_min
            refined_y = int(np.mean(y_coords)) + fine_y_min
    
    # Créer un masque pour la position affinée
    sun_radius = max(2, int(width / 100))
    refined_sun_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(max(0, refined_y - sun_radius), min(height, refined_y + sun_radius)):
        for j in range(max(0, refined_x - sun_radius), min(width, refined_x + sun_radius)):
            if (j - refined_x) ** 2 + (i - refined_y) ** 2 <= sun_radius ** 2:
                refined_sun_mask[i, j, 0] = 255  # Marquer en rouge
    
    return refined_x, refined_y, refined_sun_mask