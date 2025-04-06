import numpy as np
import datetime
import os
import cv2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from cloud_detection import *
from sun_position_identification import *

def extract_features(unwrapped_img, timestamp):
    """
    Extract features from a sky image for PV output prediction.
    
    Parameters:
    -----------
    sky_image : numpy.ndarray
        Original sky image
    timestamp : datetime.datetime
        Timestamp of the image
        
    Returns:
    --------
    features : dict
        Dictionary of extracted features
    """
    # Unwrap the sky image
     
    
    # Get sun position and mask
    sun_x, sun_y, sun_mask = detect_sun_in_unwrapped(unwrapped_img, timestamp)
    
    # Get cloud coverage and mask
    cloud_cover, cloud_mask, _ = cloud_detection_unwrapped(timestamp, unwrapped_img)
    
    # Extract solar angle information
    azimuth, zenith = solar_angle(timestamp)
    
    # Calculate time-based features
    hour = timestamp.hour + timestamp.minute / 60.0  # Hour of day as float
    day_of_year = timestamp.timetuple().tm_yday  # Day of year (1-366)
    
    # Calculate distance from sun to clouds
    cloud_to_sun_distance = calculate_cloud_sun_distance(cloud_mask, sun_x, sun_y)
    
    # Calculate cloud distribution features
    cloud_distribution = calculate_cloud_distribution(cloud_mask, unwrapped_img.shape)
    
    # Calculate image brightness statistics
    brightness_stats = calculate_brightness_stats(unwrapped_img)
    
    # Calculate cloud opacity (simplified)
    cloud_opacity = calculate_cloud_opacity(unwrapped_img, cloud_mask)
    
    # Compile all features
    features = {
        'cloud_cover': cloud_cover,
        'sun_zenith': zenith,
        'sun_azimuth': azimuth,
        'hour': hour,
        'day_of_year': day_of_year,
        'cloud_to_sun_distance': cloud_to_sun_distance,
        'cloud_distribution': cloud_distribution,
        'brightness_mean': brightness_stats['mean'],
        'brightness_std': brightness_stats['std'],
        'cloud_opacity': cloud_opacity
    }
    
    return features

def calculate_cloud_sun_distance(cloud_mask, sun_x, sun_y):
    """
    Calculate the minimum distance from sun to clouds.
    
    Parameters:
    -----------
    cloud_mask : numpy.ndarray
        Binary mask of clouds
    sun_x, sun_y : int, int
        Sun position coordinates
        
    Returns:
    --------
    min_distance : float
        Minimum distance from sun to cloud pixels
    """
    if sun_x is None or sun_y is None:
        return -1  # Sun not visible
    
    # Get cloud pixel coordinates
    cloud_pixels = np.where(cloud_mask[:,:,1] > 0)
    
    if len(cloud_pixels[0]) == 0:
        return 1000  # No clouds
    
    # Calculate distances to all cloud pixels
    distances = np.sqrt((cloud_pixels[1] - sun_x)**2 + (cloud_pixels[0] - sun_y)**2)
    
    # Return minimum distance
    return np.min(distances) if len(distances) > 0 else 1000

def calculate_cloud_distribution(cloud_mask, img_shape):
    """
    Calculate cloud distribution in the image.
    
    Parameters:
    -----------
    cloud_mask : numpy.ndarray
        Binary mask of clouds
    img_shape : tuple
        Shape of the image
        
    Returns:
    --------
    distribution : float
        Cloud distribution metric (0-1)
    """
    height, width = img_shape[:2]
    
    # Divide image into 4 quadrants
    quadrants = [
        cloud_mask[:height//2, :width//2, 1],
        cloud_mask[:height//2, width//2:, 1],
        cloud_mask[height//2:, :width//2, 1],
        cloud_mask[height//2:, width//2:, 1]
    ]
    
    # Calculate cloud coverage in each quadrant
    quadrant_coverage = [np.sum(q) / (q.size * 255) for q in quadrants]
    
    # Calculate standard deviation of coverage across quadrants
    # Higher std means less uniform distribution
    distribution_uniformity = 1 - np.std(quadrant_coverage)
    
    return distribution_uniformity

def calculate_brightness_stats(image):
    """
    Calculate brightness statistics of the image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        RGB image
        
    Returns:
    --------
    stats : dict
        Dictionary of brightness statistics
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate statistics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    return {
        'mean': mean_brightness,
        'std': std_brightness
    }

def calculate_cloud_opacity(image, cloud_mask):
    """
    Calculate cloud opacity (simplified).
    
    Parameters:
    -----------
    image : numpy.ndarray
        RGB image
    cloud_mask : numpy.ndarray
        Binary mask of clouds
        
    Returns:
    --------
    opacity : float
        Cloud opacity metric (0-1)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Get cloud pixels
    cloud_pixels = cloud_mask[:,:,1] > 0
    
    if np.sum(cloud_pixels) == 0:
        return 0  # No clouds
    
    # Calculate mean brightness of cloud pixels
    cloud_brightness = np.mean(gray[cloud_pixels])
    
    # Normalize to 0-1 (higher value = more opaque)
    # Assuming clouds are bright, so higher brightness = higher opacity
    opacity = cloud_brightness / 255.0
    
    return opacity

def generate_synthetic_pv_output(features):
    """
    Generate synthetic PV output based on features.
    This is a placeholder for a real physics-based model or ML model.
    
    Parameters:
    -----------
    features : dict
        Dictionary of features
        
    Returns:
    --------
    pv_output : float
        Synthetic PV output (0-1, normalized)
    """
    # Base output determined by sun position
    base_output = np.cos(np.radians(features['sun_zenith']))
    
    # Adjust for time of day (bell curve centered at noon)
    time_factor = 1 - 0.8 * ((features['hour'] - 12) / 6) ** 2
    time_factor = max(0.2, min(time_factor, 1.0))
    
    # Reduce output based on cloud cover
    cloud_factor = 1 - 0.8 * features['cloud_cover']
    
    # Additional reduction if clouds are near the sun
    sun_cloud_factor = 1.0
    if features['cloud_to_sun_distance'] < 100:
        sun_cloud_factor = max(0.3, features['cloud_to_sun_distance'] / 100)
    
    # Calculate final output
    pv_output = base_output * time_factor * cloud_factor * sun_cloud_factor
    
    # Add some noise to make it realistic
    pv_output = max(0, min(1, pv_output + np.random.normal(0, 0.05)))
    
    return pv_output

def train_pv_prediction_model(sky_images, timestamps, pv_outputs=None):
    """
    Train a model to predict PV output from sky images.
    
    Parameters:
    -----------
    sky_images : numpy.ndarray
        Array of sky images
    timestamps : numpy.ndarray
        Array of corresponding timestamps
    pv_outputs : numpy.ndarray, optional
        Array of corresponding PV outputs. If None, synthetic data is generated.
        
    Returns:
    --------
    model : RandomForestRegressor
        Trained prediction model
    scaler : StandardScaler
        Feature scaler
    """
    # Extract features from all images
    feature_list = []
    for i in range(len(sky_images)):
        features = extract_features(sky_images[i], timestamps[i])
        feature_list.append([
            features['cloud_cover'],
            features['sun_zenith'],
            features['sun_azimuth'],
            features['hour'],
            features['day_of_year'],
            features['cloud_to_sun_distance'],
            features['cloud_distribution'],
            features['brightness_mean'],
            features['brightness_std'],
            features['cloud_opacity']
        ])
    
    X = np.array(feature_list)
    
    # Create target data
    if pv_outputs is None:
        y = np.array([generate_synthetic_pv_output(extract_features(sky_images[i], timestamps[i])) 
                     for i in range(len(sky_images))])
    else:
        y = pv_outputs
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    return model, scaler

def predict_pv_output(model, scaler, sky_image, timestamp):
    """
    Predict PV output for a given sky image.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained prediction model
    scaler : StandardScaler
        Feature scaler
    sky_image : numpy.ndarray
        Sky image to predict from
    timestamp : datetime.datetime
        Timestamp of the image
        
    Returns:
    --------
    pv_output : float
        Predicted PV output (0-1, normalized)
    """
    # Extract features
    features = extract_features(sky_image, timestamp)
    
    # Convert to array
    feature_array = np.array([[
        features['cloud_cover'],
        features['sun_zenith'],
        features['sun_azimuth'],
        features['hour'],
        features['day_of_year'],
        features['cloud_to_sun_distance'],
        features['cloud_distribution'],
        features['brightness_mean'],
        features['brightness_std'],
        features['cloud_opacity']
    ]])
    
    # Scale features
    feature_array_scaled = scaler.transform(feature_array)
    
    # Make prediction
    pv_output = model.predict(feature_array_scaled)[0]
    
    return pv_output

def visualize_results(unwrapped_img, timestamps, predictions, actual=None, n_samples=5):
    """
    Visualize prediction results.
    
    Parameters:
    -----------
    sky_images : numpy.ndarray
        Array of sky images
    timestamps : numpy.ndarray
        Array of corresponding timestamps
    predictions : numpy.ndarray
        Array of predicted PV outputs
    actual : numpy.ndarray, optional
        Array of actual PV outputs
    n_samples : int
        Number of samples to visualize
    """
    # Randomly select samples to visualize
    indices = np.random.choice(len(unwrapped_img), min(n_samples, len(unwrapped_img)), replace=False)
    
    plt.figure(figsize=(15, 4 * n_samples))
    
    for i, idx in enumerate(indices):
        # Plot original image
        plt.subplot(n_samples, 3, i*3 + 1)
        plt.imshow(cv2.cvtColor(unwrapped_img[idx], cv2.COLOR_BGR2RGB))
        plt.title(f"Sky Image - {timestamps[idx].strftime('%Y-%m-%d %H:%M')}")
        plt.axis('off')
        
        _, cloud_mask, sun_mask = cloud_detection_unwrapped(timestamps[idx], unwrapped_img)
        
        # Plot unwrapped image with clouds and sun
        plt.subplot(n_samples, 3, i*3 + 2)
        combined_img = unwrapped_img.copy()
        cloud_indices = cloud_mask[:,:,1] > 0
        combined_img[cloud_indices, 1] = 255
        if sun_mask is not None:
            sun_indices = sun_mask[:,:,0] > 0
            combined_img[sun_indices, 0] = 255
        plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        plt.title("Unwrapped Image with Cloud (green) and Sun (red)")
        plt.axis('off')
        
        # Plot PV output
        plt.subplot(n_samples, 3, i*3 + 3)
        plt.bar(['Predicted'], [predictions[idx]], color='blue', alpha=0.7)
        if actual is not None:
            plt.bar(['Actual'], [actual[idx]], color='green', alpha=0.7)
        plt.ylim(0, 1)
        plt.title(f"PV Output: {predictions[idx]:.3f}")
    
    plt.tight_layout()
    plt.show()
def process_and_predict_pv_output(sunny_images, sunny_timestamps, cloudy_images, cloudy_timestamps):
    """
    Process sunny and cloudy sky images and predict PV output.
    
    Parameters:
    -----------
    sunny_images : numpy.ndarray
        Array of sunny sky images
    sunny_timestamps : numpy.ndarray
        Array of corresponding sunny timestamps
    cloudy_images : numpy.ndarray
        Array of cloudy sky images
    cloudy_timestamps : numpy.ndarray
        Array of corresponding cloudy timestamps
        
    Returns:
    --------
    model : RandomForestRegressor
        Trained prediction model
    scaler : StandardScaler
        Feature scaler
    predictions_sunny : numpy.ndarray
        Predicted PV outputs for sunny images
    predictions_cloudy : numpy.ndarray
        Predicted PV outputs for cloudy images
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    
    # Since we may have many images, let's use a subset for training to speed things up
    max_train_samples = min(100, min(len(sunny_images), len(cloudy_images)))
    
    # Use a subset of sunny and cloudy images
    train_sunny_idx = np.random.choice(len(sunny_images), min(max_train_samples//2, len(sunny_images)), replace=False)
    train_cloudy_idx = np.random.choice(len(cloudy_images), min(max_train_samples//2, len(cloudy_images)), replace=False)
    
    train_images = np.concatenate([sunny_images[train_sunny_idx], cloudy_images[train_cloudy_idx]])
    train_timestamps = np.concatenate([sunny_timestamps[train_sunny_idx], cloudy_timestamps[train_cloudy_idx]])
    
    print(f"Training with {len(train_images)} images...")
    
    # Extract features from training images
    print("Extracting features...")
    features_list = []
    for i in range(len(train_images)):
        try:
            # Extract basic features without detailed processing
            # This simplifies the feature extraction to avoid potential errors
            if len(train_images[i].shape) == 3:
                gray = cv2.cvtColor(train_images[i], cv2.COLOR_BGR2GRAY)
            else:
                gray = train_images[i].copy()
            
            # Calculate brightness
            brightness_mean = np.mean(gray)
            brightness_std = np.std(gray)
            
            # Get time-based features
            hour = train_timestamps[i].hour + train_timestamps[i].minute / 60.0
            day_of_year = train_timestamps[i].timetuple().tm_yday
            
            # Get simplified cloud cover
            try:
                cloud_cover, _, _ = cloud_detection_unwrapped(train_timestamps[i], train_images[i])
            except Exception as e:
                print(f"Cloud detection failed for image {i}: {e}")
                cloud_cover = 0.5  # Default value
            
            # Get sun position (simplified)
            azimuth, zenith = solar_angle(train_timestamps[i])
            
            # Create feature vector
            features = [
                cloud_cover,
                zenith, 
                azimuth,
                hour,
                day_of_year,
                brightness_mean,
                brightness_std
            ]
            
            features_list.append(features)
        except Exception as e:
            print(f"Error processing training image {i}: {e}")
            # Add a row of zeros as a placeholder
            features_list.append([0, 0, 0, 0, 0, 0, 0])
    
    X = np.array(features_list)
    
    # Generate synthetic PV output for training
    # Higher values for sunny images, lower for cloudy
    print("Generating synthetic PV outputs...")
    y = np.array([
        0.8 * np.cos(np.radians(60 * (hour - 12) / 6)) * (1 - 0.7 * cloud_cover)
        for cloud_cover, _, _, hour, _, _, _ in features_list
    ])
    
    # Clip values to 0-1 range
    y = np.clip(y, 0, 1)
    
    # Train a simple RandomForest model
    print("Training model...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    # Make predictions
    print("Making predictions...")
    
    # Predict for sunny images
    predictions_sunny = np.zeros(len(sunny_images))
    for i in range(len(sunny_images)):
        try:
            # Extract simplified features for prediction
            if len(sunny_images[i].shape) == 3:
                gray = cv2.cvtColor(sunny_images[i], cv2.COLOR_BGR2GRAY)
            else:
                gray = sunny_images[i].copy()
            
            brightness_mean = np.mean(gray)
            brightness_std = np.std(gray)
            
            hour = sunny_timestamps[i].hour + sunny_timestamps[i].minute / 60.0
            day_of_year = sunny_timestamps[i].timetuple().tm_yday
            
            try:
                cloud_cover, _, _ = cloud_detection_unwrapped(sunny_timestamps[i], sunny_images[i])
            except:
                cloud_cover = 0.2  # Default for sunny
            
            azimuth, zenith = solar_angle(sunny_timestamps[i])
            
            features = [
                cloud_cover,
                zenith, 
                azimuth,
                hour,
                day_of_year,
                brightness_mean,
                brightness_std
            ]
            
            # Make prediction
            features_scaled = scaler.transform([features])
            predictions_sunny[i] = model.predict(features_scaled)[0]
        except Exception as e:
            print(f"Error predicting for sunny image {i}: {e}")
            predictions_sunny[i] = 0.7  # Default value for sunny
    
    # Predict for cloudy images
    predictions_cloudy = np.zeros(len(cloudy_images))
    for i in range(len(cloudy_images)):
        try:
            # Extract simplified features for prediction
            if len(cloudy_images[i].shape) == 3:
                gray = cv2.cvtColor(cloudy_images[i], cv2.COLOR_BGR2GRAY)
            else:
                gray = cloudy_images[i].copy()
            
            brightness_mean = np.mean(gray)
            brightness_std = np.std(gray)
            
            hour = cloudy_timestamps[i].hour + cloudy_timestamps[i].minute / 60.0
            day_of_year = cloudy_timestamps[i].timetuple().tm_yday
            
            try:
                cloud_cover, _, _ = cloud_detection_unwrapped(cloudy_timestamps[i], cloudy_images[i])
            except:
                cloud_cover = 0.7  # Default for cloudy
            
            azimuth, zenith = solar_angle(cloudy_timestamps[i])
            
            features = [
                cloud_cover,
                zenith, 
                azimuth,
                hour,
                day_of_year,
                brightness_mean,
                brightness_std
            ]
            
            # Make prediction
            features_scaled = scaler.transform([features])
            predictions_cloudy[i] = model.predict(features_scaled)[0]
        except Exception as e:
            print(f"Error predicting for cloudy image {i}: {e}")
            predictions_cloudy[i] = 0.3  # Default value for cloudy
    
    # Simplified visualization
    print("Visualizing results...")
    visualize_results_simple(sunny_images, sunny_timestamps, predictions_sunny, "Sunny Images")
    visualize_results_simple(cloudy_images, cloudy_timestamps, predictions_cloudy, "Cloudy Images")
    
    print(f"Average predicted PV output - Sunny: {np.mean(predictions_sunny):.4f}, Cloudy: {np.mean(predictions_cloudy):.4f}")
    
    return model, scaler, predictions_sunny, predictions_cloudy

def visualize_results_simple(images, timestamps, predictions, title, n_samples=3):
    """
    Simplified visualization function that doesn't rely on cloud detection
    """
    import matplotlib.pyplot as plt
    
    # Select a few random images
    if len(images) > n_samples:
        indices = np.random.choice(len(images), n_samples, replace=False)
    else:
        indices = range(len(images))
    
    plt.figure(figsize=(12, 4 * len(indices)))
    
    for i, idx in enumerate(indices):
        # Display the image
        plt.subplot(len(indices), 2, i*2 + 1)
        if len(images[idx].shape) == 3:
            plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[idx], cmap='gray')
        plt.title(f"Image - {timestamps[idx].strftime('%Y-%m-%d %H:%M')}")
        plt.axis('off')
        
        # Display the prediction
        plt.subplot(len(indices), 2, i*2 + 2)
        plt.bar(['PV Output'], [predictions[idx]], color='blue')
        plt.ylim(0, 1)
        plt.title(f"Predicted PV Output: {predictions[idx]:.3f}")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()