# Cloud & Contrail Detection Platform – Team 266

## Overview
This project, developed by Team 266, focuses on analyzing satellite images to detect clouds, contrails, and sun positions. It also predicts photovoltaic (PV) energy production based on these detections. The platform combines machine learning models with a user-friendly interface to provide real-time analysis and historical data visualization.​

## Features
**Cloud Detection**: Identifies cloud formations in satellite images.

**Contrail Segmentation**: Uses YOLOv11 for precise contrail detection.

**Sun Position Estimation**: Calculates the sun's position based on image metadata.

**PV Production Prediction**: Estimates solar energy output considering cloud coverage and sun position.

**Interactive Interface**: Allows users to upload images, view analysis results, and access historical data.​

Installation
Clone the Repository:

```
git clone https://github.com/clairecc94/PI2.git
cd PI2
```
Frontend 

```
cd frontend
npm install
npm start
```
Access the Application:

Open your browser and navigate to http://localhost:3000 to use the platform.

## Usage
Upload Satellite Image: Navigate to the "Real-Time Analysis" section and upload a satellite image.

View Analysis: The platform will process the image and display detected clouds, contrails, and sun position.

PV Prediction: Based on the analysis, the platform will estimate the potential PV energy production.

Access Historical Data: Use the "Scan History" section to view previous analyses.

## Challenges Faced
Data Quality: Variations in satellite image quality affected detection accuracy.

Model Generalization: Ensuring models performed well across diverse datasets was challenging.

Integration: Seamless communication between frontend and backend required careful API design.

Performance Optimization: Balancing model accuracy with processing speed was crucial for real-time analysis.​

## Future Improvements
Enhanced Models: Incorporate more advanced models for better detection accuracy.

User Customization: Allow users to select preferred models or adjust detection parameters.

Mobile Support: Optimize the interface for mobile devices.

Extended Data Sources: Integrate additional satellite data providers for broader coverage.

Automated Alerts: Implement notification systems for significant analysis results.​

## Contributing
We welcome contributions! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss proposed modifications.​

## License
This project is licensed under the MIT License. See the LICENSE file for details.​

## Acknowledgements
Special thanks to our partner and CTS for their support and guidance throughout the project.​

## References
Nie, Y., Sun, Y., Chen, Y., Orsini, R., & Brandt, A. (2020). PV power output prediction from sky images using convolutional neural network: The comparison of sky condition-specific sub-models and an end-to-end model. Journal of Renewable and Sustainable Energy, 12(4)
