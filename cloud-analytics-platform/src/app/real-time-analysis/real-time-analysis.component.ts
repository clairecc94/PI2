import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import axios from 'axios';

@Component({
  standalone: true,
  selector: 'app-real-time-analysis',
  imports: [CommonModule],
  templateUrl: './real-time-analysis.component.html',
  styleUrls: ['./real-time-analysis.component.scss'],
})
export class RealTimeAnalysisComponent {
  imageUrl: string | ArrayBuffer | null = null;
  imageResult: any; // To hold the result from the Roboflow API
  loading: boolean = false;
  error: string | null = null;

  //private roboflowUrl = 'https://detect.roboflow.com/contrails-detection-6hngf/2/rf_F5cjmGqfC4TgBuRCOiC5hoZYKBt2';

  private model = 'contrails-detection-6hngf';
  private version = '2';
  private apiKey = 'ezLS5FQ0wcs0SzBf9Nj7';

  constructor() {}

  // Method to handle file selection
  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        this.imageUrl = reader.result;
        this.processImage(reader.result as string); // Process the selected image
      };
      reader.readAsDataURL(file);
    }
  }

  // Method to process the selected image with Roboflow API
  processImage(base64Image: string): void {
    this.loading = true;
    this.error = null;

    const url = `https://outline.roboflow.com/${this.model}/${this.version}`;
    const base64Data = base64Image.replace(/^data:image\/[a-z]+;base64,/, '');

    // Using axios to send the image to the Roboflow API
    axios({
      method: 'POST',
      url,
      params: {
        api_key: this.apiKey,
      },
      data: base64Data,
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    })
      .then((response) => {
        this.imageResult = response.data;
        console.log('Segmentation result:', this.imageResult);
        this.loading = false;

        // TODO: Add canvas drawing here if you want to overlay the mask
      })
      .catch((error) => {
        this.error = 'Error processing image: ' + error.message;
        this.loading = false;
        console.error(error);
      });
  }
}
