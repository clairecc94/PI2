import { Component, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms'; // <-- add this if not already there
import axios from 'axios';

@Component({
  standalone: true,
  selector: 'app-real-time-analysis',
  imports: [CommonModule, FormsModule],
  templateUrl: './real-time-analysis.component.html',
  styleUrls: ['./real-time-analysis.component.scss'],
})
export class RealTimeAnalysisComponent {
  imageUrl: string | ArrayBuffer | null = null;
  imageResult: any; // To hold the result from the Roboflow API
  loading: boolean = false;
  error: string | null = null;
  confidenceThreshold: number = 0.5;

  private model = 'contrails-detection-6hngf';
  private version = '2';
  private apiKey = 'ezLS5FQ0wcs0SzBf9Nj7';

  // Class-to-color mapping
  protected classColors: { [key: string]: string } = {
    'contrail young': 'rgba(255, 0, 0, 0.4)',       // red
    'contrail old': 'rgba(255,20,147, 0.4)',        // pink
    'contrail veryold': 'rgba(128, 0, 128, 0.4)',   // purple
  };

// List of classes to skip
  private ignoredClasses: string[] = [
    'sun',
    'parasite',
    'unknow',
  ];

  constructor() {}

  @ViewChild('maskCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('imageRef') imageRef!: ElementRef<HTMLImageElement>;

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

        // If image already loaded, draw now
        if (this.imageRef?.nativeElement.complete) {
          this.drawMasks();
        }
      })
      .catch((error) => {
        this.error = 'Error processing image: ' + error.message;
        this.loading = false;
        console.error(error);
      });
  }

  onImageLoad(img: HTMLImageElement): void {
    setTimeout(() => {
      this.updateCanvasSize();
      if (this.imageResult) {
        this.drawMasks();
      }
    }, 0); // Run after layout
  }

  updateCanvasSize(): void {
    const img = this.imageRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
  }

  get classColorEntries() {
    return Object.entries(this.classColors).map(([className, color]) => ({
      class: className,
      color: color
    })).filter(entry => !this.ignoredClasses.includes(entry.class));
  }

  drawMasks(): void {
    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx || !this.imageResult) return;

    const img = this.imageRef.nativeElement;

    // Roboflow image original size
    const originalWidth = this.imageResult.image.width;
    const originalHeight = this.imageResult.image.height;

    // Displayed image size
    const displayWidth = img.clientWidth;
    const displayHeight = img.clientHeight;

    const xScale = displayWidth / originalWidth;
    const yScale = displayHeight / originalHeight;

    // Clear previous masks
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    this.imageResult.predictions.forEach((prediction: any) => {
      if (prediction.confidence < this.confidenceThreshold) return;

      const className = prediction.class;
      if (this.ignoredClasses.includes(className)) return;

      const points = prediction.points;

      ctx.beginPath();
      points.forEach((pt: any, index: number) => {
        const x = pt.x * xScale;
        const y = pt.y * yScale;
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.closePath();

      // Use class color or default gray
      const color = this.classColors[className] || 'rgba(150, 150, 150, 0.4)';
      ctx.fillStyle = color;
      ctx.fill();

      //ctx.strokeStyle = 'red';
      //ctx.lineWidth = 2;
      //ctx.stroke();
    });
  }
}
