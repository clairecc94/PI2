import { Component, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import axios from 'axios';
import { ScanHistoryService } from '../scan-history/scan-history.service';

@Component({
  standalone: true,
  selector: 'app-real-time-analysis',
  imports: [CommonModule, FormsModule],
  templateUrl: './real-time-analysis.component.html',
  styleUrls: ['./real-time-analysis.component.scss'],
})
export class RealTimeAnalysisComponent {
  imageUrl: string | ArrayBuffer | null = null;
  imageResult: any;
  loading: boolean = false;
  error: string | null = null;
  confidenceThreshold: number = 0.5;

  private model = 'contrails-detection-6hngf';
  private version = '2';
  private apiKey = 'ezLS5FQ0wcs0SzBf9Nj7';

  protected classColors: { [key: string]: string } = {
    'contrail young': 'rgba(255, 0, 0, 0.4)',
    'contrail old': 'rgba(255,20,147, 0.4)',
    'contrail veryold': 'rgba(128, 0, 128, 0.4)',
  };

  private ignoredClasses: string[] = ['sun', 'parasite', 'unknow'];

  constructor(private scanHistoryService: ScanHistoryService) {}

  @ViewChild('maskCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('imageRef') imageRef!: ElementRef<HTMLImageElement>;

  // Triggered when a file is selected
  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        this.imageUrl = reader.result;
        this.processImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }

  // Process the selected image
  processImage(base64Image: string): void {
    this.loading = true;
    this.error = null;

    const url = `https://outline.roboflow.com/${this.model}/${this.version}`;
    const base64Data = base64Image.replace(/^data:image\/[a-z]+;base64,/, '');

    // Send image data to the model for processing
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

        if (this.imageRef?.nativeElement.complete) {
          this.drawMasks();
        }

        // Save the scan result to history
        this.saveScanToHistory(base64Image, this.imageResult);
      })
      .catch((error) => {
        this.error = 'Error processing image: ' + error.message;
        this.loading = false;
        console.error(error);
      });
  }

  // Save the processed scan to history
  // Save the processed scan to history
  saveScanToHistory(base64Image: string, result: any): void {
    const scanEntry = {
      timestamp: new Date().toISOString(),  // Change id to timestamp for easier sorting
      imageUrl: base64Image,  // Store the base64 image data
      predictions: result.predictions.map((p: any) => ({
        class: p.class,
        confidence: p.confidence,
        color: this.classColors[p.class] || 'rgba(150, 150, 150, 0.4)'  // Set color based on class
      })),
    };
  
    const existingHistory = JSON.parse(localStorage.getItem('scanHistory') || '[]');
    existingHistory.unshift(scanEntry);
    localStorage.setItem('scanHistory', JSON.stringify(existingHistory));
  }
  


  // This method will extract the canvas content as a base64 image (modified version)
  getModifiedImageUrl(): string {
    const canvas = this.canvasRef.nativeElement;
    return canvas.toDataURL('image/png');  // Get the canvas content as a base64-encoded PNG image
  }

  // Generate a summary of the scan results
  getScanSummary(result: any): string {
    const classes = result.predictions.map((p: any) => p.class);
    const counts = classes.reduce((acc: any, cls: string) => {
      acc[cls] = (acc[cls] || 0) + 1;
      return acc;
    }, {});
    return Object.entries(counts).map(([k, v]) => `${v}× ${k}`).join(', ');
  }

  // Handle image load event
  onImageLoad(img: HTMLImageElement): void {
    setTimeout(() => {
      this.updateCanvasSize();
      if (this.imageResult) {
        this.drawMasks();
      }
    }, 0);
  }

  // Update the canvas size based on the image dimensions
  updateCanvasSize(): void {
    const img = this.imageRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
  }

  // Filter out ignored classes and return the remaining class-color pairs
  get classColorEntries() {
    return Object.entries(this.classColors).map(([className, color]) => ({
      class: className,
      color: color
    })).filter(entry => !this.ignoredClasses.includes(entry.class));
  }

  // Draw the predicted masks on the canvas
  drawMasks(): void {
    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx || !this.imageResult) return;

    const img = this.imageRef.nativeElement;
    const originalWidth = this.imageResult.image.width;
    const originalHeight = this.imageResult.image.height;
    const displayWidth = img.clientWidth;
    const displayHeight = img.clientHeight;

    const xScale = displayWidth / originalWidth;
    const yScale = displayHeight / originalHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Iterate over the predictions and draw the masks on the canvas
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

      const color = this.classColors[className] || 'rgba(150, 150, 150, 0.4)';
      ctx.fillStyle = color;
      ctx.fill();
    });
  }
}
