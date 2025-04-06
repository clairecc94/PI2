import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-real-time-analysis',
  imports: [CommonModule],
  templateUrl: './real-time-analysis.component.html',
  styleUrls: ['./real-time-analysis.component.scss'],
})
export class RealTimeAnalysisComponent {
  imageUrl: string | ArrayBuffer | null = null;

  // Method to handle file selection
  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        this.imageUrl = reader.result;
      };
      reader.readAsDataURL(file);
    }
  }
}
