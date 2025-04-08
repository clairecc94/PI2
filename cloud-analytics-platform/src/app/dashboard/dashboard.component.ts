import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent {
  // List of AI pipeline components displayed on the dashboard
  components = [
    {
      icon: '📥',
      title: 'Data Loading',
      description: 'The pipeline begins by loading contrail and cloud images for training from the specified directory.'
    },
    {
      icon: '📁',
      title: 'Data Splitting',
      description: 'The dataset is split into training and validation sets. The validation set is moved to a separate folder for testing.'
    },
    {
      icon: '🔧',
      title: 'Model Training',
      description: 'The YOLOv11 model is trained on the labeled contrail and cloud images to learn object detection.'
    },
    {
      icon: '⚙️',
      title: 'Model Evaluation',
      description: 'After training, the model is evaluated using validation data to assess accuracy and performance.'
    },
    {
      icon: '🔍',
      title: 'Detection',
      description: 'The trained model is used to detect contrail and cloud objects in new test images.'
    },
    {
      icon: '📊',
      title: 'Visualization',
      description: 'Detection results are visualized, with the model output overlaid on the input images for inspection.'
    }
  ];
}
