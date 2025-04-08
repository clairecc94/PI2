import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  standalone: true,
  selector: 'app-scan-history',
  imports: [CommonModule, FormsModule],
  templateUrl: './scan-history.component.html',
  styleUrls: ['./scan-history.component.scss'],
})
export class ScanHistoryComponent implements OnInit {
  scans: Scan[] = [];  // Declare scans array with Scan type

  ngOnInit(): void {
    // Load scan history from localStorage (or an initial empty array)
    const storedScans = localStorage.getItem('scanHistory');
    if (storedScans) {
      this.scans = JSON.parse(storedScans);
    }
  }

  // Clear the scan history and update localStorage
  clearHistory(): void {
    this.scans = [];
    localStorage.removeItem('scanHistory'); // Remove scan history from localStorage
  }
}

// Define the Scan interface outside of the component class
export interface Scan {
  timestamp: string;
  imageUrl: string;
  predictions: { class: string, confidence: number, color: string }[];
}
