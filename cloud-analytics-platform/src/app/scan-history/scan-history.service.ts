import { Injectable } from '@angular/core';

export interface Scan {
  imageUrl: string;
  modifiedImageUrl: string;  // Ajout du champ pour l'image modifiée
  predictions: { class: string; confidence: number; color: string }[];
  timestamp: number;
}

@Injectable({
  providedIn: 'root',
})
export class ScanHistoryService {
  private scans: Scan[] = [];

  constructor() {}

  // Get all scans
  getScans(): Scan[] {
    return this.scans;
  }

  // Add a new scan to history with both original and modified images
  addScan(imageUrl: string, modifiedImageUrl: string, predictions: { class: string; confidence: number; color: string }[]): void {
    const scan: Scan = {
      imageUrl,
      modifiedImageUrl,  // Stocke l'image modifiée
      predictions,
      timestamp: Date.now(), // Store the timestamp when the scan is added
    };
    this.scans.push(scan);
  }
}
