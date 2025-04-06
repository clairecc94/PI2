import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-scan-history',
  templateUrl: './scan-history.component.html',
  styleUrls: ['./scan-history.component.scss'],
  imports: [CommonModule]
})
export class ScanHistoryComponent implements OnInit {
  scanHistory: any[] = [];  // Tableau contenant l'historique des scans

  ngOnInit(): void {
    // Simulate scan history
    this.scanHistory = [
      { id: 1, date: '2024-12-01', result: 'Analysis successful', description: 'Everything is in order, no anomalies detected.' },
      { id: 2, date: '2024-12-02', result: 'Analysis in progress', description: 'The analysis is in progress, results will be available shortly.' },
      { id: 3, date: '2024-12-03', result: 'Anomaly detected', description: 'An anomaly has been found, verification required.' },
      { id: 4, date: '2024-12-04', result: 'Analysis successful', description: 'Scans were completed successfully without errors.' },
      { id: 5, date: '2024-12-05', result: 'Analysis in progress', description: 'The data is currently being processed.' },
    ];
  }

  viewScanDetails(scan: any): void {
    // Logique pour afficher plus de détails d'un scan
    console.log('Affichage des détails pour le scan:', scan);
    // Tu pourrais ajouter ici un affichage modale ou une redirection vers une page détaillant le scan
  }
}
