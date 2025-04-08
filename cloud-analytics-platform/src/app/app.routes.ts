import { Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard.component';
import { RealTimeAnalysisComponent } from './real-time-analysis/real-time-analysis.component';
import { ScanHistoryComponent } from './scan-history/scan-history.component';
import { CloudDetectionComponent } from './pages/cloud-detection/cloud-detection.component';



// Définir les routes sous la constante APP_ROUTES
export const APP_ROUTES: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' }, // Redirection par défaut
  { path: 'dashboard', component: DashboardComponent },
  { path: 'real-time-analysis', component: RealTimeAnalysisComponent },
  { path: 'scan-history', component: ScanHistoryComponent },
  { path: 'cloud-detection', component: CloudDetectionComponent }
];
