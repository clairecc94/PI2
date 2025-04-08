import { Component } from '@angular/core';
import { Router } from '@angular/router';  // Importer le Router pour la navigation
import { RouterModule } from '@angular/router';
import { APP_ROUTES } from './app.routes';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  standalone: true,
  imports: [RouterModule]
})
export class AppComponent {
  title = 'Cloud Analytics Platform';

  constructor(private router: Router) {}

  navigateTo(route: string): void {
    this.router.navigate([route]); // Navigation vers la route spécifiée
  }
}
