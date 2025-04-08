import { bootstrapApplication } from '@angular/platform-browser';
import { provideRouter } from '@angular/router';  // Importer provideRouter pour configurer le routeur
import { provideClientHydration } from '@angular/platform-browser';  // Optionnel, pour la gestion du client hydraté
import { AppComponent } from './app/app.component';  // Importer ton composant principal
import { APP_ROUTES } from './app/app.routes';  // Importer tes routes


// Bootstrapping de l'application avec la configuration du routeur
bootstrapApplication(AppComponent, {
  providers: [
    provideRouter(APP_ROUTES),  // Fournir les routes à Angular
    provideClientHydration(),   // Optionnel pour une gestion du côté client
  ]
}).catch(err => console.error(err));
