import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { APP_ROUTES } from './app.routes';  // Importation correcte de APP_ROUTES
import { provideClientHydration, withEventReplay } from '@angular/platform-browser';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }), // Fournit la détection des changements de zone (optimisation des performances)
    provideRouter(APP_ROUTES),  // Fournit le routeur avec la configuration des routes définie dans app.routes.ts
    provideClientHydration(withEventReplay())  // Assure la "client hydration" pour le rendu côté client après SSR, avec replay des événements
  ]
};
