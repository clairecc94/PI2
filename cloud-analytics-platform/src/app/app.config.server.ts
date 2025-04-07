import { mergeApplicationConfig, ApplicationConfig } from '@angular/core';
import { provideServerRendering } from '@angular/platform-server';
import { provideServerRoutesConfig } from '@angular/ssr';
import { appConfig } from './app.config';
import { serverRoutes } from './app.routes.server';

const serverConfig: ApplicationConfig = {
  providers: [
    provideServerRendering(),  // Fournir le rendu côté serveur pour SSR
    provideServerRoutesConfig(serverRoutes)  // Configurer les routes côté serveur pour SSR
  ]
};

export const config = mergeApplicationConfig(appConfig, serverConfig);  // Fusionner les configurations côté client et côté serveur
