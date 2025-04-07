import { Component } from '@angular/core';

@Component({
  selector: 'app-cloud-detection',
  templateUrl: './cloud-detection.component.html',
  styleUrls: ['./cloud-detection.component.scss']  // ✅ Utilise bien SCSS
})
export class CloudDetectionComponent {
  // Indices des curseurs
  sunnyIndex: number = 0;
  cloudyIndex: number = 0;

  // Données pour les images "sunny" et "cloudy"
  timeStampsSunny: string[] = [];     // Optionnel : à remplir si tu veux afficher les timestamps
  skyImagesSunny: string[] = [];      // Chemins des images "sunny"
  timeStampsCloudy: string[] = [];    // Optionnel : à remplir si tu veux afficher les timestamps
  skyImagesCloudy: string[] = [];     // Chemins des images "cloudy"

  // Vidéos et PV output
  videoSunny: string = 'assets/videos/sunny_video.mp4';
  videoCloudy: string = 'assets/videos/cloudy_video.mp4';

  constructor() {
    // Initialiser les chemins d'images sunny
    for (let i = 0; i < 1779; i++) {
      this.skyImagesSunny.push(`assets/sunny/sunny_${i}.png`);
      this.timeStampsSunny.push(`Timestamp ${i}`); // Optionnel : à remplacer par des vrais timestamps
    }

    // Initialiser les chemins d'images cloudy
    for (let i = 0; i < 1605; i++) {
      this.skyImagesCloudy.push(`assets/cloudy/cloudy_${i}.png`);
      this.timeStampsCloudy.push(`Timestamp ${i}`); // Optionnel
    }
  }

  // Récupère le chemin d'une image en fonction de l'index et du type
  getImagePath(index: number, type: string): string {
    if (type === 'sunny') {
      return this.skyImagesSunny[index] || '';
    } else if (type === 'cloudy') {
      return this.skyImagesCloudy[index] || '';
    }
    return '';
  }

  // Mise à jour index sunny
  onSunnyIndexChange(event: Event) {
    const target = event.target as HTMLInputElement;
    this.sunnyIndex = parseInt(target.value, 10);
  }

  // Mise à jour index cloudy
  onCloudyIndexChange(event: Event) {
    const target = event.target as HTMLInputElement;
    this.cloudyIndex = parseInt(target.value, 10);
  }
}
