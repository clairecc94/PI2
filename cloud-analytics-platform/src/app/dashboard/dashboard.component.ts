import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  imports: [CommonModule],
  selector: 'app-dashboard',
  standalone: true,
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent {
  // Properties related to the dashboard logic
  solarEnergy = true;
  cloudCover = true;
  cloudType = true;
  windSpeed = true; // Wind speed
  humidity = true; // Humidity
  precipitationRisk = true; // Precipitation risk

  // Placeholder data for display
  solarEnergyInfo: string = "The available solar energy today is 450 W/m² with an expected 10% increase in the afternoon.";
  cloudCoverInfo: string = "The cloud cover is at 60%, with light shower risks in the evening.";
  cloudTypeInfo: string = "Cumuliform clouds are increasing, with a forecast of greater coverage in the afternoon.";
  windSpeedInfo: string = "Wind speed: 20 km/h, mainly blowing from the southwest.";
  humidityInfo: string = "Current humidity: 75%, with a risk of condensation in the late afternoon.";
  precipitationRiskInfo: string = "Precipitation risk: 40% with thunderstorms expected in the afternoon.";

  // Data for solar energy
  solarEnergyWatts: number = 450;  // Example value
  solarEnergyIncrease: number = 10;  // Example increase

  toggleSection(section: string) {
    if (section === 'solarEnergy') {
      this.solarEnergy = !this.solarEnergy;
    } else if (section === 'cloudCover') {
      this.cloudCover = !this.cloudCover;
    } else if (section === 'cloudType') {
      this.cloudType = !this.cloudType;
    } else if (section === 'windSpeed') {
      this.windSpeed = !this.windSpeed;
    } else if (section === 'humidity') {
      this.humidity = !this.humidity;
    } else if (section === 'precipitationRisk') {
      this.precipitationRisk = !this.precipitationRisk;
    }
  }
}
