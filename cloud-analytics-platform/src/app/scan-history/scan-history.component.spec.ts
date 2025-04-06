import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ScanHistoryComponent } from './scan-history.component';

describe('ScanHistoryComponent', () => {
  let component: ScanHistoryComponent;
  let fixture: ComponentFixture<ScanHistoryComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ScanHistoryComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ScanHistoryComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
