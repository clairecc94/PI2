import { ComponentFixture, TestBed } from '@angular/core/testing';
import { RealTimeAnalysisComponent } from './real-time-analysis.component';

describe('RealTimeAnalysisComponent', () => {
  let component: RealTimeAnalysisComponent;
  let fixture: ComponentFixture<RealTimeAnalysisComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RealTimeAnalysisComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RealTimeAnalysisComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
