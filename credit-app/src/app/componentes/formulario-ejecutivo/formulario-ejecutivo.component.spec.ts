import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { FormularioEjecutivoComponent } from './formulario-ejecutivo.component';

describe('FormularioEjecutivoComponent', () => {
  let component: FormularioEjecutivoComponent;
  let fixture: ComponentFixture<FormularioEjecutivoComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ FormularioEjecutivoComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(FormularioEjecutivoComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
