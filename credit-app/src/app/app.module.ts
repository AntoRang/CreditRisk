import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { FormularioComponent } from './componentes/formulario/formulario.component';

import { ReactiveFormsModule } from '@angular/forms';
import { MatSelectModule } from '@angular/material/select';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

import { HttpClientModule } from '@angular/common/http';
import { SpinnerComponent } from './spinner/spinner.component';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { EncabezadoComponent } from './componentes/comunes/encabezado/encabezado.component';
import { PrincipalComponent } from './componentes/principal/principal.component';
import { EjecutivoComponent } from './componentes/ejecutivo/ejecutivo.component';
import { ClienteComponent } from './componentes/cliente/cliente.component';
import { APP_ROUTING } from './app.routes';
import { FormularioEjecutivoComponent } from './componentes/formulario-ejecutivo/formulario-ejecutivo.component';

@NgModule({
  declarations: [
    AppComponent,
    FormularioComponent,
    SpinnerComponent,
    EncabezadoComponent,
    PrincipalComponent,
    EjecutivoComponent,
    ClienteComponent,
    FormularioEjecutivoComponent
  ],
  imports: [
    BrowserModule,
    ReactiveFormsModule,
    MatSelectModule,
    BrowserAnimationsModule,
    HttpClientModule,
    NgbModule,
    APP_ROUTING
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
