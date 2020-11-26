import { RouterModule, Routes } from '@angular/router';

//Se tienen que importar todas las rutas que vamos a utilizar
import { PrincipalComponent } from './componentes/principal/principal.component';
import { EjecutivoComponent } from './componentes/ejecutivo/ejecutivo.component';
import { ClienteComponent } from './componentes/cliente/cliente.component';
import { FormularioComponent } from './componentes/formulario/formulario.component';
import { FormularioEjecutivoComponent } from './componentes/formulario-ejecutivo/formulario-ejecutivo.component';
import { LoginComponent } from './componentes/login/login.component';


const APP_ROUTES: Routes = [

  //Se inicializan todas las rutas que vamos a utilizar.
  { path: 'principal', component: PrincipalComponent },
  { path: 'ejecutivo', component: EjecutivoComponent },
  { path: 'cliente', component: ClienteComponent },
  { path: 'cliente/formulario', component: FormularioComponent},
  { path: 'login', component: LoginComponent },
  { path: 'ejecutivo/formulario-ejecutivo', component: FormularioEjecutivoComponent },
  { path: '**', pathMatch: 'full', redirectTo: 'principal' }

];

//Se tiene que importar APP_ROUTING en el archivo app.module.ts como un import,
//y dentro del arreglo de imports[]
export const APP_ROUTING = RouterModule.forRoot(APP_ROUTES);
