import { Component, OnInit } from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms'

@Component({
  selector: 'app-formulario',
  templateUrl: './formulario.component.html',
  styleUrls: ['./formulario.component.css']
})
export class FormularioComponent implements OnInit {

  formulario:FormGroup;

  constructor() {
    this.formulario = new FormGroup({
      'nombre': new FormControl('', [Validators.required, Validators.minLength(8)]),
      'age': new FormControl('', [Validators.required]),
      'sex': new FormControl('', [Validators.required]),
      'housing': new FormControl('', [Validators.required]),
      'saving_account': new FormControl('', [Validators.required]),
      'checking_accounts': new FormControl('', [Validators.required]),
      'duration': new FormControl('', [Validators.required]),
      'purpose': new FormControl('', [Validators.required]),
      'correoElectronico': new FormControl('', [Validators.required, Validators.email])
    });
   }

  ngOnInit(): void {

  }
  guardar(){
    console.log(this.formulario.value);
  }

}
