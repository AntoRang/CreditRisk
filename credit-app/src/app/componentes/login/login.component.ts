import { Component, OnInit } from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';


@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {

  formulariologin: FormGroup;

  constructor() {
    this.formulariologin = new FormGroup({
      'usuario': new FormControl('', [Validators.required]),
      'contraseña': new FormControl('', [Validators.required]),
    });
   }

  ngOnInit(): void {
  }

}
