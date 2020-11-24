import { Component, OnInit } from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms'
import { HttpClient } from '@angular/common/http';
import { SpinnerService } from 'src/app/spinner/spinner.service';
import { delay } from 'rxjs/operators';
import { NgbModalConfig, NgbModal } from '@ng-bootstrap/ng-bootstrap';

@Component({
  selector: 'app-formulario-ejecutivo',
  templateUrl: './formulario-ejecutivo.component.html',
  styleUrls: ['./formulario-ejecutivo.component.css']
})
export class FormularioEjecutivoComponent implements OnInit {
  formularioEjecutivo:FormGroup;
  showAlert = false;
  min_ammount: number;
  max_ammount: number;
  risk: string;

  constructor(private http: HttpClient, private spinnerService: SpinnerService, private modalService: NgbModal, config: NgbModalConfig) { 
    this.formularioEjecutivo = new FormGroup({
      'nombre': new FormControl('', [Validators.required, Validators.minLength(8)]),
      'Age': new FormControl('', [Validators.required]),
      'Sex': new FormControl('', [Validators.required]),
      'Housing': new FormControl('', [Validators.required]),
      'Saving accounts': new FormControl('', [Validators.required]),
      'Checking account': new FormControl('', [Validators.required]),
      'Duration': new FormControl('', [Validators.required]),
      'Purpose': new FormControl('', [Validators.required]),
      'correoElectronico': new FormControl('', [Validators.required, Validators.email])
    });
  }

  ngOnInit(): void {
  }
  guardar(content){
    this.spinnerService.requestStarted();
    console.log(this.formularioEjecutivo.value);
    var jsonformulario = this.formularioEjecutivo.value;

    this.http.post('http://127.0.0.1:5000/credit-request', jsonformulario).pipe(delay(3000)).toPromise().then(data => {
      // Successed responce with the server
      this.spinnerService.requestEnded();
      this.min_ammount = data['Min. ammount'];
      this.max_ammount = data['Max. ammount'];
      switch(data['Risk']){
        case 'low':
          this.risk= 'Bajo';
          break;
        case 'moderate':
          this.risk= 'Mediano';
          break;
        case 'high':
          this.risk= 'Alto';
          break;
        case 'risky':
          this.risk = 'Riesgoso';
          break;
        default:
          this.risk = data['risk'];
          break;
      }
      console.log(data)
      this.modalService.open(content);
    }).catch((e) => {
      // Failed responce with the server
      this.spinnerService.resetSpinner();
      console.log('handle error here instead', e)
    });

  }

}
