# Elementos necesarios

## NodeJS
Es necesario descargar [NodeJS](https://nodejs.org/en/)

En consola correr los siguientes comandos para verificar la instalación haya sido correctamente. Puede variar la versión si es que ya se tiene instalado.

`node --version`  

(Output) v12.16.1

`npm -v` 

(Output) 6.14.5

## Typescript
Instalar [Typescript](https://www.typescriptlang.org/) con el siguiente comando 

`npm install -g typescript@latest`

Validar la instalación con el siguiente comando `tsc -v`

## Angular
Instalar [Angular CLI](https://cli.angular.io/) para preparar el ambiente es necesario el siguiente comando (puede tardar varios minutos)

`npm install -g @angular/cli`

O si ya se tiene instalado, para actualizar, se utilizan los siguientes comandos

`npm uninstall -g angular-cli`

`npm cache verify`

`npm install -g @angular/cli@latest`

## Dirigirse a la carpeta credit-app

`cd credit-app`

Para correr la applicación es necesario correr el siguiente comando

`ng serve`

Después dirigirse al web browser de tu preferencia y dirigirse a la siguiente dirección `http://localhost:4200/`