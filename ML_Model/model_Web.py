"""Vinculación con flask"""
import flask
import numpy as np
import joblib

app= flask.Flask(__name__)

@app.route('/prueba/<dato>',methods=['POST'])
def ejemplo_servicio(dato):
    contenido=flask.request.json
    print(contenido)
    ejemplo=np.array([2.67716444, 0.66994484, 1.67683077, -0.13364358, -0.70614255,  1.8910083, 0.75438593, -1.45490372])
    modelo = joblib.load('model_dt.joblib')
    respuesta = modelo.predict(ejemplo.reshape(1,-1))
    print(respuesta)
    return flask.jsonify({'resultad':respuesta[0]})
