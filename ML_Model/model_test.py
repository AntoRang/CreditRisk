# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:27:44 2020

@author: jesus
"""

import numpy as np
import joblib

ejemplo = np.array([2.67716444, 0.66994484, 1.67683077, -0.13364358, -0.70614255,  1.8910083, 0.75438593, -1.45490372])
modelo = joblib.load('model_dt.joblib')
respuesta = (modelo.predict(ejemplo.reshape(1,-1)))[0]
val= int(respuesta)
print(type(respuesta))
print(type(val))
