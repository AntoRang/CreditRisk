# -*- coding: utf-8 -*-
"""Evidencia2Código_Equipo#5

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19f0zUqRfGipvffUQhZ9wI0_1EjG_BSZB

#Data Mining & Modelo de Aprindizaje: Equipo #5 - Riesgo Crediticio

El siguiente notebook presenta la extracción de información (data mining, así como la selección, entrenamiento y optmización del modelo de aprendizaje automaplicable para nuestro proyecto: Riesgo Crediticio. 

Las etapas que se explican dentro del notebook siguen una estructura similar a la metodología CRISP:
1. Comprensión del negocio 
2. Comprensión de los datos (Selección de tipo de tarea)
3. Preparación de los datos (Datamining)
4. Modelado de la información (Selección de algoritmo de M.L.)
5. Evaluación del modelo (Optimización) 

Es importante mecionar que la base de datos se obtuvo del siguiente enlace (de uso gratuito): https://www.kaggle.com/nitinkhandagale/credit-risk-with-label


Así mismo, mencionar que el código que fungió como esquema se encuentra en el repositiorio de la siguiente liga: https://github.com/akhil12028/Bank-Marketing-data-set-analysis

##1. Comprensión del Negocio

Debido a la situación actual, muchas personas por diferentes razones se ven en la necesidad de pedir un crédito en el banco. Según el INEGI, en un estudio de Créditos y cuentas bancarias de las unidades económicas, reporta que cada vez más las unidades económicas recurren a un crédito o financiamiento con las instituciones financieras, por lo que la gama de opciones de préstamos se adecuan a las necesidades del cliente, mediante el uso y manejo de cuentas bancarias. En 2014, se pudo observar una distribución de las unidades económicas según su condición de obtención de créditos, donde el 16.5% sí obtuvo acceso al financiamiento, y en su contraparte, el 83.5% no accedió a ningún tipo de crédito, préstamo o financiamiento. El número de unidades económicas que no contó con crédito bancario fue de 3 527 571, donde se destaca que 54.9% no lo necesitaron, 32.4% no contaron con un crédito bancario por los altos intereses, 14.2% por otras causas y 7.4% lo solicitaron, pero no cumplieron con los requisitos.

Por su parte, los bancos tienen la problemática de poder atender a la alta demanda de gente y con poco personal para hacerlo de forma correcta. Por lo cual, los bancos tienen la necesidad de poder clasificar los tipos de créditos de forma eficiente con los recursos disponibles en este momento.
Sin embargo, existe la herramienta denominada aprendizaje automático (Machine Learning), la cual es es una subárea de la Inteligencia Artificial, que ha demostrado una alta eficiencia en situaciones que requieran entender, visualizar y comprender información de forma apropiada. De esta manera, la problemática podría resumirse en la necesidad de optimizar el proceso de selección de solicitudes de préstamos crediticios por parte de entidades bancarias, con el fin de otorgar recursos a las personas “más apropiadas” según los parámetros de medición.

Con base en la información anterior, se ve como un área de oportunidad el uso de algoritmos de aprendizaje automático para automatizar el proceso de selección las candidaturas de solicitud de préstamos crediticios, optimizando el proceso que hoy se hace en en las entidades bancarias. 
Se podría utilizar tanto por entidades bancarias, como por personas físicas. El primer grupo se vería beneficiado al clasificar a las personas que cumplan satisfactoriamente los requisitos, optimizando los tiempos de resolución de las solicitudes. En su contraparte, las personas físicas lo podrían utilizar para corroborar que su documentación es suficiente para comenzar su solicitud de manera formal, con esto podrán tener una mayor probabilidad de éxito de aprobación.

En el caso de las entidades bancarias, estas podrían optimizar su proceso de selección, dejando a los analistas en otras áreas con mayor demanda. En el caso de las personas físicas, les ayudaría en dos sentidos, en primer lugar, ahorrarían tiempo en sus procesos de solicitud; así mismo, su historial crediticio no se perjudica al no tener solicitudes denegadas, pues tendrían la oportunidad de corregir sus documentos antes de iniciar su proceso.

Por tal motivo, se plantea la posibilidad de desarrollar aplicación web donde se requiera ingresar ciertos parámetros, y con base en esa información puntual, plantear un porcentaje de aceptación de la solicitud de crédito, así mismo, se desplegará la información que señale las razones por la cual se decretó esa decisión. 
La información que proyectaría el resultado dentro de la plataforma incluirá los siguientes puntos:

●	Clasificación del riesgo por la cantidad de préstamo solicitado: Riesgoso o No riesgoso.

●	Los factores influyeron en la decisión (por ejemplo género, uso, cantidad de ahorros, etc.).

●	Porcentaje de fiabilidad de la solución.

##2. Comprensión de los Datos

Como se mencionó previamente, la información de estudio toma como punto de partida la base de datos que se encuentran en el siguiente enlace: https://www.kaggle.com/nitinkhandagale/credit-risk-with-label

Dentro del mismo sitio web de Kaggle se puede apreciar un resumen de los datos en la pestaña con el título Column. Es de ahí de donde se parte la primera extracción de la información referente a los tipos de datos.

### *Exploración y Descripción de los Datos*

Se puede apreciar que existen en total existen 10 parámetros, más su índice. A continuación, se presenta la clasificación de los parámetros dependiendo del tipo de dato que manejan.

**Datos Cualitativos**

  ●	Datos Nominales

*   Sex: Male/Female

●	Datos Ordinales

*   Housing: Free/ Own / Rent
*   Saving accounts: Little/ Moderate / Quite Rich / Rich
*   Checking accounts: Little / Moderate / Rich
*   Purpose
*   Risk: Bad/Good

**Datos Cuantitativos**

●	Datos de Numéricos

*   Age
*   Credit amount
*   Duration
*   Job


### *Tarea a Desarrollar*

Con base en la información que se planteó en la sección Comprensión del Negocio, y en los tipos de datos, se puede definir la tarea para este proyecto, como un ejercicio de clasificación. Definirlo de esta manera es lo más conveniente, ya que el algoritmo de aprendizaje automático nos permitirá binarizar el resultado final, convirtiéndolo en un tipo de crédito Riesgoso o No riesgoso, tal cómo se tiene planteado en el objetivo del proyecto. Este tipo de tarea es adecuado para un aprendizaje automático supervisado, ya que nuestro dataset ya viene etiquetado con el tipo de riesgo y podemos aprovechar esta información para hacer un modelo de clasificación.

##3. Preparación de los Datos

La limpieza de los datos se hará de forma en que se permita extraer la información más relevante y dearrollar un buen algoritmo de M.L.

No obsante, en esta primera sección se estudiarán los elementos de manera general. En el apartado de "Evaluación de modelo" se modificarán denuevo los datos de forma que se genere el mejor 'accuracy' posible.

###3.1. Importe de Librerias y declaración de funciones
"""

#Instalación de libreria "Pandas_profiling"
# ! pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

# Commented out IPython magic to ensure Python compatibility.
#Librerias necesarias: 'pandas', 'numpy', 'seaborn', 'matplotlib', ''google' & 'sklea 

# import pandas_profiling
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing

# from google.colab import drive
import matplotlib.pyplot as plt
from sklearn import metrics as sk_metrics
# %matplotlib inline

#Función para hacer la matriz de confusión, se necesita el atributo de salida, el modelo de GridSearch y el nombre del modelo.
def make_mc(target, model, name): 
    labels_Y = []
    labels = target.value_counts().index
    for i in labels:
        labels_Y.append(i)

    # Calculamos la matriz de confusión 
    confusion_matrix = sk_metrics.confusion_matrix(Y, model.predict(X))

    df_confusion_matrix = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

    # mostramos los resultados de la matriz de confusión
    plt.figure(figsize=(8, 8))
    sns.heatmap(df_confusion_matrix, annot=True, cbar=False, cmap='Oranges', linewidths=1, linecolor='black')
    plt.xlabel('Etiquetas predichas', fontsize=15)
    plt.xticks(fontsize=16)
    plt.ylabel('Etiquetas verdaderas', fontsize=15)
    plt.yticks(fontsize=16)
    plt.title(name)

#Función para transformar de parámetros nominales/ordinales a valores numéricos
#Función para transformar de parámetros nominales/ordinales a valores numéricos
def categorize(data):
    df = data.copy()
    LE = preprocessing.LabelEncoder()
    df['Sex'] = LE.fit_transform(df['Sex'])
    df['Housing'] = LE.fit_transform(df['Housing'])
    df['Saving accounts'] = LE.fit_transform(df['Saving accounts'])
    df['Checking account'] = LE.fit_transform(df['Checking account'])
    df['Purpose'] = LE.fit_transform(df['Purpose'])
    df['Risk'] = LE.fit_transform(df['Risk'])
    df["Credit category"] = df["Credit category"].replace(['low', 'moderate', 'high', 'risky'], [0,1,2,3])
    return df

#Función para calcular el 'information value' que ayuda a eliminar columnas inescesarias
def calc_iv(df, feature, target):
  lst = []
  for i in range(df[feature].nunique()):
    val = list(df[feature].unique())[i]
    lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])
  data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Bad'])
  data = data[data['Bad'] > 0]
  data['Share'] = data['All'] / data['All'].sum()
  data['Bad Rate'] = data['Bad'] / data['All']
  data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
  data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
  data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
  data['IV'] = (data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])).sum()
  data = data.sort_values(by=['Variable', 'Value'], ascending=True)
  return data['IV'].values[0]

#Función para eliminar 'outliers'
def remove_outliers(df, column , min, max):
  outlier = df[(df[column] >= max)|(df[column] <= min)].index
  df.drop(outlier, inplace=True)
  return df

# Función para normalizar campos númericos con la estandarización z-score 
def z_score(feature):
    new_feature = (feature - feature.mean()) / feature.std()
    return new_feature

# Función para normalizar campos númericos con la estandarización simple feature scaling 
def simple_feature_scaling(feature):
    new_feature = feature / feature.max()
    return new_feature

# Función para normalizar campos númericos con la estandarización min-max
def simple_feature_scaling(feature):
    new_feature = (feature - feature.min()) / (feature.max() - feature.min())
    return new_feature

"""###3.2. Extracción de Datos"""

#Carga los archivos de nuestro Google Drive en el notebook de Colab
# drive.mount('/content/drive/')

#Ruta del archivo donde esta la base de datos
#Se por la ruta donde se ubique el archivo
path = '../Dataset/Riesgo Crediticio.csv'

#Guarda la información del archivo en la variable 'df_riesgo'
#Configura que todos los valores 'nulos' se aprecien como 'NaN'
df = pd.read_csv(path, na_values =['NA'])


"""###3.3. Modificación de los Datos"""

#Eliminar la columna 'Unamed' que no incluye información
df.drop(['Unnamed: 0'], axis=1, inplace=True)


#Clasificación en cuatro categorías del 'Credit amount': bajo, moderado, alto, riesgoso
tipo_riesgo=['low','moderate','high','risky']
credito= pd.cut(df["Credit amount"], bins=len(tipo_riesgo), labels=tipo_riesgo)
df['Credit category']= pd.Series.to_frame(credito) 


#Valores en la separación de 'Credit category'
bins_credito = np.linspace(min(df["Credit amount"]), max(df["Credit amount"]), len(tipo_riesgo)+1)


#Conteo de valores nulos
# df.isnull().sum()

#Adicición de categoría 'n/a' para rellenar datos nulos de los parámetros 'Checking account' y 'Saving accounts'
df["Checking account"] = df["Checking account"].fillna("n/a")
df["Saving accounts"] = df["Saving accounts"].fillna("n/a")

#Verificación del intercambio de datos nulos
# df.isnull().sum()

"""###3.4. Visualización de los Datos"""

#Conteo del numeros total de información
# df.shape

#Gráficas con la cuenta de los valores de las respectivas clases 
#Histogramas para valores numéricos
#Gráfico de Barra para valores categóricos/nominales

#Age
# sns.set(style="ticks", color_codes=True)
# sns.distplot(df['Age'])

# #Duration
# sns.set(style="ticks", color_codes=True)
# sns.distplot(df['Duration'])

# #Credit amount
# sns.set(style="ticks")
# sns.distplot(df["Credit amount"])

# #Sex
# sns.set(style="ticks")
# sns.countplot(x='Sex', data=df)

# #Job
# sns.set(style="ticks")
# sns.countplot(x='Job', data=df)

# #Housing
# sns.set(style="ticks")
# sns.countplot(x='Housing', data=df)

# #Saving Accounts
# sns.set(style="ticks")
# sns.countplot(x='Saving accounts', data=df)

# #Checking Account
# sns.set(style="ticks")
# sns.countplot(x='Checking account', data=df)

# #Purpose
# sns.set(style="ticks")
# sns.countplot(x='Purpose', data=df)

# #Risk
# sns.set(style="ticks")
# sns.countplot(x='Risk', data=df)

# #Credit ammount category
# sns.set(style="ticks")
# sns.countplot(x='Credit category', data=df)

#Age category
#sns.set(style="ticks")
#sns.countplot(x='Age category', data=df)

#Duration category
#sns.set(style="ticks")
#sns.countplot(x='Duration category', data=df)

#Observar el rango de cada clase de los parámetros: 'Credit amount category', 'Age category', 'Duration category'
#Credit amount category
clases_credito= pd.cut(df["Credit amount"], bins=len(tipo_riesgo)).value_counts()
# clases_credito

#Age category
#clases_edad= pd.cut(df["Age"], bins=len(rango_edad)).value_counts()
#clases_edad

#Duration category
#clases_duracion= pd.cut(df["Duration"], bins=len(rango_duracion)).value_counts()
#clases_duracion

#Descripción estadística de los parámetros (tanto numéricos, como nominales y ordinales)
# df.describe(include='all')

"""###3.5. Normalización a Datos Numéricos"""

#Conversión a valores numéricos
df = categorize(df)


#Normalizar valores con z-score
df['Age'] = z_score(df['Age'])
df['Sex'] = z_score(df['Sex'])
df['Job'] = z_score(df['Job'])
df['Housing'] = z_score(df['Housing'])
df['Saving accounts'] = z_score(df['Saving accounts'])
df['Checking account'] = z_score(df['Checking account'])
df['Credit amount'] = z_score(df['Credit amount'])
df['Duration'] = z_score(df['Duration'])
df['Purpose'] = z_score(df['Purpose'])
df['Risk'] = z_score(df['Risk'])


"""###3.6. Visualización de Parámetros Innescesarios"""

#Conteo total de datos


#Relación entre los datos
#pandas_profiling.ProfileReport(df)


#Obtención de 'information value' de cada parámetro con respecto a 'Credit category'

"""###3.7. Eliminación de Outliers"""


#Eliminación de outliers en los parámetros pertinentes: 'Age' & 'Credit amount'
df = remove_outliers(df=df, column='Age' , min = df["Age"].min(), max=65)
df = remove_outliers(df=df, column='Duration', min = df["Duration"].min(), max=60)
df = remove_outliers(df=df, column='Credit amount' , min= df["Credit amount"].min(), max=17000)


#Reordenar los índices a partir de 1
orden_id = pd.Series(range(1, df.shape[0]+1 ))
df = df.set_index(orden_id)


"""###3.8. Datos Finales"""

#Eliminación de columnas: 'Job', 'Age', 'Duration', Risk' y 'Credit amount'
df = df.drop(columns=['Risk', 'Credit amount'])

df.to_csv('../Dataset/CreditRiskETL.csv')