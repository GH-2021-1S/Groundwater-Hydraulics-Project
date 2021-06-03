![Unal](http://occidente.co/wp-content/uploads/2017/11/Logo.png)


# Estimación de parámetros hidrodinámicos en acuíferos confinados

_Este es un proyecto desarrollado por estudiantes del curso Hidráulica Subterránea de la Universidad Nacional de Colombia - sede Bogotá. Esta herramienta, en base a información de pruebas de bombeo y recuperación en acuíferos confinados, permite filtrar datos anómalos; graficar pruebas diagnóstico; reconocer la existencia de efectos piel y almacenamiento; ajustar de manera automática los rangos en el tiempo donde es posible aplicar los métodos de Theis y Cooper-Jacob y finalmente estimar parámetros como la transmisividad y el almacenamiento del acuífero._

### Pre-requisitos 📋

_Para hacer uso de la herramienta es necesario contar con un ordenador que soporte el lenguaje de programación [Python](https://www.python.org/ "Python") y se recomienda utilizar el entorno de Desarrollo integrado [Spyder](https://www.spyder-ide.org/ "Spyder"). Este programa utiliza las bibliotecas numpy, pandas, scipy y matplotlib, por lo que estas deben estar descargadas y habilitadas. Adicionalmente, es necesario contar con un software de edición de hojas de cálculo, como Microsoft Excel, que permita crear archivos csv donde se consignaran los datos de las pruebas de bombeo y recuperación._

## Hacer Uso del Programa 🔧

_Para hacer uso del programa se recomienda seguir los pasos descritos a continuación:_

1.	Crear una carpeta donde se descargarán los archivos “prueba_trabajo.py”, “Datos_entrada_b.csv” y “Datos_entrada_r.csv”. 

2.	En base a los datos de la prueba de bombeo a analizar, se deben consignar los tiempos y descensos registrados en el archivo “Datos_entrada_b.csv” con la variable tiempo en días en la primera columna y los descensos en metros en la segunda columna (El archivo “Datos_entrada_b.csv” posee valores de prueba de bombeo a manera de ejemplo, estos deben ser borrados y reemplazados con los datos de la prueba de bombeo a analizar).

3.	De igual manera, si se tienen valores de prueba de recuperación, estos deben ser consignados en el archivo “Datos_entrada_r.csv” donde la primera columna es tiempo en días y la segunda descensos (Recuperación) en metros (No olvidar borrar y reemplazar valores ejemplo en el archivo; En caso de no tener valores de recuperación dejar el archivo vacío).

4.	 Se debe abrir el script “prueba_trabajo.py” en el software Spyder donde se deben introducir los valores de caudal de bombeo “Q” en metros por día (Línea 85) y el valor del radio al piezómetro r en metros (Línea 86). En caso de no tener valor de radio al piezómetro r, no será posible estimar el almacenamiento en el acuífero.

5.	El usuario debe elegir la función objetivo con la que se realizara la calibración teniendo la oportunidad de elegir entre la función NSE, R², Y MSE. Para elegir la función objetivo deseada debe modificarse la línea 317 en donde se define “n_fun_obj” para elegir la función NSE debe igualarse a cero, para elegir R² debe igualarse a 1 y para MSE debe igualarse a 2. 

6.	Finalmente, debe ejecutarse el programa el cual entregara 7 resultados. 

## Resultados 📌
1. Graficas diagnóstico de Datos observados (crudos) vs datos seleccionados con criterio Coeficiente de Variación.

2. Resultados de la calibración con la función objetivo seleccionada y la selección del mejor ajuste.

3. Sensibilidad de parámetros por dotty plots.

4. Análisis de incertidumbre.

5. Mapas de calor para sensibilidad de datos.

6. Análisis de efecto piel o almacenamiento comparando Theis vs observado.

7. Análisis de recuperación.

## Base Conceptual 📖

### Theis

![Theis](https://latex.codecogs.com/gif.latex?s%28r%2Ct%29%3D%5Cfrac%7BQ%7D%7B4%5Cpi%20T%7DW%28u%29)

![u](https://latex.codecogs.com/gif.latex?u%3D%5Cfrac%7Br%5E2%20S%7D%7B4Tt%7D)

![Wu](https://latex.codecogs.com/gif.latex?W%28u%29%3D-Ei%28-u%29)

### Cooper-Jacob

![cooper](https://latex.codecogs.com/gif.latex?s%28r%2Ct%29%3D%5Cfrac%7B2.303Q%7D%7B4%5Cpi%20T%7DLog%28%5Cfrac%7B2.25%20T%20t%7D%7Br%5E2%20S%7D%29 "cooper")

### Derivada por el Metodo de Bourdet

![Bourdet](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20s%7D%7B%5Cpartial%20lnT%7D%20%3D%20%5Cfrac%7B%28%5CDelta%20s_%7Bi-1%7D/%20%5CDelta%20ln%20T_%7Bi-1%7D%29%5CDelta%20ln%20T_%7Bi%20&plus;1%7D&plus;%28%5CDelta%20s_%7Bi&plus;1%7D/%20%5CDelta%20ln%20T_%7Bi&plus;1%7D%29%5CDelta%20ln%20T_%7Bi-1%7D%7D%7B%5CDelta%20ln%20T_%7Bi-1%7D&plus;%5CDelta%20ln%20T_%7Bi&plus;1%7D%7D)

### Coeficiente de Variación

![CV](https://latex.codecogs.com/gif.latex?CV%3D%5Cfrac%7B%5Csigma%7D%7B%5Cbar%7Bx%7D%7D)

### Funciones objetivo

#### NSE

Nash–Sutcliffe model efficiency coefficient

![NSE](https://latex.codecogs.com/gif.latex?NSE%3D1-%5Cfrac%7B%5Csum_%7Bt%3D1%7D%5E%7BT%7D%28Q_%7Bm%7D%5E%7Bt%7D-Q_%7B0%7D%5E%7Bt%7D%29%5E2%7D%7B%5Csum_%7Bt%3D1%7D%5E%7BT%7D%20%28Q_0%5Et-%5Cbar%7BQ_0%7D%29%5E2%7D)

#### R²

Coeficiente de determinación

![R²](https://latex.codecogs.com/gif.latex?R%5E2%3D1-%5Cfrac%7B%5Csigma_r%5E2%7D%7B%5Csigma%5E2%7D)

#### MSE
                    
![MSE](https://latex.codecogs.com/gif.latex?MSE%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28Y_i-%5Chat%20Y_i%29%5E2)

## Autores ✒️
* **Iris Juliana Barreto** - *Estudiante Maestría en Recursos Hidráulicos - UNAL* 
* **Diego Ricardo Higuera** - *Estudiante Ingeniería Civil - UNAL*
* **Dónoban Steven Rojas** - *Estudiante Maestría en Recursos Hidráulicos - UNAL* 
* **Leonardo David Donado** - *Profesor Titular Universidad Nacional de Colombia*

### Licencia 📄
_Puedes ejecutar, estudiar, copiar, modificar y mejorar esta herramienta, solo no te olvides de dar crédito y referenciar a sus autores._
