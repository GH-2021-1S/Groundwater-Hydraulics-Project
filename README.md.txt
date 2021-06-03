![Unal](http://occidente.co/wp-content/uploads/2017/11/Logo.png)
#Estimación de parámetros hidrodinámicos en acuíferos confinados
_Este es un proyecto desarrollado por estudiantes del curso Hidráulica Subterránea de la Universidad Nacional de Colombia - sede Bogotá. Esta herramienta, en base a información de pruebas de bombeo y recuperación en acuíferos confinados, permite filtrar datos anómalos; graficar pruebas diagnóstico; reconocer la existencia de efectos piel y almacenamiento; ajustar de manera automática los rangos en el tiempo donde es posible aplicar los métodos de Theis y Cooper-Jacob y finalmente estimar parámetros como la transmisividad y el almacenamiento del acuífero. _

### Pre-requisitos 📋
_Para hacer uso de la herramienta es necesario contar con un ordenador que soporte el lenguaje de programación [Python](https://www.python.org/ "Python") y se recomienda utilizar el entorno de Desarrollo integrado [Spyder](https://www.spyder-ide.org/ "Spyder"). Este programa utiliza las bibliotecas numpy, pandas, scipy y matplotlib, por lo que estas deben estar descargadas y habilitadas. Adicionalmente, es necesario contar con un software de edición de hojas de cálculo, como Microsoft Excel, que permita crear archivos csv donde se consignaran los datos de las pruebas de bombeo y recuperación. _

##Hacer Uso del Programa 🔧
_Para hacer uso del programa se recomienda seguir los pasos descritos a continuación: _

1.	Crear una carpeta donde se descargarán los archivos “prueba_trabajo.py”, “Datos_entrada_b.csv” y “Datos_entrada_r.csv”. 

2.	En base a los datos de la prueba de bombeo a analizar, se deben consignar los tiempos y descensos registrados en el archivo “Datos_entrada_b.csv” con la variable tiempo en días en la primera columna y los descensos en metros en la segunda columna (El archivo “Datos_entrada_b.csv” posee valores de prueba de bombeo a manera de ejemplo, estos deben ser borrados y reemplazados con los datos de la prueba de bombeo a analizar).

3.	De igual manera, si se tienen valores de prueba de recuperación, estos deben ser consignados en el archivo “Datos_entrada_r.csv” donde la primera columna es tiempo en días y la segunda descensos (Recuperación) en metros (No olvidar borrar y reemplazar valores ejemplo en el archivo; En caso de no tener valores de recuperación dejar el archivo vacío).

4.	 Se debe abrir el script “prueba_trabajo.py” en el software Spyder donde se deben introducir los valores de caudal de bombeo “Q” en metros por día (Línea 85) y el valor del radio al piezómetro r en metros (Línea 86). En caso de no tener valor de radio al piezómetro r, no será posible estimar el almacenamiento en el acuífero.

5.	El usuario debe elegir la función objetivo con la que se realizara la calibración teniendo la oportunidad de elegir entre la función NSE, R², Y MSE. Para elegir la función objetivo deseada debe modificarse la línea 317 en donde se define “n_fun_obj” para elegir la función NSE debe igualarse a cero, para elegir R² debe igualarse a 1 y para MSE debe igualarse a 2. 

6.	Finalmente, debe ejecutarse el programa el cual entregara 7 resultados. 

##Resultados 📌
1. Graficas diagnóstico de Datos observados (crudos) vs datos seleccionados con criterio Coeficiente de Variación.

2. Resultados de la calibración con la función objetivo seleccionada y la selección del mejor ajuste.

3. Sensibilidad de parámetros por dotty plots.

4. Análisis de incertidumbre.

5. Mapas de calor para sensibilidad de datos.

6. Análisis de efecto piel o almacenamiento comparando Theis vs observado.

7. Análisis de recuperación.

##Base Conceptual 📖

####Theis
$$s(r,t)=\frac{Q}{4\pi T}W(u)$$

$$u=\frac{r^2 S}{4Tt}$$

$$W(u)=-Ei(-u)$$
####Cooper-Jacob
$$s(r,t)=\frac{2.303Q}{4\pi T}Log(\frac{2.25 T t}{r^2 S})$$

####Derivada por el Metodo de Bourdet
$$\frac{\partial s}{\partial lnT} = \frac{(\Delta s\_{i-1}/  \Delta ln T\_{i-1})\Delta ln T\_{i +1}+(\Delta s\_{i+1}/  \Delta ln T\_{i+1})\Delta ln T\_{i-1}}{\Delta ln T\_{i-1}+\Delta ln T\_{i+1}} $$
####Coeficiente de Variación
$$CV=\frac{\sigma}{\bar{x}} $$
####Funciones objetivo
######NSE
Nash–Sutcliffe model efficiency coefficient

$$NSE=1-\frac{\sum\_{t=1}^{T}(Q\_{m}^{t}-Q\_{0}^{t})^2}{\sum_{t=1}^{T} (Q_0^t-\bar{Q_0})^2} $$
######R²
Coeficiente de determinación

$$R^2=1-\frac{\sigma_r^2}{\sigma^2}$$
######MSE
                    
$$MSE=\frac{1}{n}\sum_{i=1}^{n}(Y_i-\hat Y_i)^2 $$
## Autores ✒️
* **Iris Juliana Barreto** - *Estudiante Maestría en Recursos Hidráulicos - UNAL* 
* **Diego Ricardo Higuera** - *Estudiante Ingeniería Civil - UNAL*
* **Dónoban Steven Rojas** - *Estudiante Maestría en Recursos Hidráulicos - UNAL* 
* **Leonardo David Donado ** - *Profesor Titular Universidad Nacional de Colombia*

## Licencia 📄
_Puedes ejecutar, estudiar, copiar, modificar y mejorar esta herramienta, solo no te olvides de dar crédito y referenciar a sus autores. _
