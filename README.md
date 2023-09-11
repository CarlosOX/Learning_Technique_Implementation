# Learning_Technique_Implementation

Para este proyecto, creé una técnica de regresión lineal utilizando un conjunto de datos de Kaggle en el que no utilicé marcos de trabajo como TensorFlow o scikit-learn.
Hice uso del paradigma de Programación Orientada a Objetos (POO), la biblioteca pandas y numpy para programar el algoritmo desde cero. Además, utilicé matplotlib para visualizar
los resultados de la implementación.

El conjunto de datos utilizado para probar el algoritmo tiene como objetivo predecir la relación entre la altura y el peso de los atletas olímpicos, con estos datos probaremos
nuestra regresión lineal. 

La explicación a detalle del modelo se encuentra en el archivo LinearRegression.py donde cada línea de código esta comentada para su entendimiento. En general la implementación
consta de la siguiente forma:

 ### Crear una Clase LinearRegression donde se reciban los hiperparámetros (learning rate, epochs, b)

  Dentro de esta clase, tendremos las siguientes funciones:
  
      -  set_data(self,x_train,y_train)  <-  Esta funcion se encarga de preparar los datos para el algoritmo, recibe los datos de entrenamiento y asigna pesos aleatorios para los parametros
      
      -  h(self,params,x_train)  <- Esta funcion hace la parte h(x):wx + wx2 ... del gradient descent 
      
      -  GD(self)  <-  Esta funcion hace el proceso de gradient descent para todos nuestros parametros,pero solamente los ajusta una vez.
      
      -  show_errors(self) <- Esta funcion es para visualizar el error entre cada epoch y verificar si se reduce o no
      
      -  fit(self) <- Esta funcion hace todo el proceso de optimizacion completo, ya que las funciones anteriores se hicieron para una corrida
      
      -  predict(self,x_test,y_test) <- En esta funcion hacemos un predict de datos nuevos para ver que tan efectivo es el algoritmo

  #### Posteriormente, se crea un objeto de linearRegression donde usamos nuestos datos de entrenamiento y de prueba. Finalmente observamos la gráfica del error a lo largo de las épocas y nuestra linea de regresión con respecto a todos los datos. 
