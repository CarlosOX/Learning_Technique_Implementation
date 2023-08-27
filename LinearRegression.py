import numpy as np  # importing numpy to work with arrays and matrix multiplication
import matplotlib.pyplot as plt # importing matplotlib to visualize the algorithm behavior
import random  # importing random to set random values to our parameters
import pandas as pd  # importing pandas to load the dataset


"""""
Esta clase busca tener una estructura similar a como se
implementaría un modelo con un framework o una librería como scikit learn

"""""
class LinearRegression:
    def __init__(self, learning_rate=0.00001, max_epochs=100, b = 0):   # Aqui el objeto recibe como parámetros iniciales el learning rate y las epochs
        self.lr = learning_rate
        self.epochs = max_epochs    # Inicializamos las variables que se utilizarán en todas las funciones de la clase-
        self.params =[] # Parametros
        self.b = b
        self.errors = []  # Variable que guarda los errores a lo largo de los epochs

        self.x_train = None  #Variable donde guardamos nuestra variable x del  dataset
        self.y_train= None    #Variable donde guatdamos nuestra variable y del dataset
        self.x_test = None
        self.y_test = None

        self.fig = None   # Esta es para crear la figura de las graficas de matplotlib
    
    def set_data(self,x_train,y_train):     #Esta funcion se encarga de preparar los datos para el algoritmo, recibe los datos de entrenamiento
        self.x_train=x_train.to_numpy()    # Transformamos el dataframe en un arreglo numpy para su manipulacion
        self.y_train=y_train.to_numpy()   # Lo mismo para los datos en y 
        for i in range(len(self.x_train[0])):    #Aqui inicializamos los parametros de forma aleatoria
            self.params.append([random.random()])# El numero de parametros esta definido por cuantas x hay (no cuantos datos)
        print("Parametros iniciales: ",self.params)           # la guardamos en una lista de listas para que el producto punto de h(x) salga bien
                       
            
            
    def h(self,params,x_train):  #Esta funcion hace la parte h(x):wx + wx2 ... del gradient descent 
    
        h_x = np.dot(x_train,params)
        h = h_x + self.b
        return h# Hacemos un producto punto que va a devolver una lista con las sumas de h(x) de cada uno de nuestros valores del dataset.
        
    
    def GD(self):  # Esta funcion hace el proceso de gradient descent para todos nuestros parametros,pero solamente los ajusta una vez.
                   #para obtener los nuevos valores de los parametros utilice la siguiente ecuacion :
                  #  theta_new= theta - ( a  / m *sumatoria[(h(xi)-y)xi] )          
        temp=list(self.params) #Creamos una lista donde guardamos los parametros 
        h_x=self.h(self.params,self.x_train) # Realizamos el proceso de h(x) mencionado anteriormente
        resta =h_x-self.y_train #Aqui hago la resta de h(xi)-y, reste las listas para evitar hacer ciclos for
        x_acum=np.dot(resta.T,self.x_train) #Aqui hacemos una multiplicacion de la transpuesta de la resta(para que se sumen todas las restas de las variable dependientes) para todas las x, estamos haciendo ya la sumatoria completa para todos los datos
        x_acum_pot = self.lr/len(self.x_train) # aqui sacamos la parte de la ecuacion por separado(a  / m ) ya que es un unico valor
        theta_new = x_acum_pot * x_acum.T  # aqui ya obtenemos el nuevo valor de nuestros parametros (a  / m *sumatoria[(h(xi)-y)xi] )
        temp = self.params-theta_new # aqui ya hacemos la resta para obtener el nuevo parametro en una variable, en este caso temp
        #print(theta_new)
        tem=list(temp)  # Haremos una lista de nuestros parametros actualizados
        return temp # esta funcion retorna esos parametros ya que la utilizaremos en fit para irlos ajustando por cada epoch
        
    def show_errors(self):   #Esta funcion es para visualizar el error entre cada epoch y verificar si se reduce o no
                            # Para eso utilice la ecuacion de Mean Square Error -> 1/n (sumatoria(Y-Y_real)**2)
        h_x=self.h(self.params,self.x_train)   # obtenemos nuestro valor y, el cual es h(x)
        error_acum =0
        error = h_x-self.y_train   # Obtenemos la diferencia entre la y real y la y estimada
        cuadrados = error**2    #Obtenemos el cuadrado de los errores
        error_acum = np.sum(cuadrados)   #Hacemos la sumatoria de los errores de todos los datos
        mean_error_param=error_acum/len(self.x_train) # obtenemos el promedio de los errores 
        self.errors.append(mean_error_param)  # lo guardamos en una lista que obtendra todos los errores para poder visualizarlos despues
       
    
    def fit(self):  #Esta funcion hace todo el proceso de optimizacion completo, ya que las funciones anteriores se hicieron para una corrida
        
        for i in range (self.epochs):    # creamos un ciclo de las epochs y llamamos la funcion de optimizacion Gradient Descent
            self.params=self.GD()
 
            self.show_errors()              #Igual obtenemos los errores despues de cada actualizacion de los parametros
        print("parametros finales:",self.params)

        self.fig = plt.figure(figsize=(10, 6))

        ax1 = self.fig.add_subplot(121)
        ax1.plot(self.errors)  # hacemos el plot del error
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Mean Squared Error')    
        
    def predict(self,x_test,y_test):       #En esta funcion hacemos un predict de datos nuevos para ver que tan efectivo es el algoritmo
        
        self.x_test=x_test.to_numpy()    # Transformamos el dataframe en un arreglo numpy para su manipulación
        self.y_test=y_test.to_numpy()             
        idx_rand = random.randint(1,len(self.x_test))    #obtenemos un indice random para escoger un valor aleatorio 
        x_real = self.x_test[idx_rand]
        y_real =self.y_test[idx_rand]               
        x_p = np.dot(x_real,self.params)          #realizamos la prediccion del nuevo valor y = wx +b
        y_predict = x_p + self.b
        print("Estatura real: ", x_real)
        print("Peso real: ", y_real)
        print("Peso estimado: ", y_predict )

        ax2 = self.fig.add_subplot(122)
        
        ax2.scatter(self.x_train,self.y_train, label='Data')    #visualizamos los datos de entrenamiento con un scatter plot
        
        x = np.linspace(0, 240)                         #creamos valores de x entre 0 y 240, la cual representa la estatura
        y = self.params*x+self.b                        #Visualizamos la ecuacion de regresion lineal obtenida con los parametros finales
      
        ax2.plot(x,y.flatten())            #hacemos un flatten de y para que tenga el mismo shape que x
        ax2.set_xlabel('Height')
        ax2.set_ylabel('Weight')
        ax2.legend()
        plt.show()
         

df = pd.read_csv('olympics.csv')
df_shuffled=df.sample(frac=1).reset_index(drop=True)
df_shuffled=df_shuffled.dropna(subset=["Height","Weight"])
df_x= df_shuffled[["Height"]]
df_y= df_shuffled[["Weight"]]

df_x_train = df_x[:40000]
df_y_train = df_y[:40000]
df_x_test = df_x[40000:]
df_y_test=df_y[40000:]

############# IMPLEMENTANDO EL MODELO ################

model = LinearRegression(0.000001,1000,0) 

model.set_data(df_x_train,df_y_train)

model.fit()

model.predict(df_x_test,df_y_test)

