import numpy as np
import pandas as pd
from graficacion import Grafica

def maxpositive_to_one_rest_to_neg(vec): #Si el maximo es positivo lo pone en 1 y el resto en -1, si el maximo es negativo pone todo en -1
    index_max = np.argmax(vec)
    if vec[index_max] < 0: return -1 * np.ones_like(vec)
    else: 
        aux = -1 * np.ones_like(vec)
        aux[index_max] = 1
        return aux            

def sigmoid(x):
    return np.tanh(x)

def sigmoid_derivative(x):
    return (1+x)*(1-x)  

def loadData(route, cant_entradas):
    data = pd.read_csv(route)
    x = data.iloc[:, :cant_entradas].values  # Todas las filas y "cant_entradas" columnas 
    y = data.iloc[:,cant_entradas:].values  # Todas las filas y las columnas restantes, aplanado
    return [x,y]

class capa:
    def __init__(self, num_neuronas, num_entradas):
        self.num_neuronas = num_neuronas
        self.num_entradas = num_entradas
        self.W = np.random.uniform(-0.5, 0.5, (num_neuronas, (num_entradas + 1)))  # +1 columna para el bias
        self.salidas = [] #Lista de las ultimas salidas de cada neurona
        self.grad_local = [] #Lista de errores locales de cada neurona
        

class MLP:
    def __init__(self, lista_capas, cant_entradas, lr=0.01, epoca_max=100): #lista_capas es una lista con la cantidad de neuronas por capa
        self.lr = lr
        self.epoca_max = epoca_max
        #Inicializar capas
        self.capas = []
        for i in range(len(lista_capas)):
            if lista_capas[i] <= 0: raise ValueError("cantidad de neuronas invalida")
            if i == 0: self.capas.append(capa(lista_capas[i], cant_entradas))   #La primera capa se conecta a las entradas
            else:      self.capas.append(capa(lista_capas[i], lista_capas[i-1])) #El resto de las capas se conectan a la capa anterior
    
    def forward_pass(self, x): #Se recorre de la capa de entrada a la de salida
        entrada_capa = x
        entrada_capa = np.insert(entrada_capa, 0, -1)  # Agregar bias al vector de entrada
        for capa in self.capas:
            z = np.dot(capa.W, entrada_capa)    #Salida lineal
            salida_capa = sigmoid(z)    #Salida no lineal
            capa.salidas = salida_capa
            entrada_capa = np.insert(salida_capa, 0, -1)  #Conectar la salida de esta capa con la entrada de la siguiente y agregar bias
        return salida_capa #Devuelve la salida de la ultima capa
    
    def backward_pass(self, y_D): #Se recorre de la capa de salida a la de entrada
        e = y_D - self.capas[-1].salidas #Señal de error
        for i in reversed(range(len(self.capas))):
            capa = self.capas[i]
            if i == len(self.capas) - 1:
                capa.grad_local = e * sigmoid_derivative(capa.salidas) #Capa de salida
            else:
                capa_siguiente = self.capas[i + 1]
                grad_siguiente = capa_siguiente.grad_local
                W_siguiente = capa_siguiente.W[:,1:]  # Excluir pesos del bias, no se usan en el calculo del gradiente local
                capa.grad_local = np.dot(W_siguiente.T,grad_siguiente) * sigmoid_derivative(capa.salidas) #Capas ocultas

    def ajuste_pesos(self, x):
        entrada_capa = x
        entrada_capa = np.insert(entrada_capa, 0, -1)  # Agregar bias al vector de entrada
        for capa in self.capas: #Para cada capa
            for i in range(capa.W.shape[0]):  # Para cada neurona
                for j in range(capa.W.shape[1]):  # Para cada peso
                    capa.W[i, j] += self.lr * capa.grad_local[i] * entrada_capa[j] #Correccion del peso
            entrada_capa = np.insert(capa.salidas, 0, -1)  # Agregar bias para la siguiente capa
    
    def entrenamiento(self,x,y):
        vec_error_class = []
        vec_error_cuad = []
        for epoca in range(self.epoca_max):
            error_class = 0
            error_cuad = 0
            for i in range(int(x.shape[0]*0.8)): #Para cada patron de entrenamiento
                salida = self.forward_pass(x[i,:]) #Calculo de salida
                self.backward_pass(y[i, :]) #Calculo de gradientes
                self.ajuste_pesos(x[i,:]) #Ajuste de pesos
            
            cant_validacion = 0
            for i in range(int(x.shape[0]*0.8), x.shape[0]): #Validacion
                salida = self.forward_pass(x[i,:]) #Calculo de salida
                salida_arreglada = maxpositive_to_one_rest_to_neg(salida)
                if(np.any(salida_arreglada-y[i] != 0)): error_class+=1 #Error de clasificacion, todas las salidas corregidas deben ser iguales

                e = y[i,:] - salida #Señal de error
                error_cuad += np.sum(e**2)/e.shape[0] #Error cuadratico
                
                cant_validacion += 1
            
            vec_error_class.append(error_class)
            vec_error_cuad.append((error_cuad/cant_validacion)*100) 
            print(f"Epoca {epoca+1}, Error de clasificacion: {error_class}, Error cuadratico medio: {(error_cuad/cant_validacion)*100}%") 
            if(error_class == 0 and error_cuad == 0): break

        return vec_error_class, vec_error_cuad

    def test(self, x, y): #Devuelve el error de clasificacion
        error = 0
        for i in range(x.shape[0]):
            salida = self.forward_pass(x[i,:])
            salida_arreglada = maxpositive_to_one_rest_to_neg(salida)
            if(np.any(salida_arreglada-y[i] != 0)): error+=1 
        return error/x.shape[0]
