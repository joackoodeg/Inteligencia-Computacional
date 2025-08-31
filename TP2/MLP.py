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
    y = data.iloc[:,cant_entradas:].values  # Todas las filas y las columnas restantes
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
        for epoca in range(self.epoca_max):
            error_class = 0
            error_cuad = 0
            for i in range(x.shape[0]): #Para cada patron de entrenamiento
                salida = self.forward_pass(x[i,:]) #Calculo de salida
                self.backward_pass(y[i, :]) #Calculo de gradientes
                self.ajuste_pesos(x[i,:]) #Ajuste de pesos

                salida_arreglada = maxpositive_to_one_rest_to_neg(salida)
                if(np.any(salida_arreglada-y[i] != 0)): error_class+=1 #Error de clasificacion, todas las salidas corregidas deben ser iguales

                e = y[i,:] - salida #Señal de error
                error_cuad += np.sum(e**2) #Error cuadratico 
            print(f"Epoca {epoca+1}, Error de clasificacion: {error_class}, Error de cuadratico medio: {error_cuad/x.shape[0]}") 
            #a veces dan valores raros, revisar

    def test(self, x, y): #Devuelve el error de clasificacion
        error = 0
        for i in range(x.shape[0]):
            salida = self.forward_pass(x[i,:])
            salida_arreglada = maxpositive_to_one_rest_to_neg(salida)
            if(np.any(salida_arreglada-y[i] != 0)): error+=1 
        return error/x.shape[0]

if __name__ == "__main__":
    lista_capas = [2,1]  # 2 neuronas en la capa oculta, 1 en la capa de salida
    cant_entradas = 2
    mlp = MLP(lista_capas, cant_entradas,0.1,10)
    [x,y] = loadData("XOR_trn.csv",cant_entradas)  # Cargar datos de entrada
    
    mlp.entrenamiento(x,y)

    # Graficar datos de entrenamiento con heatmap
    #x_bias_train = np.hstack((np.ones((x.shape[0], 1)) * -1, x))
    #grafica_train = Grafica()
    #grafica_train.graficar(x_bias_train, y, titulo="MLP - Datos de Entrenamiento", predict_func=lambda entrada: mlp.forward_pass(entrada))
    
    [x,y] = loadData("XOR_tst.csv",cant_entradas)  # Cargar datos de entrada
    
    mlp.test(x,y)
    print("Tasa de error en test:", mlp.test(x,y))

    # Graficar los datos de testeo con heatmap
    #x_bias_test = np.hstack((np.ones((x.shape[0], 1)) * -1, x))
    #grafica_test = Grafica()
    #grafica_test.graficar(x_bias_test, y, titulo="MLP - Datos de Testeo", predict_func=lambda entrada: mlp.forward_pass(entrada))