import numpy as np
import pandas as pd

def sigmoid(x):
    return np.tanh(x)

def sigmoid_derivative(x):
    return (1+x)*(1-x)  #Derivada de tanh

def loadData(route):
    data = pd.read_csv(route)
    x = data.iloc[:, :-1].values  # Todas las filas y todas las columnas menos la última  
    y = data.iloc[:,-1].values  # Solo la última columna
    return [x,y]

class capa:
    def __init__(self, num_neuronas, num_entradas):
        self.num_neuronas = num_neuronas
        self.num_entradas = num_entradas
        self.W = np.random.uniform(-0.5, 0.5, (num_neuronas, (num_entradas + 1)))  # +1 para el bias
        self.salidas = [] #Lista de las ultimas salidas de cada neurona
        self.grad_local = [] #Lista de errores locales de cada neurona
        
        #self.bias = np.random.uniform(-0.5, 0.5, num_neuronas) ??

class MLP:
    def __init__(self, lista_capas, cant_entradas, lr=0.01, epoca_max=100): #lista_capas es una lista con la cantidad de neuronas por capa
        self.lr = lr
        self.epoca_max = epoca_max
        #Inicializar capas
        self.capas = []
        for i in range(len(lista_capas)):
            if lista_capas[i] <= 0: raise ValueError("cantidad de neuronas invalida")
            if i == 0: self.capas.append(capa(lista_capas[i], cant_entradas))
            else:      self.capas.append(capa(lista_capas[i], lista_capas[i-1]))
    
    def forward_pass(self, x):
        entrada_capa = x
        entrada_capa = np.insert(entrada_capa, 0, -1)  # Agregar bias al vector de entrada
        for capa in self.capas:
            z = np.dot(capa.W, entrada_capa)
            salida_capa = sigmoid(z)
            capa.salidas = salida_capa
            entrada_capa = np.insert(salida_capa, 0, -1)  # Agregar bias para la siguiente capa
        return salida_capa
    
    def backward_pass(self, x, y_D):
        e = y_D - self.capas[-1].salidas
        for i in reversed(range(len(self.capas))):
            capa = self.capas[i]
            if i == len(self.capas) - 1:
                capa.grad_local = e * sigmoid_derivative(capa.salidas)
            else:
                capa_siguiente = self.capas[i + 1]
                grad_siguiente = capa_siguiente.grad_local
                W_siguiente = capa_siguiente.W[:,1:]  # Excluir pesos del bias, no se usan en el calculo del gradiente local
                capa.grad_local = np.dot(W_siguiente.T,grad_siguiente) * sigmoid_derivative(capa.salidas) #w esta transpuesta para que coincidan las dimensiones

    def ajuste_pesos(self, x):
        entrada_capa = x
        entrada_capa = np.insert(entrada_capa, 0, -1)  # Agregar bias al vector de entrada
        for capa in self.capas:
            for i in range(capa.W.shape[0]):  # Para cada neurona
                for j in range(capa.W.shape[1]):  # Para cada peso
                    capa.W[i, j] += self.lr * capa.grad_local[i] * entrada_capa[j]
            entrada_capa = np.insert(capa.salidas, 0, -1)  # Agregar bias para la siguiente capa
    
    def entrenamiento(self,x,y):
        for epoca in range(self.epoca_max):
            error = 0
            for i in range(x.shape[0]):
                salida = self.forward_pass(x[i,:])
                self.backward_pass(x[i,:], y[i])
                self.ajuste_pesos(x[i,:])
                if(salida*y[i] < 0): error+=1  #Si la salida y la deseada son del mismo signo, es correcto?
            #print(f"Epoca {epoca+1}, Tasa de error: {error}")

    def test(self, x, y):
        error = 0
        for i in range(x.shape[0]):
            salida = self.forward_pass(x[i,:])
            if(salida*y[i] < 0): error+=1 #Si la salida y la deseada son del mismo signo, es correcto?
        return error/x.shape[0]

if __name__ == "__main__":
    # Ejemplo de uso
    lista_capas = [2,1]  # 2 neuronas en la capa oculta, 1 en la capa de salida
    cant_entradas = 2
    mlp = MLP(lista_capas, cant_entradas,0.1,100)
    [x,y] = loadData("XOR_trn.csv")  # Cargar datos de entrada
    
    mlp.entrenamiento(x,y)
    
    [x,y] = loadData("XOR_tst.csv")  # Cargar datos de entrada
    
    mlp.test(x,y)
    print("Tasa de error en test:", mlp.test(x,y))
    #salida = mlp.forward_pass(x[0,:])
    #print(x[0,:]," -> ",y[0],": ",salida)