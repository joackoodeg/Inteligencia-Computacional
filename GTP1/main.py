import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, cant_entradas ,learning_rate=0.01, epoca_max=1000):
        self.lr = learning_rate
        self.epoca_max = epoca_max
        self.w = np.random.uniform(-0.5,0.5,cant_entradas+1)  # Inicializa pesos aleatorios

    def entrenamiento(self, route,tasa_aceptable=0):
        self.loadData(route) 
        for epoca in range(self.epoca_max):
            # Mezclar los índices de las muestras para cada época
            # Esto asegura que el orden de las muestras sea aleatorio en cada época
            indices = np.arange(self.X.shape[0])
            np.random.shuffle(indices) 
            ind_entrenamiento = indices[:int(0.8*len(indices))]  # 80% para entrenamiento
            ind_validacion = indices[int(0.8*len(indices)):]  # 20% para validación
            
            # Entrenamiento
            for i in ind_entrenamiento:
                y_pred = self.predict(self.X[i])
                error = self.y[i] - y_pred
                self.w += self.lr * error * self.X[i]
            
            # Validación
            tasa_error = self.test_index(ind_validacion)
            # print(f"Época {epoca+1}, Tasa de error: {tasa_error*100:.2f}%")
            
            if(tasa_error <= tasa_aceptable):
                print(f"Tasa de error aceptable alcanzada: {tasa_error*100:.2f}% en la época {epoca+1}")
                break

    def predict(self, input_data): #Devuelve un int 1 o -1
        if(np.dot(self.w, input_data) >= 0):
            return 1
        else:
            return -1
        
    def test_index(self, indices): #Devuelve la tasa de error para un conjunto de índices
        correct_predictions = 0
        for i in indices:
            y_pred = self.predict(self.X[i])
            correct_predictions += (y_pred == self.y[i])
        
        tasa_error = 1-(correct_predictions / len(indices)) #0-1
        return tasa_error
    
    def test(self, route): #Devuelve la tasa de error para un conjunto de datos en un archivo csv
        self.loadData(route)
        correct_predictions = 0
        for i in range(len(self.X)):
            y_pred = self.predict(self.X[i])
            correct_predictions += (y_pred == self.y[i])
        
        tasa_error = 1-(correct_predictions / len(self.X)) #0-1
        return tasa_error
     

    def loadData(self,route):
        data = pd.read_csv(route)
        self.X = data.iloc[:, :-1].values
        # toma todas las filas (:) y todas las columnas menos la última (:-1).
        # con .values se pasa a un array de numpy
        self.X = np.insert(self.X, 0, -1, axis=1)  # Agregar bias
        # -> si axis = 0 se inserta a lo largo de las filas
        # -> si axis = 1 se inserta a lo largo de las columnas  
        self.y = data.iloc[:,-1].values  # Solo la última columna
    
    def graficar(self, X, y, titulo="Perceptrón"):
        # Extraer solo las características (sin bias)
        X_sin_bias = X[:, 1:]  # Quitar la primera columna (bias)
        
        # Separar puntos por clase
        clase_1 = X_sin_bias[y == 1]  # Puntos de clase 1
        clase_neg1 = X_sin_bias[y == -1]  # Puntos de clase -1
        
        # Crear la gráfica
        plt.figure(figsize=(8, 6))
        
        # Graficar puntos por clase con colores diferentes
        plt.scatter(clase_1[:, 0], clase_1[:, 1], c='blue', marker='o', s=50, label='Clase +1')
        plt.scatter(clase_neg1[:, 0], clase_neg1[:, 1], c='red', marker='x', s=50, label='Clase -1')
        
        # Graficar linea de separación si tenemos pesos
        if self.w is not None:
            self._dibujar_linea_separacion(X_sin_bias)
        
        plt.xlabel('Entrada 1')
        plt.ylabel('Entrada 2') 
        plt.title(titulo)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _dibujar_linea_separacion(self, X):
        # Simplemente tomar puntos de -3 a 3 en x1
        x1_vals = np.array([-3, 3])
        
        # Ecuación: -1*w0 + w1*x1 + w2*x2 = 0
        # Despejamos: x2 = (w0 - w1*x1) / w2
        w0, w1, w2 = self.w[0], self.w[1], self.w[2]
        
        x2_vals = (w0 - w1 * x1_vals) / w2
        plt.plot(x1_vals, x2_vals, 'g-', linewidth=3, label='Línea de separación')

def routine(perceptron, training_route, test_route, titulo, tasa_aceptable=0):
    print(f"{titulo} - Learning Rate: {perceptron.lr}")
    print(f"{titulo} -Épocas máximas: {perceptron.epoca_max}")

    ext = perceptron.test(test_route)
    print(f"{titulo} -Error en test externo antes del entrenamiento: {ext*100:.2f}%")
    perceptron.graficar(perceptron.X, perceptron.y, f"Datos {titulo} - Antes del entrenamiento")

    perceptron.entrenamiento(training_route,tasa_aceptable)

    ext = perceptron.test(test_route)
    print(f"{titulo} -Error en test externo: {ext*100:.2f}%")
    perceptron.graficar(perceptron.X, perceptron.y, f"Datos {titulo} - Después del entrenamiento")

if __name__ == "__main__":
    # Ej 2 - 
    perceptronOR = Perceptron(cant_entradas=2,learning_rate=0.1, epoca_max=1000)
    routine(perceptronOR, "OR_trn.csv", "OR_tst.csv", "OR", 0)
    perceptronXOR = Perceptron(cant_entradas=2, learning_rate=0.1, epoca_max=1000)
    routine(perceptronXOR, "XOR_trn.csv", "XOR_tst.csv", "XOR", 0)
    # Ej 3 -
    perceptronOR50 = Perceptron(cant_entradas=2, learning_rate=0.1, epoca_max=1000)
    routine(perceptronOR50, "OR_50_trn.csv", "OR_50_tst.csv", "OR-50", 0)
    perceptronOR90 = Perceptron(cant_entradas=2, learning_rate=0.1, epoca_max=1000)
    routine(perceptronOR90, "OR_90_trn.csv", "OR_90_tst.csv", "OR-90", 0)