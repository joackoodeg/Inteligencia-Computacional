import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_i = n_iterations
        self.w = None

    def entrenamiento(self, X, y):
        # Inicializa pesos con ceros
        self.w = np.zeros(X.shape[1])
        # shape[0] es el número de filas
        # shape[1] es el número de columnas

        for _ in range(self.n_i):
            for xi, target in zip(X, y):
                y_pred = self.predict_raw(xi)
                if y_pred != target:  # solo actualizar si hay error
                    update = self.lr * (target - y_pred)
                    self.w += update * xi
        # zip: Sirve para recorrer dos o mas arrays en paralelo (emparejando sus elementos por posicicion)
        # ej: Recorrer tres o más listas a la vez: for a, b, c in zip(lista1, lista2, lista3):

    def predict_raw(self, x):
        if np.dot(x, self.w) >= 0:
            return 1
        return -1

    def predict(self, X):
        predictions= []
        for xi in X:
            predictions.append(self.predict_raw(xi))
        return np.array(predictions)

    def loadData(self, route):
        df = pd.read_csv(route)

        X = df.iloc[:, :-1].values 
        # toma todas las filas (:) y todas las columnas menos la última (:-1).
        # con .values se pasa a un array de numpy
        X = np.insert(X, 0, -1, axis=1)  # Agregar bias
        # esto es np.insert(matriz, lugar, valor, axis) 
        # -> si axis = 0 se inserta a lo largo de las filas
        # -> si axis = 1 se inserta a lo largo de las columnas

        y = df.iloc[:, -1].values
        # toma solo la última columna
        return X, y

    def test(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y) * 100
        return accuracy
    
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
        
        # Graficar línea de separación si tenemos pesos
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

def routine(perceptron, training_route, test_route, titulo):
     # Entrenamiento
    X_train, y_train = perceptron.loadData(training_route)
    
    # Mostrar datos antes del entrenamiento
    print(f"=== DATOS DE ENTRENAMIENTO === {titulo}")
    perceptron.graficar(X_train, y_train, f"Datos {titulo} - Antes del entrenamiento")
    
    # Entrenar
    perceptron.entrenamiento(X_train, y_train)
    
    # Mostrar resultado después del entrenamiento
    print("=== RESULTADO DESPUÉS DEL ENTRENAMIENTO ===")
    perceptron.graficar(X_train, y_train, f"Datos {titulo} - Después del entrenamiento")

    # Prueba
    X_test, y_test = perceptron.loadData(test_route)
    accuracy = perceptron.test(X_test, y_test)

    print(f"Learning Rate: {perceptron.lr}")
    print(f"Number of Iterations: {perceptron.n_i}")
    print(f"Pesos finales: {perceptron.w}")
    print(f"Precisión en test: {accuracy:.2f}%")


if __name__ == "__main__":
    perceptronOR = Perceptron(learning_rate=0.01, n_iterations=1000)
    routine(perceptronOR, "OR_trn.csv", "OR_tst.csv", "OR")
    perceptronXOR = Perceptron(learning_rate=0.01, n_iterations=1000)
    routine(perceptronXOR, "XOR_trn.csv", "XOR_tst.csv", "XOR")

