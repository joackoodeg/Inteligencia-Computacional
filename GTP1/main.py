import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    perceptron = Perceptron(learning_rate=0.1, n_iterations=20)

    # Entrenamiento
    X_train, y_train = perceptron.loadData("OR_90_trn.csv")
    perceptron.entrenamiento(X_train, y_train)

    # Prueba
    X_test, y_test = perceptron.loadData("OR_90_tst.csv")
    accuracy = perceptron.test(X_test, y_test)

    print(f"Learning Rate: {perceptron.lr}")
    print(f"Number of Iterations: {perceptron.n_i}")
    print(f"Pesos finales: {perceptron.w}")
    print(f"Precisión en test: {accuracy:.2f}%")
