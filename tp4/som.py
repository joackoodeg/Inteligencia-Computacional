import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, rows, cols, dim, lr=0.1, r_init=None, seed=0):
        self.rows, self.cols, self.dim = rows, cols, dim
        self.lr = lr
        self.r_init = r_init
        rng = np.random.default_rng(seed)
        # pesos iniciales aleatorios en [-1,1]
        self.W = rng.uniform(-1, 1, size=(rows, cols, dim))

    def routine(self, X):
        # Fase 1
        self.lr = 0.9
        self.r_init = self.rows // 2        # radio ≈ medio mapa
        self.train(X, epochs=1000)

        # Fase 2: reducimos lr y radius poco a poco
        for t in range(1000):
            # decaimiento lineal
            self.lr = 0.9 - (0.8 * (t / 1000))      # de 0.9 a ~0.1
            self.r_init = int((self.rows // 2) * (1 - t / 1000)) + 1
            self.train(X, epochs=1)

        # Fase 3
        self.lr = 0.05
        self.r_init = 0   # solo la ganadora
        self.train(X, epochs=3000)

    def find(self, x):
        """Busca la neurona más cercana a x"""
        min_dist = float("inf")
        best_row, best_col = 0, 0
        for i in range(self.rows):
            for j in range(self.cols):
                w = self.W[i][j]
                dist = 0
                for k in range(self.dim):
                    dist += (w[k] - x[k])**2
                if dist < min_dist:
                    min_dist = dist
                    best_row, best_col = i, j
        return best_row, best_col

    def neighbors(self, row, col, radius):
        """Devuelve los índices de la vecindad cuadrada alrededor de (row, col)"""
        neigh = []
        # range(a, b) genera números desde a hasta b-1. -> radius = radius +1
        for i in range(row - radius, row + radius + 1):
            for j in range(col - radius, col + radius + 1):
                # verificar que esté dentro de la grilla
                if 0 <= i < self.rows and 0 <= j < self.cols:
                    neigh.append((i, j))
        return neigh

    def train(self, X, epochs=100, lr=None, radius=None):
        if lr is None:
            lr = self.lr
        if radius is None:
            radius = self.r_init

        for _ in range(epochs):
            for x in X:
                row, col = self.find(x)
                for i, j in self.neighbors(row, col, radius):
                    self.W[i, j] += lr * (x - self.W[i, j])

    def evaluate(self, X, y, num_classes=3):
        # Inicializar estructuras
        labels = []
        activations = np.zeros((self.rows, self.cols), dtype=int)
        class_counts = np.zeros((self.rows, self.cols, num_classes), dtype=int)

        # 1. Recorrer todas las muestras
        for idx, sample in enumerate(X):
            row, col = self.find(sample)
            labels.append(row * self.cols + col)

            # contar activación
            activations[row, col] += 1

            # sumar 1 al contador de la clase correspondiente
            class_counts[row, col, y[idx]] += 1

        labels = np.array(labels)

        # 2. Clase mayoritaria por neurona
        majority_class = np.argmax(class_counts, axis=2)   # el índice con más votos
        majority_class[activations == 0] = -1              # neuronas vacías → -1

        return labels, activations, majority_class


    def plot(self, X, title="SOM"):
        # 1. Para cada dato, buscar la neurona ganadora (BMU)
        labels = []
        for x in X:
            row, col = self.find(x)
            labels.append(row * self.cols + col)  # índice de la neurona ganadora

        labels = np.array(labels)

        # 2. Dibujar los datos, coloreados por su BMU
        plt.scatter(X[:,0], X[:,1], c=labels, cmap="tab20", s=10)

        # 3. Dibujar las conexiones entre neuronas (rejilla)
        for i in range(self.rows):
            plt.plot(self.W[i,:,0], self.W[i,:,1], "k-")   # filas
        for j in range(self.cols):
            plt.plot(self.W[:,j,0], self.W[:,j,1], "k-")   # columnas

        # 4. Dibujar los centroides de las neuronas
        plt.scatter(self.W[:,:,0], self.W[:,:,1], c="red", marker="x")

        plt.title(title)
        plt.axis("equal")
        plt.show()

