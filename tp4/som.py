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


    def plot(self, X, title="SOM"):
        plt.scatter(X[:,0], X[:,1], c="lightgray", s=10)
        # pesos
        for i in range(self.rows):
            plt.plot(self.W[i,:,0], self.W[i,:,1], "k-")   # filas
        for j in range(self.cols):
            plt.plot(self.W[:,j,0], self.W[:,j,1], "k-")   # columnas
        plt.scatter(self.W[:,:,0], self.W[:,:,1], c="red", marker="x")
        plt.title(title)
        plt.axis("equal")
        plt.show()
