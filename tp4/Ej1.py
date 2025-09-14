import numpy as np
import matplotlib.pyplot as plt
from som import SOM
import pandas as pd

if __name__ == "__main__":
    # Cargar datos
    X_circ = pd.read_csv("circulo.csv", header=None).values   # 2 columnas (x,y)

    # Crear SOM
    som = SOM(rows=10, cols=10, dim=2, lr=0.2)

    # ----- Fase 1: Ordenamiento global -----
    som.lr = 0.9
    som.r_init = som.rows // 2        # radio ≈ medio mapa
    som.train(X_circ, epochs=1000)    # 500–1000 épocas

    # ----- Fase 2: Transición -----
    # reducimos lr y radius poco a poco
    for t in range(1000):
        # decaimiento lineal
        som.lr = 0.9 - (0.8 * (t / 1000))      # de 0.9 a ~0.1
        som.r_init = int((som.rows // 2) * (1 - t / 1000)) + 1
        som.train(X_circ, epochs=1)            # entreno de a una época

    # ----- Fase 3: Ajuste fino -----
    som.lr = 0.05
    som.r_init = 0   # solo la ganadora
    som.train(X_circ, epochs=3000)  # hasta convergencia (~3000 épocas)

    # Graficar resultado final
    som.plot(X_circ, "SOM con vecindad cuadrada (3 fases)")
