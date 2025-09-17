from som import SOM
import pandas as pd

def routine(route):
    # Cargar datos
    data = pd.read_csv(route, header=None).values   # 2 columnas (x,y)

    # Crear SOM
    som = SOM(rows=3, cols=3, dim=2, lr=0.2)

    # Fase 1
    som.lr = 0.9
    som.r_init = som.rows // 2        # radio ≈ medio mapa
    som.train(data, epochs=300)    # 500–1000 épocas

    # Fase 2
    # reducimos lr y radius poco a poco
    for t in range(300):
        # decaimiento lineal
        som.lr = 0.9 - (0.8 * (t / 1000))      # de 0.9 a ~0.1
        som.r_init = int((som.rows // 2) * (1 - t / 1000)) + 1
        som.train(data, epochs=1)            # entreno de a una época

    # Fase 3
    som.lr = 0.05
    som.r_init = 0   # solo la ganadora
    som.train(data, epochs=300)  # hasta convergencia (~3000 épocas)

    # Graficar resultado final
    som.plot(data, "SOM con vecindad cuadrada (3 fases)")


if __name__ == "__main__":
    #routine("te.csv") 
    routine("circulo.csv")   