import numpy as np
import matplotlib.pyplot as plt

class Grafica:
    def __init__(self, w=None):
        self.w = w

    def graficar(self, X, y, titulo="Perceptrón", predict_func=None):
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
        # Si no hay pesos pero hay función de predicción, graficar heatmap
        elif predict_func is not None:
            self._dibujar_heatmap(predict_func)
        
        plt.xlabel('Entrada 1')
        plt.ylabel('Entrada 2') 
        plt.title(titulo)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    def _dibujar_heatmap(self, predict_func):
        # Crear una malla de puntos en el espacio de entrada
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        # Agregar bias
        grid_bias = np.hstack((np.ones((grid.shape[0], 1)) * -1, grid))
        # Predecir para cada punto
        zz = np.array([predict_func(pt[1:]) for pt in grid_bias])
        zz = zz.reshape(xx.shape)
        # Graficar el mapa de calor
        plt.contourf(xx, yy, zz, levels=50, cmap='RdBu', alpha=0.3)
        plt.contour(xx, yy, zz, levels=[0], colors='green', linewidths=2)
    
    def _dibujar_linea_separacion(self, X):
        # Simplemente tomar puntos de -3 a 3 en x1
        x1_vals = np.array([-3, 3])
        
        # Ecuación: -1*w0 + w1*x1 + w2*x2 = 0
        # Despejamos: x2 = (w0 - w1*x1) / w2
        w0, w1, w2 = self.w[0], self.w[1], self.w[2]
        
        x2_vals = (w0 - w1 * x1_vals) / w2
        plt.plot(x1_vals, x2_vals, 'g-', linewidth=3, label='Línea de separación')
