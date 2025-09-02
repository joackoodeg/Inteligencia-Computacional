import numpy as np
import matplotlib.pyplot as plt

class Grafica:
    def __init__(self, w=None):
        self.w = w

    def graficar(self, X, y, titulo="Perceptrón", predict_func=None, agregar_bias=False):
        # Si se solicita agregar bias, agregarlo
        if agregar_bias:
            X = np.hstack((np.ones((X.shape[0], 1)) * -1, X))
        
        # Extraer solo las características (sin bias)
        X_sin_bias = X[:, 1:] if X.shape[1] > 2 else X  # Quitar la primera columna solo si tiene más de 2 columnas
        
        # Asegurar que y sea un array 1D
        if y.ndim > 1:
            y = y.flatten()
        
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
        # Si no hay pesos pero hay función de predicción, graficar línea de separación
        elif predict_func is not None:
            self._dibujar_linea_separacion_mlp(predict_func, X_sin_bias)

        plt.xlabel('Entrada 1')
        plt.ylabel('Entrada 2') 
        plt.title(titulo)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _dibujar_linea_separacion_mlp(self, predict_func, X_sin_bias):
        # Determinar los límites del espacio basado en los datos
        x_min, x_max = X_sin_bias[:, 0].min() - 1, X_sin_bias[:, 0].max() + 1
        y_min, y_max = X_sin_bias[:, 1].min() - 1, X_sin_bias[:, 1].max() + 1
        
        # Crear una malla de puntos en el espacio de entrada
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # Predecir para cada punto
        zz = np.array([predict_func(pt) for pt in grid])
        zz = zz.reshape(xx.shape)
        
        # Graficar solo la línea de separación (contorno de nivel 0)
        plt.contour(xx, yy, zz, levels=[0], colors='green', linewidths=3)
    
    def _dibujar_linea_separacion(self, X):
        # Simplemente tomar puntos de -3 a 3 en x1
        x1_vals = np.array([-3, 3])
        
        # Ecuación: -1*w0 + w1*x1 + w2*x2 = 0
        # Despejamos: x2 = (w0 - w1*x1) / w2
        w0, w1, w2 = self.w[0], self.w[1], self.w[2]
        
        x2_vals = (w0 - w1 * x1_vals) / w2
        plt.plot(x1_vals, x2_vals, 'g-', linewidth=3, label='Línea de separación')

    @staticmethod
    def graficar_mlp(X, y, mlp, titulo="MLP", agregar_bias=True):
        """Método estático para graficar datos con un MLP de forma simplificada"""
        grafica = Grafica()
        grafica.graficar(X, y, titulo=titulo, predict_func=lambda entrada: mlp.forward_pass(entrada), agregar_bias=agregar_bias)
