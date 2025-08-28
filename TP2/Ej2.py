from graficacion import Grafica
import numpy as np
from MLP import MLP, loadData

if __name__ == "__main__":
    # Ejemplo de uso
    # [10,50, 1] - 0.01, 1000 -> 1.6 %
    # [50, 1] -> 0.01, 200 -> 1.7%
    # [100, 1] -> 0.01, 200 -> 1.2%
    lista_capas = [10,1]  # 2 neuronas en la capa oculta, 1 en la capa de salida
    cant_entradas = 2
    [x,y] = loadData("concent_trn.csv",cant_entradas)  # Cargar datos de entrada
    mlp = MLP(lista_capas, cant_entradas,0.01,100)
    
    mlp.entrenamiento(x,y)

    # Graficar datos de entrenamiento con heatmap
    #x_bias_train = np.hstack((np.ones((x.shape[0], 1)) * -1, x))
    #grafica_train = Grafica()
    #grafica_train.graficar(x_bias_train, y, titulo="MLP - Datos de Entrenamiento", predict_func=lambda entrada: mlp.forward_pass(entrada))

    [x,y] = loadData("concent_tst.csv",cant_entradas)  # Cargar datos de entrada

    mlp.test(x,y)
    print("Tasa de error en test:", mlp.test(x,y))

    # Graficar los datos de testeo con heatmap
    #x_bias_test = np.hstack((np.ones((x.shape[0], 1)) * -1, x))
    #grafica_test = Grafica()
    #grafica_test.graficar(x_bias_test, y, titulo="MLP - Datos de Testeo", predict_func=lambda entrada: mlp.forward_pass(entrada))