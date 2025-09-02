from graficacion import Grafica
import numpy as np
from MLP import MLP, loadData
from plot2v import plot_two_vectors

if __name__ == "__main__":
    # Ejemplo de uso
    # [10,50, 1] - 0.01, 1000 -> 1.6 %
    # [50, 1] -> 0.01, 200 -> 1.7%
    # [100, 1] -> 0.01, 200 -> 1.2%
    lista_capas = [50,1]  # 2 neuronas en la capa oculta, 1 en la capa de salida
    cant_entradas = 2
    [x,y] = loadData("concent_trn.csv",cant_entradas)  # Cargar datos de entrada
    
    mlp = MLP(lista_capas, cant_entradas,0.01,100)
    
    vec_error_class, vec_error_cuad = mlp.entrenamiento(x,y)
    vec_error_class = np.array(vec_error_class)*100/vec_error_class[0] # Normalizo para que inicie en 100
    plot_two_vectors(vec_error_cuad,vec_error_class, titulo="Errores durante el entrenamiento", label1="Error Cuadratico", label2="Error de Clasificación")

    # Graficar datos de entrenamiento (sin necesidad de preparar bias manualmente)
    Grafica.graficar_mlp(x, y, mlp, titulo="MLP - Datos de Entrenamiento")

    [x,y] = loadData("concent_tst.csv",cant_entradas)  # Cargar datos de entrada

    mlp.test(x,y)
    print("Tasa de error en test:", mlp.test(x,y))

    # Graficar los datos de testeo (simplificado)
    Grafica.graficar_mlp(x, y, mlp, titulo="MLP - Datos de Testeo")