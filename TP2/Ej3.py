from MLP import MLP, loadData
import numpy as np
from plot2v import plot_two_vectors

if  __name__ == "__main__":
    lista_capas = [6,3]  # [2,3] a veces converge, a veces no
    cant_entradas = 4
    [x,y] = loadData("iris81_trn.csv",cant_entradas)  # Cargar datos de entrada
    mlp = MLP(lista_capas, cant_entradas,0.01,100)
    
    vec_error_class, vec_error_cuad = mlp.entrenamiento(x,y)
    vec_error_class = np.array(vec_error_class)*100/vec_error_class[0] # Normalizo para que inicie en 100
    plot_two_vectors(vec_error_cuad,vec_error_class, titulo="Errores durante el entrenamiento", label1="\Error Cuadratico", label2="Error de Clasificación")

    [x,y] = loadData("iris81_tst.csv",cant_entradas)  # Cargar datos de entrada
    mlp.test(x,y)
    print("Tasa de error en test:", mlp.test(x,y))