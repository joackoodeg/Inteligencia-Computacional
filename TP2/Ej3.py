from MLP import MLP, loadData
import numpy as np

import matplotlib.pyplot as plt

def plot_two_vectors(vec1, vec2, titulo="Dos vectores", label1="Vector 1", label2="Vector 2"):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(vec1, marker='o', label=label1, linestyle='-')
    plt.plot(vec2, marker='x', label=label2, linestyle='-')
    plt.title(titulo)
    plt.xlabel("Epoca")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.legend()
    plt.show()

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