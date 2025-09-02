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
