import pandas as pd
    
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_i = n_iterations

    def entrenamiento(self, X, y):
        self.w = 

    def loadData(route):
        df = pd.read_csv(route)
        X = []
        y = []

        for row in df.iterrows():
            X.append([-1 , row[0], row[1]])
            y.append(row[2])
            Perceptron.entrenamiento(X, y)

if __name__ == "__main__":
    perceptron = Perceptron()
    perceptron.loadData("OR_90_trn.csv")

    # Example usage of the Perceptron class
    print(f"Learning Rate: {perceptron.lr}")
    print(f"Number of Iterations: {perceptron.n_i}")    