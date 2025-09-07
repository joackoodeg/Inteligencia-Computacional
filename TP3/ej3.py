import numpy as np
from sklearn import model_selection 
from sklearn import datasets

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

if __name__ == "__main__":
    X,y = datasets.load_(return_X_y=True)

    kf = model_selection.KFold(n_splits=5,shuffle=True)

    exactitudes = np.zeros((5,6))
    for i,(train_i,test_i) in enumerate(kf.split(X)):
        
        adb = AdaBoostClassifier()
        adb.fit(X[train_i],y[train_i])
        exactitudes[i,0] = adb.score(X[test_i],y[test_i])

        bag = BaggingClassifier()
        bag.fit(X[train_i],y[train_i])
        exactitudes[i,1] = bag.score(X[test_i],y[test_i])


    print("ADB: Promedio: ", np.mean(exactitudes[:,0])," Varianza: ", np.var(exactitudes[:,0]))
    print("BAG: Promedio: ", np.mean(exactitudes[:,1])," Varianza: ", np.var(exactitudes[:,1]))