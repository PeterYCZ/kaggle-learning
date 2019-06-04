import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

def loaddata(path):
    data = pd.read_csv(path)
    return data

def splitxy(data,i):
    x = np.array(data.iloc[:,i+1:])
    y = np.array(data.iloc[:,i])
    return x,y

def compute_scores(X,n_components):
    pca = PCA(svd_solver='full')
    pca_scores = []
    for n in n_components:
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
    return pca_scores

if __name__ == "__main__":
    path = "./data/train.csv"
    data = loaddata(path)
    x,y = splitxy(data,1)
    n_features = 200
    n_components = np.arange(0, n_features, 5)
    sores = compute_scores(x, n_components)
    plt.plot(sores)
    plt.show()