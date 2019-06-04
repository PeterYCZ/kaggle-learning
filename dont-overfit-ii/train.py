import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def loaddata(path):
    data = pd.read_csv(path)
    return data

def splitxy(data,i):
    x = np.array(data.iloc[:,i+1:])
    y = np.array(data.iloc[:,i])
    return x,y

def compute_scores(X,n_components):
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()
    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
        fa_scores.append(np.mean(cross_val_score(fa, X, cv=5)))

    return pca_scores, fa_scores

if __name__ == "__main__":
    path = "./data/train.csv"
    data = loaddata(path)
    x,y = splitxy(data,1)
    n_features = len(x[0])
    n_components = np.arange(0, n_features, 5)
    pca_scores, fa_scores = compute_scores(x, n_components)
    print(pca_scores)
    print(fa_scores)
