import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def loaddata(path):
    data = pd.read_csv(path)
    return data

def splitxy(data,i):
    x = np.array(data.iloc[:,i+1:])
    y = np.array(data.iloc[:,i],int)
    return x,y

def reduce_features(X,n):
    pca = PCA(n_components = n)
    pca.fit(X)
    return pca

if __name__ == "__main__":
    path = "./data/train.csv"
    path_test = "./data/test.csv"
    data = loaddata(path)
    test_data = loaddata(path_test)
    x,y = splitxy(data,1)
    pca = reduce_features(x,5)
    x_train = pca.transform(x)
    x_test = test_data.iloc[:,1:]
    id = np.array(test_data.iloc[:, 0],int)
    clf = RandomForestClassifier(n_estimators=10,max_depth=None, min_samples_split=2,random_state=0)
    clf.fit(x_train,y)
    x_test = pca.transform(x_test)
    y_pred = clf.predict(x_test)
    with open("submission.csv","w") as file:
        file.write("id,target\n")
        for i in range(len(id)):
            file.write(str(id[i])+","+str(y_pred[i])+"\n")