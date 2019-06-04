import pandas as pd
from sklearn import preprocessing
from sklearn import impute
import numpy as np

train=pd.read_csv("./data/train.csv")
Nan = train.isnull().sum().to_frame()
print(Nan.loc[Nan[0] != 0])