import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
import numpy as np

X = np.random.rand(100, 1)*10
Y = X.squeeze*2.5 + 1.5 + np.random.rand(100, 1)


