import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR


class SVR_regresion:
    def __init__(self,
                 kernel='rbf',
                 C=100,
                 gamma=0.1,
                 epsilon=.1,
                 degree=3,):
        self.svr = SVR(kernel=kernel, degree=degree, C=C, gamma=gamma, epsilon=epsilon)

    def predict(self, input):
        return self.svr.predict(input)
    
    def fit(self, input, output):
        self.svr.fit(input, output)

