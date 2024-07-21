import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class Gaussian_Processes:
    def __init__(self,
                 kernel=C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)),
                 n_restarts_optimizer=10,
                 alpha=0.1
                 ):
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, alpha=alpha)

    def predict(self, input):
        return self.gp.predict(input)
    
    def fit(self, input, output):
        self.gp.fit(input, output)

