import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.linear_model import LinearRegression



class LinearRegressionModel:
    def __init__(self):
        pass
    def fit(self, x: np.ndarray, y: np.ndarray):
        self.lr= LinearRegression()
        self.lr.fit(x, y)
        self.coefficients=self.lr.coef_
        self.intercept= self.lr.intercept_
        return self.lr
    def predict(self, x: np.ndarray):
        return self.lr.predict(x)