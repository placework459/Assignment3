import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class Preprocessor:

    def __init__(self):
        pass

    def data_train_test_split(self,x: np.ndarray, y: np.ndarray, test_size: float=0.20):
        return train_test_split(x,y,  test_size=test_size, random_state=42)