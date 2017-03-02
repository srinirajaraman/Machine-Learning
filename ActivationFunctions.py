import numpy as np

def function_logistic(x):
    return 1.0 / (1 + np.exp(-x))

def function_d_sigmoid(y):
    return y * (1 - y)
