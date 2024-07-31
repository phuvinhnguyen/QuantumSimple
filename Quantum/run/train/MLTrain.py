import numpy as np

def preprocess_data(data):
    X = np.array([i.edge_attr.flatten() for i in data])
    Y = np.array([i.y for i in data]).flatten()

    return X, Y

def train(
    model,
    data,
    preprocess_data=preprocess_data
    ):
    X, Y = preprocess_data(data)
    model.fit(X, Y)
