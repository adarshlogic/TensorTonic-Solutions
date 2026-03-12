import numpy as np

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr, steps):
    
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N, D = X.shape

    # initialize parameters
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):

        # linear model
        z = X @ w + b

        # prediction probability
        p = _sigmoid(z)

        # gradients
        dw = (1/N) * (X.T @ (p - y))
        db = (1/N) * np.sum(p - y)

        # gradient descent update
        w -= lr * dw
        b -= lr * db

    return w, b