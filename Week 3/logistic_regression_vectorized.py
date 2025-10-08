import pandas as pd
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

# VECTORIZED IMPLEMENTION OF LOGISTIC REGRESSION

import numpy as np

# -------------------------------
# 1️⃣ Sigmoid Function
# -------------------------------
def sigmoid(z):
    """
    Compute the sigmoid of z
    z can be a scalar, vector, or matrix
    Returns values between 0 and 1
    """
    return 1 / (1 + np.exp(-z))


# -------------------------------
# 2️⃣ Cost Function (Vectorized)
# -------------------------------
def costFunction(x, y, w, b):
    """
    Compute the logistic regression cost function using all examples (vectorized)
    
    x : input features, shape (m, n)
    y : labels, shape (m,)
    w : weights, shape (n,)
    b : bias, scalar
    """
    m = x.shape[0]                      # number of examples
    f_wb = np.dot(x, w) + b             # linear combination for all examples, shape (m,)
    z = np.clip(sigmoid(f_wb), 1e-15, 1-1e-15) # (clip to avoid log(0))

    # Compute cost using vectorized formula
    # np.dot(y, np.log(z)) sums y_i*log(z_i) over all examples
    cost = -(np.dot(y, np.log(z)) + np.dot(1 - y, np.log(1 - z)))

    return cost / m                     # average cost over all examples


# -------------------------------
# 3️⃣ Gradient Function (Vectorized)
# -------------------------------
def computeGradient(x, y, w, b):
    """
    Compute gradients of the cost function w.r.t weights (w) and bias (b)
    
    Returns:
        d_dw : gradient w.r.t weights, shape (n,)
        d_db : gradient w.r.t bias, scalar
    """
    m = x.shape[0]              # number of examples
    f_wb = np.dot(x, w) + b     # linear combination
    z = sigmoid(f_wb)           # predicted probabilities
    error = z - y               # difference between predictions and true labels, shape (m,)

    # Vectorized gradient computation
    d_dw = x.T @ error                   # sum of (error * x_i) over all examples, shape (n,)
    d_db = np.sum(error)                 # sum of errors for bias, scalar

    # Average gradients
    d_dw /= m
    d_db /= m

    return d_dw, d_db


# -------------------------------
# 4️⃣ Gradient Descent (Vectorized)
# -------------------------------
def gradient_descent(x, y, w, b, alpha, numIter, costfunc, gradient):
    """
    Perform gradient descent to learn w and b
    
    x : input features, shape (m, n)
    y : labels, shape (m,)
    w : initial weights, shape (n,)
    b : initial bias, scalar
    alpha : learning rate
    numIter : number of iterations
    costfunc : function to compute cost
    gradient : function to compute gradients

    Returns:
        w : learned weights
        b : learned bias
        cost_history : list of cost values at intervals during training
    """
    cost_history = []                    # store cost every 10% of iterations

    for i in range(numIter):
        # 1. Compute gradients
        dj_dw, dj_db = gradient(x, y, w, b)

        # 2. Update weights and bias
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        # 3. Save cost roughly every 10% of iterations
        if i % max(1, numIter // 10) == 0:
            cost_history.append(costfunc(x, y, w, b))

    return w, b, cost_history


