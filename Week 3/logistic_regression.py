import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1+np.exp(-z))


# Cost Function (Unvectorized)
def cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = np.dot(x[i], w) + b         # linear combination for one example
        # To Prevent Exact Log 0, 1 we'll clip f_wb value
        # np.clip(value, min_val, max_val) keeps the number within a range:
        # If value < min_val, it becomes min_val
        # If value > max_val, it becomes max_val
        # Otherwise, it stays the same
        z = np.clip(sigmoid(f_wb), 1e-15, 1-1e-15)
        # 1e-15 -> almost 0, 1- 1e-15 -> almost 1
        cost+= -(y[i] * np.log(z) + (1-y[i]) * np.log(1-z))
        cost/= m
    return cost


# 3️⃣ Gradient Function (Unvectorized)
# -------------------------------
def computeGradient(x, y, w, b):
    """
    Compute gradients of the cost function w.r.t weights (w) and bias (b) using loops
    
    Returns:
        d_dw : gradient w.r.t weights, shape (n,)
        d_db : gradient w.r.t bias, scalar
    """
    m = x.shape[0]
    d_dw = np.zeros_like(w)   # initialize weight gradients
    d_db = 0                  # initialize bias gradient

    # Loop over all examples
    for i in range(m):
        f_wb = np.dot(x[i], w) + b       # linear combination for one example
        z = sigmoid(f_wb)                # predicted probability
        error = z - y[i]                 # difference between prediction and true label

        d_dw += error * x[i]             # accumulate gradient for weights
        d_db += error                    # accumulate gradient for bias

    d_dw /= m   # average gradient over all examples
    d_db /= m
    return d_dw, d_db   


def gradient_descent(x, y, w_init, b, num_iters, alpha, costFunction, gradient):
    j_history = []
    w = w_init

    for i in range(num_iters):
        dj_dw, dj_db = gradient(x, y, w, b)
        w-= alpha * dj_dw
        b-= alpha * dj_db

        if i % max(1, num_iters//10)==0:
            j_history.append(costFunction(x, y, w, b))

    return w, b, j_history


# PROJECT

data = pd.read_csv('default.csv')

x_train = data[['age','income','credit_score']].values
y_train = data['target']

w = w = np.zeros(x_train.shape[1])
b = 0
iterations = 10000
learning_rate = 0.01

final_w, final_b, cost_history = gradient_descent(x_train, y_train, w, b, iterations,
                                            learning_rate, cost_function, computeGradient)

print('Initial Cost: ', cost_history[0])
print('Final Cost: ', cost_history[-1])
print(f'Fitted Parameters\nW: {final_w}\nb: {final_b}')

