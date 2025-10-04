
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('houses.csv')

# Cost Function

def costFunction(x, y, w, b):
    m = x.shape[0]
    f_wb = x @ w +b
    cost = np.sum((f_wb - y)**2)
    return cost/(2*m)

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    f_wb = x @ w +b
    error = f_wb - y
    d_dw = x.T @ error
    d_db = np.sum(error)

    return d_dw/m, d_db/m


def gradient_descent(x, y, w, b, num_iters, learning_rate, gradient, cost_function):
    cost_history = []
    for i in range(num_iters):
        dj_dw, dj_db = gradient(x, y, w, b)
        w-= learning_rate * dj_dw
        b-= learning_rate * dj_db

        if i % 100 == 0:  # Record cost every 100 iterations
            cost_history.append(cost_function(x, y, w, b))
    return w, b, cost_history

x_train = data[['Size', 'Bedrooms', 'Age']].values
y_train = data['Price'].values

w = np.zeros(x_train.shape[1])
b = 0

total_cost = costFunction(x_train, y_train , w , b)
dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)

final_w, final_b, cost_history = gradient_descent(x_train, y_train , w , b, 1000, 0.01,
                                     compute_gradient, costFunction)

# Plot 1: Cost function over iterations
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(range(0, 1000, 100), cost_history, 'b-')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function History')
plt.grid(True)

# Plot 2: Actual vs Predicted prices
plt.subplot(1, 3, 2)
y_pred = x_train @ final_w + final_b
plt.scatter(y_train, y_pred, alpha=0.7)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted')
plt.grid(True)

# Plot 3: Residuals
plt.subplot(1, 3, 3)
residuals = y_train - y_pred
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)

plt.tight_layout()
plt.show()

# 2. Feature vs Price plots
plt.figure(figsize=(15, 4))

# Size vs Price
plt.subplot(1, 3, 1)
plt.scatter(x_train[:, 0], y_train, alpha=0.7, label='Actual')
plt.scatter(x_train[:, 0], y_pred, alpha=0.7, label='Predicted', marker='x')
plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Size vs Price')
plt.legend()
plt.grid(True)

# Bedrooms vs Price
plt.subplot(1, 3, 2)
plt.scatter