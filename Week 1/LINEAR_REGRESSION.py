import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Compute Cost

def compute_cost(x,y,w,b):
    cost_squared = 0
    m = x.shape[0]
    for i in range(m):
        f_wb = w*x[i] + b
        cost_squared+= (f_wb-y[i])**2
    
    total_cost = 1/(2*m) * cost_squared
    return total_cost

#  Computing Gradient

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    d_dw = 0
    d_db = 0

    for i in range(m):
        f_wb = w*x[i] + b
        d_dw += (f_wb-y[i])*x[i]
        d_db += f_wb-y[i]
    
    dj_dw = d_dw/m
    dj_db = d_db/m

    return dj_dw, dj_db


# Compute Gradient Descent

def gradient_descent(x, y, w, b, num_iters, alpha, compute_cost, gradient_function):
    j_history = [] # To see if cost function actually decreases

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x,y,w,b)
        w = w - alpha* dj_dw
        b = b - alpha* dj_db

        cost = compute_cost(x,y,w,b)
        j_history.append(cost)

    return w,b,j_history


# PROJECT IMPLEMENTION

data = pd.read_csv("house_size_price.csv")

x = data["Size_sqft"].values
y = data["Price"].values

w = 0
b = 0
alpha = 0.0000001
iterations = 1000

final_w, final_b, history = gradient_descent(x, y, w, b, iterations, alpha,
                                              compute_cost, compute_gradient)

print(f"Final parameters: w = {final_w}, b = {final_b}")
print(f"First cost: {history[0]}")
print(f"Final cost: {history[-1]}")


# 5. Plot cost vs iterations

plt.plot(history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()

# Fitted Line

plt.scatter(x, y, label="Data")
plt.plot(x, final_w * x + final_b, color="red", label="Fitted line")
plt.xlabel("Size (sqft)")
plt.ylabel("Price ($1000s)")
plt.legend()
plt.show()


