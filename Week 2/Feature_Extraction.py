# import numpy as np

# # Features: [Size (sqft), Bedrooms]
# x = np.array([[1000, 2],
#               [1500, 3],
#               [2000, 4],
#               [2500, 5]])

# # Target: House price in 1000s
# y = np.array([200, 300, 400, 500])


# # NO SCALING

# # Initialize weights
# w = np.zeros(2)
# b = 0

# # Learning rate
# alpha = 0.0000001

# # Simulate 1 iteration of gradient descent
# m = x.shape[0]
# for i in range(m):
#     err = (x[i,0]*w[0] + x[i,1]*w[1] + b) - y[i]
#     w[0] -= alpha * err * x[i,0]  # weight for size
#     w[1] -= alpha * err * x[i,1]  # weight for bedrooms
#     b -= alpha * err

# print("Weights without scaling:", w)

# # Min-Max scaling
# x_scaled = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
# # print("Scaled x:\n", x_scaled)

# # Initialize weights
# w = np.zeros(2)
# b = 0
# alpha = 0.1  # can use bigger learning rate now

# # 1 iteration of gradient descent
# for i in range(m):
#     err = (x_scaled[i,0]*w[0] + x_scaled[i,1]*w[1] + b) - y[i]
#     w[0] -= alpha * err * x_scaled[i,0]
#     w[1] -= alpha * err * x_scaled[i,1]
#     b -= alpha * err

# print("Weights with scaling:", w)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def costFunction(x, y, w, b):
    m = x.shape[0]
    f_wb = x @ w +b
    cost = np.sum((f_wb - y)**2)
    return cost/(2*m)

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    f_wb = x @ w + b
    error = f_wb - y
    d_dw = x.T @ error
    d_db = np.sum(error) 
    return d_dw/m, d_db/m

def gradient_descent(x, y, w, b, num_iters, learning_rate, cost_function, gradient):
    j_history = []
    for i in range(num_iters):
        dj_dw, dj_db = gradient(x, y, w, b)
        w-= learning_rate * dj_dw
        b-= learning_rate * dj_db

        j_history.append(cost_function(x, y, w, b))

    return w,b, j_history



# IMPLEMENTING

# Feature 1: Size of house (large values)
size = np.array([1000, 2000, 3000, 4000, 5000])

# Feature 2: Number of bedrooms (small values)
bedrooms = np.array([1, 2, 3, 4, 5])

# Target: Price (just for example)
price = np.array([300, 500, 700, 900, 1100])

# Combine features
x = np.column_stack((size, bedrooms))
y = price



# initialize parameters
w = np.zeros(x.shape[1])
b = 0
alpha = 0.00000001   # very small, otherwise it may diverge
num_iters = 25

total_cost = costFunction(x, y, w, b)
dj_dw, dj_db = compute_gradient(x, y, w, b)
final_w, final_b, j_history = gradient_descent(x, y, w, b, num_iters, alpha, 
                                               costFunction, compute_gradient)

final_w, final_b, J_hist = gradient_descent(
    x, y, w, b, num_iters, alpha, costFunction, compute_gradient
)



print ("BEFORE FEATURE SELECTION")
print("Final weights:", final_w)
print("Final bias:", final_b)
print("Final cost:", J_hist[-1])



# IMPLEMENTING FEATURE SCALING 

x_bar = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
w = np.zeros(x_bar.shape[1])
# print("\nOriginal: ", x)
# print("\nScaled: ", x_bar)

print ("AFTER FEATURE SELECTION")
# MUST CHANGE LEARNING RATE 

total_cost = costFunction(x_bar, y, w, b)
dj_dw, dj_db = compute_gradient(x_bar, y, w, b)
w_final, b_final, j_history = gradient_descent(x_bar, y, w, b, num_iters, 0.1, 
                                               costFunction, compute_gradient)

w_final, b_final, J_hist = gradient_descent(
    x_bar, y, w, b, num_iters, 0.1, costFunction, compute_gradient
)
print("Final weights:", w_final)
print("Final bias:", b_final)
print("Final cost:", J_hist[-1])



# PLOTTING

# FINAL PREDICTIONS before and after scaling
y_pred_before = x @ np.array(final_w) + final_b
y_pred_after = x_bar @ np.array(w_final) + b_final

# PLOT
plt.figure(figsize=(10,5))

# Before Scaling
plt.subplot(1,2,1)
plt.scatter(y, y_pred_before, color="red")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="blue", linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Before Feature Scaling")

# After Scaling
plt.subplot(1,2,2)
plt.scatter(y, y_pred_after, color="green")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="blue", linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("After Feature Scaling")

plt.tight_layout()
plt.show()



# LEARNING CURVE OVER TIME

# Learning curve BEFORE feature scaling
plt.figure(figsize=(10,4))
plt.plot(J_hist, color='red')
plt.xlabel("Iterations")
plt.ylabel("Cost J(W,B)")
plt.title("Learning Curve Before Feature Scaling")
plt.show()

# Learning curve AFTER feature scaling
plt.figure(figsize=(10,4))
plt.plot(j_history, color='green')
plt.xlabel("Iterations")
plt.ylabel("Cost J(W,B)")
plt.title("Learning Curve After Feature Scaling")
plt.show()
