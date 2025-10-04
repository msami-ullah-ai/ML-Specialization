import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# # SALARY PREDICTION 

# def cost_function(x, y, w, b):
#     cost = 0
#     m = x.shape[0]
#     for i in range(m):
#         f_wb = w*x[i]+b
#         cost+= (f_wb - y[i])**2
#     total_cost = 1/(2*m) * cost
#     return total_cost

# def compute_gradient(x, y, w, b):
#     m = x.shape[0]
#     d_dw = 0
#     d_db = 0
#     for i in range(m):
#         f_wb = w * x[i] + b
#         d_dw += (f_wb - y[i])*x[i]
#         d_db += f_wb - y[i]
#     d_dw /= m
#     d_db /= m
#     return d_dw, d_db

# def gradient_descent(x, y, w, b, alpha, num_iters, cost_function, gradient):
#     j_history = []
#     for i in range(num_iters):
#         dj_dw, dj_db = gradient(x, y, w, b)
#         w = w - (alpha * dj_dw)
#         b = b - (alpha * dj_db)

#         cost = cost_function(x, y, w, b)
#         j_history.append(cost)
    
#     return w, b, j_history


# # IMPLEMENTATION

# data = pd.read_csv("salary_data.csv")
# x_train = data["YearsExperience"].values
# y_train = data["Salary"].values

# w = 0
# b = 0
# learning_rate = 0.01

# cost_calculation = cost_function(x_train, y_train, w, b)
# d_w, d_b = compute_gradient(x_train, y_train, w, b)
# final_w, final_b , parameters = gradient_descent(x_train, y_train, w, b, learning_rate,
#                                                  1000, cost_function, compute_gradient)


# plt.scatter(x_train, y_train, color="blue", label="Training Data")
# y_pred = final_w * x_train + final_b
# plt.plot(x_train, y_pred, color="red", label="Fitted Line")

# plt.xlabel("Years of Experience")
# plt.ylabel("Salary")
# plt.legend()
# plt.show()


