# import numpy as np
# import pandas as pd


# def cost_function(x, y, w, b):
#     cost = 0.0
#     m = x.shape[0]
#     for i in range(m):
#         f_wb = np.dot(w * x[i]) + b
#         cost+= (f_wb - y[i])**2
#     total_cost = cost / (2*m)
#     return total_cost

# def compute_gradient(x, y, w, b):
#     m,n = x.shape
#     d_dw = np.zeros((n,)) 
#     d_db = 0
#     for i in range(m):
#         f_wb = (np.dot(w, x[i]) + b) - y[i]
#         for j in range(n):
#             d_dw += f_wb * x[i,j]
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

#         if i<100000:      # prevent resource exhaustion 
#             j_history.append( cost_function(x, y, w, b))
    
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




# DISTANCE VS CALORIES

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# data = pd.read_csv("calories.csv")
# x_train = data["Distance_km"].values
# y_train = data["Calories"].values


# def cost_function(x, y, w, b):
#     m = x_train.shape[0]
#     cost = 0

#     for i in range(m):
#         f_wb = w * x[i] + b
#         cost += (f_wb - y[i])**2
#     total_cost = cost/(2*m)
#     return total_cost

# def compute_gradient(x, y, w, b):
#     m = x.shape[0]
#     d_dw = 0
#     d_db = 0

#     for i in range(m):
#         f_wb = w * x[i] + b
#         d_dw += (f_wb - y[i])*x[i]
#         d_db += f_wb - y[i]
    
#     d_dw/= m
#     d_db/= m

#     return d_dw, d_db


# def gradient_descent(x, y, w, b, alpha, num_iters, costFunction, gradient):
#     j_history = []
#     for i in range(num_iters):
#         dj_dw, dj_db = gradient(x, y, w, b)
#         w = w - (alpha * dj_dw)
#         b = b - (alpha * dj_db)

#         if i<1000:
#             j_history.append(costFunction(x, y, w, b))
    
#     return w, b, j_history


# w = 0
# b = 0

# num_iterations = 1000
# learning_rate = 0.001

# total_cost = cost_function(x_train, y_train, w, b)
# d_w, d_b = compute_gradient(x_train, y_train, w, b)
# final_w, final_b, parameters = gradient_descent(x_train, y_train, w, b, learning_rate, 
#                                         num_iterations, cost_function, compute_gradient)

# plt.scatter(x_train, y_train, color = "green", label = "Data")
# actual_y = final_w * x_train + final_b
# plt.plot(x_train, actual_y, color = "red", label = "Fitted Line")
# plt.xlabel("Distance")
# plt.ylabel("Calories")
# plt.legend()
# plt.show()




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # ============================
# # 1. Load Data
# # ============================
# data = pd.read_csv("calories.csv")
# x_train = data["Distance_km"].values
# y_train = data["Calories"].values

# # ============================
# # 2. Define Cost Function
# # ============================
# def cost_function(x, y, w, b):
#     m = x.shape[0]
#     cost = 0
#     for i in range(m):
#         f_wb = w * x[i] + b
#         cost += (f_wb - y[i])**2
#     return cost / (2 * m)

# # ============================
# # 3. Compute Gradient
# # ============================
# def compute_gradient(x, y, w, b):
#     m = x.shape[0]
#     d_dw = 0
#     d_db = 0
#     for i in range(m):
#         f_wb = w * x[i] + b
#         d_dw += (f_wb - y[i]) * x[i]
#         d_db += (f_wb - y[i])
#     return d_dw / m, d_db / m

# # ============================
# # 4. Gradient Descent
# # ============================
# def gradient_descent(x, y, w, b, alpha, num_iters, costFunction, gradient):
#     j_history = []
#     for i in range(num_iters):
#         dj_dw, dj_db = gradient(x, y, w, b)
#         w -= alpha * dj_dw
#         b -= alpha * dj_db

#         j_history.append(costFunction(x, y, w, b))
#         if i % 100 == 0:
#             print(f"Iteration {i}: Cost {j_history[-1]:.4f}, w={w:.4f}, b={b:.4f}")
#     return w, b, j_history

# # ============================
# # 5. Train Model
# # ============================
# w, b = 0, 0
# num_iterations = 1000
# learning_rate = 0.01

# final_w, final_b, cost_history = gradient_descent(
#     x_train, y_train, w, b, learning_rate, num_iterations,
#     cost_function, compute_gradient
# )

# print("\nTraining complete.")
# print(f"Final parameters: w = {final_w:.4f}, b = {final_b:.4f}")

# # ============================
# # 6. Plot Cost Convergence
# # ============================
# plt.plot(range(num_iterations), cost_history)
# plt.xlabel("Iteration")
# plt.ylabel("Cost")
# plt.title("Cost Function Convergence")

# # ============================
# # 7. Plot Regression Line
# # ============================
# plt.scatter(x_train, y_train, color="green", label="Data")
# plt.plot(x_train, final_w * x_train + final_b, color="red", label="Fitted Line")
# plt.xlabel("Distance (km)")
# plt.ylabel("Calories Burned")
# plt.legend()
# plt.show()

# # ============================
# # 8. Make Predictions
# # ============================
# def predict(x, w, b):
#     return w * x + b

# print("\nPredictions:")
# for distance in [12, 15, 20]:
#     print(f"Distance {distance} km -> {predict(distance, final_w, final_b):.2f} calories")





# import matplotlib.pyplot as plt
# import numpy as np

# # Define points A and B
# A = (2, 3)
# B = (-1, 3)

# # Calculate the slope of line AB
# # slope = (y2 - y1) / (x2 - x1)
# slope_AB = (B[1] - A[1]) / (B[0] - A[0])

# # Calculate the slope of perpendicular lines
# # If slope of AB is m, then slope of perpendicular line is -1/m
# slope_perpendicular = -1 / slope_AB

# print(f"Point A: {A}")
# print(f"Point B: {B}")
# print(f"Slope of line AB: {slope_AB}")
# print(f"Slope of perpendicular lines: {slope_perpendicular}")

# # Create the plot
# plt.figure(figsize=(10, 8))

# # Plot the points
# plt.plot(A[0], A[1], 'ro', markersize=8, label=f'A{A}')
# plt.plot(B[0], B[1], 'bo', markersize=8, label=f'B{B}')

# # Create line AB
# x_range = np.linspace(-4, 4, 100)
# # Using point-slope form: y - y1 = m(x - x1)
# y_AB = slope_AB * (x_range - A[0]) + A[1]
# plt.plot(x_range, y_AB, 'g-', linewidth=2, label=f'Line AB (slope = {slope_AB})')

# # Create a perpendicular line passing through point A
# y_perp = slope_perpendicular * (x_range - A[0]) + A[1]
# plt.plot(x_range, y_perp, 'r--', linewidth=2, label=f'Perpendicular line (slope = {slope_perpendicular:.3f})')

# # Add grid and labels
# plt.grid(True, alpha=0.3)
# plt.axhline(y=0, color='k', linewidth=0.5)
# plt.axvline(x=0, color='k', linewidth=0.5)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Line AB and Perpendicular Line')
# plt.legend()

# # Set axis limits for better visualization
# plt.xlim(-4, 4)
# plt.ylim(-4, 3)

# # Add annotations for the points
# plt.annotate('A(-2, 1)', xy=A, xytext=(A[0]+0.3, A[1]+0.3), 
#             arrowprops=dict(arrowstyle='->', color='red'))
# plt.annotate('B(2, -2)', xy=B, xytext=(B[0]+0.3, B[1]-0.3), 
#             arrowprops=dict(arrowstyle='->', color='blue'))

# plt.tight_layout()
# plt.show()

# # Additional calculations and verification
# print("\n" + "="*50)
# print("DETAILED CALCULATIONS:")
# print("="*50)
# print(f"1. Slope of line AB:")
# print(f"   m = (y₂ - y₁) / (x₂ - x₁)")
# print(f"   m = ({B[1]} - {A[1]}) / ({B[0]} - ({A[0]}))")
# print(f"   m = {B[1] - A[1]} / {B[0] - A[0]}")
# print(f"   m = {slope_AB}")

# print(f"\n2. Slope of perpendicular lines:")
# print(f"   m_perp = -1 / m_AB")
# print(f"   m_perp = -1 / {slope_AB}")
# print(f"   m_perp = {slope_perpendicular}")

# print(f"\n3. Equation of line AB:")
# print(f"   Using point-slope form: y - y₁ = m(x - x₁)")
# print(f"   y - {A[1]} = {slope_AB}(x - ({A[0]}))")
# print(f"   y - {A[1]} = {slope_AB}(x + 1)")
# print(f"   y = {slope_AB}x + {slope_AB + A[1]}")

# print(f"\n4. General equation of perpendicular lines:")
# print(f"   y - y₁ = {slope_perpendicular}(x - x₁)")
# print(f"   where (x₁, y₁) is any point on the perpendicular line")



import numpy as np
import matplotlib.pyplot as plt


# LOGISTIC REGRESSION

# def sigmoid(z):
#     return 1/(1+np.exp(-z))


# def costFunction(x, y, w, b):
#     cost = 0
#     m = x.shape[0]
#     for i in range(m):
#         f_wb = np.dot(x[i], w) +b
#         z = sigmoid(f_wb)
#         cost+= (y[i] * np.log(z)) + (1-y[i])*(np.log(1-z))
#     cost/= -m
#     return cost


# def computeGradient(x, y, w, b):
#     m = x.shape[0]
#     d_dw = np.zeros_like(w)
#     d_db = 0
#     for i in range(m):
#         f_wb = np.dot(x[i], w) +b
#         z = np.clip(sigmoid(f_wb), 1e-15, 1-1e-15)
#         error = z - y[i]
#         d_dw+= error * x[i]
#         d_db+= error
    
#     d_dw/= m
#     d_db/= m
#     return d_dw, d_db



# def gradient_descent(x, y, w, b, alpha, numIter, costfunc, gradient):
#     cost_history = []
#     for i in range(numIter):
#         dj_dw, dj_db = gradient(x, y, w, b)
#         w = w - (alpha * dj_dw)
#         b = b - (alpha * dj_db)
    
#     if i % max(1, numIter // 10) == 0:
#         cost_history.append(costfunc(x, y, w, b))

#     return w, b, cost_history




# // VECTORIZED FORM

# def sigmoid(z):
#     return 1/(1+np.exp(-z))


# def costFunction(x, y, w, b):
#     m = x.shape[0]
#     f_wb = np.dot(x, w) +b
#     z = np.clip(sigmoid(f_wb), 1e-15, 1- 1e-15)
#     cost = -(np.dot(y, np.log(z))+np.dot(1-y, np.log(1-z)))
#     return cost/m


# def computeGradient(x, y, w, b):
#     m = x.shape[0]
#     f_wb = np.dot(x, w) +b
#     z = sigmoid(f_wb)
#     error = z - y
#     d_dw = x.T @ error
#     d_db = np.sum(error)
    
#     d_dw/= m
#     d_db/= m
#     return d_dw, d_db



# def gradient_descent(x, y, w, b, alpha, numIter, costfunc, gradient):
#     cost_history = []
#     for i in range(numIter):
#         dj_dw, dj_db = gradient(x, y, w, b)
#         w = w - (alpha * dj_dw)
#         b = b - (alpha * dj_db)
    
#         if i % max(1, numIter // 10) == 0:
#             cost_history.append(costfunc(x, y, w, b))

#     return w, b, cost_history

