
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # IMPLEMENTING COST FUNCTION

# def cost_function(x, y, w, b):
#     m = x.shape[0]
#     cost = 0
#     for i in range(m):
#         f_wb = np.dot(w, x[i])+b
#         cost+= (f_wb - y[i])**2
    
#     total_cost = cost / (2*m)
#     return total_cost


# def compute_gradient(x, y, w, b):
#     d_dw = 0
#     d_db = 0
#     m = x.shape[0]

#     for i in range(m):
#         f_wb = np.dot(w, x[i])+b
#         d_dw += (f_wb - y[i])*x[i]
#         d_db += f_wb - y[i]
    
#     d_dw/= m
#     d_db/= m

#     return d_dw, d_db


# def gradient_descent(x, y, w, b, num_iters, learning_rate, cost_function, gradient):
#     j_history = []

#     for i in range(num_iters):
#         dj_dw, dj_db = gradient(x, y, w, b)
#         w = w - (learning_rate * dj_dw)
#         b = b - (learning_rate * dj_db)

#         cost = cost_function(x, y, w, b)
#         j_history.append(cost)
    
#     return dj_dw, dj_db, j_history


# data = pd.read_csv("calories.csv")
# x_train = data["Distance_km"].values
# y_train = data["Calories"].values

# w = 0
# b = 0

# alpha = 0.01
# iterations = 1000

# total_cost = cost_function(x_train, y_train, w, b)
# d_dw, d_dj = compute_gradient(x_train, y_train, w, b)
# final_w, final_b, parameters = gradient_descent(x_train, y_train, w, b, iterations,
#                                                 alpha, cost_function, compute_gradient)


# plt.scatter(x_train, y_train, color="green", label="Data")
# plt.plot(x_train, final_w * x_train + final_b, color="red", label="Fitted Line")
# plt.xlabel("Distance (km)")
# plt.ylabel("Calories Burned")
# plt.legend()
# plt.show()



# PREDICTING HOUSING PRICE

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def costFunction(x, y, w, b):
#     m = x.shape[0]
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
#         d_db += (f_wb - y[i])
    
#     return d_dw/m, d_db/m


# def gradient_descent(x, y, w, b, num_iters, learning_rate, cost_function, gradient):
#     parameter_history = []

#     for i in range(num_iters):
#         dj_dw, dj_db = gradient(x, y, w, b)
#         w -= learning_rate * dj_dw
#         b -= learning_rate * dj_db

#         cost = cost_function(x, y, w, b)
#         parameter_history.append(cost)
    
#     return w, b, parameter_history


# data = pd.read_csv("houses.csv")

# x_train = data['Size (sqft)'].values
# y_train = data['Price ($1000s)'].values

# w = 0
# b = 0

# alpha = 0.01
# itr = 1000

# total_cost = costFunction(x_train, y_train, w, b)
# d_dw, d_db = compute_gradient(x_train, y_train, w, b)

# final_w, final_b, history = gradient_descent(x_train, y_train, w, b, itr, alpha,
#                                              costFunction, compute_gradient)


# print(f"Final parameters: w = {final_w:.4f}, b = {final_b:.4f}")
# print(f"Final cost: {history[-1]:.4f}")


# # Prediction function
# def predict(x, w, b):
#     return w * x + b

# # Scatter plot of training data
# plt.scatter(x_train, y_train, color='blue', label="Training Data")

# # Sort x values before plotting regression line
# x_sorted = np.sort(x_train)
# y_pred = predict(x_sorted, final_w, final_b)

# # Plot regression line
# plt.plot(x_sorted, y_pred, color='red', label="Regression Line")

# plt.xlabel("Size (sqft)")
# plt.ylabel("Price ($1000s)")
# plt.title("Housing Price Prediction")
# plt.legend()
# plt.show()








# PREDICTING HOUSING PRICE

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def costFunction(x, y, w, b):
#     m = x.shape[0]
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
#         d_db += (f_wb - y[i])
    
#     return d_dw/m, d_db/m

# def gradient_descent(x, y, w, b, num_iters, learning_rate, cost_function, gradient):
#     parameter_history = []

#     for i in range(num_iters):
#         dj_dw, dj_db = gradient(x, y, w, b)
#         w -= learning_rate * dj_dw
#         b -= learning_rate * dj_db

#         cost = cost_function(x, y, w, b)
#         parameter_history.append(cost)
    
#     return w, b, parameter_history

# # Load data
# data = pd.read_csv("houses.csv")

# x_train = data['Size (sqft)'].values
# y_train = data['Price ($1000s)'].values

# # FEATURE SCALING - CRITICAL FIX
# def feature_scaling(x):
#     return (x - np.mean(x)) / np.std(x)

# def target_scaling(y):
#     return (y - np.mean(y)) / np.std(y)

# # Scale the features and target
# x_train_scaled = feature_scaling(x_train)
# y_train_scaled = target_scaling(y_train)

# w = 0
# b = 0

# alpha = 0.1  # Increased learning rate for scaled data
# itr = 1000

# # Test cost function with scaled data
# total_cost = costFunction(x_train_scaled, y_train_scaled, w, b)
# print(f"Initial cost with scaled data: {total_cost:.4f}")

# d_dw, d_db = compute_gradient(x_train_scaled, y_train_scaled, w, b)
# print(f"Initial gradients: d_dw = {d_dw:.4f}, d_db = {d_db:.4f}")

# # Run gradient descent with SCALED data
# final_w, final_b, history = gradient_descent(x_train_scaled, y_train_scaled, w, b, itr, alpha,
#                                              costFunction, compute_gradient)

# print(f"Final parameters: w = {final_w:.4f}, b = {final_b:.4f}")
# print(f"Final cost: {history[-1]:.4f}")

# # Prediction function for scaled data
# def predict_scaled(x_scaled, w, b):
#     return w * x_scaled + b

# # Convert parameters back to original scale
# def convert_to_original_scale(w_scaled, b_scaled, x_original, y_original):
#     x_mean = np.mean(x_original)
#     x_std = np.std(x_original)
#     y_mean = np.mean(y_original)
#     y_std = np.std(y_original)
    
#     w_original = w_scaled * (y_std / x_std)
#     b_original = b_scaled * y_std + y_mean - w_scaled * (y_std / x_std) * x_mean
    
#     return w_original, b_original

# # Convert scaled parameters back to original scale
# w_original, b_original = convert_to_original_scale(final_w, final_b, x_train, y_train)
# print(f"Original scale parameters: w = {w_original:.4f}, b = {b_original:.4f}")

# # Prediction function for original scale
# def predict_original(x, w, b):
#     return w * x + b

# # Create plot with original scale data
# plt.figure(figsize=(12, 5))

# # Plot 1: Scaled data with regression line
# plt.subplot(1, 2, 1)
# plt.scatter(x_train_scaled, y_train_scaled, color='blue', label="Scaled Training Data")
# x_sorted_scaled = np.sort(x_train_scaled)
# y_pred_scaled = predict_scaled(x_sorted_scaled, final_w, final_b)
# plt.plot(x_sorted_scaled, y_pred_scaled, color='red', label="Scaled Regression Line")
# plt.xlabel("Size (scaled)")
# plt.ylabel("Price (scaled)")
# plt.title("Scaled Housing Price Prediction")
# plt.legend()

# # Plot 2: Original data with regression line
# plt.subplot(1, 2, 2)
# plt.scatter(x_train, y_train, color='blue', label="Training Data")
# x_sorted = np.sort(x_train)
# y_pred = predict_original(x_sorted, w_original, b_original)
# plt.plot(x_sorted, y_pred, color='red', label="Regression Line")
# plt.xlabel("Size (sqft)")
# plt.ylabel("Price ($1000s)")
# plt.title("Housing Price Prediction (Original Scale)")
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Print prediction examples
# print("\nPrediction Examples:")
# test_sizes = [1000, 1500, 2000, 2500, 3000]
# for size in test_sizes:
#     price = predict_original(size, w_original, b_original)
#     print(f"Size: {size} sqft -> Predicted Price: ${price:.2f} thousands")



# import pandas as pd
# import numpy as np
# import time

# data = pd.read_csv("houses.csv")
# x_train = data['Size (sqft)'].values
# y_train = data['Price ($1000s)'].values

# start_time = time.time()

# def costFunction(x, y, w, b):
#     m = x.shape[0]
#     cost = 0
#     for i in range(m):
#         f_wb = np.dot(x,y)+b
#         cost += (f_wb - y[i])**2
#     cost/= m
#     return cost

# end_time = time.time()

# print(f"Total time took {end_time - start_time:.4f}")
# w = 0
# b = 0

# print(costFunction(x_train, y_train, w, b))



# import numpy as np

# x = np.array([
#     [45,67],
#     [32,78],
#     [89,14]
# ])
# w = np.array([4,6])
# # print(x.shape)
# # print(w.shape)
# # print(x @ w)

# print(x[0:,1])

# import numpy as np
# import tensorflow as tf
# print(tf.__version__)
# from tensorflow.keras.datasets()


# Loads MNIST dataset from TensorFlow's mirror
# X_train -> 60,000 images
# y-train -> Hold Labels for each image
# X_test  -> selected 10000 images for testing
# y_test  -> 10000 labels of testing images
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

# import tensorflow as tf
# print(tf.__version__)

