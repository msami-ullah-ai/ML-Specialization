import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("houses.csv")
X_data = data[["Size", "Bedrooms", "Age"]].values

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_data)
Y = data["Price"].values

print(f"Peak to peak value in Raw data: {np.ptp(X_data, axis=0)}")
print(f"Peak to peak value in Transformed Normalized data: {np.ptp(X_norm, axis=0)}")

sgdr = SGDRegressor(max_iter=3000)
sgdr.fit(X_norm, Y)

print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_

print(f"model parameters:\nw: {w_norm}, b:{b_norm}")


# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{Y[:4]}")

