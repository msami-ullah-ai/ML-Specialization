# print("Hello World")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_train = np.array([1,2])
y_train = np.array([30000,50000])

m = x_train.shape[0]

# plt.plot(x_train, y_train, marker = "x", color = "red")
# plt.title("Relationship")
# plt.xlabel("Years of Experience")
# plt.ylabel("Total Salary")
# plt.show()

w = 10000
b = 20000

f_wb = np.zeros(m)

for num in range(len(x_train)):
    f_wb[num] = w*x_train[num] + b

trained_model = f_wb

plt.scatter(x_train, trained_model, color="red", marker = "o")
plt.show()


x_train = np.array([1,2,3,4])
y_train = np.array([30000, 50000, 70000, 90000])

m = x_train.shape[0]

f_wb = np.zeros(m)

w = 20000
b = 9000

for num in range(m):
    f_wb[num] = w*x_train[num]+b

plt.scatter(x_train, y_train, c="r", marker="o")
plt.plot(x_train, f_wb, color = "orange")
# plt.show()


# IMPLEMENTING COST FUNCTION

x_train = np.array([1, 2, 3, 4, 5])  # Years of experience
y_train = np.array([40, 50, 65, 75, 85])  # Salary in $

def cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0

    for num in range(m):
        f_wb = w*x[num] + b
        error = (f_wb-y[num])**2
        total_cost+=error

    amount = total_cost/(2*m)
    return amount

result = cost(x_train, y_train, 20,1)
print(result)


# Years of Experience (x)
x_train = np.array([1, 2, 3, 4, 5])

# Salaries (y) in $1000s
y_train = np.array([40, 50, 65, 75, 85])

def cost_function(x,y,w,b):
    m = x.shape[0]
    total = 0
    for i in range(m):
        f_wb = w*x[i]+b
        error = (f_wb-y[i])**2
        total+=error
    
    total_cost = total/(2*m)
    return total_cost


def predict(x, w, b):
    return w*x + b


# Linear Regression, Gradient Descent