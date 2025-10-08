import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    exp = 1/(1+np.exp(-z))
    return exp

arr = np.arange(-10,11)
g_z = sigmoid(arr)
np.set_printoptions(precision=4)
print("Input (z), Sigmoid(z)")
print(np.c_[arr,g_z])


# PLOTTING

fig,axe = plt.subplots(1,1,figsize=(5,3))
axe.plot(arr, g_z, c="b")
axe.set_title("Sigmoid function")
axe.set_ylabel('sigmoid(z)')
axe.set_xlabel('z')
plt.show()
