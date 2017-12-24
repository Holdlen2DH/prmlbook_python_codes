"""
==================================================
Figure 1.6 in PRML
==================================================
Figure 1.6 Plots of the solutions obtained by minimizing the sum-of-squares error function using the M = 9
polynomial for N = 15 data points (left plot) and N = 100 data points (right plot). We see that increasing the
size of the data set reduces the over-fitting problem.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

lw = 2
N = 100
train_nums = [15, 100]

noise_mu = 0
noise_sig = 0.3

poly_degree = 9

def gen_sin_data(x):
    y = np.sin(2 * np.pi * x)
    return y
def gen_noised_sin_data(x):
    y = np.sin(2 * np.pi * x) + np.random.normal(loc = noise_mu, scale = noise_sig, size = len(x))[:, np.newaxis]
    # y = np.sin(2 * np.pi * x) + np.random.normal(loc = noise_mu, scale = noise_sig, size = len(X))
    return y

x = np.linspace(0, 1, num = N)[:, np.newaxis]
y = gen_sin_data(x)



plt.figure(0, figsize = (10, 5))
for i, N_train in enumerate(train_nums):

    plt.subplot(1, 2, i + 1)
    poly_model = make_pipeline(PolynomialFeatures(degree = poly_degree), LinearRegression())

    x_train = np.linspace(0, 1, num = N_train)[:, np.newaxis]
    
    y_train = gen_noised_sin_data(x_train)
    poly_model.fit(x_train, y_train)

    y_est =poly_model.predict(x)

    plt.scatter(x_train, y_train, color = 'blue', linewidth = lw)
    plt.plot(x, y, color = 'green', linewidth = lw)
    plt.plot(x, y_est, color = 'red', linewidth = lw)
    
    plt.xlabel('x')
    plt.ylabel('t')
    plt.ylim(-2, 2)
    plt.text(0.8, 0.9, 'N = ' + str(N_train))


plt.suptitle("Plots of polynomial having the order M = %d for N = %d \
data points and N = %d data points" % (poly_degree, train_nums[0], train_nums[1]))

plt.show()

