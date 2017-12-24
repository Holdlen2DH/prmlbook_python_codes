"""
==================================================
Figure 1.5 in PRML
==================================================
Figure 1.5 Graphs of the root-mean-square error, defined by (1.3),
evaluated on the training set and on an independent test set for 
various values of M.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

lw = 2
N = 100      
N_train = 10

noise_mu = 0
noise_sig = 0.3

poly_degrees = range(10)

def gen_sin_data(x):
    y = np.sin(2 * np.pi * x)
    return y
def gen_noised_sin_data(x):
    y = np.sin(2 * np.pi * x) + np.random.normal(loc = noise_mu, scale = noise_sig, size = len(x))[:, np.newaxis]
    # y = np.sin(2 * np.pi * x) + np.random.normal(loc = noise_mu, scale = noise_sig, size = len(X))
    return y

x = np.linspace(0, 1, num = N)[:, np.newaxis]
y = gen_sin_data(x)



x_train = np.linspace(0, 1, num = N_train)[:, np.newaxis]
y_train = gen_noised_sin_data(x_train)

x_test = np.random.uniform(0, 1, N_train)[:, np.newaxis]
y_test = gen_noised_sin_data(x_test)

# X = np.random.uniform(0, 1, N)[:, np.newaxis]
# X = np.linspace(0, 1, num = N)[:, np.newaxis]
# Y = gen_noised_sin_data(X)
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.9)

print(x_train.shape)
print(y_train.shape)
rms_train = np.zeros(shape = (len(poly_degrees), 1))
rms_test = np.zeros(shape = (len(poly_degrees), 1))

plt.figure(0, figsize = (15, 8))
for i, poly_degree in enumerate(poly_degrees):

    plt.subplot(3, 4, i + 1)
    poly_model = make_pipeline(PolynomialFeatures(degree = poly_degree), LinearRegression())
    poly_model.fit(x_train, y_train)

    y_train_est = poly_model.predict(x_train)
    y_test_est = poly_model.predict(x_test)

    y_est =poly_model.predict(x)

    plt.scatter(x_train, y_train, color = 'blue', linewidth = lw)
    plt.plot(x, y, color = 'green', linewidth = lw)
    plt.plot(x, y_est, color = 'red', linewidth = lw)
    
    plt.xlabel('x')
    plt.ylabel('t')
    plt.ylim(-2, 2)
    plt.text(0.8, 0.9, 'M = ' + str(poly_degree))


    # rms_train[i,0] = np.sqrt(np.mean((y_train_est - y_train)**2, 0))
    # rms_test[i, 0] =np.sqrt(np.mean((y_test_est - y_test)**2, 0))
    rms_train[i,0] = np.sqrt(mean_squared_error(y_train, y_train_est));
    rms_test[i,0] = np.sqrt(mean_squared_error(y_test, y_test_est));


plt.suptitle("Plots of polynomial having various orders M")

plt.figure(1)
plt.plot(poly_degrees, rms_train, '-o', color = 'blue', label = 'Training')
plt.plot(poly_degrees, rms_test, '-o', color = 'red', label = 'Test')
plt.legend()
plt.title("Graphs of the root-men-square error")
plt.xlabel('M')
plt.ylabel(r"$E_{RMS}$")
plt.ylim(0, 1)
plt.show()

