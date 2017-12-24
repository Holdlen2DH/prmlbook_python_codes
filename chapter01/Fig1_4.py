"""
==================================================
Figure 1.4 in PRML
==================================================
Figure 1.4 Plots of polynomial having various orders M, shown as 
red curves, fitted to the data set shown  in Figure 1.2
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

lw = 2
N = 100      
N_train = 10;

noise_mu = 0
noise_sig = 0.2

poly_degrees = [0, 1, 3, 9]

x = np.linspace(0, 1, num = N)[:, np.newaxis]
y = np.sin(2 * np.pi * x) 

# Generate sample data
x_train = np.linspace(0, 1, num = N_train)[:, np.newaxis]
# add noise
z = np.sin(2 * np.pi * x_train) + np.random.normal(loc = noise_mu, scale = noise_sig, size = N_train)[:, np.newaxis]


plt.figure(figsize = (10, 7))
plt.suptitle('Plots of polynomials having various orders M')
for i in range(len(poly_degrees)):
    plt.subplot(2, 2, i + 1)
    # polynoimal regression
    poly_degree = poly_degrees[i]
    poly_model = make_pipeline(PolynomialFeatures(degree = poly_degree), LinearRegression())
    poly_model.fit(x_train, z)
    z_est = poly_model.predict(x)
    
    plt.scatter(x_train, z, color = 'blue', linewidth = lw)
    plt.plot(x, y, color = 'green', linewidth = lw)
    plt.plot(x, z_est, color = 'red', linewidth = lw)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.text(0.8, 0.9, 'M = ' + str(poly_degree))
    
plt.show()

