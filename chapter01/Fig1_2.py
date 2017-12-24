
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

lw = 2
N = 100      
N_train = 10

noise_mu = 0
noise_sig = 0.2

poly_degree = 9

rng = np.random.RandomState(0)
x = np.linspace(0, 1, num = N)
y = np.sin(2 * np.pi * x) 

# Generate sample data
x_train = np.linspace(0, 1, num = N_train)
# add noise
# z = np.sin(2 * np.pi * x_train) + (noise_sig * np.random.randn(N_train) + noise_mu) 
z = np.sin(2 * np.pi * x_train) + np.random.normal(loc = noise_mu, scale = noise_sig, size = N_train)


plt.scatter(x_train, z, color = 'blue', linewidth = lw)
plt.plot(x, y, color = 'green', linewidth = lw)


plt.xlabel('x')
plt.ylabel('t')
plt.show()

