# importing mplot3d toolkits, numpy and matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = x+2

plt.plot(x, y)
plt.show()