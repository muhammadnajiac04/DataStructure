import matplotlib.pyplot as plt
import numpy as np

data=np.loadtxt('data.txt')

plt.plot(data[:,0],data[:,1])
plt.xlabel("x AXIS")
plt.ylabel("y AXIS")
plt.show()
