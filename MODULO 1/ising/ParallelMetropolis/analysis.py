import numpy as np
import matplotlib.pyplot as plt

mag = np.loadtxt("mag.txt", unpack=True)
mag = np.reshape(mag,[4,64])

chi = np.loadtxt("chi.txt", unpack=True)
chi = np.reshape(chi,[4,64])

plt.figure(1)
for i in range(4):
    plt.plot(np.abs(mag[i]), marker='+')
plt.figure(2)
for i in range(4):
    plt.plot(chi[i], marker='+')
plt.show()
