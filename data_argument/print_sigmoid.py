import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.arange(-9.5, 10, 0.1)
y = sigmoid(z)

plt.figure(figsize=(9,6))
plt.plot(z,y)
plt.axvline(0, c='black')
plt.axhspan(.0, 1.0, facecolor='0.93', alpha=1.0, ls=':', edgecolor='0.4')
#plt.axhline(y=.5, color='.3',alpha=1.0, ls=':')
plt.yticks([.0, .5, 1.0])
plt.ylim(-.1, 1.1)
plt.title('sigmoid', fontsize=23)
plt.xlabel('x', fontsize=19)
plt.ylabel('y', fontsize=13)
