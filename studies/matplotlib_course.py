import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10)
print(x)

y = x**2
plt.plot(x, y, 'r--')
plt.xlim(0, 4)
plt.ylim(0, 16)
plt.title('title')
plt.xlabel('x label')
plt.ylabel('y label')
plt.show()

mat = np.arange(0, 100).reshape(10, 10)

plt.imshow(mat)
plt.show()

mat = np.random.randint(0, 1000, (10, 10))

plt.imshow(mat)
plt.colorbar()
plt.show()
