import matplotlib.pyplot as plt
from studies.snnf.network import *
from studies.snnf.functions import *

g = Graph()
g.set_as_default()

A = Variable(10)
b = Variable(1)
x = Placeholder()
y = Multiply(A, x)
z = Add(y, b)

sess = Session()
result = sess.run(operation=z, feed_dict={x: 10})
print(result)

# Classification

sample_z = np.linspace(-10, 10, 100)
sample_a = sigmoid(sample_z)
plt.plot(sample_z, sample_a)
plt.show()
