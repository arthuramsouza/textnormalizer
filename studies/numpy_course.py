import numpy as np

# Arrays

arr_1 = np.array([1, 2, 3])
arr_2 = np.arange(0, 10, 2)
arr_3 = np.zeros((3, 5))
arr_4 = np.ones((3, 5))
arr_5 = np.linspace(0, 11, 5000)
arr_6 = np.random.randint(0, 1000, (3, 3))

print('1:\n' + str(arr_1) + '\n')
print('2:\n' + str(arr_2) + '\n')
print('3:\n' + str(arr_3) + '\n')
print('4:\n' + str(arr_4) + '\n')
print('5:\n' + str(arr_5) + '\n')
print('Random numbers:\n' + str(arr_6))
print('\nReshape:\n' + str(arr_6.reshape(9)))

# Matrices

mat = np.arange(0, 100).reshape(10, 10)

print(mat)
print(mat[4, 2])
print(mat[:, 0])
print(mat[5, :])
print(mat[0:3, 0:3])
print(mat[mat > 50])
