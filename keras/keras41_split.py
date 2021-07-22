import numpy as np

a = np.array(range(1, 11))
print(a)

size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)

print(dataset)

# x = dataset[:, :-1]
# y = dataset[:, -1]

# print("x : \n", x)
# print("y : ", y)