from torchvision import transforms
import torch
import numpy as np


rows, cols = 5, 5
n_elements = rows * cols

# 生成从 1 到 n_elements 的不同整数，并打乱顺序
numbers = np.random.permutation(n_elements) + 1  # 数字从 1 开始
matrix = numbers.reshape((rows, cols))

numbers = np.random.permutation(n_elements) + 1  # 数字从 1 开始
matrix2 = numbers.reshape((rows, cols))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(3, 3),
])

l_trans = transforms.Lambda(lambda data: ({"image": transform(data[0]), "label": transform(data[1])}))

print(matrix, '\n', matrix2)
m = l_trans((matrix, matrix2))
print(m["image"], '\n',  m["label"])


