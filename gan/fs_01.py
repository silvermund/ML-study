import torch

# x = 3.5
# y = x*x + 2
# print(x, y)

# # simple pytorch tensor
# x = torch.tensor(3.5)
# print(x)

# # 텐서를 이용한 간단한 연산
# y = x + 3
# print(y)

# # 파이토치 텐서
# x = torch.tensor(3.5, requires_grad=True)
# print(x)

# # x로부터 정의된 y
# y = (x-1) * (x-2) * (x-3)
# print(y)

# # 기울기 계산
# y.backward()

# print(x.grad)

# set up simple graph relating x, y, and z

x = torch.tensor(3.5, requires_grad=True)
y = x*x
z = 2*y + 3

# work out gradients
z.backward()

# what is gradient at x = 3.5
print(x.grad)

