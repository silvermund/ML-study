import torch

# x, y, z에 관한 간단한 그래프 설정
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)


x = 2*a + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y

# 기울기 계산
z.backward()

# a = 2.0일 떄 기울기
print(a.grad)


