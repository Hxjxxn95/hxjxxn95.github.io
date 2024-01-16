---
layout : post
title: SoftMax Regression
author: Hojoon_Kim
date: 2024-01-16 18:15:10 +0900
categories: [Develope, Pytorch]
tags: [Pytorch, SoftMax Regression, ML]
pin: true
math: true
---

## One-Hot Encoding
원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식이다.

```
강아지 = [1,0,0]
고양이 = [0,1,0]
병아리 = [0,0,1]
```
### One - Hot Encoding 의 무작위성
- 대부분의 다중 클래스 분류 문제가 클래스 간의 관계가 균등하기 때문에 One-Hot Encoding 은 좋은 선택지이다.
- 하지만, 모든 클래스 간의 관계에서 One - Hot Encoding 을 통해서 얻은 거리를 구해도, 전부 유클리드 거리가 동일하기 때문에 무작위석을 가지게 된다.
- 이러한 문제를 해결하기 위해서는 Word2Vec, GloVe 등의 방법을 사용한다.

## SoftMax Regression

### Multinomial Classification (다중 클래스 분류)
- 이진 분류가 두 개의 클래스 중 하나를 고르는 문제였다면, 다중 클래스 분류는 셋 이상의 클래스 중 하나를 고르는 문제이다.
- 소프트맥스 회귀는 확률의 총 합이 1이 되는 소프트맥스 함수를 이용하여 각 클래스에 대한 확률을 구하고, 가장 확률이 높은 클래스를 예측값으로 선택한다.

$$ H(x) = softmax(Wx + b) $$

### SoftMax Function (소프트맥스 함수)
- 소프트맥스 함수는 분류해야 하는 클래스의 개수를 k 라 할 때 k 차원의 벡터를 입력받아 각 클래스에 대한 확률을 추정한다.

- k 차원의 벡터에서 i 번째 원소를 $$ z_i $$ 라고하고, 정답일 확률을 $$ p_i $$ 라고 하면, 소프트맥스 함수는 아래와 같다.

$$ p_i = \frac{e^{z_i}}{\sum_{j=1}^{k}e^{z_j}} for~i = 1,2,3...k $$

$$ softmax(z) = [ \frac{e^{z_1}}{\sum_{j=1}^{k}e^{z_j}} , \frac{e^{z_2}}{\sum_{j=1}^{k}e^{z_j}} , \frac{e^{z_3}}{\sum_{j=1}^{k}e^{z_j}} , ... , \frac{e^{z_k}}{\sum_{j=1}^{k}e^{z_j}} ] $$

- 각각의 출력값은 입력값의 전부 다른 가중치를 가지고 차원 변경을 하는 것이다.

### Cross Entropy Loss (크로스 엔트로피 손실 함수)
- 크로스 엔트로피 손실 함수는 소프트맥스 회귀에서 사용하는 손실 함수이다.
- 아래에서는 $$ y $$ 는 실제값, $$ k $$ 는 클래스의 개수, $$ y_i $$ 는 실제값 원-핫 벡터의 i 번째 인덱스를 의미한다. $$ p_i $$ 는 예측값 벡터의 i 번째 인덱스를 의미한다.

$$ cost(W) = -\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{k}y_{i}log(p_{i}) $$

### SoftMax Regression with Pytorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1, 2, 1, 1],
                             [2, 1, 3, 2],
                             [3, 1, 3, 4],
                             [4, 1, 5, 5],
                             [1, 7, 5, 5],
                             [1, 2, 5, 6],
                             [1, 6, 6, 6],
                             [1, 7, 7, 7]])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])
```


#### Low Level 구현

```python
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1) #dim = 1, label = y_train.unsqueeze(1), value = 1
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros((1,3), requires_grad=True)

optimizer = optim.SGD([W,b], lr=0.1)

nb_epochs = 1000

for epoch in range(nb_epochs):
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

```

#### High Level 구현

```python
# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000

for epoch in range(nb_epochs):

    # Cost 계산 (1)
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)

    # cost로 H(x) 개선 (2)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

#### nn.Module 구현

```python
# 모델 초기화
model = nn.Linear(4, 3)

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000

for epoch in range(nb_epochs):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

#### Class 구현

```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000

for epoch in range(nb_epochs):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```




