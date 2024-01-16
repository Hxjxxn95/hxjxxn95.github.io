---
layout : post
title: Logistic Regression
author: Hojoon_Kim
date: 2024-01-16 15:15:10 +0900
categories: [Develope, ML]
tags: [Pytorch,Logistic Regression, ML]
pin: true
math: true
---

## Binary Classfication(이진 분류)

학생들의 시험 성적에 따라 결과가 합불이 되는 데이터가 있고, 이 데이터를 이용하여 합불을 예측하는 모델을 만들 수 있다.

[ table1 ] : 학생들의 시험 성적에 따른 합불 데이터

| 시험 성적 | 합불 |
|:--------:|:----:|
|  45점    |  0   |
|  50점    |  0   |
|  55점    |  0   |
|  60점    |  0   |
|  65점    |  1   |
|  70점    |  1   |
|  75점    |  1   |
|  80점    |  1   |

위와 같은 데이터를 이용하여 합불을 예측하는 모델을 만들 수 있다. 이때, 합불을 예측하는 모델을 이진 분류 모델이라고 한다.

이러한 점들을 표현하는 그래프는 S자의 형태로 표현된다. $$ Wx + b $$ 와 같은 직선 함수가 아니라 S자 형태로 표현할 수 있는 함수가 필요하다. 그래서 특정함수를 사용하여 $$ H(x) = f(Wx + b) $$ 와 같이 표현할 수 있다.
그리고 이 특정함수 중 널리 쓰이는 것이 시그모이드 함수이다.

## Sigmoid Function (시그모이드 함수)

$$ H(x) = sigmoid(Wx + b) = \frac{1}{1 + e^{-(Wx + b)}} $$

시그모이드 함수는 위와 같이 표현할 수 있다. 이 함수는 0과 1사이의 값을 가지며, 0.5를 기준으로 0과 1로 분류할 수 있다.

## Cost Function (비용 함수)

$$ cost(W) = -\frac{1}{m}\sum_{i=1}^{m}ylog(H(x)) + (1-y)(log(1-H(x))) $$

위와 같은 비용 함수를 사용하여 학습을 진행할 수 있다. 이 비용 함수는 실제값이 1일 때, 예측값이 1에 가까워지면 비용이 작아지고, 예측값이 0에 가까워지면 비용이 커지며, 실제값이 0일 때, 예측값이 0에 가까워지면 비용이 작아지고, 예측값이 1에 가까워지면 비용이 커진다.

## Pytorch로 구현하기

### Low Level 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
 
# 모델 초기화
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000

for epoch in range(nb_epochs):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

### nn.Module 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]

y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

optimizer = torch.optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000

for epoch in range(nb_epochs):

    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))
```

### Class 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 데이터

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000

for epoch in range( nb_epochs ) :
    
        # H(x) 계산
        hypothesis = model(x_train)
    
        # cost 계산
        cost = F.binary_cross_entropy(hypothesis, y_train)
    
        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    
        # 100번마다 로그 출력
        if epoch % 100 == 0:
            prediction = hypothesis >= torch.FloatTensor([0.5])
            correct_prediction = prediction.float() == y_train
            accuracy = correct_prediction.sum().item() / len(correct_prediction)
            print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
                epoch, nb_epochs, cost.item(), accuracy * 100,
            ))
```

## Reference
[https://wikidocs.net/57810](https://wikidocs.net/57810)
