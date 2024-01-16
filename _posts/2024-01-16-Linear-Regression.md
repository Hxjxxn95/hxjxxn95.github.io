---
layout : post
title: Linear Regression
author: Hojoon_Kim
date: 2024-01-16 11:15:10 +0900
categories: [Develope, ML]
tags: [Pytorch,Linear Regression, ML]
pin: true
math: true
---

## Hypothesis(가설 세우기)

$$ H(x) = Wx + b $$

``` python
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = x_train * W + b
```

## Cost Function (비용 함수)
_비용 함수(cost function) = 손실 함수(loss function) = 오차 함수(error function) = 목적 함수(objective function)_   
y = Wx + b 에서 W와 b를 구하는 것이 목표이다. 이를 위해서는 W와 b에 대한 cost function을 정의해야 한다. 

$$ mse(W,b) = \frac{1}{m}\sum_{i=1}^{m}(H(x^{(i)}) - y^{(i)})^2 $$

``` python
cost = torch.mean((hypothesis - y_train) ** 2)
```

## Gradient Descent (경사 하강법)

$$ W := W - \alpha\frac{\partial}{\partial W}cost(W) $$

- W : 접선의 기울기
- $\alpha$ : 학습률

## Pytorch 로 Linear Regression 구현하기

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1) # 랜덤 시드 고정

# 데이터
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# 모델 초기화
W = torch.zeros(1, requires_grad=True) # 가중치 W를 0으로 초기화하고, requires_grad=True를 통해 학습을 통해 값이 변경되는 변수임을 명시
b = torch.zeros(1, requires_grad=True)

# 여기까지 상태에서 예측을 하게 된다면 직선의 방정식 은 y = 0 * x + 0 이므로 모든 예측값은 0이다.

hypothesis = x_train * W + b # 가설 세우기

# 비용 함수 선언
cost = torch.mean((hypothesis - y_train) ** 2)

# 경사 하강법 구현
optimizer = optim.SGD([W, b], lr=0.01)

# gradient를 0으로 초기화
optimizer.zero_grad()
# 기울기를 초기화해야만 새로운 가중치 편향에 대해서 새로운 기울기를 구할 수 있다.
# 그렇지 않으면 기존에 계산된 기울기에 누적하여 계산한다.

# 비용 함수를 미분하여 gradient 계산
cost.backward()
# W와 b를 업데이트
optimizer.step()
```

## 다중 선형 회귀

``` python
# optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))
```

위의 코드는 x 의 개수가 3개였기 때문에 가중치를 3개를 선언해주었다. 만약 x의 개수가 100개라면 가중치도 100개를 선언해야 한다. 이는 비효율적이다. 이를 해결하기 위해 행렬 곱셈을 사용한다.

<div align="center">

$$ (x_1, x_2, x_3) \times \begin{pmatrix} w_1 \\ w_2 \\ w_3 \end{pmatrix} = (x_1w_1 + x_2w_2 + x_3w_3) $$

</div>

$$ H(X) = XW $$

``` python
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                              [93, 88, 93],
                              [89, 91, 90],
                              [96, 98, 100],
                              [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))
```

## nn.Module로 구현하는 선형 회귀

``` python
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 모델 초기화 및 선언 input_dim = 3, output_dim = 1
model = nn.Linear(3, 1)

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20

for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))
```

## 클래스로 파이토치 모델 구현하기

class 형태의 모델 구현 방식은 nn.Module 을 상속받는다. 그리고 __init__()에서 모델의 구조와 동적을 정의하는 생성자를 정의한다. super() 함수를 부르면 여기서 만든 클래스는 nn.Module 클래스의 속성들을 가지고 초기화 된다. forward() 함수는 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 함수이다.
foward() 함수는 model 객체를 데이터와 함께 호출하면 자동으로 실행이 된다.

$$ forward : \hat{y} = xW + b $$

``` python
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                              [93, 88, 93],
                              [89, 91, 90],
                              [96, 98, 100],
                              [73, 66, 70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델을 클래스로 구현

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(3,1)

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20

for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))
```


