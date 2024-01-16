---
layout : post
title: Mini Batch and Data Load
author: Hojoon_Kim
date: 2024-01-16 13:15:10 +0900
categories: [Develope, ML]
tags: [Pytorch, ML, Mini Batch, Data Load]
pin: true
math: true
---

## Mini Batch and Batch Size

머신러닝은 병렬 학습을 하기에 적합한 분야이다. 병렬 학습을 하기 위해서는 데이터를 여러 개로 나누어서 학습을 시켜야 한다. 이때 데이터를 나누는 단위를 **배치(batch)** 라고 한다. 배치는 보통 2의 제곱수로 설정한다. 배치의 크기를 배치 크기(batch size) 라고 한다. 배치 크기는 보통 2의 제곱수로 설정한다. 배치 크기가 2의 제곱수이면 메모리 관리 측면에서 효율적이다. 또한 배치의 크기가 너무 작으면 학습 속도가 느려지고, 너무 크면 학습이 잘 되지 않는다.

- 데이터 전체에 대해서 한 번에 Gradient Descent를 수행하는 방법을 **Batch Gradient Descent** 라고 한다.
- 데이터를 나누어서 Gradient Descent를 수행하는 방법을 **Mini-Batch Gradient Descent** 라고 한다.
- Batch Gradient Descent 는 전체 데이터로 최적화시키므로 안정적으로 되지만 시간이 오래 걸린다.
- Mini-Batch Gradient Descent 는 전체 데이터가 아닌 일부 데이터로만 계산하므로 최적화의 정확도가 떨어지지만, 계산 시간이 빠르다.

## Data Load

Pytorch에서는 데이터를 불러오는 방법으로 2가지를 제공한다.

- torch.utils.data.Dataset : 데이터셋을 나타내는 추상 클래스
- torch.utils.data.DataLoader : 데이터셋을 불러오는 클래스

### Dataset

Dataset은 데이터셋을 나타내는 추상 클래스이다. Dataset을 상속받아서 직접 데이터셋을 만들 수 있다. Dataset을 상속받아서 만들어진 클래스는 반드시 아래의 3개 함수를 구현해야 한다.

- __len__(self) : 데이터셋의 총 데이터 수를 반환하는 함수
- __getitem__(self, idx) : 데이터셋에서 특정 1개의 샘플을 가져오는 함수
- __add__(self, other) : 두 데이터셋을 연결하는 함수

``` python
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CutomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y
```
TensorDataset 을 사용하면 위의 코드를 아래와 같이 간단하게 구현할 수 있다.
``` python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

dataset = TensorDataset(x_train, y_train)
```
### DataLoader

DataLoader는 데이터셋에서 미니 배치만큼 데이터를 로드하게 만들어주는 역할을 한다. DataLoader는 반드시 데이터셋을 입력으로 받는다. 그리고 다음의 인자들을 사용할 수 있다.

- dataset : 데이터셋을 입력으로 받는다.
- batch_size : 미니 배치의 크기를 정한다. 통상적으로 2의 제곱수를 사용한다.
- shuffle : Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꾼다.
- drop_last : 마지막 배치를 버릴 것인지를 정한다. 데이터의 수가 배치 크기로 나누어 떨어지지 않는 경우 마지막 배치를 버릴 수 있다.

``` python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    drop_last=True
)
```