---
layout : post
title: Tensor Manipulate
author: Hojoon_Kim
date: 2024-01-16 10:15:10 +0900
categories: [Develope, Pytorch]
tags: [Pytorch, Tensor, Manipulate]
pin: true
---
## Tensor

### 2D Tensor
|t| = (batch size , dim)
### 3D Tensor ( in Vision )
|t| = (batch size , width , height)
### 3D Tensor ( in NLP )
|t| = (batch size , length , dim)

#### Example
````python
string = [[나는 사과를 좋아해], [나는 바나나를 좋아해],[나는 사과를 싫어해] , [나는 바나나를 싫어해]]
````
위와 같은 문장이 있다고 하자. 이 문장을 토큰화하고, 각 토큰을 임베딩하면 아래와 같은 텐서가 만들어진다.

````python
tensor = [[[나는],[사과를],[좋아해]], [[나는],[바나나를],[좋아해]],[[나는],[사과를],[싫어해]] , [[나는],[바나나를],[싫어해]]]
````
이 텐서의 크기는 (4,3) 이다. 이 텐서를 임베딩하면 아래와 같은 텐서가 만들어진다.

````python
tensor = [[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]], [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]] , [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]] , [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]]]
````
이 텐서의 크기는 (4,3,3) 이다. 

## Broadcasting

### 1D Tensor + 2D Tensor

1D Tensor와 2D Tensor를 더하면 1D Tensor의 각 원소가 2D Tensor의 각 행에 더해진다.
반대의 경우에 대해서도 마찬가지이다.
#### ex1
````python
t1 = torch.FloatTensor([1,2])
t2 = torch.FloatTensor([[1,2],[3,4]])
t1 + t2
t2 + t1
````

````python
tensor([[2., 4.],
        [4., 6.]])
tensor([[2., 4.],
        [4., 6.]])
````
#### ex2
````python
```python
t1 = torch.FloatTensor([[1,2]])
t2 = torch.FloatTensor([[1],[2]])
t1 + t2
```

````python
tensor([[2., 3.],
        [3., 4.]])
````

## Multiplication vs Matrix Multiplication

### Multiplication(일반 곱셈)

#### ex1
````python
t1 = torch.FloatTensor([[1,2],[3,4]])
t2 = torch.FloatTensor([[1],[2]])
t1 * t2
````
````python
tensor([[1., 2.],
        [6., 8.]])
````

#### ex2
````python
t1 = torch.FloatTensor([[1,2],[3,4]])
t2 = torch.FloatTensor([[1,2],[3,4]])
t1 * t2
````
````python
tensor([[ 1.,  4.],
        [ 9., 16.]])
````
### Matrix Multiplication(행렬 곱셈)

#### ex1
````python
t1 = torch.FloatTensor([[1,2],[3,4]])
t2 = torch.FloatTensor([[1],[2]])
t1.matmul(t2)
````
````python
tensor([[ 5.],
        [11.]])
````
#### ex2
````python
t1 = torch.FloatTensor([[1,2],[3,4]])
t2 = torch.FloatTensor([[1,2],[3,4]])
t1.matmul(t2)
````
````python
tensor([[ 7., 10.],
        [15., 22.]])
````

## Mean ( 평균 )

### ex1
````python
t = torch.FloatTensor([1,2])
t.mean()
````
````python
tensor(1.5000)
````
### ex2 ( 모든 차원 제거 )
````python
t = torch.FloatTensor([[1,2],[3,4]])
t.mean()
````
````python
tensor(2.5000)
````
### ex3 ( 첫 번째 차원 제거 )
````python
t = torch.FloatTensor([[1,2],[3,4]])
t.mean(dim=0)
````
````python
tensor([2., 3.])
````
### ex4 ( 마지막 차원 제거 )
````python
t = torch.FloatTensor([[1,2],[3,4]])
t.mean(dim=1)
````
````python
tensor([1.5000, 3.5000])
````
### ex5 ( 마지막 차원 제거 )
````python
t = torch.FloatTensor([[1,2],[3,4]])
t.mean(dim=-1)
````
````python
tensor([1.5000, 3.5000])
````
Sum(합) , Max(최대값) , Argmax(최대값 인덱스) 도 Mean과 같은 방식으로 사용할 수 있다.

## View(Reshape)

파이토치 텐서의 뷰(View)는 넘파이에서의 리쉐이프(Reshape)와 같은 역할을 한다. 텐서의 크기(size)나 모양(shape)을 변경하고 싶을 때 사용한다.

````python
t = np.array([[[0,1,2],
               [3,4,5]],
              [[6,7,8],
               [9,10,11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
print(ft.view([-1,3]))
print(ft.view([-1,3]).shape)
````
````python
torch.Size([2, 2, 3])
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]])
torch.Size([4, 3])
````


### 규칙

 1. view 는 기본적으로 변경 전과 변경 후의 텐서 안의 우너소의 개수가 유지되어야 한다.
 2. -1 을 사용하면 다른 차원으로부터 해당 값을 유추한다.

## Squeeze(차원 축소)

```python
t = torch.FloatTensor([[0],[1],[2]])
print(t)
print(t.shape)
print(t.squeeze())
print(t.squeeze().shape)
```

```
tensor([[0.],
        [1.],
        [2.]])
torch.Size([3, 1])
tensor([0., 1., 2.])
torch.Size([3])
```

## Unsqueeze(차원 추가)

```python
t = torch.FloatTensor([0,1,2])
print(t.shape)
print(t.unsqueeze(0))
print(t.unsqueeze(0).shape)
print(t.view(1,-1))
print(t.view(1,-1).shape)
print(t.unsqueeze(1))
print(t.unsqueeze(1).shape)
```

```
torch.Size([3])
tensor([[0., 1., 2.]])
torch.Size([1, 3])
tensor([[0., 1., 2.]])
torch.Size([1, 3])
tensor([[0.],
        [1.],
        [2.]])
torch.Size([3, 1])
```

## Concatenate

```python
x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])
print(torch.cat([x,y],dim=0))
print(torch.cat([x,y],dim=1))
```

```
tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])
tensor([[1., 2., 5., 6.],
        [3., 4., 7., 8.]])
```

## Stacking

```python
x = torch.FloatTensor([1,4])
y = torch.FloatTensor([2,5])
z = torch.FloatTensor([3,6])
print(torch.stack([x,y,z]))
print(torch.stack([x,y,z],dim=1))
```

```
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
tensor([[1., 2., 3.],
        [4., 5., 6.]])
```

## Ones and Zeros Like

```python
x = torch.FloatTensor([[0,1,2],[2,1,0]])
print(x)
print(torch.ones_like(x))
print(torch.zeros_like(x))
```

```
tensor([[0., 1., 2.],
        [2., 1., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

## In-place Operation

```python
x = torch.FloatTensor([[1,2],[3,4]])
print(x.mul(2.))
print(x)
print(x.mul_(2.))
print(x)
```

```
tensor([[2., 4.],
        [6., 8.]])
tensor([[1., 2.],
        [3., 4.]])
tensor([[2., 4.],
        [6., 8.]])
tensor([[2., 4.],
        [6., 8.]])
```






