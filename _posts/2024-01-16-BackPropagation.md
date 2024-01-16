---
layout : post
title: Back Propagation
author: Hojoon_Kim
date: 2024-01-16 19:15:10 +0900
categories: [Develope, DL]
tags: [Pytorch, DL, Back Propagation]
pin: true
math: true
mermaid: true
---
## 인공 신경망 (Artificial Neural Network)
인공 신경망은 입력층, 은닉층, 출력층 총 3개의 층을 가진다. 은닉층은 보통 2개 이상의 층을 가진다. 은닉층이 2개 이상인 인공 신경망을 **심층 신경망(Deep Neural Network)** 이라고 한다.
<div> 
    <img src="https://wikidocs.net/images/page/37406/nn1_final.PNG" width="400"/>
</div>

## Forward Propagation
<div> 
    <img src="https://wikidocs.net/images/page/37406/nn2_final_final.PNG" width="400"/>
</div>

파란 숫자는 입력값(0.1, 0.25), 빨간 숫자는 각 가중치의 값을 의미.

$$ z_1 = W_1x_1 + W_2x_2 + b_1 $$
$$ z_2 = W_3x_1 + W_4x_2 + b_2 $$

$$ z_1 $$ 과 $$ z_2 $$ 는 각각 시그모이드 함수를 지나 $$ h_1 $$ 과 $$ h_2 $$ 가 된다.

$$ h_1 = sigmoid(z_1) $$ , $$ h_2 = sigmoid(z_2) $$


## Back Propagation
순전파가 입력층에서 출력층으로 향하는 것이라면 역전파는 반대로 출력층에서 입력층의 방향으로 계산하면서 가중치를 업데이트 해주는 과정이다. 역전파는 손실 함수를 미분하여 이전 층으로 전달하는 과정이다.

역정파 과정에서 사용되는 미분 공식은 다음과 같다.

$$ \frac{\partial E}{\partial W} = \frac{\partial E}{\partial y} \frac{\partial y}{\partial z} \frac{\partial z}{\partial W} $$

위의 식에서 우변의 각 항에 대해서 순서대로 계산하면 다음과 같다.

$$ \frac{\partial E}{\partial y} = \frac{y - t}{y(1-y)} $$

[위키독스](https://wikidocs.net/60682)

