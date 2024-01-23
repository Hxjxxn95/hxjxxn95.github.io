---
layout : post
title: Language Model
author: Hojoon_Kim
date: 2024-01-21 15:30:10 +0900
categories: [Develope, NLP]
tags: [Pytorch, DL, LanguageModel]
pin: true
math: true
mermaid: true
---
언어모델(Language Model)이란  단어 시퀀스에 확률을 할당하는 모델이다. 이전 단어들이 주어졌을 때 다음 단어를 예측하는 모델이다. 마치 수능 영어 과목에 있는 빈칸 채우기 문제와 같은 것이다.

## 통계적 언어 모델(Statistical Language Model, SLM)

### 문장에 대한 확률
문장 "I go to school"의 확률을 구해보자. 이 문장의 확률은 다음과 같이 구할 수 있다.

$$P(I\ go\ to\ school) = P(I)\ P(go\ |\ I)\ P(to\ |\ I\ go)\ P(school\ |\ I\ go\ to)$$

일반화하게 되면 다음과 같다.

$$P(W) = P(w_1\ w_2\ w_3\ ... \ w_n) = \prod_{i=1}^{n}P(w_i\ |\ w_1\ ...\ w_{i-1})$$

그렇다면 이전 단어들이 주어졌을 때 다음 단어를 예측하는 확률을 어떻게 구할 수 있을까? 이를 위해 조건부 확률을 사용한다.

$$P(w_i\ |\ w_1\ ...\ w_{i-1}) = \frac{P(w_1\ ...\ w_{i-1}\ w_i)}{P(w_1\ ...\ w_{i-1})}$$

하지만 이런 방법은 기계에 매우 방대한 양을 학습시켜서 가능하게 하는 것이다. 만약 단어 시퀀스가 존재하지 않게 된다면 확률을 계산할 수 없다. 이런 문제를 희소 문제(Sparsity Problem)라고 한다. 이를 해결하기 위해 n-gram 이나 스무딩이나 백오프 등의 기법을 사용하게 된다.

### N-gram 언어 모델(N-gram Language Model)
SLM의 희소 문제를 해결하기 위해 사용하는 방법이다. n-gram은 n개의 연속적인 단어 나열을 의미한다. 예를 들어 "I go to school"이라는 문장이 있을 때, n을 2로 한다면 "I go", "go to", "to school"이 n-gram이 된다. 

$$ P(w | go\ to) = \frac{P(go\ to\ w)}{P(go\ to)}$$

이를 일반화하면 다음과 같다.

$$ P(w | w_{n-1}\ ...\ w_{1}) = \frac{P(w_{n-1}\ ...\ w_{1}\ w)}{P(w_{n-1}\ ...\ w_{1})}$$

이 모델은 희소성 문제는 어느정도 해결할 수는 있지만 여전히 희소성 문제가 존재하게 된다. 또한 trade-off가 존재한다. n을 크게 하면 희소성 문제는 해결할 수 있지만 n이 커질수록 모델 사이즈가 커지고 학습 데이터가 적어지는 문제가 발생한다. 또한 언어 모델을 학습시키기 위한 수집된 데이터의 도메인에 따라 모델 성능의 차이가 발생한다. 이를 해결하기 위해 딥러닝 모델을 활용하기도 한다.

## 언어 모델 평가 방법(Evaluation metric) - PPL
언어 모델의 성능을 평가하는 방법으로 PPL(Perplexity)를 사용한다. PPL은 확률의 역수이다. PPL이 낮을수록 언어 모델의 성능이 좋다고 판단한다. PPL은 다음과 같이 정의된다.

$$PPL(W) = P(w_1\ w_2\ w_3\ ... \ w_n)^{-\frac{1}{n}} = \sqrt[n]{\frac{1}{P(w_1\ w_2\ w_3\ ... \ w_n)}}$$

### 분기 계수(Branching factor)
분기 계수는 선택할 수 있는 가능한 경우의 수를 의미한다. 예를 들어 "I go to school"이라는 문장이 있을 때, "I" 다음에 올 수 있는 단어는 "go", "am", "went" 등이 있다. 이때 분기 계수는 3이 된다. 이를 일반화하면 다음과 같다.

$$b = \frac{1}{n}\sum_{i=1}^{n}b_i$$

PPL 은 분기 계수와 같다고 볼 수 있다. 분기 계수가 높을수록 PPL이 높아지게 된다. 즉, PPL이 높다는 것은 선택할 수 있는 가능한 경우의 수가 많다는 것을 의미한다. 이는 언어 모델의 성능이 좋지 않다는 것을 의미한다.


## Reference
[위키독스 - 딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155)