---
layout : post
title: 상관분석(Correlation Analysis)
author: Hojoon_Kim
date: 2024-03-18 19:00:10 +0900
categories: [Statistic]
tags: [Statistic, Correlation Analysis]
pin: true
math: true
---
## 상관 관계(Correlation)

- 자료들 사이에 존재하는 어떤 상호관계

## 공분산(Covariance)

공분산은 두 변수 사이의 관계를 나타내는 지표이다. 공분산은 두 변수의 단위에 영향을 받기 때문에, 단위에 의존하지 않는 상관계수를 사용한다.

$$ \sigma_{xy} = Cov(X, Y) = E((X - \mu_x)(Y - \mu_y)) = E(XY) - \mu_x\mu_y $$

## 상관계수(Correlation Coefficient)

- 상관계수는 두 변수 사이의 선형적 관계의 강도와 방향을 나타내는 지표이다. 
- 상관계수는 -1 ~ 1 사이의 값을 가지며, 0에 가까울수록 두 변수 사이의 관계가 약하다.
- 상관계수가 1 또는 -1에 가까울수록 두 변수 사이의 관계가 강하다.
- 상관계수가 0이면 두 변수 사이에 선형적 관계가 없다.
- 상관계수는 단위에 영향을 받지 않는다.(더하거나 곱해도 변하지 않는다.)
- 상관계수는 두 변수 사이의 관계를 나타내는 지표이기 때문에 인과관계를 나타내지 않는다.
- 상관계수는 두 변수 사이의 선형적 관계만을 나타낸다.(비선형적 관계는 나타내지 않는다.)

$$ \rho_{xy} = \frac{\sigma_{xy}}{\sigma_x\sigma_y} = \frac{Cov(X, Y)}{\sqrt{Var(X)Var(Y)}} $$

- X, Y : 독립 -> $\rho_{xy} = 0$ & $\rho_{yx} = 0$
- X, Y : 양적 관계 -> $\rho_{xy} > 0$ & $\rho_{yx} > 0$
- X, Y : 음적 관계 -> $\rho_{xy} < 0$ & $\rho_{yx} < 0$
- -1 <= $\rho_{xy}$ <= 1

## 피어슨 상관계수(Pearson Correlation Coefficient)

$$ r = \frac{1}{n-1} \sum_{i=1}^{n}(\frac{x_i - \bar{x}}{s_x})(\frac{y_i - \bar{y}}{s_y}) = \frac{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2\frac{1}{n-1}\sum_{i=1}^{n}(y_i - \bar{y})^2} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}} $$

$$ r_{XY} = \frac{S_{XY}}{\sqrt{S_{XX}S_{YY}}} $$

$$ S_{XX} = \sum_{i=1}^{n}(x_i - \bar{x})^2, S_{YY} = \sum_{i=1}^{n}(y_i - \bar{y})^2, S_{XY} = \sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y}) $$

### 피어슨 상관계수의 성질

- -1 <= r <= 1
- $r_{XY} = r_{YX} = r$
- r = 0 또는 0에 가까울 때 -> 두 변수 사이의 상관관계가 약하다
- 0 < r < 1 -> 양상관, 0 > r > -1 -> 음상관

## 이변량 정규분포(Bivariate Normal Distribution)

$$ f(x, y) = \frac{1}{2\pi\sigma_x\sigma_y\sqrt{1-\rho^2}}exp(-\frac{1}{2(1-\rho^2)}[\frac{(x-\mu_x)^2}{\sigma_x^2} - 2\rho\frac{(x-\mu_x)(y-\mu_y)}{\sigma_x\sigma_y} + \frac{(y-\mu_y)^2}{\sigma_y^2}]) $$

### 이변량 정규분포의 성질

- X, Y : 독립 -> $\rho = 0$
- X, Y : 양적 관계 -> $\rho > 0$
- X, Y : 음적 관계 -> $\rho < 0$

## 상관계수의 검정

모집단 : 모상관계수가 $\rho$인 이변량 정규분포를 따를 때, 표본상관계수 r은 모 상관계수 $\rho$에 대한 점추정량이다.

- $ H_0 : \rho = 0 $ (두 변수 사이에 상관관계가 없다.)
- 검정통계량 : $ T_0 = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}} \sim t(n-2) $
- 가설 검정 : 
  - $ H_1 : \rho > 0 $ 일 때, $ T_0 > t_{\alpha}(n-2) $ 이면 귀무가설 기각
  - $ H_1 : \rho < 0 $ 일 때, $ T_0 < -t_{\alpha}(n-2) $ 이면 귀무가설 기각
  - $ H_1 : \rho \neq 0 $ 일 때, $ |T_0| > t_{\frac{\alpha}{2}}(n-2) $ 이면 귀무가설 기각

### 그 밖의 상관계수들

| | 범주형 변수| 순위형 변수 | 양적 변수 |
|:---:|:---:|:---:|:---:|
| 범주형 변수 | Phi 계수, 유관 계수, Lambda 계수, 사분상관계수 |  | |
| 순위형 변수 | 등위양분상관계수 | Spearman 상관계수, Kendall 상관계수 | |
| 양적 변수 |  Cramer 상관계수 |  | 피어슨 상관계수 |
