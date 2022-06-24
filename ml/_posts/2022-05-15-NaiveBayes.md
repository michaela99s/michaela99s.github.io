---
layout: post
title: 나이브 베이즈
description: >
  나이브 베이즈 모델을 배워봅니다.
sitemap: false
hide_last_modified: true
---

조건부 확률과 베이즈 정리를 이용한 머신러닝 알고리즘을 **나이브 베이즈** (Naive Bayes)라고 합니다. 

조건부 확률은 P(A|B)와 같이 표현하며 이는 사건 B가 발생하였을 때, 사건 A도 발생할 확률을 의미합니다. 

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

베이즈 정리는 어떤 사건 A에 대한 관련될 것이라고 생각되는 사건 B의 사전확률에 근거해 사건 A의 조건부 확률을 추론합니다. 사건 B의 사전확률은 P(B), 사건 A의 사전확률은 P(A)입니다. 식으로 나타내면 다음과 같습니다: 

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

# 나이브 베이즈(Naive Bayes)

## scikit-learn

나이브 베이즈는 사이킷런의 `naive_bayes`에 있습니다. 분류 문제에서는 주로 가우시안 나이브 베이즈(GaussianNB) 알고리즘이 사용됩니다.  가우시안은 가우스 분포, 즉 정규분포상에서의 발생 확률을 계산합니다. 피쳐가 연속형 변수일 경우 정규분포상에서 우도(likelihood)를 계산합니다. 

```python
class sklearn.naive_bayes.GaussianNB(*, priors=None, var_smoothing=1e-09)
```

`var_smoothing`이 주요 하이퍼파라미터이며 안정적인 연산을 위하여 분산에 더해지는 모든 피쳐의 최대 분산 비율을 의미합니다.

회귀 문제에서는 `naive_bayes`의 알고리즘이 잘 맞지 않기 때문에 `linear_model`에 있는 Bayesian Regressors 중 `BayesianRidge` 모델을 사용합니다.

```python
class sklearn.linear_model.BayesianRidge(*, n_iter=300, tol=0.001, 
alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, 
alpha_init=None, lambda_init=None, compute_score=False, 
fit_intercept=True, normalize='deprecated', copy_X=True, verbose=False) 
```

`BayesianRidge`의 하이퍼파라미터에서는 감마분포의 파라미터를 설정합니다. 

- `alpha_1`: 감마 분포의 $\alpha$ 파라미터 사전 설정

- `lambda_1`: 감마 분포의 $\lambda$파라미터 사전 설정

## 분류 예시

타이타닉 생존 분류를 가우시안 나이브 베이즈를 사용하여 진행해보겠습니다.

```python
from sklearn.naive_bayes import GaussianNB
```

```python
gnb = GaussianNB()
gnb.fit(X_train, y_train)

print(f"트레인셋 정확도: {gnb.score(X_train, y_train):.4f}")
print(f"테스트셋 정확도: {gnb.score(X_test, y_test):.4f}")
```

```
트레인셋 정확도: 0.7992
테스트셋 정확도: 0.7765
```

### Hyperparameter Tuning

`var_smoothing`값을 조정하여 모델의 성능을 개선해보겠습니다. 

#### Grid Search

```python
param_grid = {'var_smoothing': np.arange(0, 10, 1)}

grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best Parameter: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")
```

```
Best Parameter: {'var_smoothing': 0}
Best Score: 0.7935
Test Score: 0.7765
```

#### Random Search

```python
param_distribs = {'var_smoothing': randint(low=0, high=20)}

random_search = RandomizedSearchCV(GaussianNB(), param_distributions = param_distribs,
                                   n_iter=100, cv=5)
random_search.fit(X_train, y_train)

print(f"Best Parameter: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_:.4f}")
print(f"Test Score: {random_search.score(X_test, y_test):.4f}")
```

```
Best Parameter: {'var_smoothing': 0}
Best Score: 0.7935
Test Score: 0.7765
```

그리드 서치와 랜덤 서치의 결과 모두 `var_smoothing`이 0에서 결정되었습니다. 기본값 역시 0과 매우 가까운 값이기 때문에 모델이 개선되었다고 하기 어렵습니다. 

이와 같이 나이브 베이즈 모델은 뛰어난 성능을 보이지는 않습니다. 그러나 다른 모델들과의 성능 비교를 위한 기저 모델로 많이 활용되기 때문에 알아둘 필요가 있습니다.

## 회귀 예시

보스턴 집값 데이터셋에 나이브 베이즈 기법을 적용해보겠습니다.

```python
from sklearn.linear_model import BayesianRidge
```

```python
model = BayesianRidge()
model.fit(X_train_minmax, y_train)
pred_bn = model.predict(X_test_minmax)

MSE_train = mean_squared_error(y_train, model.predict(X_train_minmax))
MSE_test = mean_squared_error(y_test, pred_bn)

print(f"Train R2: {model.score(X_train_minmax, y_train):.4f}")
print(f"Test  R2: {model.score(X_test_minmax, y_test):.4f}")
print(f"Train RMSE: {np.sqrt(MSE_train):.4f}")
print(f"Test  RMSE: {np.sqrt(MSE_test):.4f}")
```

```
Train R2: 0.7308
Test  R2: 0.7113
Train RMSE: 4.1387
Test  RMSE: 3.8406
```

### Hyperparameter Tuning

`alpha_1`값과 `lambda_1`값을 조정하여 모델의 성능을 개선해봅시다.

#### Grid Search

```python
param_grid = {'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 3, 4],
              'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 3, 4]}
grid_search = GridSearchCV(BayesianRidge(), param_grid, cv=5)
grid_search.fit(X_train_minmax, y_train)

print(f"Best Parameter: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")
print(f"Train Score: {grid_search.score(X_train_minmax, y_train):.4f}")
print(f"Test  Score: {grid_search.score(X_test_minmax, y_test):.4f}")
```

```
Best Parameter: {'alpha_1': 1e-06, 'lambda_1': 2}
Best Score: 0.7135
Train Score: 0.7307
Test  Score: 0.7113
```

#### Random Search

```python
param_distribs = {'alpha_1': randint(low=1e-6, high=10),
                  'lambda_1': randint(low=1e-6, high=10)}
random_search = RandomizedSearchCV(BayesianRidge(),
                                   param_distributions=param_distribs, 
                                   n_iter=20, cv=5)
random_search.fit(X_train_minmax, y_train)

print(f"Best Parameter: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_:.4f}")
print(f"Train Score: {random_search.score(X_train_minmax, y_train):.4f}")
print(f"Test  Score: {random_search.score(X_test_minmax, y_test):.4f}")
```

```
Best Parameter: {'alpha_1': 3, 'lambda_1': 2}
Best Score: 0.7135
Train Score: 0.7307
Test  Score: 0.7113
```

회귀모델에서도 나이브 베이즈 기법은 하이퍼파라미터의 튜닝을 통해서 성능이 크게 개선되지 않았습니다. 분류 문제에서 나이브 베이즈 기법은 기저 모델로 활용되지만 회귀에서는 잘 사용되지 않습니다. 
