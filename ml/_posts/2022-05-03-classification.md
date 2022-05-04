---
layout: post
title: 머신러닝 맛보기 - 분류
description: >
  분류분석에 대한 기초를 다집니다. 
  이 포스트는 데이터캠퍼스의 [공개적 빅데이터 분석기사 실기]를 따라 작성되었습니다.
sitemap: false
hide_last_modified: true
---

분류분석에 대하여 알아보겠습니다. 

# 유방암 데이터 분석하기 

## EDA
분석의 대상이 되는 데이터셋을 로드합니다. 여기에서는 위스콘신 대학의 유방암 데이터를 불러오겠습니다. 이 데이터셋에서 목적변수는 Class이며, 유방암이 발병한 경우 1, 아닌 경우 0의 값을 갖는 범주형 변수입니다. 따라서 유방암 데이터셋을 통해 이진분류(binary classification) 문제를 수행하게 됩니다.

```python
import pandas as pd
data = pd.read_csv('breast-cancer-wisconsin.csv')
data.head()
```

|     | code    | Clump_Thickness | Cell_Size | Cell_Shape | Marginal_Adhesion | Single_Epithelial_Cell_Size | Bare_Nuclei | Bland_Chromatin | Normal_Nucleoli | Mitoses | Class |
| --- | ------- | --------------- | --------- | ---------- | ----------------- | --------------------------- | ----------- | --------------- | --------------- | ------- | ----- |
| 0   | 1000025 | 5               | 1         | 1          | 1                 | 2                           | 1           | 3               | 1               | 1       | 0     |
| 1   | 1002945 | 5               | 4         | 4          | 5                 | 7                           | 10          | 3               | 2               | 1       | 0     |
| 2   | 1015425 | 3               | 1         | 1          | 1                 | 2                           | 2           | 3               | 1               | 1       | 0     |
| 3   | 1016277 | 6               | 8         | 8          | 1                 | 3                           | 4           | 3               | 7               | 1       | 0     |
| 4   | 1017023 | 4               | 1         | 1          | 3                 | 2                           | 1           | 3               | 1               | 1       | 0     |

```python
data.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 683 entries, 0 to 682
Data columns (total 11 columns):
 #   Column                       Non-Null Count  Dtype
---  ------                       --------------  -----
 0   code                         683 non-null    int64
 1   Clump_Thickness              683 non-null    int64
 2   Cell_Size                    683 non-null    int64
 3   Cell_Shape                   683 non-null    int64
 4   Marginal_Adhesion            683 non-null    int64
 5   Single_Epithelial_Cell_Size  683 non-null    int64
 6   Bare_Nuclei                  683 non-null    int64
 7   Bland_Chromatin              683 non-null    int64
 8   Normal_Nucleoli              683 non-null    int64
 9   Mitoses                      683 non-null    int64
 10  Class                        683 non-null    int64
dtypes: int64(11)
memory usage: 58.8 KB
```

데이터셋 안에 결측치는 없으며 총 683행의 데이터로 구성되어 있음을 확인할 수 있습니다. 

각 class의 빈도를 확인해보도록 하겠습니다.

```python
data['Class'].value_counts()
```

```
0    444
1    239
Name: Class, dtype: int64
```

정상인 경우가 444건, 유방암 환자인 경우가 239건이 있습니다. 정상인 경우가 발병인 경우보다 많아 데이터의 불균형이 우려됩니다. 

### 분석준비

데이터셋의 구성을 알아보았으니 이제 분석을 위하여 데이터셋을 피쳐(features)와 레이블(label)로 나누도록 하겠습니다. 피쳐 변수는 결과값에 영향을 주는 변수들로, 회귀문제에서 독립변수와 유사한 개념입니다. 마찬가지로 레이블은 결과값을 나타내는 변수이므로 종속변수와 유사한 개념이라고 볼 수 있습니다. 

유방암 발병 여부를 분류해내는 문제에서 레이블 변수는 Class가 되겠습니다.  피쳐들은 Class를 제외한 변수에 해당합니다. 피쳐와 레이블을 나누는 방법에는 여러방법이 있습니다. 

편의를 위해서 컬럼명 리스트를 구한 다음, 피쳐 변수 X를 분리하도록 하겠습니다.

```python
data.columns
```

Index(['code', 'Clump_Thickness', 'Cell_Size', 'Cell_Shape',
       'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei',
       'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class'],
      dtype='object')

이렇게 컬럼명을 구한 다음 피쳐변수들을 분리하면 복사와 붙여넣기를 통해 보다 편리하게 진행할 수 있게 됩니다. 

```python
# 1. 칼럼명으로 나누기
X = data[['Clump_Thickness', 'Cell_Size', 'Cell_Shape', 
          'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 
          'Bare_Nuclei','Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses']]

# 2. 칼럼 위치인덱스로 나누기
X = data[data.columns[1:-1]]

# 3. loc 함수 사용하기
X = data.loc[:, 'Clump_Thickness':'Mitoses']
```

이외에도 방법들이 있지만 대표적으로 3 가지를 알아보았습니다. 위의 세 가지 방법 중 하나를 취사선택해서 적용하면 피쳐들의 데이터를 분리해낼 수 있습니다.  

라벨 데이터 y도 피쳐 데이터 X를 분리한 것과 같은 방법으로 분리해낼 수 있습니다.  

```python
y = data[['Class']]
```

#### 데이터셋 분할

훈련을 위한 트레인 데이터(train data)와 검증을 위한 테스트 데이터(test data)로 나누어줍니다. 사이킷런은 머신러닝을 위한 유용한 패키지입니다. 사이킷런(https://scikit-learn.org/stable/) 을 이용하여 머신러닝 분석을 진행하겠습니다. 

```python
from sklearn.model_selection import train_test_split
```

```python
sklearn.model_selection.train_test_split(*arrays, test_size=None, 
train_size=None, random_state=None, shuffle=True, stratify=None)
```

몇 가지 매개변수를 살펴보도록 하겠습니다.

- test_size, train_size: 검증데이터 및 훈련데이터의 비율을 0과 1사이의 값으로 설정합니다. 보통 둘 중 하나만 설정해주면 나머지 하나는 자동으로 합하여 1이 되도록 설정됩니다. 두 값이 모두 None인 경우, 검증비율은 자동으로 0.25가 됩니다.

- stratify: None이 아니면 데이터가 층화되어 나누어집니다. 즉,  stratify의 기준을 설정해준다면 그 기준 데이터의 비율에 따라 훈련데이터와 검증데이터가 나뉘게 됩니다. 예를 들어, 전체 데이터에서 목적데이터의 비율이 30:70이면, 트레인 데이터와 테스트 데이터 각각의 목적데이터도 30:70을 갖게 됩니다.

앞서 유방암 데이터의 라벨 변수에 불균형이 있음을 확인하였습니다. 따라서 stratify=y로 설정하여 훈련데이터와 검증데이터를 나누도록 하겠습니다. 

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, 
                                                    random_state=42)
```

```python
print(X_train.shape)
print(X_test.shape)

print(y_train.mean())
print(y_test.mean())
```

```
(512, 9)
(171, 9)
Class    0.349927
dtype: float64
Class    0.349609
dtype: float64
Class    0.350877
dtype: float64
```

훈련데이터와 검증데이터의 비율은 512:171로 대략 4:1입니다. test_size와 train_size를 따로 설정해주지 않았기 때문에 자동적으로 4:1의 비율로 나누어진 것입니다.
또한 전체 y와 y_train, y_test의 평균값이 모두 약 0.35로 나타나는데 이는 stratify=y로 설정해주었기 때문에 y의 비율이 트레인셋과 테스트셋에서도 유지되는 것입니다. 

#### 데이터 전처리

- 데이터 표준/정규화

- 범주자료 one-hot encoding

- 특성변수 축약

##### 데이터 정규화/표준화

다양한 피쳐들은 측정단위가 다르기 때문에 레이블을 분류하는데 단위의 영향이 발생할 수 있습니다. 따라서 단위를 맞추어주어야 레이블 분류에 영향을 미치는 정도가 동일해질 수 있습니다. 이를 데이터 정규화라고 합니다. 정규화 방법에는 표준정규화(Standard), 민맥스(MinMax), 로버스트(Robust) 스케일링 등 다양한 방법이 있습니다. 여기에서는 표준정규화와 민맥스 스케일링을 살펴보겠습니다. 정규화에 대한 라이브러리는 사이킷런의 preprocessing에 있습니다. 

```python
from sklearn.preprocessing import MinMaxScaler

scaler_minmax = MinMaxScaler()
scaler_minmax.fit(X_train)
X_scaled_minmax_train = scaler_minmax.transform(X_train) 

pd.DataFrame(X_scaled_minmax_train).describe()
```

|       | 0          | 1          | 2          | 3          | 4          | 5          | 6          | 7          | 8          |
| ----- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| count | 512.000000 | 512.000000 | 512.000000 | 512.000000 | 512.000000 | 512.000000 | 512.000000 | 512.000000 | 512.000000 |
| mean  | 0.372830   | 0.231988   | 0.242839   | 0.205078   | 0.241319   | 0.285590   | 0.269314   | 0.199002   | 0.067491   |
| std   | 0.317836   | 0.334781   | 0.332112   | 0.319561   | 0.242541   | 0.404890   | 0.265289   | 0.331503   | 0.190373   |
| min   | 0.000000   | 0.000000   | 0.000000   | 0.000000   | 0.000000   | 0.000000   | 0.000000   | 0.000000   | 0.000000   |
| 25%   | 0.111111   | 0.000000   | 0.000000   | 0.000000   | 0.111111   | 0.000000   | 0.111111   | 0.000000   | 0.000000   |
| 50%   | 0.333333   | 0.000000   | 0.000000   | 0.000000   | 0.111111   | 0.000000   | 0.222222   | 0.000000   | 0.000000   |
| 75%   | 0.555556   | 0.361111   | 0.444444   | 0.333333   | 0.333333   | 0.583333   | 0.444444   | 0.222222   | 0.000000   |
| max   | 1.000000   | 1.000000   | 1.000000   | 1.000000   | 1.000000   | 1.000000   | 1.000000   | 1.000000   | 1.000000   |

민맥스 스케일링 결과, 모든 변수의 최소값은 0, 최대값은 1로 바뀌게 됨을 확인할 수 있습니다. 

```python
from sklearn.preprocessing import StandardScaler

scaler_standard = StandardScaler()
scaler_standard.fit(X_train)
X_scaled_standard_train = scaler_standard.transform(X_train) 

pd.DataFrame(X_scaled_standard_train).describe()
```

|       | 0             | 1             | 2             | 3             | 4             | 5             | 6             | 7             | 8             |
| ----- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| count | 5.120000e+02  | 5.120000e+02  | 5.120000e+02  | 5.120000e+02  | 5.120000e+02  | 5.120000e+02  | 5.120000e+02  | 5.120000e+02  | 5.120000e+02  |
| mean  | 6.938894e-18  | 6.938894e-18  | -2.775558e-17 | -2.775558e-17 | -4.857226e-17 | 6.938894e-18  | -2.081668e-17 | -2.775558e-17 | -1.734723e-18 |
| std   | 1.000978e+00  | 1.000978e+00  | 1.000978e+00  | 1.000978e+00  | 1.000978e+00  | 1.000978e+00  | 1.000978e+00  | 1.000978e+00  | 1.000978e+00  |
| min   | -1.174173e+00 | -6.936309e-01 | -7.319088e-01 | -6.423777e-01 | -9.959361e-01 | -7.060427e-01 | -1.016165e+00 | -6.008881e-01 | -3.548677e-01 |
| 25%   | -8.242452e-01 | -6.936309e-01 | -7.319088e-01 | -6.423777e-01 | -5.373756e-01 | -7.060427e-01 | -5.969255e-01 | -6.008881e-01 | -3.548677e-01 |
| 50%   | -1.243886e-01 | -6.936309e-01 | -7.319088e-01 | -6.423777e-01 | -5.373756e-01 | -7.060427e-01 | -1.776856e-01 | -6.008881e-01 | -3.548677e-01 |
| 75%   | 5.754680e-01  | 3.860715e-01  | 6.076347e-01  | 4.017410e-01  | 3.797454e-01  | 7.360871e-01  | 6.607941e-01  | 7.011454e-02  | -3.548677e-01 |
| max   | 1.975181e+00  | 2.296314e+00  | 2.282064e+00  | 2.489978e+00  | 3.131108e+00  | 1.766180e+00  | 2.756993e+00  | 2.418624e+00  | 4.903108e+00  |

표준정규화 결과, 모든 피쳐에서 평균은 0에 가까워졌고 표준편차는 1에 가까워졌음을 확인할 수 있습니다. 

검증데이터에 대해서도 마찬가지로 정규화를 진행합니다. 

```python
X_scaled_minmax_test = scaler_minmax.transform(X_test)

pd.DataFrame(X_scaled_minmax_test).describe()
```

|       | 0          | 1          | 2          | 3          | 4          | 5          | 6          | 7          | 8          |
| ----- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| count | 171.000000 | 171.000000 | 171.000000 | 171.000000 | 171.000000 | 171.000000 | 171.000000 | 171.000000 | 171.000000 |
| mean  | 0.411306   | 0.259909   | 0.256010   | 0.198181   | 0.269006   | 0.274204   | 0.278752   | 0.233918   | 0.065627   |
| std   | 0.298847   | 0.357544   | 0.332700   | 0.315307   | 0.259557   | 0.405891   | 0.292578   | 0.360958   | 0.199372   |
| min   | 0.000000   | 0.000000   | 0.000000   | 0.000000   | 0.000000   | 0.000000   | 0.000000   | 0.000000   | 0.000000   |
| 25%   | 0.222222   | 0.000000   | 0.000000   | 0.000000   | 0.111111   | 0.000000   | 0.000000   | 0.000000   | 0.000000   |
| 50%   | 0.444444   | 0.000000   | 0.111111   | 0.000000   | 0.111111   | 0.000000   | 0.222222   | 0.000000   | 0.000000   |
| 75%   | 0.555556   | 0.444444   | 0.444444   | 0.222222   | 0.388889   | 0.444444   | 0.444444   | 0.388889   | 0.000000   |
| max   | 1.000000   | 1.000000   | 1.000000   | 1.000000   | 1.000000   | 1.000000   | 1.000000   | 1.000000   | 1.000000   |

```python
X_scaled_standard_test = scaler_standard.transform(X_test)

pd.DataFrame(X_scaled_standard_test).describe()
```

|       | 0          | 1          | 2          | 3          | 4          | 5          | 6          | 7          | 8          |
| ----- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| count | 171.000000 | 171.000000 | 171.000000 | 171.000000 | 171.000000 | 171.000000 | 171.000000 | 171.000000 | 171.000000 |
| mean  | 0.121175   | 0.083483   | 0.039700   | -0.021605  | 0.114263   | -0.028149  | 0.035612   | 0.105430   | -0.009802  |
| std   | 0.941174   | 1.069038   | 1.002747   | 0.987654   | 1.071204   | 1.003453   | 1.103943   | 1.089918   | 1.048292   |
| min   | -1.174173  | -0.693631  | -0.731909  | -0.642378  | -0.995936  | -0.706043  | -1.016165  | -0.600888  | -0.354868  |
| 25%   | -0.474317  | -0.693631  | -0.731909  | -0.642378  | -0.537376  | -0.706043  | -1.016165  | -0.600888  | -0.354868  |
| 50%   | 0.225540   | -0.693631  | -0.397023  | -0.642378  | -0.537376  | -0.706043  | -0.177686  | -0.600888  | -0.354868  |
| 75%   | 0.575468   | 0.635234   | 0.607635   | 0.053701   | 0.609026   | 0.392723   | 0.660794   | 0.573367   | -0.354868  |
| max   | 1.975181   | 2.296314   | 2.282064   | 2.489978   | 3.131108   | 1.766180   | 2.756993   | 2.418624   | 4.903108   |

검증데이터를 변환할 떄에는 fit 과정이 필요하지 않습니다. 스케일러는 훈련데이터를 기준으로 만드는 것이기 때문입니다. 따라서 훈련데이터를 통해 생성된 스케일러를 기준으로 검증데이터를 변환해주기만 하면 됩니다. 
따라서 정규화된 검정데이터의 기술통계량은 예상과는 다르게 나타날 수도 있습니다. 여기에서는 표준정규화의 경우가 평균은 0, 표준편차는 1이라는 예상과는 달리 나타나고 있습니다. 그러나 이는 전혀 문제가 되지 않습니다. 

### 모델링

#### 모델 학습

분류분석을 위한 알고리즘은 매우 다양합니다. 여기에서는 가장 기초적인 로지스틱 회귀(Logistic Regression)를 이용하여 이진분류를 수행하도록 하겠습니다. 
로지스틱 회귀는 사이킷런의 linear_model에 있습니다. 

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_scaled_minmax_train, y_train)
```

훈련 결과를 평가하기 위하여 정확도를 알아보도록 합시다.

```python
print(f"Accuracy(train): {model.score(X_scaled_minmax_train, y_train):.4f}")
print(f"Accuracy(test) : {model.score(X_scaled_minmax_test, y_test):.4f}")
```

```
Accuracy(train): 0.9727
Accuracy(test) : 0.9591
```

훈련셋과 테스트셋 모두에서 정확도가 매우 높게 나타났습니다. 그러나 모델을 정확도 지표로만 평가하기에는 한계가 있습니다. 따라서 다른 평가지표들도 살펴보아야 합니다. 사이킷런의 metrics에는 다양한 평가지표들을 제공하고 있습니다. 

모델의 예측값을 저장한 후, 평가지표들을 적용해보도록 하겠습니다.

```python
pred_train = model.predict(X_scaled_minmax_train)
pred_test = model.predict(X_scaled_minmax_test)
```

먼저 혼동행렬(confusion matrix)를 보겠습니다.

```python
from sklearn.metrics import confusion_matrix

confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)

confusion_test = confusion_matrix(y_test, pred_test)
print("검증데이터 오차행렬:\n", confusion_test)
```

```
훈련데이터 오차행렬:
 [[328   5]
 [  9 170]]
검증데이터 오차행렬:
 [[106   5]
 [  2  58]]
```

보다 상세한 평가지표를 알아보기 위해서는 분류 리포트를 확인해볼 수도 있습니다.

```python
from sklearn.metrics import classification_report

cfreport_train = classification_report(y_train, pred_train)
print("분류예측 리포트(train):\n", cfreport_train)

cfreport_test = classification_report(y_test, pred_test)
print("분류예측 리포트(test):\n", cfreport_test)
```

```python
분류예측 리포트(train):
               precision    recall  f1-score   support

           0       0.97      0.98      0.98       333
           1       0.97      0.95      0.96       179

    accuracy                           0.97       512
   macro avg       0.97      0.97      0.97       512
weighted avg       0.97      0.97      0.97       512

분류예측 리포트(test):
               precision    recall  f1-score   support

           0       0.98      0.95      0.97       111
           1       0.92      0.97      0.94        60

    accuracy                           0.96       171
   macro avg       0.95      0.96      0.96       171
weighted avg       0.96      0.96      0.96       171
```

마지막으로 ROC 지표가 있습니다. 

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, model.decision_function(X_scaled_minmax_test))
roc_auc = roc_auc_score(y_test, model.decision_function(X_scaled_minmax_test))
roc_auc
```
```
0.9923423423423423
```
ROC 커브를 시각화 해보겠습니다. 

```python
import matplotlib.pyplot as plt

plt.title("Receiver Operating Characteristic")
plt.xlabel("False Positive Rate(1 - Specificity)")
plt.ylabel("True Positive Rate(Sensitivity)")

plt.plot(fpr, tpr, 'b', label = f'Model (AUC = {roc_auc: 0.2f})')
plt.plot([0,1], [1,1], 'y--')
plt.plot([0,1], [0,1], 'r--')

plt.legend(loc='lower right')
plt.show()
```

![](data:image/png;base64,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6dKlRRKjJ4JE77wTisQNHAiVKnmROOfysPPOOzNx4kT++OMPAD755BOqV69erDEMHTqU5s2bs0vCF7TBgwdz+OGHM3jw4Njruf3225k/fz4TJ07k+++/55133mHFihVFFueSJUu46667+Pbbb/nuu++466678vzwvuiii7j//vuZMGECJ598Mg899BAA/fr1A2DChAl88sknXHfddWyIrlZ0796dPn36FEmcftcQwMKFcMUV8Prr0KIFvP9+aBx2rgTr2RPGji3adbZoEaqlF+S4447jgw8+4LTTTmPw4MF069aN//73v0D48LvggguYPn06O+20E3379qVZs2ZkZ2fTrVs35s2bR+vWrbGEL1ovvfQSTzzxBH/++SetWrWiT58+lCtXLt/tDxo0iEsuuWTj+5UrV/LFF1/w2Wef0blzZ+66664C92HVqlX069ePGTNmsMMOOwCw5557csYZZxR8AGIaNmwYHTt2ZI899gCgY8eOfPTRR3Tr1m2z+X766Sfatm27cZ5jjjmGu+++m8mTJ9O+fXsAqlWrxm677cbo0aNp2bIlXbp0oU2bNtx6663bHKefEQAsXw6ffAL33hu6kfQk4FxSXbt25ZVXXmH16tWMHz+eVq1abZz2j3/8gwMPPJDx48dz3333ce655wJw1113cfjhhzNp0iROPvlkZs+eDcCUKVN49dVX+fLLLxk7dizlypVj0KBBSbf/5ZdfcvDBB298/+6779KpUycaNGhA5cqVGTNmTIH7MG3aNGrVqrXZWUV+rrnmGlq0aLHF6/7770+63Lx586hZs+bG9zVq1GDevHlbzNekSRPeffddAF5//XXmzJkDQPPmzRkyZAjr1q1jxowZjBkzZuO03XffnTVr1pCdnV1g/AXJ3DOC2bPDA2G33BLKQsyeHS4HOVdKxPnmnirNmjVj5syZDB48mOOOO26zaV988QVvvvkmAO3btyc7O5vly5czcuRI3nrrLQCOP/54dt99dwCGDx/OmDFjOCSq0vvHH39QrVq1pNtfsmQJlRL+XwcPHszVUbte165dGTx4MAcffHC+d9AU9s6axx57rFDzF1b//v256qqruPvuu+nSpQvbb789ABdccAFTpkwhKyuL2rVrc+ihh252plStWjV++eUXKm/jnYwpTQSSOgG9gXLAc2Z2f67pOwAvAAcD2cCZZjYzlTFt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)

이상으로 모델의 성능을 평가하는 지표들을 살펴보았습니다. 각 지표들에 대한 상세한 설명은 다른 포스트를 통해서 보강하도록 하겠습니다. 
여기에서는 모델을 평가하는 지표로는 정확도만 있는 것이 아니므로 다양한 여러 지표를 살펴보아야 한다는 것이 중요합니다. 

### 마무리

지금까지의 내용을 정리하여 csv 파일로 저장하도록 하겠습니다.

먼저, 훈련 레이블 데이터에 예측값, 예측확률 컬럼을 추가해보겠습니다.

```python
y_train['y_pred'] = pred_train

prob_train = model.predict_proba(X_scaled_minmax_train)
y_train[['y_prob0', 'y_prob1']] = prob_train
y_train
```

|     | Class | y_pred | y_prob0  | y_prob1  |
| --- | ----- | ------ | -------- | -------- |
| 131 | 0     | 0      | 0.981014 | 0.018986 |
| 6   | 0     | 0      | 0.768191 | 0.231809 |
| 0   | 0     | 0      | 0.966431 | 0.033569 |
| 269 | 0     | 0      | 0.988880 | 0.011120 |
| 56  | 1     | 1      | 0.203161 | 0.796839 |
| ... | ...   | ...    | ...      | ...      |
| 515 | 1     | 1      | 0.021270 | 0.978730 |
| 216 | 1     | 0      | 0.895961 | 0.104039 |
| 312 | 1     | 1      | 0.113440 | 0.886560 |
| 11  | 0     | 0      | 0.987405 | 0.012595 |
| 268 | 0     | 0      | 0.990470 | 0.009530 |

512 rows × 4 columns

`predict_proba()`는 레이블의 예측 확률을 반환해주는 함수입니다. 표에서 y_prob0는 해당 데이터의 Class가 0으로 예측될 확률, y_prob1은 해당 데이터의 Class가 1로 예측될 확률을 나타냅니다. 둘 중 높은 확률을 가진 Class로 최종 예측 y_pred가 결정됩니다.

같은 방식으로 검정데이터에도 예측 정보를 추가해줍니다.

```python
y_test['y_pred'] = pred_test

prob_test = model.predict_proba(X_scaled_minmax_test)
y_test[['y_prob0', 'y_prob1']] = prob_test
y_test
```

|     | Class | y_pred | y_prob0  | y_prob1  |
| --- | ----- | ------ | -------- | -------- |
| 541 | 0     | 0      | 0.955893 | 0.044107 |
| 549 | 0     | 0      | 0.970887 | 0.029113 |
| 318 | 0     | 0      | 0.943572 | 0.056428 |
| 183 | 0     | 0      | 0.979370 | 0.020630 |
| 478 | 1     | 1      | 0.001305 | 0.998695 |
| ... | ...   | ...    | ...      | ...      |
| 425 | 1     | 1      | 0.006201 | 0.993799 |
| 314 | 1     | 1      | 0.067440 | 0.932560 |
| 15  | 1     | 1      | 0.436887 | 0.563113 |
| 510 | 0     | 0      | 0.983410 | 0.016590 |
| 351 | 0     | 0      | 0.987405 | 0.012595 |

171 rows × 4 columns

마지막으로 검정 데이터에 대하여 피쳐들과 레이블, 예측 정보를 합쳐서 csv파일로 저장합니다.

```python
total_test = pd.concat([X_test, y_test], axis=1)
total_test.to_csv("classification_test.csv")
total_test
```

|     | Clump_Thickness | Cell_Size | Cell_Shape | Marginal_Adhesion | Single_Epithelial_Cell_Size | Bare_Nuclei | Bland_Chromatin | Normal_Nucleoli | Mitoses | Class | y_pred | y_prob0  | y_prob1  |
| --- | --------------- | --------- | ---------- | ----------------- | --------------------------- | ----------- | --------------- | --------------- | ------- | ----- | ------ | -------- | -------- |
| 541 | 5               | 2         | 2          | 2                 | 1                           | 1           | 2               | 1               | 1       | 0     | 0      | 0.955893 | 0.044107 |
| 549 | 4               | 1         | 1          | 1                 | 2                           | 1           | 3               | 2               | 1       | 0     | 0      | 0.970887 | 0.029113 |
| 318 | 5               | 2         | 2          | 2                 | 2                           | 1           | 2               | 2               | 1       | 0     | 0      | 0.943572 | 0.056428 |
| 183 | 1               | 2         | 3          | 1                 | 2                           | 1           | 3               | 1               | 1       | 0     | 0      | 0.979370 | 0.020630 |
| 478 | 5               | 10        | 10         | 10                | 6                           | 10          | 6               | 5               | 2       | 1     | 1      | 0.001305 | 0.998695 |
| ... | ...             | ...       | ...        | ...               | ...                         | ...         | ...             | ...             | ...     | ...   | ...    | ...      | ...      |
| 425 | 10              | 4         | 3          | 10                | 4                           | 10          | 10              | 1               | 1       | 1     | 1      | 0.006201 | 0.993799 |
| 314 | 8               | 10        | 3          | 2                 | 6                           | 4           | 3               | 10              | 1       | 1     | 1      | 0.067440 | 0.932560 |
| 15  | 7               | 4         | 6          | 4                 | 6                           | 1           | 4               | 3               | 1       | 1     | 1      | 0.436887 | 0.563113 |
| 510 | 3               | 1         | 1          | 2                 | 2                           | 1           | 1               | 1               | 1       | 0     | 0      | 0.983410 | 0.016590 |
| 351 | 2               | 1         | 1          | 1                 | 2                           | 1           | 2               | 1               | 1       | 0     | 0      | 0.987405 | 0.012595 |

171 rows × 13 columns
