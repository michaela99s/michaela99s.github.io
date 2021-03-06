---
layout: post
title: 랜덤포레스트
description: >
  랜덤포레스트 모델을 알아봅니다.
sitemap: false
hide_last_modified: true
---

랜덤포레스트는 의사결정나무 기반의 앙상블 알고리즘입니다. 각 의사결정나무는 훈련셋의 서로 다른 임의적인 부분집합에 대하여 훈련합니다. 모든 개별 나무로부터 예측값을 받고 가장 많은 표를 받은 클래스를 예측합니다. 회귀의 경우에는 각 나무의 예측을 평균한 값을 예측합니다.

# 랜덤 포레스트 Random Forest

랜덤 포레스트의 과정은 다음과 같습니다. 부트스트래핑을 통하여 N 개의 샘플링 데이터셋을 생성합니다. 각 데이터셋에서 임의의 변수를 선택합니다. M개의 변수가 있을 때 보통 sqrt(M) 또는 M/3 개의 변수들을 선택합니다. 의사결정 트리들을 종합하여 앙상블 모델을 만들고 OOB 오류를 통해 오분류율을 평가합니다. 

랜덤포레스트 알고리즘은 나무가 성장함에 따라 randomness가 추가됩니다. 노드를 분리할 때 최적 피처를 찾는 대신 피쳐들의 임의의 부분집합 중 최적의 피쳐를 찾습니다. 이는 나무의 다양성을 더해줍니다. 이는 분산을 줄여주지만 편향성을 높입니다. 그러나 전반적으로 더 나은 모델이 됩니다. 정리하자면, 랜덤포레스트는 예측 편향을 줄이고 과적합을 방지할 수 있으며 이상치에 영향을 적게 받습니다.

## scikit-learn

랜덤 포레스트는 사이킷런의 `ensemble`에 있습니다.

```python
class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, 
criterion='gini', max_depth=None, min_samples_split=2, 
min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', 
max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
oob_score=False, n_jobs=None, random_state=None, verbose=0, 
warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
```

```python
class sklearn.ensemble.RandomForestRegressor(n_estimators=100, *, 
criterion='squared_error', max_depth=None, min_samples_split=2, 
min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, 
max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
oob_score=False, n_jobs=None, random_state=None, verbose=0, 
warm_start=False, ccp_alpha=0.0, max_samples=None)
```

기본적으로 랜덤포레스트 하이퍼파라미터에는 의사결정나무와 공통되는 것이 많습니다. 

- n_estimators: 나무의 수를 정합니다. 몇 개의 의사결정나무 모델의 결과값을 평균으로 분류나 회귀를 할 지 정합니다. 

- max_features: {"sqrt", "log2", None}, int or float
  최적의 분할을 찾을 때 고려할 피쳐의 수를 정합니다. 즉, 몇 개의 피쳐를 반영할 것인가입니다.

## 분류 예시

타이타닉 생존 여부를 랜덤 포레스트로 분류해보겠습니다.

```python
from sklearn.ensemble import RandomForestClassifier
```

```python
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

print(f"트레인셋 정확도: {rf_clf.score(X_train, y_train):.4f}")
print(f"테스트셋 정확도: {rf_clf.score(X_test, y_test):.4f}")
```

```
트레인셋 정확도: 0.9803
테스트셋 정확도: 0.8156
```

랜덤 포레스트에서도 오버피팅이 발생하고 있지만 이전에 살펴보았던 모델들에 비해 테스트셋의 정확도가 준수하게 나타납니다. 

### Hyperparameter Tuning

하이퍼파라미터 중 `n_estimators`와 `max_featues`를 조정하여 모델을 개선하도록 하겠습니다.

#### Grid Search

```python
param_grid = {'n_estimators': range(100, 1000, 100),
              'max_features': ['auto', 'sqrt', 'log2']}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best Parameter: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")
```

```
Best Parameter: {'max_features': 'auto', 'n_estimators': 500}
Best Score: 0.8034
Test Score: 0.8045
```

최적 하이퍼파라미터로서 `n_estimators`는 500으로 기본값인 100에 비해 커지게 되었습니다. 이는 기본보다 많은 나무를 고려하게 됨을 의미합니다.

#### Random Search

```python
param_distribs = {'n_estimators': randint(low=100, high=1000),
                  'max_features': ['auto', 'sqrt', 'log2']}

random_search = RandomizedSearchCV(RandomForestClassifier(),
                                   param_distributions=param_distribs, n_iter=20, cv=5)
random_search.fit(X_train, y_train)

print(f"Best Parameter: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_:.4f}")
print(f"Test Score: {random_search.score(X_test, y_test):.4f}")
```

```
Best Parameter: {'max_features': 'sqrt', 'n_estimators': 526}
Best Score: 0.8034
Test Score: 0.8101
```

랜덤 서치의 결과 `n_estimators`는 526으로 그리드 서치보다 더 많은 나무를 고려하게 됩니다. 그 결과, 더 높은 정확도의 모델을 얻을 수 있었습니다. 이 모델을 이용하여 피쳐 중요도를 계산해보겠습니다.

### 피쳐 중요도

```python
ftr_importances_values = random_search.best_estimator_.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=X.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)

plt.title("Feature Importances")
sns.barplot(x=ftr_sort, y=ftr_sort.index)
plt.show()
```

![](data:image/png;base64,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fd/v8/HxqTFts9ku+ZrT6byq7dbX93ury/Dhw4mIiGDJkiXcfPPN+Pr6MmDAACorK2ss9/0LG66mdqfTSatWrdi+fXuted9uNzY2lq+++or169ezbt060tLSmD59Op9//jk333xzvd5HGoaCRRqlxMREsrOziY+Pr/MS2E8//ZSHHnqI5ORk4OIvp9zcXFq3bu1axtfXl+rq6hrrRUZGAnD8+HHXa4WFhRw7dqxedeXk5FwyDBvK1q1bGTt2rGt68+bNrhFKt27dOH36NHv37nW9duHCBT7//PMa61zKt7/Aq6ur8fLyAnBt6//+7/+46667APjmm29qjaKu5L/+678A+OSTT7j//vtrzU9MTOTcuXNUVFTQvXv3Orfj5+fHT37yE37yk5+QlpZG69at+eCDD2pcBCDup6vCpFF69tln2bdvHw8//DDbtm3j4MGDrF+/ngkTJpCfnw9Ap06d+PDDD9m2bRt79+4lJSWlRlgA/OAHP2DHjh0cOHCAoqIiqqqq8Pf3p3///rzyyivs3r2bHTt28Mgjj+Dn53fFul5++WU+/PBDJk+ezK5duzhw4AD/+Mc/ePzxxxvs6qS///3vLF68mP3797No0SLeffddpkyZAsCQIUPo1asXv/jFL9i0aRN79uzhkUceoaKigieffPKy2/3BD34AwJo1azh16hSlpaWEhoYSGRnJsmXLyM3NZcuWLTz44IP4+/tfVc3t27fnoYceYuzYsbz99tscOHCA7du3u65oGzJkCElJSYwcOZIPPviA/Px8duzYwaJFi1i2bBkAf/rTn1i2bBm7d+/m8OHD/OUvf6GkpKTGYT9pHBQs0ih16dKFzZs3U1payl133UXXrl355S9/yfnz5133uSxYsIC2bdsyePBghg4dSmxsbK2/hqdMmUJERAQJCQlERkayadMmAJYvX05gYCD9+vXjgQceICUlhZtuuumKdQ0ePJh169aRnZ3NwIED6dGjB5MmTSIoKKjWYawb5YUXXiArK4uEhARmzZrFK6+8wr333gtcPPz0wQcf0LlzZ+6++25uu+02Tpw4QWZmJhEREZfd7m233caECRP41a9+RVRUFE899RR2u5333nuPAwcO0KNHD8aMGcPEiRPr9Vl934oVK/jVr37Fb3/7W7p06cK9997LwYMHXXWvWbOGkSNHMmnSJFf9H3/8MfHx8QCE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E0NxcTG5ubkAOBwOjh496s4WREQ8UqMdsWzatImf/exnNV7r3bs3x44dIywsjMmTJxMeHs4tt9xCQEAA3t7eTJkyhRUrVlBeXk51dTXDhg3j5ptvdlMHIiKeyWaMMe4u4mpVVFTQokULSkpKePbZZ0lLSyMkJOSat7fif3dYV5yIiMUa44O+YmJi6pzXaEcslzNnzhzKyspwOBzcd9991xUqIiJirSYZLDNmzHB3CSIiUodGe/JeRESaJgWLiIhYSsEiIiKWUrCIiIilmuTlxlY7fvy4u0toEBERERQVFbm7jAbhKb16Sp/gOb02lT4vd7mxRiwiImIpBYuIiFhKwSIiIpZSsIiIiKUULCIiYikFi4iIWKpJfleY1RYuXOjuEuQ7nn76aXeXICLXQSMWERGxlIJFREQspWARERFLKVhERMRSChYREbGUgkVERCylYBEREUspWERExFINeoPkqFGjiIuLw+l0Ehsby7hx4/Dz87vksqtXr6ZFixaMGDGiIUsUEZHr1KAjFl9fX+bNm8f8+fPx9vYmMzOzId9eREQagNu+0qVz584cOXIEgH/961989NFH2Gw24uLiGD9+fI1ls7KyWLt2LQ6Hg9atWzN+/Hj8/PzYsmULf/3rX7Hb7QQEBPDSSy9x9OhRli5disPhwBjDlClTuOmmm9zRooiIR3JLsFRXV7Nr1y569uzJ0aNHycjIIC0tjeDgYEpLS2st37t3b5KSkgB45513WLduHT/96U/561//ynPPPUdYWBhlZWUAZGZmMmzYMAYOHIjD4cDpdNbaXlZWFllZWQDMmTPnBnYqIuJ5GjRYKisrmTp1KgBdunRhyJAhZGZm0qdPH4KDgwEIDAystd7Ro0d55513KCsro6KigoSEBAA6derEkiVL6Nu3L7179wagY8eOZGRkcPr0aXr37n3J0UpSUpIrqERExFoNGizfnmO5WkuWLGHq1Km0a9eODRs2kJOTA0BKSgr79+9n586dpKamMmfOHAYMGED79u3ZuXMns2fPJiUlhe7du1vdioiI1MHtlxt3796drVu3UlJSAnDJQ2EVFRWEhobicDj47LPPXK+fOHGCDh06MGrUKIKDgzl9+jQnT56kdevWDBs2jMTERA4fPtxgvYiISCN4HsvNN9/Mvffey4wZM7Db7bRr145x48bVWGbUqFE8++yzBAcH06FDB86fPw/A22+/TUFBAXAxoNq2bcuHH37Ip59+ipeXFyEhIYwcObLBexIR8WQ2Y4xxdxHulpqa6u4S5DuseNBXREQERUVFFlTTuHlKn+A5vTaVPmNiYuqc5/ZDYSIi0rwoWERExFIKFhERsZSCRURELKVgERERSylYRETEUgoWERGxlNtvkGwMrLhvoiloKtfHi0jTphGLiIhYSsEiIiKWUrCIiIilFCwiImIpBYuIiFhKV4UB9n3z3V1CgzgD0GWKu8sQkWZOIxYREbGUgkVERCylYBEREUspWERExFIKFhERsZSCRURELKVgERERSylYRETEUm69QTIjI4ONGzdit9ux2WykpKSwdu1ahg8fTps2bRg9ejRvvfVWrfVyc3N58803qaqqwuFw0LdvX5KTk93QgYiIfJ/bgiU3N5cdO3Ywd+5cfHx8KC4uxuFw8Otf//qK6y5ZsoRJkybRrl07nE4nx48fb4CKRUSkPtwWLGfPniUoKAgfHx8AgoODAZgxYwajR48mPj4egDfffJPs7GxCQkKYOHEiwcHBFBcXExoaCoDdbqdNmzYArF69mpMnT3LixAlKSkoYMWIESUlJbuhORMRzue0cS0JCAqdPn2bChAm88cYb7N27t9YyFy5cID4+nvT0dLp27cp7770HwN13383EiROZN28emZmZVFZWutY5cuQIL774Ir/73e/429/+xpkzZ2ptNysri9TUVFJTU29cgyIiHsptI5YWLVowd+5c9u3bR05ODgsWLOChhx6qsYzNZqNfv34ADBw4kN///vcA3H///QwYMIDs7Gw2btzIpk2bmDFjBgCJiYn4+vri6+tLt27dyMvLo1evXjW2m5SUpJGMiMgN4taT93a7nW7dutGtWzfi4uLYsGHDZZe32Wyun6Ojo4mOjmbo0KE88cQTlJSU1FrmUtMiInJjue1Q2PHjxykoKHBNHzp0iMjIyBrLGGPYunUrABs3bqRz584A7Ny5E2MMAAUFBdjtdlq2bAnA9u3bqayspKSkhJycHNe5GhERaRhuG7FUVFSwfPlyysrK8PLyIjo6mpSUFNLT013L+Pn5kZeXR0ZGBsHBwUyaNAmATz/9lJUrV+Lr64uXlxfjx4/Hbr+YkW3btuWll16ipKSE++67j7CwMLf0JyLiqWzm2z/9m4HVq1fTokULRowYcVXrnVjrOQ+/cnrIg74iIiIoKipydxk3nKf0CZ7Ta1PpMyYmps55uvNeREQs1aweTay770VE3E8jFhERsZSCRURELKVgERERSylYRETEUgoWERGxVLO6Kuxa6d4OERHraMQiIiKWUrCIiIilFCwiImIpBYuIiFhKwSIiIpbSVWHAXzY/5e4S6u2hfovdXYKIyGVpxCIiIpZSsIiIiKUULCIiYikFi4iIWErBIiIillKwiIiIpRQsIiJiKQWLiIhYyq03SI4aNYq4uDicTiexsbGMGzcOPz+/a95eYWEhc+fOZf78+RZWKSIiV8OtIxZfX1/mzZvH/Pnz8fb2JjMzs17rVVdX3+DKRETkWjWar3Tp3LkzR44c4YsvviAjIwOHw0FQUBDjx48nJCSE1atXc/LkSQoLCwkPD2fMmDEsW7aMwsJCAJ544glCQ0NxOp387//+L7m5uYSFhTFt2jR8fX3d3J2IiOdoFMFSXV3Nrl276NmzJ507d2bmzJnYbDbWrl3LmjVreOSRRwD45ptvSEtLw9fXlwULFtC1a1emTp2K0+mkoqKC0tJSCgoKmDBhAr/+9a9JT09n69at3H777TXeLysri6ysLADmzJnT4P2KiDRnbg2WyspKpk6dCkCXLl0YMmQIx48f59VXX+Xs2bM4HA6ioqJcyycmJrpGH3v27OGppy5+eaTdbicgIIDS0lKioqJo164dALfccgunTp2q9b5JSUkkJSXd4O5ERDyTW4Pl23Ms37V8+XKGDx9OYmIiOTk5vPfee6559Tmx7+Pj4/rZbrdTWVlpXcEiInJFje5y4/LycsLCwgD417/+VedyP/zhD/nkk08AcDqdlJeXN0h9IiJyeY3iHMt3/fznPyc9PZ2WLVvSvXt318n57xszZgyvv/4669atw26388tf/pKQkJCGLVZERGqxGWOMu4twt3l/HenuEurteh70FRERQVFRkYXVNF6e0qun9Ame02tT6TMmJqbOeY3uUJiIiDRtChYREbGUgkVERCylYBEREUspWERExFIKFhERsZSCRURELNXobpB0h+u5N0RERGrSiEVERCylYBEREUspWERExFIKFhERsZSCRURELKWrwoDQr/PcXUKdznZq7+4SRESuikYsIiJiKQWLiIhYSsEiIiKWUrCIiIilFCwiImIpBYuIiFhKwSIiIpZSsIiIiKWueIPkqFGjiIuLc03379+fe+65p14bz8nJ4aOPPiI1NfWaC5wxYwajR48mPj7+qtddsmQJP/rRj+jTp881v7+IiFydKwaLr68v8+bNa4haanE6nW55XxERuXbX/JUu48aNo3///nz55Zd4eXmRkpLCqlWrOHHiBP/93//NnXfeCcD58+eZPXs2J06coFu3bjzxxBPY7XaWLVvGgQMHqKyspE+fPiQnJ7u227dvX/79738zYsQI1/s5nU7++Mc/Eh4eTnJyMn/5y1/Yu3cvVVVV3HXXXfz4xz/GGMPy5cvJzs4mIiICb299Y42ISEO74m/eyspKpk6d6pq+99576devHwARERHMmzePN998k6VLl5KWlkZVVRVTpkxxBUteXh7p6elERkYyc+ZMtm3bRp8+fXjwwQcJDAzE6XTy8ssvc/jwYdq2bQtAUFAQc+fOBSAzM5Pq6moWLlxIXFwcI0eOJCsri4CAAGbPnk1VVRXPP/88CQkJHDx4kOPHj7NgwQLOnTvH5MmTGTx4cK2esrKyyMrKAmDOnDnX+RGKiMh3XdehsMTERADi4uKoqKjA398ff39/vL29KSsrA6B9+/a0bt0auHh+5quvvqJPnz5s3ryZtWvXUl1dzdmzZ/nmm29cwfJtcH1r2bJl9O3bl5EjRwKwe/dujhw5wtatWwEoLy+noKCAffv20b9/f+x2O2FhYXTv3v2SdSclJZGUlHTFD0dERK7edR0r+vZQk91ux8fHx/W63W6nurq6zvUKCwv56KOPmD17NoGBgSxZsoSqqirXfD8/vxrLd+zYkZycHIYPH46vry/GGB577DF69uxZY7kvv/zyetoREREL3PDLjfPy8igsLMTpdLJlyxY6d+5MeXk5LVq0ICAggHPnzrFr167LbmPIkCHceuutLFiwgOrqanr27Mknn3yCw+EA4Pjx41RUVNClSxe2bNmC0+nk7Nmz5OTk3Oj2RETke676HEvPnj156KGH6v0G7du3509/+pPr5H2vXr2w2+20a9eOSZMmER4eTqdOna64neHDh1NeXs6iRYt4+umnKSwsZPr06QAEBwczdepUevXqxZ49e5g0aRIRERF07Nix3nWKiIg1bMYY4+4i3O38+k/dXUKdrHzQV0REBEVFRZZtrzHzlF49pU/wnF6bSp8xMTF1ztOd9yIiYikFi4iIWErBIiIillKwiIiIpRQsIiJiKQWLiIhYSsEiIiKW0tf/Yu29IiIink4jFhERsZSCRURELKWvdBEREUt5/IglNTXV3SU0GPXa/HhKn+A5vTaHPj0+WERExFoKFhERsZTHB4snPaJYvTY/ntIneE6vzaFPnbwXERFLefyIRURErKVgERERSzXrr3TZtWsXK1aswOl0MnToUO65554a86uqqli8eDH5+fkEBQUxceJEoqKiAHj//fdZt24ddrudxx57jJ49ezZ8A1fhWnstLCxk0qRJrseMdujQgZSUFDd0UD9X6nPv3r2sXLmSw4cPM3HiRPr06eOat2HDBjIyMgAYOXIkgwYNasDKr9719Dpq1Cji4uKAi4+6nT59ekOWflWu1Off//531q5di5eXF8HBwTz55JNERkYCzW+fXq7XprRPMc1UdXW1eeqpp8yJEydMVVWVeeaZZ8zRo0drLPOPf/zDvPbaa8YYYzZu3GjS09ONMcYcPXrUPPPMM6aystKcPHnSPPXUU6a6urrBe6iv6+n15MmTZvLkyQ1e87WoT58nT540hw4dMosWLTJbtmxxvV5SUmLGjRtnSkpKavzcWF1Pr8YY8/DDDzdkudesPn3++9//NhUVFcYYY/75z3+6/u82x31aV6/GNJ19aowxzfZQWF5eHtHR0bRu3Rpvb2/69evH9u3bayzzxRdfuP7C6dOnD3v27MEYw/bt2+nXrx8+Pj5ERUURHR1NXl6eG7qon+vptSmpT59RUVG0bdsWm81W4/Vdu3bRo0cPAgMDCQwMpEePHuzatasBq78619NrU1KfPrt3746fnx9wcUR95swZoHnu07p6bWqabbCcOXOG8PBw13R4eHitnfTdZby8vAgICKCkpKTWumFhYY16B19PrwCFhYVMmzaNF198kX379jVc4VepPn3Wd93msE8vp6qqitTUVJ577jm2bdt2I0q0xNX2uW7dOtdh6ea+T7/bKzSdfQrN/ByLXFloaChLly4lKCiI/Px85s2bx/z58wkICHB3aXIdli5dSlhYGCdPnuTll18mLi6O6Ohod5d1XT799FPy8/OZMWOGu0u54S7Va1Pap812xBIWFsbp06dd06dPnyYsLKzOZaqrqykvLycoKKjWumfOnKm1bmNyPb36+PgQFBQEwC233ELr1q0pKChouOKvQn36rO+6zWGfXml9gNatW9O1a1cOHTpkdYmWqG+f2dnZvP/++0ybNg0fH59Lrttc9umlev12fWj8+xSacbDEx8dTUFBAYWEhDoeDzZs3k5iYWGOZH/3oR2zYsAGArVu30q1bN2w2G4mJiWzevJmqqioKCwspKCigffvG+zCw6+m1uLgYp9MJwMmTJykoKKB169YN3UK91KfPuvTs2ZPdu3dTWlpKaWkpu3fvbtRX+l1Pr6WlpVRVVQFQXFzM119/TZs2bW5kudesPn0ePHiQZcuWMW3aNFq1auV6vTnu07p6bUr7FJr5nfc7d+5k5cqVOJ1OBg8ezMiRI3n33XeJj48nMTGRyspKFi9ezMGDBwkMDGTixImuX6oZGRmsX78eu93OmDFjuPXWW93czeVda69bt25l9erVeHl5Ybfb+fnPf17vX2DucKU+8/Ly+P3vf09ZWRk+Pj6EhISQnp4OXDxm/f777wMXL00dPHiwO1u5omvt9euvv+b111/HbrfjdDq5++67GTJkiLvbqdOV+kxLS+PIkSOEhIQANS+1bW77tK5em9o+bdbBIiIiDa/ZHgoTERH3ULCIiIilFCwiImIpBYuIiFhKwSIiIpZSsIiIiKUULCIiYqn/B4g8b4o+WxN2AAAAAElFTkSuQmCC)

랜덤 포레스트의 피쳐 중요도는 운임과 성별, 나이가 가장 높게 나타납니다. 

## 회귀 예시

보스턴 집값을 랜덤포레스트 회귀로 예측해봅니다.

```python
from sklearn.ensemble import RandomForestRegressor
```

```python
rf = RandomForestRegressor()
rf.fit(X_train_minmax, y_train)
pred_rf = rf.predict(X_test_minmax)

MSE_train = mean_squared_error(y_train, rf.predict(X_train_minmax))
MSE_test = mean_squared_error(y_test, pred_rf)

print(f"Train R2: {rf.score(X_train_minmax, y_train):.4f}")
print(f"Test R2: {rf.score(X_test_minmax, y_test):.4f}")
print(f"Train RMSE: {np.sqrt(MSE_train):.4f}")
print(f"Test RMSE: {np.sqrt(MSE_test):.4f}")
```

```python
Train R2: 0.9774
Test  R2: 0.8846
Train RMSE: 1.1980
Test  RMSE: 2.4275
```

랜덤포레스트를 사용하였을 때에도 의사결정나무와 같이 오버피팅된 양상을 확인할 수 있지만 그 정도는 의사결정나무만큼 심하지 않습니다.

### Hyperparameter Tuning

`n_estimators`와 `max_features`를 조정하여 최적의 모델을 탐색해 봅시다.

#### Grid Search

```python
param_grid = {'n_estimators': range(100, 500, 100), 
              'max_features': ['auto', 'sqrt', 'log2']}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_train_minmax, y_train)

print(f"Best Parameter: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")
print(f"Train Score: {grid_search.score(X_train_minmax, y_train):.4f}")
print(f"Test  Score: {grid_search.score(X_test_minmax, y_test):.4f}")
```

```
Best Parameter: {'max_features': 'log2', 'n_estimators': 300}
Best Score: 0.8718
Train Score: 0.9830
Test  Score: 0.8791
```

#### Random Search

```python
param_distribs = {'n_estimators': randint(low=100, high=500),
                  'max_features': ['auto', 'sqrt', 'log2']}
random_search = RandomizedSearchCV(RandomForestRegressor(),
                                   param_distributions=param_distribs, 
                                   n_iter=20, cv=5)
random_search.fit(X_train_minmax, y_train)

print(f"Best Parameter: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_:.4f}")
print(f"Train Score: {random_search.score(X_train_minmax, y_train):.4f}")
print(f"Test  Score: {random_search.score(X_test_minmax, y_test):.4f}")
```

```
Best Parameter: {'max_features': 'sqrt', 'n_estimators': 229}
Best Score: 0.8741
Train Score: 0.9826
Test  Score: 0.8811
```

### 피쳐 중요도

그리드 서치 결과를 이용하여 피쳐 중요도를 살펴보겠습니다.

```python
rf_best = grid_search.best_estimator_

ftr_importances_values = rf_best.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=X.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)

plt.title("Feature Importances")
sns.barplot(x=ftr_sort, y=ftr_sort.index)
plt.show()
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgMAAAFXCAYAAAA/LE0rAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAf+ElEQVR4nO3df3TP9f//8dvLfrL5sWnWKsqP9/iU/H7nx0JEhEmEin2oFol32Vjzux2/sjdFEu/OIbyZJjJ5k3QknPeand6LvHkrRZEPbX43i/16Pb5/dHp9eZthtr3m9bhezukce/283/ekXc/z9eLlMMYYAQAAa1Vy9wAAAMC9iAEAACxHDAAAYDliAAAAyxEDAABYjhgAAMBy3u4eALBdw4YNFR4erkqV/n+bN27cWDNmzCjR4+3du1dr167V1KlTS2vEqzRs2FBpaWkKDg4us+coypo1a5SXl6dBgwaV6/MCno4YACqA5cuXl9oP1h9++EGZmZml8lgVTUZGhv70pz+5ewzA4xADQAV26NAhzZgxQ+fOnVNhYaGioqL01FNPyel0aubMmfrmm2+Uk5MjY4ymT5+uu+66S/Pnz1d2drbGjx+vPn36aNq0adq4caMkKT093fX1O++8oz179igrK0sNGzbUnDlztGjRIn322WdyOp26++679frrrys0NPSa8x07dkxDhgxRmzZttGfPHhUUFOi1117T6tWrdfjwYTVu3FhvvfWWjh8/rqioKD300EP69ttvZYzRlClT1KpVK+Xn52vWrFlKS0uTl5eXmjRpovHjxyswMFCdO3dWkyZN9N133yk2Nlbbtm1Tamqq/P391a1bN02ZMkWnT5/WyZMndffdd2vevHmqWbOmOnfurCeffFJpaWk6ceKEHn/8cb322muSpLVr12rp0qWqVKmSgoKClJiYqLCwMG3btk2LFi1Sfn6+/P39FR8fr+bNm+vQoUOaOHGi8vLyZIzRU089xZkJeB4DwK3Cw8NNr169TO/evV3/nTp1yuTn55sePXqYffv2GWOM+fXXX83jjz9udu/ebb7++mvzl7/8xRQWFhpjjHnvvffM8OHDjTHGfPTRR2bYsGHGGGN27dplevbs6Xquy7+eP3++6datm8nPzzfGGJOSkmJGjx7t+jo5OdlER0dfc+bTp0+bn3/+2YSHh5utW7caY4yZMmWK6dSpk8nOzjaXLl0yERERJiMjw3W7DRs2GGOM2b59u4mIiDB5eXnm7bffNqNGjTJ5eXmmsLDQjBs3zkyePNkYY0ynTp3MggULXM8bHx9vFi9ebIwxZtmyZea9994zxhjjdDpNdHS0WbJkiet+s2bNMsYY88svv5gHH3zQHD161Bw4cMC0bt3aHD9+3BhjzNKlS83kyZPNjz/+aHr16mXOnDljjDHm4MGDJiIiwuTk5Jjx48e7nicrK8uMHj3a9X0HPAVnBoAKoKiXCX744QcdPXpUEyZMcF126dIl/ec//9Gzzz6r6tWrKzk5WT///LPS09MVEBBw08/brFkzeXv//r+BL774Qv/+97/Vr18/SZLT6dTFixev+xg+Pj7q3LmzJKlOnTpq3ry5AgMDJUm1atXS+fPnVatWLVWvXl2RkZGSpI4dO8rLy0vfffeddu7cqZiYGPn4+EiSoqKiNHLkSNfjt2rVqsjnHTJkiP71r39p6dKl+umnn/T999+radOmrusfffRRSVJoaKhq1qyp8+fP66uvvtLDDz+ssLAwSdLQoUMlSUlJScrKynJ9LUkOh0NHjx5V165dFR8fr71796pt27aaNGnSFe/vADwBMQBUUIWFhapWrZo+/vhj12WnTp1S1apVtX37ds2YMUPPPfecHn30UdWrV08bNmy46jEcDofMZR8/kp+ff8X1VapUcf3a6XQqOjpazz77rCQpLy9P58+fv+6cPj4+cjgcV3xdFC8vryu+djqd8vLyktPpvOryy+e8fMbLzZ49W3v37lW/fv3UunVrFRQUXLGrn5+f69d/fB+8vLyumPXSpUv6v//7PzmdTrVt21bz5s1zXXfixAnVqlVLjRo10pYtW/Tll18qLS1N7777rpKTk1WnTp1ivivA7YW8BSqounXrys/PzxUDJ06cUK9evbRv3z6lpqaqU6dOevbZZ/Xggw9q69atKiwslPT7D92CggJJUnBwsI4fP67Tp0/LGKOtW7de8/kefvhhrV27VhcuXJAkvf32267X2UvDmTNntHPnTknStm3b5OPjo/DwcLVv317JycnKz8+X0+lUUlKSIiIiinyMy3f75z//qSFDhqhPnz6qWbOmvvzyS9f34Fpat26ttLQ0ZWVlSZKSk5M1e/ZstWnTRqmpqTp06JAkaceOHerdu7dyc3M1ZswYffLJJ+rZs6def/11BQYG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)

RM과 LSTAT의 상대적 중요도가 크게 나타나고 있습니다. 

## 종합 정리

랜덤 포레스트는 적절히 하이퍼파라미터를 조정한다면, 과대적합을 방지하면서도 좋은 성능의 모델을 얻을 수 있습니다. 랜덤 포레스트는 개별 알고리즘보다 좋은 성능을 보이면서도 다른 앙상블 기법에 비해 간단하기 때문에 가장 많이 사용됩니다. 
