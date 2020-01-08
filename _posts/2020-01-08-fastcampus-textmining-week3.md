텍스트 분석을 위한 머신러닝 week3
==============

3주차에는 텍스트를 분류하는 여러가지 알고리즘의 개념, 원리과 그 장단점 등에 대해 배웠습니다.
수학적인 개념을 이해해야만 알고리즘의 원리가 와닿아서 수식을 하나하나 뜯어보면서 공부해야 했던 점이 3주차 수업 내용을 복습하면서 가장 힘든 부분이었습니다.

## 1. Logistic regression
- 대표적인 binary classification 알고리즘입니다.
- positive class에 속할 확률을 점수처럼 0에서 1 사이로 표현하기 때문에 확률모형처럼 이용할 수 있습니다.
- softmax regression은 logistic regression의 multi-class 버전입니다. 일반적으로 logistic regression은 두 클래스의 경계면을 학습한다고 알려져 있는 반면, softmax regression으로 표현하면 𝜃(𝒊)를 클래스 𝒊의 대표벡터로 해석할 수 있습니다.
- 하지만 cosine measure처럼 𝜃(𝒊)의 방향이 중요하며, 아래 그림과 같이 두 클래스의 데이터가 같은 방향에 있으면 𝜃(𝒊)만으로 두 클래스를 구분하기 어려울 수 있습니다.
<p>
<img src='https://user-images.githubusercontent.com/52257022/71971499-1aff5380-324e-11ea-81a2-c9b3a2de06dc.png'>
</p>
- 클래스별로 잘 분산된 데이터는 대표벡터가 잘 학습되며, 대표벡터는 각 점이 해당 클래스에 속할 확률이 가장 크도록 학습됩니다.

## Regularization
- p-norm은 벡터의 크기를 정의하는 방법입니다.
<p>
<img src ='https://user-images.githubusercontent.com/52257022/71985801-9bce4780-326e-11ea-9d3f-2de0f6de32bf.png'>
</p>
- regularization은 overfitting 방지 효과가 있으며, *L1* 은 _입력변수의 계수 중 일부를 0으로 만들어 중요한 영향을 끼치는 변수를 선택할 수 있게_ 해주며 *L2*는 _경계면을 날카롭지 않게_ 만들어줍니다.

### L1 Regularization
<p>
<img src = 'https://user-images.githubusercontent.com/52257022/71972595-4b47f180-3250-11ea-96b6-3250a0f3c8c7.png'>
</p>
몇 개의 변수만 선택적으로 0이 아닌 계수를 가지도록 하기 때문에 sparse modeling이라고도 불립니다.
- L1 regularization을 이용한 방법인 Lasso regression은 중요한 변수를 데이터 기반으로 추출합니다. 따라서 lasso 모델의 성질을 이용하면 키워드를 추출할 수 있습니다.
- Lasso 모델이 선택하는 키워드는 두 가지 조건을 만족합니다:
	- 분별력이 좋으면서,
	- 많은 문장에서 등장한 단어를 우선적으로 선택합니다.
- Lasso model은 성능을 저하하지 않으면서 모델이 이용하는 feature의 개수를 최소화합니다. 그렇기 때문에 term frequency vector로 표현된 문서의 종류를 분류하는 태스크에서는 적은 수의 단어만을 이용하여 문서 카테고리를 잘 맞출 수 있습니다.

### L2 Regularization
<p>
<img src = 'https://user-images.githubusercontent.com/52257022/71972689-7e8a8080-3250-11ea-8b56-76e598072ff3.png'>
</p>

## 2. Feed forward neural network
- clustering 과 비슷하지만, k-means는 hard하게 분리하는 알고리즘이라면 feed forward neural network는 부드러운 편입니다.
- Neural network의 hidden layer는 linear inseparable한 데이터를 linear separable한 공간으로 바꿔줍니다. (한 hidden layer는 하나의 '지역')
- 또한 sigmoid 함수는 데이터가 boolean에 가깝게 표현되도록 변형시킵니다.
- Activation function은 hidden layer에서 softmax를 적용하는 것과 비슷한 효과를 보이며, 마지막 layer는 softmax regression을 이용하여 각 클래스에 속할 확률을 나타냅니다.

## 3. k-Nearest Neighbor classifier
- k-NN 분류기는 query points와 가장 가까운 k개의 데이터를 찾은 뒤, 그 점들의 label 중에서 숫자가 가장 많은 label을 리턴합니다.
- term frequency 벡터로 표현된 문서는 유크리디안보다는 코사인이 선호됩니다. (유클리디안은 공통된 단어의 유무를 표현하기 어렵기 때문)
- k-NN classifier의 두 가지 어려움:
	- computation cost가 높습니다. (인덱싱 방법으로 해결 가능합니다.)
	- 좋은 유사도 함수/ 벡터 표현이 전제되어야 합니다.

## 4. Support Vector Machine
- SVM은 n개의 데이터 중에서 중요한 몇 개의 포인트 (support vector로 불림)를 선택하여 분류합니다.
- 여러 개의 경계면을 가질 수 있으며 그 중 margin, 여백이 가장 큰 경계면이 가장 안전한 경계면이기 때문에 선택됩니다.
* hard margin solution: 오류가 없는 경계면
* soft margin solution: 데이터의 오류를 고려한 경계면
- knn 과 비슷하지만, 당연한 classification은 제쳐두고 meaningful한 것들을 고려하는 알고리즘입니다.
- 내적은 공통된 단어 유무를 잘 표현하므로 linear kernel 만으로도 문서 분류에서 좋은 성능을 낼 수 있습니다.
- non-linear한 분류는 kernel 기법 사용 (andrew ng: 두 데이터 포인트 간의 유사도로 해석해도 무관)
	- Kernel trick 은 non-linear 한 데이터 공간을 SV를 이용하여 linear 공간으로 변환합니다.
	Query vector q 는 SV 와의 유사도 벡터 𝑲(𝒙,𝒒) 로 표현됩니다. 𝑲(𝒙,𝒒) 벡터가 가중치 벡터 𝜶 와 내적이 되는 것과 같습니다.
* LASSO 는 유용한 feature, SVM은 data 선택


## 5. Naïve Bayes Classifier
- document classification을 하고자 할 때 기본적으로 성능이 나오는지 확인해 보기에 좋은 분류기입니다.
- bayes rule : 
<p>
<img src = 'https://user-images.githubusercontent.com/52257022/71983183-8dc9f800-3269-11ea-9128-887b0a5e6de7.png'>
</p>
- 각 문서 종류 𝒚 에서의 단어 비율 𝑷(𝒙|𝒚)의 누적곱을 이용합니다.
- 문서 분류에서는 '특정 단어가 있는가?'가 중요하며, naive bayes는 이 가정과 공식이 잘 일치합니다. (직접적으로 확인하기 때문에 works good as a baseline)
- 각 클래스별 단어 확률 분포만 학습하면 되기 때문에 학습 속도가 빠르다는 장점이 있습니다. 하지만 단어 빈도만을 고려하기 때문에 키워드 추출은 불가능합니다.

## 6. Decision Tree
- 여러 단계의 decision node로 이뤄진 ‘플로우차트’ 같은 분류기입니다.
- 아래 그림과 같이 의사결정나무의 각 지역은 규칙으로 표현/정의되며, 재귀적으로 leaf를 만들어 지역을 나눕니다. 
<p>
<img src = 'https://user-images.githubusercontent.com/52257022/71983460-311b0d00-326a-11ea-858f-9ed53718ded4.png'>
</p>

* Decision Tree 사용의 장점:
  - 텍스트 분석 이외의 태스크에서는 좋은 성능을 내는 경우가 많습니다. 
  - 변수를 독립적으로 보기 때문에 scaling이나 missing value등에 영향을 적게 받습니다.
  - ifelse 룰로 표현되어, 해석이 용이한 규칙이 도출됩니다.
* Decision Tree 사용의 단점:
  - 노이즈와 데이터의 분포에 민감합니다.
  - logistic regression은 사선으로 표현될 수 있지만 decision tree는 비선형 관계를 표현하기 위해 나무의 깊이가 깊어져야 하므로 비효율적일 수 있습니다. (overfitting의 가능성이 있기 때문에 최대 깊이를 사전에 지정하므로써 일종의 regularization이 가능)
  - 문서 분류의 핵심은 특정 단어가 등장 했는가이기 때문에 decision tree가 추구 방향과 다릅니다.

## 7. Ensemble tree
- bagging (Bootstrap AGGregatING) 과 boosting이 있습니다.

#### Bagging
- 각 모델들이 과적합 되어있을 때 학습 성능이 뛰어날 경우 노이즈까지도 패턴으로 학습되지만, 이들 예측값의 평균은 과적합이 적을 가능성이 있다는 점에 주목합니다.
- 여러 개의 모델을 겹겹이 쌓아 정밀한 decision 단면을 학습합니다. 
- 변별력이 높은 변수에만 집중될 때에는 해당 변수를 제외하고 모델마다 서로 다른 변수를 이용하도록 한뒤 종합할 수 있습니다. (feature bagging)
* out of bag error
- Random forest에서 변수의 중요도를 측정하는 방법으로, 변수 하나를 망치고 perturbation (에러 평균)을 취해서 예측값이 흔들리지 확인합니다.

#### Boosting
- 과적합된 모델을 종합하여 대체적인, 안정적 예측을 하고 residual(나머지 부분)을 해결할 수 있는 모델을 하나씩 더해가는 방법입니다.
𝑦 = 𝑓1 𝑥 + 𝑒1
𝑒1 = 𝑓2 𝑥 + 𝑒2
𝑒2 = 𝑓3 𝑥 + 𝑒3
-> 𝑦 = 𝑓1 𝑥 + 𝑓2 𝑥 + 𝑓3 𝑥 + 𝑒3

* Adaptive boosting
각 모델마다 학습 데이터의 중요도를 다르게 정의하여 모든 데이터를 학습에 이용하되 몇몇 데이터 포인트들의 loss를 더 중요하게 취급하여 해당 점들을 잘 맞추는 모델을 학습하고, 이를 합쳐 최종 예측을 합니다.

* XGBoost
데이터 가중치를 계산하는 방식에 gradient를 이용하는 gradient boosting의 계산 과정을 발전시킨 모델입니다. XGBoost에서 사용하는 모델은 leaf node에 클래스 가중치를 부가하는 의사결정나무입니다.

* LightGBM
XGBoost를 더 경량, 고도화한 gradient boosting입니다. 
매 base model 마다 gradient 기반으로 학습데이터를 샘플링합니다.
Exclusive Feature Bundling 이라는 방법을 이용하여 features 의 개수도 줄여 효율적으로 모델을 학습합니다. 비슷한 정확도에 훨씬 빠른 학습 시간을 보입니다.

## 8. Evaluation measurement
1. precision
2. recall : 실제로 얼마나 positive로 예측되었는지에 대한 비율입니다. (recall 높이려면 전부 positive로 계산하면 되고, 동시에 precision이 내려감. precision과 역관계)
-> 둘을 같이 생각하는게 좋음
3. F-1 measure
4. Accuracy
<p>
<img src = 'https://user-images.githubusercontent.com/52257022/71985282-a6d4a800-326d-11ea-94ea-0e62d15458ad.png'>
</p>

## Summary ☺
- Classifiers 는 input 𝑥 를 이용하여 label 𝑦 를 분류하는 패턴을 학습합니다.
- 벡터 공간 𝑥 에 선형 (linear) 경계선을 그어 레이블을 판별합니다.
- 선형으로 분류가 어렵다면 input 을 분류가 쉬운 (linear separable) 공간의 벡터로
변형합니다. 그 결과 원 공간에서는 비선형의 경계선이 학습됩니다.
- 텍스트 마이닝 문제 중에는 선형 모형으로 풀 수 있는 경우가 많습니다.
- 문서 분류에 가장 중요한 정보는 '어떤 단어 (or n-gram) 이 존재하는가' 입니다.
- Bigram + Logistic regression / Naïve Bayes 는 문서 분류에 있어 기본이 되는 모델입니다.
