텍스트 분석을 위한 머신러닝 week2
==============

2주차 수업에서는 키워드 추출을 위한 방법들인 PMI와 n-gram extraction, 그리고 여러가지 clustering에 대해 배습니다.

# # Keyword Extraction
## What IS a keyword?
정량적 평가, 기준, 상황/문맥 모두 모호하므로 내 task에서 keyword의 목적이 무엇인지 먼저 정의하는 것이 중요합니다.
- saliency (coverage) : 빈도 기준, 한 집합을 대표하는 키워드는 집합의 문서에 자주 등장할 것이라는 전제
- distinctiveness(discriminative power) : 분별, 한 키워드를 통해 다른 집합과 구분할 수 있다고 전제
*이 둘은 역관계에 가까우며, 상황에 따라 적절히 사용하는 것이 중요합니다.

## 1. Keyword with Proportion ration

### 상대적 출현을 이용한 키워드 추출
- 한 단어를 기준으로 *관심있는 문서 집합* 에서의 단어 등장 비율과 *비교대상 문서 집합* 에서의 등장 비율을 이용해 키워드 점수를 정의합니다.
<p>
 <img src = 'https://user-images.githubusercontent.com/52257022/71542505-976e7880-29aa-11ea-85a6-9dd04901a5dd.png'>

- 여기서 제안된 점수는 (0,1) 사이의 keyword score를 가집니다. 0.5 정도의 score를 가질 때에는 키워드가 아니라고 판단할 수 있는데, 그 이유는 두 개의 문서가 뒤바뀌어도 영향이 없기 때문입니다. score가 0.5~1 사이 될수록 키워드로서의 영향력이 높다고 할 수 있습니다.


### PMI (point mutual information)
키워드간 상관성을 표현하는 index입니다.
<p>
  <img src = 'https://patentimages.storage.googleapis.com/07/b8/a8/3a125e705b079e/PCTKR2015012704-appb-M000001.png'>

- 서로 독립인지 판단할 수 있으며, 
- 로그를 사용하여 양/음 상관성을 판단합니다
- 하지만 infrequent 한 것에 예민하여 이를 보완하기 위해 smoothing PMI나 textrank를 사용하기도 합니다.

+ nlp에서 semantic 정보는 전부 co-occurence에서 파생됨, 핵심정보

## 2. n-gram extraction
- bi-gram 까지 주로 많이 쓰며 bi-gram 만으로도 문맥을 표현하기에 충분합니다.
(읽어보기: Wang, S. and Manning, C. D. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. In Proceedings of the 50th Annual Meeting of the Association for Computational 31 Linguistics)
- n-gram 은 세 개 이상의 단어 조합을 하나의 단위로 취급하며, 추출 방법이 다양한 만큼 계산 과정에서 많은 메모리를 필요로 합니다.
	- n-gram에 대해 빈도수를 계산하는 방법도 있지만 조사로 시작하거나 원하지 않는 단어들이 잡힐 수 있습니다.
	- 이를 방지하기 위해 noun으로만 시작하게 설정하거나, Rpart로 시작하는 것으로 설정하면 해결할 수 있습니다.

** Tfidf vectorizer 사용은 지양하는 것이 좋습니다!

## 3. Document clustering

### 1) clustering
- 데이터에서 비슷한 객체들을 하나의 그룹으로 묶습니다
- 각 객체들이 어떤 군집으로 할당되어야 하는지에 대한 정답 정보가 없기 때문에 unsupervised 알고리즘으로 봅니다
- 각 객체들의 유사도 (객체간 거리) 정보를 이용하여 군집화 합니다
- 군집 내 객체들은 서로 비슷하며 군집 간의 객체들은 서로 다르다는 점에 주목합니다 

### 2) k-means
- 유사도 : n개의 데이터 X에 대하여 두 데이터간 정의되는 임의의 거리입니다
- k : 그룹의 개수, 각 그룹을 centriod vector로 표현한 뒤 이를 업데이트하는 과정을 거칩니다
- 문서 군집화에서는 거리 척도가 중요하며, 문서간 유사도에서는 공통된 단어의 유무가 가장 중요한 정보입니다. (euclidean 거리는 이 정보를 고려하지 않기 때문에 사용을 지양해야 합니다)

#### k-means 단점 & solutions :
1. 군집의 모양이 centroid를 중심으로 한 구형임을 가정합니다
- k-means ensemble로 해결 가능합니다
2. initial point에 따라 군집의 모양이 달라질 수 있습니다
- 특정 지역에 initial point가 몰려있지 않다면 큰 문제가 되지 않으며, 저차원 데이터에 적합한 k-means ++ 사용 또한 가능합니다. (고차원에서 '가깝다'는 유의미하지만, '멀다'는 의미를 가지지 않습니다)
- 또한 term frequency representation의 특징을 이용하면 널리 퍼진 initial point를 빠르게 찾을 수 있습니다.
3. 적절한 군집의 개수를 사용자가 직접 정의해야 합니다
- 예상되는 군집의 개수보다 크게 k의 개수를 설정한 뒤, 후처리로 비슷한 군집을 병합할 수 있습니다.
4. 노이즈 데이터에 민감한 경우가 발생합니다
- 모든 점을 반드시 군집으로 assign 하기 때문에 일단 가장 가까운 군집에 할당되어 centriod를 크게 움직입니다.
- 따라서 데이터의 노이즈 (텍스트 데이터의 경우 길이가 극단적으로 길거나 짧은 문서 등) 를 미리 처리하면 해결 가능합니다.

* More about k-means
- k-means 는 heuristic한 방법이며 너무 여러번 반복할 필요가 없습니다. (default max_iter가 300으로 설정되어 있는 경우, 20~30사이로 바꾸어 주어도 거의 수렴합니다.)
- 재차 강조되는 부분은, k-means 를 사용할 때에는 euclidean 이 아닌 cosine (유닛 벡터 간의 내적) 을 이용하는 것이 좋습니다. 
유클리디안은 한 백터가 0이어도 영향력을 가지게 되며, 공통의 단어를 묻는 태스크가 아니므로 사용하지 않는 것이 좋습니다.
- uniform effect : iteration 회수가 많아져 overfitting이 되는 경우, 각 군집의 볼륨이 달라서 나눠먹기 하려는 경향이 생깁니다. 작은 cluster가 점점 큰 cluster를 잠식해가는 경우를 볼 수 있습니다. (minor 한 군집을 처음에 잘 잡으려면 k를 크게 설정합니다)

#### GMM
- Gaussian을 이용하기 때문에 euclidean에서 정의되며, 따라서 문서에는 사용하지 않는 것이 좋습니다.
- 군집 사이에 밀도 차이가 있을 경우에 적합합니다.
- 구 형태의 군집만을 표현하는 k-means와는 달리 skewness를 학습하여 타원형 모양의 군집도 표현 가능합니다.

#### BGMM
- 군집의 구분이 잘 되는 데이터에서 잘 작동합니다 (bayesian 은 노이즈가 많으면 X)
- 문서 군집 외의 태스크에서 좋은 성능을 보입니다.

#### Hierarchical clustering
- 군집화 방식: 거리가 가장 가까운 두 집합을 하나로 묶으며 모든 집합이 하나가 될 때 까지 merging을 반복합니다.
- Outlier의 영향을 덜 받는다는 장점이 있습니다.
- 최대 계산 횟수가 n제곱이므로 많은 계산 공간과 시간을 필요로 합니다.

#### DBSCAN
- not suitable for document clustering
- 파라미터 변화에 매우 민감합니다.
	*HDBSCAN
- less sensitive, but still not suitable

## 4. Summary
* k-means는 centroid를 중심으로 구형의 군집을 만듦
	- euclidean 이용할 경우 구 형태의 군집
	- Cosine 이용할 경우, 벡터의 각도를 기준으로 만들어진 partition
* Hierarchical clustering, DBSCAN은 복잡한 모양의 데이터용
	- sparse vector + Cosine의 공간은 복잡하지 않음
	- 단순한 알고리즘이 빠르며 안정적
* 고차원 벡터에서는 매우 가까운 거리만 의미를 지님
	- kmeans 이용 시 k 가 지나치게 작을 경우 먼 문서들이 하나의 클러스터에 할당될 수 있기 때문에 불안정한 학습이 될 수 있음
	- 고차원 벡터의 경우 충분히 큰 k로 군집화를 수행한 뒤, 동일한 의미를 지니는 군집들을 하나로 묶는 후처리 방식을 추천
* 불필요하거나 여러가지 군집에 공통적으로 나타나는 단어들을 제거하는 것은 군집화 알고리즘에 도움이 됨
	- df가 지나치게 높거나 낮은 단어
	- 변별력이 없는 단어는 term document matrix에서 제거한 뒤 학습
* soynlp 패키지 사용법 몇 가지
- word extractor : 띄어쓰기 정보 막 없애버리면, "오늘의 날씨는" 에서 "의 날씨" 가 자주 쓰여 단어로서의 점수를 높게 가질 수 있다는 점을 주의해야 합니다.
- noun extraction : 긴 단어부터 모아서 삭제하고, 그 다음을 보는 원리로 작동합니다.
	ex) 아이디어 -> 아이디 -> 아이
