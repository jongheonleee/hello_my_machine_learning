#  Hello My First Machine Learning 🤖

## 📌 00. 전체 틀 파악 

#### 🧑🏻‍🏫 Machine Learning Algorithms(머신러닝 주요 알고리즘)!

<img src="https://github.com/user-attachments/assets/04812333-c296-427f-928a-707e0c5092ea" width="800" height="500"/>


<br>
<br>

## 📌 01. Machine Learning 주요 용어 정리 

> - 인공지능(artificial intelligence) : 학습하고 추론할 수 있는 지능을 가진 컴퓨터 시스템을 만드는 기술
> - 강인공지능 : 인공일반지능이라고도 하고 사람의 지능과 유사함
> - 약인공지능 : 특정 분야에서 사람을 돕는 보조 AI(자율주행, 음성 비서)
> - #### 머신러닝 : 데이터에서 규칙을 학습하는 알고리즘을 연구하는 분야
> - #### 딥러닝 : 인공신경망을 기반으로 한 머신러닝 분야를 일컬음
> - 코랩과 노트북 : 웹 브라우저에서 텍스트와 프로그램 코드를 자유롭게 작성 할 수 있는 온라인 에디터로 이를 코랩 노트북 또는 노트북이라 부름
> - #### 이진 분류(binary classification) : 머신러닝에서 여러개의 종류(혹은 클래스) 중 하나를 구별해 내는 문제를 분류(classification)라고 부르며 2개의 종류(클래스) 중 하나를 고르는 문제를 '이진 분류'라고함
> - #### 특성(feature) : 데이터를 표현하는 특징. DB 테이블에서 칼럼을 의미함
> - 맷플롯립(matplotlib) : 파이썬에서 과학 계산용 그래프를 그리는 대표 패키지
> - #### k-최근접 이웃 알고리즘(k-Nearest Neighbors Algorithm, KNN) : 인접한 샘플을 기반으로 예측을 수행함. 해당 샘플의 주위에 어떤 클래스가 있는지, 또 그 중에서 다수결의 원칙으로 가장 많은 클래스를 판단하여 해당 샘플의 클래스를 그 클래스로 예측함
> - #### 훈련(training) : 머신러닝 알고리즘이 데이터에서 규칙을 찾는 과정 또는 모델에 데이터를 전달하여 규칙을 학습하는 과정

<br>
<br>

 
> - #### 지도 학습(supervised learning) : 지도 학습은 입력(데이터)과 타깃(정답)으로 이뤄진 훈련 데이터가 필요하며 새로운 데이터를 예측하는 데 활용함.
> - #### 비지도 학습(unsupervised learning) : 타깃 데이터 없이 입력 데이터만 있을 때 사용. 데이터 속에서 규칙과 패턴을 파악해내는 용도로 사용함
> - #### 훈련 데이터(training data) : 지도 학습의 경우 필요한 입력(데이터)과 타깃(정답)을 합쳐 놓은 것
> - 훈련 세트와 테스트 세트 : 모델을 훈련할 때는 훈련 세트를 사용하고 평가는 테스트 세트로 함. 테스트 세트는 전체 데이터에서 20~30%
> - 샘플링 편향(sampling bias) : 훈련 세트와 테스트 세트에 샘플이 고르게 섞여 있지 않을 때 샘플링 편향이 발생.
> - 넘파이 : 파이썬의 대표적인 배열 라이브러리로 고차원의 배열을 손쉽게 만들고 조작할 수 있는 간편한 도구를 많이 제공함. 공식 명칭은 NumPy
> - 배열 인덱싱(array indexing) : 넘파이 기능으로 여러 개의 인덱스로 한 번에 여러 개의 원소를 선택할 수 있음
> - #### 데이터 전처리(data preprocessing) : 머신러닝 모델에 훈련 데이터를 주입하기 전 가공하는 단계로 특성값을 일정한 기준으로 맞추어 주는 작업. 데이터를 표현하는 기준이 다르면 알고리즘을 올바르게 예측할 수 없음. 데이터에서 사용할 샘플의 특성들의 스케일의 크기가 다를 경우 그래프의 정확도는 감소함. 데이터 전처리를 통해 좀 더 표준화된 그래프를 그릴 수 있음
> - 브로드캐스팅(broadcasting) : 조건을 만족하면 모양이 다른 배열 간의 연산을 가능하게 해 주는 기능  


<br>
<br>

> - #### 회귀(regression) : 클래스 중 하나로 분류하는 것이 아닌 임의의 어떤 숫자(값)를 예측하는 문제
> - #### k-최근접 이웃 분류 vs k-최근접 이웃 회귀
>   - <img src="https://github.com/user-attachments/assets/f2326ec6-c311-4a0c-95c6-4fc7621883e6" width="400" height="400"/>
> - #### 결정계수(R^2, coefficient of determination) : 회귀 모델에서 예측의 적합도를 0과 1 사이의 값으로 계산하는 것으로 1에 가까울수록 완벽함
> - #### 과대적합(overfitting) : 모델의 훈련 세트 점수가 테스트 세트 점수보다 훨씬 높은 경우를 의미함
> - #### 과소적합(underfitting) : 모델의 훈련 세트와 테스트 세트 점수가 모두 동일하게 낮거나 테스트 세트 성능이 오히려 더 높은 경우를 의미함
> - #### 선형회귀(linear regression) : 특성(feature)이 하나인 경우 어떤 직선을 학습하는 알고리즘
> - #### 가중치(weight) : 선형 회귀가 학습한 직선의 기울기를 의미함
> - #### 다항회귀(polynomial regression) : 다항식을 사용하여 특성과 타깃 사이의 관계를 나타낸 선형 회귀. a*x^2 + b*y + c -> 곡선을 나타냄
> - #### 다중회귀(multiple regression) : 여러 개의 특성을 사용한 회귀. y=a*x0 + b*x1 + c*x2 + ... + k*xn -> 평면을 나타냄
> - 변환기(transformer) : 특성을 만들거나 전처리하는 사이킷런의 클래스로 타깃 데이터 없이 입력 데이터를 변환함
> - #### 릿지 회소(ridge regression) : 규제가 있는 선형 회귀 모델 중 하나로 모델 객체를 만들 때 alpha 매개변수로 규제의 강도를 조절함. alpha 값이 크면 규제 강도가 세지므로 계수 값이 줄고 조금 더 과소적합되도록 유도하여 과대적합을 완화시킴
> - #### 하이퍼파라미터(hyperparameter) : 머신러닝 모델이 학습할 수 없고 사람이 지정하는 파라미터
> - #### 라쏘 회소(lasso regression) : 또 다른 규제가 있는 선형 회귀 모델로 alpha 매개변수로 규제의 강도를 조절함. 릿지와 달리 계수 값을 아예 0으로 만들 수 있음


<br>
<br>

> - 다중 분류(multi-class classification) : 타깃 데이터에 2개 이상의 클래스가 포함된 문제
> - #### 로지스틱 회귀(logistic regression) : 선형 방정식을 사용한 분류 알고리즘으로 선형 회귀와 달리 시그모이드 함수나 소프트맥스 함수를 사용하여 클래스 확률을 출력
> - #### 시그모이드 함수(sigmoid function) : 시그모이드 함수 또는 로지스틱 함수(logisitc regression)라고 부르며 선형 방정식의 출력을 0과 1 사이의 값으로 압축하여 이진 분류를 위해 사용, 이진 분류일 경우 시그모이드 함수의 출력이 0.5보다 크면 양성 클래스, 0.5보다 작으면 음성 클래스로 판단
> - 불리언 인덱싱(boolean indexing) : 넘파이 배열은 True, False 값을 전달하여 행을 선택할 수 있으며 이를 불리언 인덱싱이라고함
> - #### 소프트맥스 함수(softmax function) : 여러 개의 선형 방정식의 출력값을 0~1 사이로 압축하고 전체 합이 1이 되도록 만들며 이를 위해 지수 함수를 사용하기 때문에 정규화된 지수 함수라고함
> - #### 확률적 경사 하강법(Stochastic Gradient Descent) : 훈련 세트에서 랜덤하게 하나의 샘플을 선택하여 손실 함수의 경사를 따라 최적의 모델을 찾는 알고리즘
> - #### 애포크(epoch) : 확률적 경사 하강법에서 훈련 세트를 한 번 모두 사용하는 과정
> - 미니배치 경사 하강법(minibatch gradient descent) : 1개가 아닌 여러 개의 샘플을 사용해 경사 하강법을 수행하는 방법으로 실전에서 많이 사용
> - 배치 경사 하강법(batch gradient descent) : 한 번에 전체 샘플을 사용하는 방법으로 전체 데이터를 사용하므로 가장 안정적인 방법이지만 그만큼 컴퓨터 자원을 많이 사용함. 또한, 어떤 경우는 데이터가 너무 많아 한 번에 전체 데이터를 모두 처리할 수 없을지도 모름
> - #### 손실 함수(loss function) : 어떤 문제에서 머신러닝 알고리즘이 얼마나 엉터리인지 측정하는 기준
> - #### 로지스틱 손실 함수(logistic loss function) : 양성 클래스(타깃 = 1)일 때 손실은 -log(예측 확률)로 계산하며, 1 확률이 1에서 멀어질수록 손실은 아주 큰 양수가 됨. 음성 클래스(타깃 = 0)일 때 손실은 -log(1-예측 확률)로 계산함. 이 예측 확률이 0에서 멀어질수록 손실을 아주 큰 양수가됨
> - 크로스엔트로피 손실 함수(cross-entropy loss function) : 다중 분류에서 사용하는 손실 함수
> - 힌지 손실(hinge loss) : 서포트 벡터 머신(SVM)이라 불리는 또 다른 머신러닝 알고리즘을 위한 손실 함수로 널리 사용하는 머신러닝 알고리즘 중 하나. SGDClassifier가 여러 종류의 손실 함수를 loss 매개변수에 지정하여 다양한 머신러닝 알고리즘을 지원함
 
<br>
<br>

> - #### 결정 트리(Decision Tree) : 질문을 하나씩 던져 정답을 맞춰가며 학습하는 알고리즘. 비교적 예측 과정을 이해하기 쉬움
> - #### 검증 세트(validation set) : 하이퍼파라미터 튜닝을 위해 모델을 평가할 때, 테스트 세트를 사용하지 않기 위해 훈련 세트에서 다시 떼어 낸 데이터 세트
> - #### 교차 검증(cross validation) : 훈련 세트를 여러 폴드로 나눈 다음 한 폴드가 검증 세트의 역할을 하고 나머지 폴드에서 모델을 훈련함
>  - 이렇게 모든 폴드에 대해 검증 점수를 얻어 평균하는 방법으로 교차 검증을 이용하면 검증 점수가 안정적이며, 훈련에 더 많은 데이터를 사용할 수 있음
> - 그리드 서치(Grid Search) : 하이퍼파라미터 탐색을 자동화해 주는 도구
> - 랜덤 서치(Random Search) : 랜덤 서치는 연속적인 매개변수 값을 탐색할 때 유용함
> - 정형 데이터 vs 비정형 데이터(structured data vs unstructured data) : 특정 구조로 이루어진 데이터를 정형 데이터라고 하고, 반면 정형화되기 어려운 사진이나 음악 등을 비정형 데이터라고함
> - #### 앙상블 학습(ensemble learning) : 여러 알고리즘(예, 결정 트리)을 합쳐서 성능을 높이는 머신러닝 기법
> - #### 랜덤 포레스트(Random Forest) : 대표적인 결정 트리 기반의 앙상블 학습 방법, 안정적인 성능 덕분에 널리 사용됨. 부트스트랩 샘플을 사용하고 랜덤하게 일부 특성을 선택하여 트리를 만드는 것이 특징임
> - 부트스트랩 샘플(bootstrap sample) : 데이터 세트에서 중복을 혀용하여 데이터를 샘플링하는 방식
> - 엑스트라 트리(extea trees) : 랜덤 포레스트와 비슷하게 동작하며 결정 트리를 사용하여 앙상블 모델을 만들지만, 부트스트랩 샘플을 사용하지 않는 대신 랜덤하게 노드를 분할하여 과대적합을 감소시킴
> - 그레이디언트 부스팅(gradient boosting) : 깊이가 얕은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블 하는 방법. 깊이가 얕은 결정 트리를 사용하기 때문에 과대적합에 강하고 일반적으로 높은 일반화 성능을 기대할 수 있음
> - 히스토그램 기반 그레이디언트 부스팅(Histogram-based Gradient Boosting) : 그레이디언트 부스팅의 속도를 개선한 것으로 과대적합을 잘 억제하며 그레이디언트 부스팅보다 조금 더 높은 성능을 제공. 안정적인 결과와 높은 성능으로 매우 인기가 높음 
 
<br>
<br>

> - #### 히스토그램(histogram) : 값이 발생한 빈도를 그래프로 표시한 것으로 보통 x축이 값의 구간(계급)이고, y축은 발생 빈도(도수) 임
> - #### 군집(clutering) : 비슷한 샘플끼리 그룹으로 모으는 작업으로 대표적인 비지도 학습 작업 중 하나
> - #### k-평균 알고리즘(k-means algorithm) : 처음에 랜덤하게 클러스터 중심을 정하여 클러스터를 만들고 그다음 클러스터의 중심을 이동하여 다시 클러스터를 결정하는 식으로 반복해서 최적의 클러스터를 구성하는 알고리즘
> - 이너셔(inertia) : k-평균 알고리즘은 클러스터 중심과 클러스터에 속한 샘플 사이의 거리를 잴 수 있는데 이 거리의 제곱 합을 이너셔라고 함. 즉 클러스터의 샘플이 얼마나 가깝게 있는지를 나타내는 값임
> - #### 차원 축소(dimensionality reduction) : 데이터를 가장 잘 나타내는 일부 특성을 선택하여 데이터 크기를 줄이고 지도 학습 모델의 성능을 향상시킬 수 있는 방법
> - #### 주성분 분석(principal component analysis, PCA) : 차원 축소 알고리즘의 하나로 데이터에서 가장 분산이 큰 방향을 찾는 방법이며 이런 방향을 주성분이라 함. 원본 데이터를 주성분에 투영하여 새로운 특성을 만들 수 있음
 
<br>
<br>

> - #### 인공신경말(artificial neural network, ANN) : 생물학적 뉴런에서 영감을 받아 만든 머신러닝 알고리즘, 신경망은 기존의 머신러닝 알고리즘으로 다루기 어려웠더 이미지, 음성, 텍스트 분야에서 뛰어난 성능을 발휘하면서 크게 주목을 받고 있으며 종종 딥러닝이라고도 부름
> - #### 딥러닝(deep learning) : 딥러닝은 인공신경망과 거의 동의어로 사용되는 경우가 많으며 혹은 심층 신경망을 딥러닝이라고 부름. 심층 신경망은 여러 개의 층을 가진 인공신경망임
> - #### 텐서플로(TensorFlow) : 구글이 만든 딥러닝 라이브러리로 CPU와 GPU를 사용해 인공신경망 모델을 효율적으로 훈련하며 모델 구축과 서비스에 필요한 다양한 도구를 제공함. 텐서플로 2.0부터는 신경망 모델을 빠르게 구성할 수 있는 케라스를 핵심 API로 채택. 케라스를 사용하면 간단한 모델에서 아주 복잡한 모델까지 손쉽게 만들 수 있음
> - #### 활성화 함수(activation function) : 소프트맥스와 같이 뉴런의 선형 방정식 계산 결과에 적용되는 함수
> - #### 원-핫 인코딩(one-hot encoding) : 타깃값을 해당 클래스만 1이고 나머지는 모두 0인 배열로 만드는 것. 다중 분류에서 크로스 엔트로피 손실 함수를 사용하려면 0, 1, 2와 같이 정수로 된 타깃값을 원-핫 인코딩으로 변환해야 함
> - #### 은닉층(hidden layer) : 입력과 출력층 사이에 있는 모든 층을 은닉층이라고 부름
> - #### 심층 신경망(deep neural network, DNN) : 2개 이상의 층을 포함한 신경망으로 종종 다층 인공 신경망, 심층 신경망, 딥러닝을 같은 의미로 사용함
> - #### 렐루 함수(ReLU function) : 입력이 양수일 경우 마치 활성화 함수가 없는 것처럼 그냥 입력을 통과시키고 음수일 경우에는 0으로 만드는 함수
> - #### 옵티마이저(optimizer) : 신경망의 가중치와 절편을 학습하기 위한 알고리즘 또는 방법. 케라스에는 다양한 경사 하강법 알고리즘이 구현되어 있으며 대표적으로 SGD, 네스테로프 모멘텀, RMSprop, Adam 등이 있음
> - 적응적 학습률(adaptive learning rate) : 모델이 최적점에 가까이 갈수록 안정적으로 수렴하도록 학습률을 낮추도록 조정하는 방법. 이런 방식들은 학습률 매개변수를 튜닝하는 수고를 덜 수 있는 것이 장점
> - 드랍아웃(dropout) : 훈련 과정에서 층에 있는 일부 뉴런을 랜덤하게 꺼서(즉 뉴런의 출력을 0으로 만들어) 과대적합을 막음
> - 콜백(callback) : 케라스에서 훈련 과정 중간에 어떤 작업을 수행할 수 있게 하는 객체로 keras, callbacks 패키지 아래에 있는 클래스를 fit()의 callbacks 매개변수에 리스트로 전달하여 사용

 
<br>
<br>


## 📌 02. Machine Learning 주요 학습 내용 정리 


<br>
<br>

- [머신러닝 알고리즘]
- <img src="https://github.com/user-attachments/assets/ee80dcfd-420e-4bd8-b169-e2e123e63155" width="500" height="500"/>
- <img src="https://github.com/user-attachments/assets/d0de570d-23de-4e1d-b720-8574b7384230" width="500" height="500"/>
- <img src="https://github.com/user-attachments/assets/496bef19-1b7e-4e8f-a7ed-96cfec29ee8d" width="500" height="500"/>


<br>
<br>


### 📖 머신러닝 학습 요약

<br>

> ### 👉 머신러닝? '데이터에서 규칙, 패턴을 도출하는 알고리즘을 통해 특정 작업을 처리하는 것'

<br>

- 머신러닝은 크게 3가지 종류로 구분
  - [머신러닝 세 가지 학습]
  - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/4f60e285-be54-4f26-9a10-df748bf17015" width="500" height="500"/>
  - (1) 지도 학습 -> 정답이 있는 것
    - 정답이 있는 데이터. 즉, 레이블된 데이터(타깃)를 활용
    - 피드백이 빠름
    - 예측, 생성 용이
  - (2) 비지도 학습 -> 정답이 없는 것
    - 정답이 없는 데이터, 레이블 및 타깃이 없음
    - 피드백 없음(정답 x)
    - 데이터에서 패턴 찾기
  - (3) 강화 학습 -> 보상 시스템으로 의사결정 학습
    - 의사결정 학습
    - 보상시스템 활용, 좋은 결정하도록 유도


> ### 👉 표기법과 규칙, 익숙해지도록 많이 보기

<br>

- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/2bbfa557-5c72-42eb-b978-d36e67294929" width="500" height="500"/>
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/49561765-4627-447b-9032-cccb916cad95" width="500" height="500"/>
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/a51f13b5-ebf0-48ee-88d8-793aade84618" width="500" height="500"/>
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/65797c78-5792-4708-97f0-0bc5c4c2f4f2" width="500" height="500"/>

<br>

> ### 👉 머신 러닝 전반적인 작업 흐름

<br>

- [머신러닝 작업 흐름]
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/6ce9ea68-e56a-446e-8a44-2dc87c5e0873" width="500" height="500"/>

- 전처리
  - 스케일 : 일관성 유지. 예를들어, cm, inch, m^2 ... -> 'cm'로 통일시키는 것. 이 작업을 스케일 맞춘다라고 표현
  - 차원 축소 : 특성 개수를 줄임. 예를들어, 3차원 -> 2차원
  - 훈련용/테스트용 데이터 분리

 - 예측 모델 훈련과 선택
   - 대표적인 분류 지표 : 정확도
     - 정확도 : 정확히 분류된 샘플 비율
   - 파라미터 : 사람이 설정하는 값. 모델 성능과 직결됨.
     - 모델 성능을 상세하게 조정하기 위해 하이퍼파라미터 기법 활용
   - 교차 검증 : 좋은 모델 판단 과정
   

<br>

### 머신 러닝 교과서에 나온 주요 용어

OvA : 이진 분류를 중첩해서 사용 -> 다중 분류 처리 

<br>


### 머신 러닝 교과서에 나온 알고리즘

- 1. 기초 선형 분류기
  - 퍼셉트론 
  - 아달린(선형 활성화 함수, 경사하강법)
 
 
- 2. 기본 분류기 알고리즘
  - 로지스틱 회귀 : 시그모이드 함수
  - SVM : 마진 최대화
  - 결정 트리 : 소수의 질문, 카테고리 분류

### 주요 알고리즘 그림

- [퍼셉트론]
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/2acd368f-b760-4fd2-97ff-9b9610e341bc" width="500" height="500"/>
- 가중치 자동 업데이트, 성능 향상, 예측 
- 퍼셉트론 규칙
  - 자동으로 최적의 가중치를 학습하는 알고리즘 제안
  - 이 가중치는 뉴런의 출력 신호를 낼지 말지를 결정하기 위해 입력 특성에 곱하는 계수임
  - 즉, 해당 알고리즘을 사용하여 새로운 데이터 포인터가 한 클래스에 속하는지 아닌지 예측 가능


<br>

- [아달린]
- <img src="https://github.com/user-attachments/assets/e20c7da1-4d2a-4930-b613-27ea753a6e1b" width="500" height="500"/>

<br>
  
- [로지스틱 회귀]
- <img src="https://github.com/user-attachments/assets/ce1e75a9-9a8e-4ccf-a621-0d2fe46fce4f" width="500" height="500"/>

- [서포트 벡터 머신(SVM)]
- <img src="https://github.com/user-attachments/assets/c70f2a51-396f-4954-bd75-651ec61fe234" width="500" height="500"/>
- <img src="https://github.com/user-attachments/assets/e0b68d4b-6252-4bc3-a93b-40ad5fce5d65" width="500" height="500"/>


- [커널 SVM]
- <img src="https://github.com/user-attachments/assets/ab4d718d-1c94-4eb9-8aff-1a4facb7c8f8" width="500" height="500"/>

- [결정 트리]
- <img src="https://github.com/user-attachments/assets/58192f0e-15da-4965-b9e0-91efe1fa2a47" width="500" height="500"/>
- <img src="https://github.com/user-attachments/assets/c9d1645f-a940-4e4c-9aeb-e7434f77c208" width="500" height="500"/>
- 어떤 항목에 대한 관측값과 목표값을 연결시켜주는 예측 모델로서 결정 트리를 사용
- 특성 공간을 사각 격자로 나누기 때문에 복잡한 결정 경계를 만들 수 있음
- 깊어질 수록 과대적합 우려 -> 가지치기


- [랜덤 포레스트(앙상블)]
- <img src="https://github.com/user-attachments/assets/84e644c2-f32b-4972-a332-eabe10151887" width="500" height="500"/>
- 앙상블? n개를 표준내서 사용하는 것 -> 과대적합 방지 일반화 시킴
- 랜덤 포레스트는 결정 트리의 앙상블임. n개의 (깊은) 결정 트리를 평균내는 것
- 개개의 트리는 분산이 높은 문제가 있지만, 앙상블은 견고한 모델을 만들어 일반화 성능을 높이고 과대적합의 위험을 줄임
- 랜덤 포레스트의 학습 과정 
  - (1) n개의 랜덤한 부트스트랩 샘플을 받음(중복 허용, m개 샘플 선택)
  - (2) 부트스트랩 샘플에서 결정트리 학습
      - 중복 x, 랜덤하게 d개 선택
      - 목적 함수를 기준으로 최선의 분할을 만드는 특성을 사용해서 노드 분할
  - (3) (1) ~ (2) k 번 반복
  - (4) 각 트리의 예측을 모아 다수결 투표로 클래스 레이블 할당



- [KNN]
- <img src="https://github.com/user-attachments/assets/2712a99b-16e0-4fcc-adc7-a4e003243145" width="500" height="500"/>
- 분류하려는 포인트와 가장 근접한 샘플 k개 찾음, 새로운 데이터 포인트의 클래스 레이블은 이 k개의 최근접 이웃에서 다수결 투표로 결정
- 데이터에서 판별 함수를 학습 시키는 대신 훈련 데이터 셋을 메모리에 저장함
- 적절한 k개를 찾는 것이 중요함, 과대적합/과소적합의 올바른 균형을 잡기 위해 중요함
- KNN 학습 과정
  - (1) 숫자 k와 거리 측정 기준을 선택
  - (2) 분류하려는 샘플에서 k개의 최근접 이웃을 찾음
  - (3) 다수결투표를 통해 클래스 레이블 할당

- 모수 모델
  - 새로운 데이터 포인트를 분류할 수 있는 함수를 학습하기 위함
  - 훈련 데이터 셋에서 모델 파라미터를 추정함

- 비모수 모델
  - 훈련 데이터가 늘어남에 따라 파라미터 개수도 늘어남(랜덤 포레스트, 앙상블, KNN)

- 차원의 저주
  - 고정된 크기의 훈련 데이터셋이 차원이 늘어남에 따라 특성 공간이 점점 희소해지는 현상 

<br>

- 순차 특성 선택 알고리즘
  - 탐욕적 탐색 알고리즘 기반
  - 초기 d 차원의 특성 공간을 k < d 인 k 차원의 특성 부분 공간으로 축소함
  - 주어진 문제에 가장 관련이 높은 특성 부분 집합을 자동으로 선택

- SBS(순차 후진 선택)
  - 새로운 특성의 부분 공간이 목표하는 특성 개수가 될 때까지 전체 특성에서 순차적으로 특성을 제거함
    - 각 단계에서 제거했을 때 성능 손실이 최소가 되는 특성 제거
  - 모델 성능을 가능한 적게 희생, 초기 특성의 부분 공간으로 차원 축소, 과대적합의 문제 해결
  - 학습 과정
    - (1) 알고리즘을 k = d로 초기화 (d = Xd)
    - (2) 조건 x- = argmaxj(Xm-x)를 최대화 하는 특성 x- 결정
    - (3) 특성 집합에서 x-를 제거
    - (4) k가 목표하는 특성 개수가 되면 종료

- [PCA]
- <img src="https://github.com/user-attachments/assets/1e491b81-73d6-4b1b-9908-ec487a10a514" width="500" height="500"/>

- [LDA]
- <img src="https://github.com/user-attachments/assets/06aee3ae-746c-4fef-bb25-1e3178c980d5" width="500" height="500"/>
- <img src="https://github.com/user-attachments/assets/a2cba6cf-eb16-442e-8c36-048b557ec038" width="500" height="500"/>

- [KPCA]
- <img src="https://github.com/user-attachments/assets/d4d25b6f-8c60-44c3-9676-fed50107eec5" width="500" height="500"/>

- 툭성 추출을 위한 세 개의 기본적인 차원 축소 기법
  - (1) PCA : 비지도학습, 직교하는 특성 축을 따라 분산이 최대가 되는 저차원 부분 공간으로 데이터를 투영함
  - (2) LDA : 지도학습, 훈련 데이터셋에 있는 클래스 정보를 사용하여 선형 특성 공간에서 클래스 구분 능력을 최대화함
  - (3) KPCA : 커널 트릭과 고차원 특성 공간으로의 가상 투영을 통하여 비선형 특성을 가진 데이터셋을 저차원 부분 공간으로 극적으로 압축함 
 
### 📋 몰입 리스트
