# 🤖 machine_learning

<br>

### 몰입할 때 참고했던 서적과 교육
> ### 📚 참고 서적 및 사이트
> ### 1. 머신러닝 교과서
> 도서 링크 : https://product.kyobobook.co.kr/detail/S000001834604
> ### 2. 남궁성의 백엔드 데브 캠프 1기(AI 융합)


<br>

### ⭐️ 목표
> ### 추후에 AI 프로젝트를 위함, 그리고 언젠가는 AI 쪽으로 커리어 전환이 일어날거임
> ### 시대에 흐름에 맞춰 가기 위함

<br>
<br>

### 🫵🏻 머신러닝의 핵심 
> ### 주요 알고리즘이 어떤 상황에 쓰이는지

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

### 📋 목차
> ### 📌 1. 컴퓨터는 데이터에서 배운다
> ### 📌 2. 간단한 분류 알고리즘 훈련

<br>

### 📌 1. 컴퓨터는 데이터에서 배운다

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
