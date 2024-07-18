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

<br>

> ### 👉 (1) 지도 학습 => 레이블된 데이터 -> [머신러닝 알고리즘] -> 모델, 새로운 값 -> [모델] -> 예측 값 

<br>

- [지도 학습]
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/b1c73a1d-4a98-4467-844d-368f5d7a1b61" width="500" height="500"/>
- 주요 목적은 레이블된 데이터에서 모델을 학습함, 새로운 데이터로부터 예측을 만듦


<br>

> ### 👉 (1) 지도 학습 작업은 크게 2가지 -> 예측 & 분류 / 회귀

<br>

- 지도 학습은 크게 2가지로 구분
  - (1) 분류 : 클래스 레이블 예측 -> 데이터 분류
    - [분류]
    - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/12f78cf0-a0a9-47b1-a336-60380ca30092" width="500" height="500"/>
    
  - (2) 회귀 : 연속적인 출력 값 예측 -> 그래프 그려서 예측 
    - [회귀]
    - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/3869067d-0994-4a03-8573-692140e5210d" width="500" height="500"/>

<br>

> ### 👉 (2) 강화 학습 -> 보상 & 상태 => 의사결정

<br>

- [강화 학습]
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/70419b6f-c579-4e24-bba0-f06b86ac55f2" width="500" height="500"/>
- 강화 학습은 환경(sw 환경)과 상호 작용하여 시스템(에이전트) 성능 향상 시키는 것이 목적
  - 성능 향상 : 판단력 향상 
  - 보상 시스템 : 좋은 판단 -> 보상, 그렇지 않으면 안줌
  - 상태 : sw 환경 상태를 의미, 에이전트의 행동에 따라 sw 환경 상태는 변동됨
    - 보상 & 상태를 기반으로 에이전트는 특정 행동을 수행함 

<br>

> ### 👉 (3) 비지도 학습 -> 그룹핑/차원 축소

<br>

- 비지도학습은 크게 2 가지 형태의 작업 처리
  - (1) 군집 : 서브그룹 찾기
    - [군집]
    - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/a60d0032-6e5e-4967-acaf-e5a9627de69f" width="500" height="500"/>

  - (2) 차원 축소 : 데이터 압축
    - [차원 축소]
    - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/892a40c3-7e1d-4b11-b86c-d7c203a02a5b" width="500" height="500"/>

<br>

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
<br>

### 📌 2. 간단한 분류 알고리즘 훈련

### 2-1. 인공 뉴런 : 초기 머신 러닝의 간단한 역사

<br>

- [뇌의 신경 세포]
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/1a3d5daa-ec10-4c8e-a686-0e91f3f35ed1" width="500" height="500"/>
- 합쳐진 신호가 특정 임계 값을 넘으면 출력 신호가 생성되고 축삭 돌기를 이용하여 전달함
- 퍼셉트론 규칙
  - 자동으로 최적의 가중치를 학습하는 알고리즘 제안
  - 이 가중치는 뉴런의 출력 신호를 낼지 말지를 결정하기 위해 입력 특성에 곱하는 계수임
  - 즉, 해당 알고리즘을 사용하여 새로운 데이터 포인터가 한 클래스에 속하는지 아닌지 예측 가능

<br>

- [인공 뉴런의 수학적 정의]
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/e35c53dd-ef67-4afd-b690-33dcd805f4a5" width="500" height="500"/>
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/94b3483b-5c6d-40be-a06f-1f9f5113e2bc" width="500" height="500"/>
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/5236d1ce-e06b-4e57-97b6-633f7e26bcc0" width="500" height="500"/>
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/70f4c1c2-1974-45c4-b71b-160091cda0fd" width="500" height="500"/>

<br>

- 퍼셉트론 알고리즘 : 가중치 자동 업데이트, 성능 향상, 예측 
  - [퍼셉트론 알고리즘]
  - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/2acd368f-b760-4fd2-97ff-9b9610e341bc" width="500" height="500"/>

- 퍼셉트론 학습 규칙 -> 출력을 하거나 하지 않는 경우로만 나뉨
  - [퍼셉트론 학습 규칙]
  - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/9eaf2a13-bbfb-422e-ad4b-c5dde643745a" width="500" height="500"/>

  <br>
  
  - [수식 추가 및 상세 설명]
  - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/3efaa074-664a-4749-914a-8c7ace6c9223" width="500" height="500"/>
 
  <br>

  - [클래스 레이블 예측 성공]
  - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/dc8395bc-54c3-434b-a2d3-6144c0c89f81" width="500" height="500"/>
  - 타깃값 = 예측값 -> 가중치 변경 x

  <br>

  - [클래스 레이블 예측 실패]
  - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/d4f5ddac-d915-45ed-8284-8133ed5fbfce" width="500" height="500"/>
  - 타깃값 != 예측값 -> 가중치 변경 o

  <br>

  - [퍼셉트론 알고리즘 구현]
  - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/4eaa6629-4883-47c6-a027-b0b20637f346" width="500" height="500"/>
  - <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/20740135-ddc8-4668-a7bb-34addf3308a1" width="500" height="500"/>

- 퍼셉트론은 두 클래스가 선형적으로 구분되고 학습률이 충분히 작을 때만 수렴이 보장
- [선형 구분 데이터셋/그렇지 못한 데이터셋]
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/06e322cf-0195-4ccc-9213-3cec023fe9d2" width="500" height="500"/>


- [2024.07.16]
- <img src="https://github.com/user-attachments/assets/a436ad06-88ba-4763-92b4-c50bd369fea0" width="500" height="500"/>
- <img src="https://github.com/user-attachments/assets/4a513e49-bd9d-45f3-8088-abe1b78251d1" width="500" height="500"/>
- <img src="https://github.com/user-attachments/assets/bfb50509-088c-4592-a25d-99562c26d39a" width="500" height="500"/>



### 주요 알고리즘 그림

- [퍼셉트론]
- <img src="https://github.com/jongheonleee/machine_learning/assets/87258372/2acd368f-b760-4fd2-97ff-9b9610e341bc" width="500" height="500"/>

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
 
### 📋 몰입 리스트
