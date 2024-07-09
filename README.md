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

### 📋 몰입 리스트
