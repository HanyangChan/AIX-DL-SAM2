# AIX-DL-SAM2
Project for AI+X: DeepLearning, for 2025-2 semester

SAM2 모델을 활용한 음식 사진 object segmentation과 칼로리 계산 모델

- 황병찬 서울 데이터사이언스학부 2022069407 - 데이터 수집 및 코드 구축 및 블로그 작성
- 엄태훈 서울 신소재공학부 2024082624 - 코드 구현 및 보고서 작성
- 김연준 서울 융합전자공학부 2017027674 - 코드 구현


## I. 프로젝트 개요
### Motivation 
Facebook AI (현 meta AI)에서 개발한 visual segmentation 모델 SAM(Segment Anything Model)은 사전 학습된 파운데이션 모델(Pre-trained foundation model)으로 이미지 속에 있는 객체를 segmentation을 하고 기존의 컴퓨터 비전에서의 문제인 데이터 부족이라는 문제를 극복하기 위한 파운데이션 모델을 구축하였다. 이후 2024년 9월 출시한 SAM2는 이미지를 넘어서 동영상 객체에 대해서도 segmentation을 효과적으로 실행할 수 있는 능력을 보여준다. 따라서 SAM2를 적용할 데이터셋을 탐색하는 중 kaggle의 음식 사진 데이터셋을 발견하여 이를 통해서 SAM2를 적용할 계획이다. 최종적인 목표로 각각의 음식들과 이에 상응하는 칼로리를 매칭하여 전체적인 칼로리를 계산할 수 있는 모델을 구현 해보고자 한다. 

### What do you want to see at the end?
기존 Yolo 기반 모델이 아닌 SAM2 모델을 선택한 이유로는 우선 프롬프트 기반의 인스턴스 분할이 가능하며 특히 SAM2에는 모델이 보지 못한 객체에서도 사용자의 프롬프트에 따라 분할이 가능한(Class-agnostic) 장점이 있다. 또한 SAM2의 뛰어난 generalization 성능을 이용하여 파인튜닝을 통한 음식의 객체를 분석할 예정이다. 이를 통해서 음식의 사진과 음식의 용량을 파악하고 입력 이미지의 칼로리가 어떻게 될 것인지 예측하는 모델을 구현할 예정이다.  

## II. 데이터셋 구축
### 데이터 수집 및 구성 
kaggle에 음식 사진을 수집하여 데이터셋을 구축한다. 
- 음식 사진 분류: [Food Image Classification Dataset](https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset)

### 디렉토리 구조
```
data/
├── train/
│   ├── 도넛/
│   └── 피자/
|   └── .../
|   └── .../
├── val/
│   ├── 도넛/
│   └── 피자/
|   └── .../
|   └── .../
└── test/
    ├── 도넛/
    └── 피자/
    └── .../
    └── .../
```

### 전처리

## III. 모델: SAM2 상세 설명



## IV. 학습 과정


### 전체 흐름 개요

Meta AI의 SAM2를 위주로 하되, 속도나 객체 탐지를 하는데 있어서 문제가 생기면 Yolo기반을 활용하여 객체를 탐지하고 바운더리 박스를 생성하고 나서 바운더리 박스를 SAM2의 프롬프트로 입력하여 탐지된 객체에 대한 픽셀 단위의 정밀한 마스크를 생성하는 방향을 진행할 예정이다.


