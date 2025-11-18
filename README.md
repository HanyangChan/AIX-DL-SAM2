# AIX-DL-SAM2
Project for AI+X: DeepLearning, for 2025-2 semester

SAM2 모델을 활용한 음식 사진 object segmentation과 칼로리 계산 모델

- 황병찬 서울 데이터사이언스학부 2022069407 - 데이터 수집 및 코드 구축 및 블로그 작성
- 엄태훈 서울 신소재공학부 2024082624 - 코드 구현 및 보고서 작성
- 김연준 서울 융합전자공학부 2017027674 - 


## I. 프로젝트 개요
Facebook AI (현 meta AI)에서 개발한 visual segmentation 모델 SAM(Segment Anything Model)은 사전 학습된 파운데이션 모델(Pre-trained foundation model)으로 이미지 속에 있는 객체를 segmentation을 하고 기존의 컴퓨터 비전에서의 문제인 데이터 부족이라는 문제를 극복하기 위한 파운데이션 모델을 구축하였다. 이후 2024년 9월 출시한 SAM2는 이미지를 넘어서 동영상 객체에 대해서도 segmentation을 효과적으로 실행할 수 있는 능력을 보여준다. 따라서 SAM2를 적용할 데이터셋을 탐색하는 중 kaggle의 음식 사진 데이터셋을 발견하여 이를 통해서 SAM2를 적용할 계획이다. 최종적인 목표로 각각의 음식들과 이에 상응하는 칼로리를 매칭하여 전체적인 칼로리를 계산할 수 있는 모델을 구현 해보고자 한다. 

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
│   ├── 도넛/
│   └── 피자/
|   └── .../
|   └── .../
```

### 전처리

## III. 모델: SAM2 상세 설명



## IV. 학습 과정


### 전체 흐름 개요
