SAM2 모델을 활용한 음식 사진 object segmentation과 칼로리 계산 모델

- 황병찬 서울 데이터사이언스학부 2022069407 - 데이터 수집 및 코드 구축 및 블로그 작성
- 엄태훈 서울 신소재공학부 2024082624 - 코드 구현 및 보고서 작성
- 김연준 서울 융합전자공학부 2017027674 - 코드 구현


## I. 프로젝트 개요
### Motivation 
Facebook AI (현 meta AI)에서 개발한 visual segmentation 모델 SAM(Segment Anything Model)은 사전 학습된 파운데이션 모델(Pre-trained foundation model)으로 이미지 속에 있는 객체를 segmentation을 하고 기존의 컴퓨터 비전에서의 문제인 데이터 부족이라는 문제를 극복하기 위한 파운데이션 모델을 구축하였다. 이후 2024년 9월 출시한 SAM2는 이미지를 넘어서 동영상 객체에 대해서도 segmentation을 효과적으로 실행할 수 있는 능력을 보여준다. 따라서 SAM2를 적용할 데이터셋을 탐색하는 중 kaggle의 음식 사진 데이터셋을 발견하여 이를 통해서 SAM2를 적용할 계획이다. 최종적인 목표로 각각의 음식들과 이에 상응하는 칼로리를 매칭하여 전체적인 칼로리를 계산할 수 있는 모델을 구현 해보고자 한다. 

### Problem with current food segmentation apps
기존의 칼로리 측정앱을 사용해보면서 갖고 있는 문제점들에 대해서 알아보고자 하였다. 따라서 카카오 헬스케어에서 운영중인 "파스타"라는 칼로리 측정어플을 이용해보았다. 어플을 사용해보면서 가장 큰 문제점으로는 같은 음식이면 크기에 상관없이 항상 일정한 칼로리로 측정을 하는 문제점이 존재한다. 아래 사진들과 같이 떡볶이라는 음식 사진을 측정할 때 음식의 양과 재료들이 다르지만 항상 일정하게 509kcal로 측정을 하고 있다.

<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/f94b57e8-17e8-4165-ae00-8a6b14386bdd" />
<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/c120e213-07f0-4e98-a223-db38b06ab7a0" />
<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/d225a83f-06e7-4aeb-9437-aee8ddf4a832" />

또한 음식의 칼로리를 측정할 때 전체 칼로리 하나만 측정이 되고 음식 내에 세부적인 칼로리 등에 대해서는 측정이 되지 않거나 아예 무시되는 개선사항이 존재한다. 아래 사진에서는 스테이크 뿐만 아니라 랍스타와 채소들도 있지만 오직 스테이크 하나만을 측정하고 있다. 

<img width="300" height="600" alt="image" src="https://github.com/user-attachments/assets/9117ab09-01c9-45ed-be04-4ab47dbba713" />


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

<img width="1248" height="313" alt="image" src="https://github.com/user-attachments/assets/34321423-3113-4d0e-941d-f5a555b44bf3" />

본 프로젝트에서는 SAM2를 채택하였으며 장점과 모델의 특징으로는 다음과 같다:
SAM 2는 이미지나 동영상에서 원하는 객체만 선택해서 분할(segmentation)하기 위한 모델이다.
가장 큰 특징으로는 프롬프팅이 가능하며 `포인트 클릭`, `박스 지정`, `마스크 제공` 같은 프롬프트 기능이 존재한다. 

### `이미지 인코더`

역할

-동영상 프레임이 들어올 때마다 해당 프레임을 특징(feature) 토큰으로 변환한다.

-한 번의 상호작용 과정 전체에서 이미지 인코더는 프레임마다 한 번만 실행되며, 이후 단계들은 이 feature를 계속 활용한다.

-계층적 구조이기 때문에 멀티스케일 특징(고해상도·저해상도)을 모두 사용할 수 있다.

### `메모리 어텐션 (Memory attention)`

해당 모델에서 메모리란 
-과거 프레임에서 모델이 예측한 결과들(마스크)과 그 프레임의 특징 정보들을 저장한 공간.

-이 메모리는 다음 프레임의 segmentation에 도움을 준다.

동작 방식 
각 프레임을 처리할 때 다음 순서로 진행된다.

1.현재 프레임의 이미지 feature를 transformer block에 입력

2.Self-attention으로 현재 프레임 내부 정보를 먼저 정리

3.저장된 메모리(과거 프레임 특징 + 예측) 와 cross-attention
→ 과거 정보를 참고하여 현재 프레임의 segmentation을 더 정확히 만듦

4.MLP로 출력 정제

즉, 과거 프레임의 정보와 현재 프레임의 특징과 새로운 프롬프트
→ 이 세 가지를 조합해서 현재 프레임의 최종 임베딩을 만든다.


### `프롬프트 인코더`

SAM 1과 동일한 방식 사용

사용자가 클릭한 좌표, bounding box, 또는 기존 마스크를 임베딩으로 변환

클릭/박스는 학습된 embedding + 위치 정보

마스크는 conv 네트워크를 통해 임베딩 후 프레임 임베딩에 더함

### `마스크 디코더`

프레임 임베딩 + 프롬프트 임베딩을 함께 입력

이를 양방향 transformer block 여러 개로 업데이트하면서 최종 segmentation 마스크 생성

여러 마스크가 필요한 경우(예: 단일 클릭으로 ambiguous한 경우)
→ 여러 개의 후보 마스크 출력

동영상에서도 프레임별로 여러 마스크를 예측할 수 있음

<img width="900" height="387" alt="image" src="https://github.com/user-attachments/assets/84c7eefd-6c03-4512-ba24-096edcde7211" />

디코더 디자인은  프롬프트와 프레임 임베딩을 업데이트하는 양방향 transformer block을 쌓는다. 만약 여러 마스크가 있을 수 있는 모호한 프롬프트, 즉 단일 클릭의 경우, 여러 마스크를 예측한다. 모호성이 프레임 전체로 확장될 수 있는 동영상의 경우, 모델은 각 프레임에서 여러 마스크를 예측한다. 후속 프롬프트가 모호성을 해결하지 못하면, 모델은 현재 프레임에 대해 예측된 IoU(mlp의 output)가 가장 높은 마스크만 전파한다.



## IV. 학습 과정

<img width="1248" height="681" alt="image" src="https://github.com/user-attachments/assets/c3f115e3-0849-4ac9-9406-40e8e80dacb5" />




SAM2를 통해서 물체를 인식하고 EfficientNet V2를 이용해서 물체를 분류(classification)할 예정이다. EfficientNet은 기존의 CNN이나 ResNet에서 정확도와 효율에서 좋은 모습을 보이기 때문에 선택하였다. ImageNet을 학습한 Pretrained된 EfficientNet을 사용하였다. 



- **프레임워크** : PyTorch
- **손실 함수** : CrossEntropy
- **옵티마이저** : Adam (lr = 0.0001)
- **scheduler** : StepLR (gamma=0.1)
- **Proportion of train/vaildation : 80/20%

다음과 같은 세팅으로 우선 아무 학습없이 하였을 때 분류하였을 때 다음과 같은 결과가 나왔다. 

<img width="540" height="796" alt="image" src="https://github.com/user-attachments/assets/0615e562-48f4-4579-8ae7-47dc7e2856a0" />

이후 10번의 에포크를 돌린다음 결과는 다음과 같다. 

<img width="700" height="65" alt="image" src="https://github.com/user-attachments/assets/0d0f2368-d991-429a-a388-813b51901bbd" />

## V. 분석 및 시각화

### Confusion Matrix

<img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/4632600e-627d-452f-862d-f398b6b04708" />

Confusion Matrix는 모델이 각 음식 클래스를 얼마나 정확하게 맞추는지 보여주는 혼동 행렬이다.
오른쪽 아래 정방향으로의 대각선은 맞춘 개수를, 대각선 바깥쪽은 틀린 개수를 의미한
다. 어떤 클래스끼리 헷갈리는지를 한눈에 보여주는 자료라고 볼 수 있다.

- ‘Donut’->’Donut’: 30번 맞춤
- ‘Taco’->’Taco: 29번 맞춤
- ‘Baked Potato’->’Baked Potato’: 18번 맞춤

### Grad-CAM

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/5e2519d0-a28d-44ce-9f29-653b956a6064" />

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/2320c771-0cce-4997-a9c6-48c424d7daa6" />

왼쪽 사진은 원본 이미지고 오른쪽 사진은 Grad-Cam 결과 이미지다. Grad-Cam은 모델
이 이미지를 보고 예측을 내릴 때 어느 부분을 가장 중요하게 봤는지를 시각화한 자료다.
빨간색~노란색은 모델이 강하게 주목한 부분을, 파란색은 별로 주목하지 않은 부분을 의
미한다.

첫번째 만두 사진에서는 만두 형태의 윗부분과 골격을 강하게 참고해서 ‘momos’
라고 판단했다는 의미이고 두번째 치즈 케이크 사진에서는 케이크 형태의 중심을 강하게
참고해서 ‘cheesecake’라고 판단했다는 의미이다.

### t-SNE

<img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/411d778e-f1bb-4fc1-bf67-4aff3e7e0975" />

t-SNE는 이미지 임베딩을 2D에 압축해 시각화한 것이다. 비슷한 종류의 음식 데이터는
가까운 위치에 모이고 다른 종류의 음식은 멀리 떨어져 있음을 보여준다. 따라서 이 시
각화 자료는 모델이 추출한 특징으로 음식을 얼마나 잘 구분했는지 보여주는 그래프다.

## VI. 전체 흐름 개요 및 SAM2 모델 활용

Meta AI의 SAM2를 위주로 하되 SAM2가 설치가 되지않았거나 문제가 생겼을 때 opencv가 대신 작동이 되도록 설정을 해두었다.

SAM2 라이브러리를 이용하여 다음과 같이 불러올 수 있다. 

`from sam2.build_sam import build_sam2`

`from sam2.sam2_image_predictor import SAM2ImagePredictor`

실행 위치 (Code Location):

`backend/main.py`
: 
predict
 함수 안에서 
run_sam2_inference
를 호출하여 SAM2를 실행합니다. 여기서 그리드 포인트를 생성하고 결과를 수집합니다.
backend/sam2_utils.py
: 실제로 SAM2 모델을 로드하고(
load_sam2_model
) 추론을 수행하는(
run_sam2_inference
) 코드가 들어있습니다.



## VII. 웹서비스 배포 (11/22 진행중)
프론트앤드는 CSS, react와 자바스크립트를 활용하여 개발하였으며 pytest를 활용한 API endpoint 그리고 fastAPI를 활용하여 배포를 진행하였다. 
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/4b4b0854-aca0-43a5-b0ca-5a85dbe4350a" />

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/7b962e11-d181-40e0-86bf-af8a42d76354" />

### 실행
실행 방법으로는 다운 폴더에서 터미널을 두 번 입력해야한다. 

```Bash
cd backend
python -m uvicorn main:app --reload
```

```Bash
cd frontend
npm rum dev
```
이후 `http://localhost:5173`와 같은 url을 통해서 접속할 수 있다. 

## VIII (임시 11/25 기준 문제점과 한계점)
<img width="932" height="810" alt="image" src="https://github.com/user-attachments/assets/5429d806-95fe-457f-a8e4-51510b60bddd" />

다음과 같이 처음 보는 사진들에 대해서 잘 인식을 하지못하는 모습을 보여주고 있다. 우선 기초적인 틀을 만들기 위해서 웹에서는 SAM2모델는 이후 추가할 예정이므로 물체를 인식하는 능력이 엄청 뛰어나지 않으며 기존 이미지에서 오버피팅되는 모습을 보여준다. 따라서 이후 SAM2을 연결할 예정이며 성능이 좋지 않은 경우 이미지 데이터셋을 파인 튜닝하여 성능을 향상시킬 예정이다.  

## 수정 사항 
캐글에 있는 데이터셋을 통하여 efficientNet을 학습시키였고 sam2를 바탕으로 물체를 segementation하도록 하였다. 여러 물체가 있는 경우에도 서로 구별하여 잘 인식하고 있으며 각각의 칼로리를 계산해서 더하고 있다. 

<img width="1404" height="742" alt="image" src="https://github.com/user-attachments/assets/c9a4cd66-69a6-4e39-b21c-950f09fdf94a" />


