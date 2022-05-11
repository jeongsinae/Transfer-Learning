# Stanford Car 데이터 셋에서 모델 별 전이학습 유무에 따른 정확도 차이
ResNet50, ResNet18, AlexNet을 사용하여 정확도를 측정함 | 전이학습 유무로 총 6번의 실험을 진행함 

## 사용 모델
ResNet50, ResNet18, AlexNet을 사용하였고 전이학습 유무로 한번 더 학습시킴

## 데이터셋
Stanford Car Dataset by classes folder을 사용
>	> 총 16,185개의 이미지
>	> 196개의 종류
>	> 학습 이미지 8,144개
>	> 테스트 이미지 8.041개
>	> 제조사, 모델, 제조연도 ( ex) BMW M3 2012)

다운로드 링크 : <https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset>

## 연구 목적
심층 신경망을 사용하여 세분화된 분류 작업을 진행하고 다양한 모델을 적용하여 성능을 평가 및 비교 분석

## Model
### AlexNet
### GoogleNet
### DenseNet
### ResNet

## 전이학습
학습 데이터가 부족한 분야의 모델 구축을 위해 데이터가 풍부한 분야에서 훈련된 모델을 재사용하는 머신러닝 학습 기법

## 결과
<img src="https://user-images.githubusercontent.com/49273782/167881155-f709c080-ed14-424a-8057-598e13be2efa.png"  width="500" height="370">
<img src="(https://user-images.githubusercontent.com/49273782/167881271-fcf045f0-d2e8-413c-aaec-ead733164864.png"  width="500" height="370">

## 분석 및 고찰
전이학습을 진행한 모델 ( (평균) 77.83% )  > 전이학습을 진행하지 않은 모델 ( (평균) 54.33% )

