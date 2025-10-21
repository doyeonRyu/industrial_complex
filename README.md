# 산업 단지 전력 사용량 예측 모델

> Project: 산업 단지 전력 사용량 예측 모델   
Author: 유도연     
Created Date: 2025-10-13   
Last Modified: 2025-10-14   
> 
- 파일 동작 순서
    - n01_load_data.py: 추출된 데이터 병합하여 산업체마다 하나의 데이터셋으로 완성
    - n02_data_preprocessing.py: 데이터 전처리
    - n03_train_model.py: 모델 학습 과정
    - n04_evaluate_and_visualize.py: 모델 평가 및 시각화
    - n05_tun_model.py: 모델 튜닝

## n01_load_data

> 추출된 데이터 병합하여 산업체마다 하나의 데이터셋으로 완성
> 
- 한 파일의 여러 테이블에서 15분 단위 시계열 데이터만을 추출
- 데이터셋 병합하여 하나의 데이터셋으로 완성
- `data/preprocessed/{산업체명}_시계열_데이터({기간}).csv` 형태로 저장

## n02_data_preprocessing

> 데이터 점검 및 전처리 과정
> 

### data_inspection

1. 컬럼명 변경
2. 변수별 시각화
    - 요일별 평균 사용량 시각화
    - 월별 평균 사용량 시각화
    - 각 변수별 사용량과의 상관관계 산점도
    - 각 변수별 사용량과의 상관관계 히트맵
- 시각화된 그래프는 `plots/{폴더명}/{파일명}.png` 형태로 저장
- 데이터 프레임 저장

### data_preprocessing

3. Train / Valid / Test 분할

4. X, y 분리

5. 이상치, 결측치 탐색 및 처리

6. 로그 변환

7. (정규화), 표준화

- minmax scaler 생략
- z-score standardization만 진행
- 출력된 데이터 정보 `data/preprocessed/{산업체명}/{산업체명}_data_inspection_and_preprocessing_log.txt` 형태로 저장
- train_origin, valid_origin, test_origin, train_preprocessed, valid_preprocessed, test_preprocessed, train_preprocessed_y, valid_preprocessed_y, test_preprocessed_y `data/preprocessed/{산업체명}/{산업체명}_{변수명}.csv` 형태로 저장

## n03_train_model

> 모델 학습 과정
> 

Functions:

- load_data: 전처리 완료된 CSV 파일과 스케일러 로드
- sliding_window: 슬라이딩 윈도우 생성
- build_dataloader: DataLoader 생성
- build_model: 모델 초기화
- train_model: 모델 훈련

최고 성능 모델 저장

- 하이브리드 모델
    - `results/{산업체명}/{산업체명}_hybrid_{model1}_with_{model2}_({input},{output}).pth`
    - `results/{산업체명}/{산업체명}_hybrid_{model2}_({input},{output}).pth`
- 단일 모델
    - `results/{산업체명}/{산업체명}*_{model}_*({input},{output}).pth`

## n04_evaluate_and_visualize

> 모델 평가 및 시각화
> 

### Evaluate

- 모델의 MAE, RMSE, MAPE, R2 평가

### Visualize

- 원하는 날짜만큼 예측 결과 시각화
- ouput_window는 최신 걸로 덮어 씌움
- 한 step 뒤로 예측되는 착각
- 시각화 결과
    - 하이브리드 모델: `plots/{산업체명}/[{산업체명] {시각화 날짜}days_{데이터 타입}_forecast({모델1 명}_with_{모델2 명})_({input},{output}).png`
    - 단일 모델: `plots/{산업체명}/[{산업체명}] {시각화 날짜}days_{데이터 타입}_forecast({모델2 명})_({input},{output).png`