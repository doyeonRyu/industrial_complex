# 산업 단지 전력 사용량 예측 모델

> Project: 산업 단지 전력 사용량 예측 모델   
Author: 유도연     
Created Date: 2025-10-13   
Last Modified: 2025-11-23   
> 
- 파일 동작 순서
    - n01_load_data.py: 추출된 데이터 병합하여 산업체마다 하나의 데이터셋으로 완성
    - n02_data_preprocessing.py: 데이터 전처리
    - n03_train_model(_longformer).py: 모델 학습 과정
    - n04_evaluate_and_visualize(_longformer).py: 모델 평가 및 시각화

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
- 데이터 프레임 리턴

### data_preprocessing

3. 관련 변수 추가   

4. Train / Valid / Test 분할   

5. X, y 분리    

6. 이상치, 결측치 탐색 및 처리   

7. 로그 변환   

8. 정규화, 표준화    
    - robust scaling만 진행   

- 출력된 데이터 정보 `data/preprocessed/{산업체명}/{산업체명}_data_inspection_and_preprocessing_log.txt` 형태로 저장
- train_origin, valid_origin, test_origin, train_preprocessed, valid_preprocessed, test_preprocessed, train_preprocessed_y, valid_preprocessed_y, test_preprocessed_y `data/preprocessed/{산업체명}/{산업체명}_{변수명}.csv` 형태로 저장
- longformer 일 떄: mark 데이터 추가   

## n03_train_model

> 모델 학습 과정
> 

Functions:

- load_data: 전처리 완료된 CSV 파일과 스케일러 로드   
- build_sliding_window: 슬라이딩 윈도우 생성   
- initiate_model: 모델 초기화   
- train: 모델 훈련   
- evaluate: 검증 데이터셋으로 모델 평가   

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

- 모델의 MAE, RMSE, MAPE_filtered, sMAPE, R2, PAPE, HR, lag 평가   
    - mape_filtered: 실제값이 0인 값 제외   

### Visualize
1. plot_predictions_chained
    - 원하는 날짜만큼 예측 결과 시각화
    - ouput_window는 최신 걸로 덮어 씌움
    - 한 step 뒤로 예측되는 듯한 형태 가능성

    - 시각화 결과
        - 하이브리드 모델: `plots/{산업체명}/[{산업체명] {시각화 날짜}days_{데이터 타입}_forecast({모델1 명}_with_{모델2 명})_({input},{output}).png`
        - 단일 모델: `plots/{산업체명}/[{산업체명}] {시각화 날짜}days_{데이터 타입}_forecast({모델2 명})_({input},{output).png`

2. plot_single_prediction
    - 한 window 실행 결과 시각화
    - 시작 시점 선택 가능  
    - 시각화 결과 