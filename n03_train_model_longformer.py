"""
==============================================================================
File: n03_train_model_longformer.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-11-19

Description: longformer(Informer, Autoformer)용 예측 모델 초기화 및 훈련 과정
    Functions:
        - load_data: 전처리 완료된 CSV 파일과 스케일러 로드
        - build_sliding_window_longformer: 슬라이딩 윈도우 생성
        - initiate_model_longformer: 모델 초기화
        - train_longformer: 모델 훈련
        - evaluate_longformer: 검증 데이터셋으로 모델 평가
Note
    - 코드 실행 시 {industry_name}, {model1}, {model2} 인자 설정
        - 실행 예시: python n03_train_model_longformer.py --industry_name 광명금속 (--model1 CNN) --model2 Informer
    - 모델: CNN + Informer / CNN + Autoformer / Informer 단독 / Autoformer 단독
==============================================================================
"""

import torch
import torch.nn as nn
import os
import time
import argparse
from torch.utils.data import DataLoader, TensorDataset

from utils.load_data import load_data
from utils.sliding_window import build_sliding_window_longformer
from utils.model import initiate_model_longformer
from utils.train import train_longformer
from utils.evaluate import evaluate_longformer
from utils.weighted_loss import WeightedHuberLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 명령행 인자 파서 설정 함수
def parse_args():
    parser = argparse.ArgumentParser(description="전력 사용량 예측 모델(longformer) 훈련 스크립트")
    parser.add_argument("--industry_name", type=str, default=None, help="결과 파일명에 사용할 산업체 이름")
    parser.add_argument("--model1", type=str, default=None, help="모델1 유형 선택 (CNN 또는 None)")
    parser.add_argument("--model2", type=str, default=None, help="모델2 유형 선택 (longformer 타입)")
    return parser.parse_args()

# 모델 훈련 함수
def train_model(path, input_window, label_len, output_window, 
                criterion, optimizer_type, num_epochs, batch_size,
                model1, model2, device, threshold_value):
    """
    Function: train_model
        - 하이브리드 or 단일 시계열 예측 모델 훈련
    Parameters:
        - path: str
            - 데이터가 저장된 디렉토리 경로
        - input_window: int
            - 입력 시퀀스 길이
        - output_window: int
            - 출력 시퀀스 길이
        - criterion:
            - 손실 함수
        - optimizer_type: str
            - 옵티마이저 유형 ("Adam", "SGD", "RMSprop" 등)
        - num_epochs: int
            - 학습 epoch 수
        - batch_size: int
            - 배치 크기
        - model1: str or None, "CNN" or None
        - model2: str, "longformer"
        - device: cuda or cpu
        - threshold_value: float,
            - 피크 구간 가중치 부여 임계값 (0 ~ 1 사이 값, 0이면 피크 가중치 없음)
    Returns:
        - None
    """
    # 1. 데이터 로드 # path에 longformer 포함 시 longformer용 데이터 로드됨
    _, _, _, train_preprocessed, valid_preprocessed, _, train_x_mark, valid_x_mark, test_x_mark, _, _, _, _, _ = load_data(path, model_type="longformer")

    # 2. 슬라이딩 윈도우, torch.Tensor 변환
    #    torch_x.shape: [N(window 수), L(input window 길이), F(feature 수)]
    #    torch_y.shape: [N(window 수), output_window] or [N(window 수),] (output_window=1일 때)
    torch_train_x_enc, torch_train_x_mark_enc, torch_train_x_dec, torch_train_x_mark_dec, torch_train_y = build_sliding_window_longformer(
        model2,
        train_preprocessed, # 일반 feature
        train_x_mark, # 날짜 시간 feature
        seq_len=input_window,
        label_len=label_len,
        pred_len=output_window,
        target_col="usage_kWh",
    )

    # print(train_x_enc.shape, train_x_dec.shape, train_y_target.shape, train_x_mark_enc.shape, train_x_mark_dec.shape)
    torch_valid_x_enc, torch_valid_x_mark_enc, torch_valid_x_dec, torch_valid_x_mark_dec, torch_valid_y = build_sliding_window_longformer(
        model2,
        valid_preprocessed, # 일반 feature
        valid_x_mark, # 날짜 시간 feature
        seq_len=input_window,
        label_len=label_len,
        pred_len=output_window,
        target_col="usage_kWh",
    )
    # 3. DataLoader 생성
    #    DataLoader: 배치 단위로 데이터 로드, 셔플링, 병렬 처리 지원
    train_ds = TensorDataset(torch_train_x_enc, torch_train_x_dec, torch_train_y, torch_train_x_mark_enc, torch_train_x_mark_dec)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_ds = TensorDataset(torch_valid_x_enc, torch_valid_x_dec, torch_valid_y, torch_valid_x_mark_enc, torch_valid_x_mark_dec)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # 4. 모델 초기화
    model1, model2 = initiate_model_longformer(model1, model2, torch_train_x_enc, torch_train_x_dec, input_window, label_len, output_window, device)

    # train 데이터의 상위 threshold_value 분위값을 피크 임계값으로 설정
    if threshold_value != 0:
        peak_threshold = torch.quantile(torch_train_y, threshold_value)

        # 피크 기반 WeightedHuberLoss 사용
        criterion = WeightedHuberLoss(
            delta=1.0,
            peak_threshold=peak_threshold.item(),
            peak_weight=4 # 피크 구간 가중치 
        )

    # 5. 옵티마이저 설정
    if model1 is not None: 
        # 하이브리드 모델의 경우
        optimizer = torch.optim.__dict__[optimizer_type](list(model1.parameters()) + list(model2.parameters()), lr=1e-4)
    else: # 단일 시계열 예측 모델의 경우
        optimizer = torch.optim.__dict__[optimizer_type](list(model2.parameters()), lr=1e-4)
    
    # 6. 모델 저장 경로 및 best loss 초기화
    path_name = os.path.basename(os.path.normpath(path))
    best_loss = float('inf')

    if not os.path.exists(f'results/{path_name}'):
        os.makedirs(f'results/{path_name}')

    if model1 is not None: 
        # 하이브리드 모델의 경우
        best_model1_path = f'results/{path_name}/{path_name}_hybrid_{model1.__class__.__name__}_with_{model2.__class__.__name__}_({input_window},{output_window}).pth'
        best_model2_path = f'results/{path_name}/{path_name}_hybrid_{model2.__class__.__name__}_with_{model1.__class__.__name__}_({input_window},{output_window}).pth'
    else: # 단일 시계열 예측 모델의 경우
        best_model1_path = None
        best_model2_path = f'results/{path_name}/{path_name}_{model2.__class__.__name__}_({input_window},{output_window}).pth'

    # 피크 가중치 경로 수정 (_peak_weight 추가)
    if threshold_value != 0:
        best_model2_path = best_model2_path.replace('.pth', f'_peak_weight.pth')
        if best_model1_path is not None:
            best_model1_path = best_model1_path.replace('.pth', f'_peak_weight.pth')

    # 7. 학습 루프
    print("\n======================================================================\n")
    if threshold_value != 0:
        print(f"Training started for {path_name} with peak weight...\n")
    else:
        print(f"Training started for {path_name}...\n")

    total_start_time = time.time()

    # Epoch 반복
    for epoch in range(1, num_epochs + 1):
        # 1) 학습
        #    train_loss: epoch 당 평균 손실 (train set)
        train_loss = train_longformer(train_loader, model1, model2, criterion, optimizer, device)

        # 2) 검증
        #    overall_loss: epoch 당 평균 손실 (valid set)
        overall_loss = evaluate_longformer(valid_loader, model1, model2, criterion, device, output_window)

        # 3) best model 저장
        if overall_loss < best_loss:
            best_loss = overall_loss
            if model1 is not None:
                torch.save(model1.state_dict(), best_model1_path)
                torch.save(model2.state_dict(), best_model2_path)
            else:
                torch.save(model2.state_dict(), best_model2_path)
            print(f"[{path_name}] ({epoch:03d}) Saved best | valid loss: {best_loss:.6f}")

        # 4) 출력: train / valid 평균 손실
        print(f"[{path_name}] ({epoch:03d}) train {train_loss:.6f} | valid {overall_loss:.6f}")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    if model1 is not None:
        # 하이브리드 모델의 경우
        print(f"\nTraining complete. [{path_name} | {model1.__class__.__name__} + {model2.__class__.__name__} | (input_window={input_window}, output_window={output_window})] Best valid loss: {best_loss:.6f}")
        print(f"Total training time: {total_duration/60:.2f} minutes\n")
    else: # 단일 시계열 예측 모델의 경우
        print(f"\nTraining complete. [{path_name} | {model2.__class__.__name__} | (input_window={input_window}, output_window={output_window})] Best valid loss: {best_loss:.6f}")
        print(f"Total training time: {total_duration/60:.2f} minutes\n")
    
    print("======================================================================\n")

# main 실행 블록
if __name__ == "__main__":
    "코드 실행 시 {industry_name}, {model1}, {model2} 설정"
    args = parse_args()

    industry_name = args.industry_name
    if industry_name is None:
        industry_name = "광명금속"
    path = f"data/preprocessed/{industry_name}/"

    input_window = 36 # 15분 단위, 24 = 6시간 + 하루 (24*4=96) 36 + 96 
    label_len = 18  # Informer 디코더의 label_len 설정
    output_window = 32 # 15분 단위, 24 = 6시간

    model1 = args.model1 # "CNN" or None
    if model1 is None:
        model1 = None
    model2 = args.model2 # "Informer" or "Autoformer"
    
    num_epochs = 500
    batch_size = 512
    criterion = nn.HuberLoss(delta=1.0, reduction="mean") # Huber Loss
    optimizer_type = "Adam" # "Adam", "SGD", "RMSprop"
    threshold_value = 0.9 # 상위 10% # 피크 구간 가중치 부여 임계값 (0 ~ 1 사이 값, 0이면 피크 가중치 없음)

    train_model(path, input_window, label_len, output_window, 
                criterion, optimizer_type, num_epochs, batch_size,
                model1, model2, device, threshold_value=threshold_value)