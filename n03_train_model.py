"""
==============================================================================
File: n03_train_model.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-11-19

Description: 하이브리드 혹은 단일 시계열 예측 모델 초기화 및 훈련 과정
    Functions:
        - load_data: 전처리 완료된 CSV 파일과 스케일러 로드
        - build_sliding_window: 슬라이딩 윈도우 생성
        - initiate_model: 모델 초기화
        - train: 모델 훈련
        - evaluate: 검증 데이터셋으로 모델 평가
Note
    - 코드 실행 시 {industry_name}, {model1}, {model2} 인자 설정
        - 실행 예시: python n03_train_model.py --industry_name 광명금속 (--model1 CNN) --model2 LSTM
    - 모델: CNN + LSTM / CNN + Transformer / LSTM 단독 / Transformer 단독 (longformer 부터는 longformer 파일 사용)
==============================================================================
"""

import torch
import torch.nn as nn
import os
import time
import argparse
from torch.utils.data import DataLoader, TensorDataset

from utils.load_data import load_data
from utils.sliding_window import build_sliding_window
from utils.model import initiate_model
from utils.train import train
from utils.evaluate import evaluate
from utils.weighted_loss import WeightedHuberLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 명령행 인자 파서 설정 함수
def parse_args():
    parser = argparse.ArgumentParser(description="전력 사용량 예측 모델 훈련 스크립트")
    parser.add_argument("--industry_name", type=str, default=None, help="결과 파일명에 사용할 산업체 이름")
    parser.add_argument("--model1", type=str, default=None, help="모델1 유형 선택 (CNN 또는 None)")
    parser.add_argument("--model2", type=str, default=None, help="모델2 유형 선택 (LSTM 또는 Transformer)")
    return parser.parse_args()
    
# 모델 훈련 함수
def train_model(path, input_window, output_window, 
                criterion, optimizer_type, num_epochs, batch_size,
                model1, model2, device, threshold_value):
    """
    Function: train_model
        - 하이브리드 or 단일 시계열 예측 모델 훈련
    Parameters:
        - path: str
            - 데이터가 저장된 디렉토리 경로
        - input_window: int, 
            - 입력 시퀀스 길이
        - output_window: int, 
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
        - model2: str, "LSTM" or "Transformer"
        - device: cuda or cpu
        - threshold_value: float,
            - 피크 구간 가중치 부여 임계값 (0 ~ 1 사이 값, 0이면 피크 가중치 없음)
    Returns:
        - None
    """

    # 1. 데이터 로드
    _, _, _, train_preprocessed, valid_preprocessed, _, _, _, _, _, _ = load_data(path)

    # 2. 슬라이딩 윈도우, torch.Tensor 변환
    #    torch_x.shape: [N(window 수), L(input window 길이), F(feature 수)]
    #    torch_y.shape: [N(window 수), output_window] or [N(window 수),] (output_window=1일 때)
    torch_train_x, torch_train_y = build_sliding_window(
        train_preprocessed, input_window, output_window,
        target_col="usage_kWh", keep_target_in_x=True
    )
    torch_valid_x, torch_valid_y = build_sliding_window(
        valid_preprocessed, input_window, output_window,
        target_col="usage_kWh", keep_target_in_x=True
    )

    # 3. DataLoader 생성
    #    DataLoader: 배치 단위로 데이터 로드, 셔플링, 병렬 처리 지원
    train_ds = TensorDataset(torch_train_x, torch_train_y)
    valid_ds = TensorDataset(torch_valid_x, torch_valid_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())

    # 4. 모델 초기화
    model1, model2 = initiate_model(model1, model2, train_preprocessed, input_window, output_window, device)

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
        train_loss = train(train_loader, model1, model2, criterion, optimizer, device)

        # 2) 검증
        #    overall_loss: epoch 당 평균 손실 (valid set)
        overall_loss = evaluate(valid_loader, model1, model2, criterion, device, output_window)

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
    "코드 실행 시 {industry_name}, {model1}, {model2} 인자 설정"
    args = parse_args()
    
    industry_name = args.industry_name
    if industry_name is None:
        industry_name = "광명금속"
    path = f"data/preprocessed/{industry_name}/"

    input_window = 36 # 15분 단위, 24 = 6시간 + 하루 (24*4=96) 36 + 96 
    output_window = 32 # 15분 단위, 24 = 6시간

    model1 = args.model1 # "CNN" or None
    if model1 is None:
        model1 = None
    model2 = args.model2 # "LSTM" or "Transformer_encoder" or "Transformer"

    num_epochs = 500
    batch_size = 512
    criterion = nn.HuberLoss(delta=1.0, reduction="mean") # Huber Loss
    optimizer_type = "Adam" # "Adam", "SGD", "RMSprop"
    threshold_value = 0 # 상위 10% # 피크 구간 가중치 부여 임계값 (0 ~ 1 사이 값, 0이면 피크 가중치 없음)

    train_model(path, input_window, output_window, 
                criterion, optimizer_type, num_epochs, batch_size,
                model1, model2, device, threshold_value=threshold_value)