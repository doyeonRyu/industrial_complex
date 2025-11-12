"""
==============================================================================
File: n03_train_model.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-10-14

Description: 하이브리드 혹은 단일 시계열 예측 모델 초기화 및 훈련 과정
    Functions:
        - load_data: 전처리 완료된 CSV 파일과 스케일러 로드
        - sliding_window: 슬라이딩 윈도우 생성
        - build_dataloader: DataLoader 생성
        - build_model: 모델 초기화
        - train_model: 모델 훈련
Note
    - main 모듈 변수 설정 후 실행
    - 모델: CNN + LSTM / CNN + Transformer / LSTM 단독 / Transformer 단독 (추가 가능)
    - 산업체마다 데이터 특성이 다를 수 있음. 수정 필요 - 현재는 메인텍 2공장의 포멧을 따름
    - 각 모델의 파라미터는 글로벌 변수로 설정
==============================================================================
"""

import torch
import torch.nn as nn
import os
import time

from utils.sliding_window import build_sliding_window
from utils.dataloader import build_dataloader
from utils.load_data import load_data
from utils.model import build_model
from utils.train import train
from utils.evaluate_metrics import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 피크 구간 가중치 부여한 Huber Loss 클래스
class WeightedHuberLoss(nn.Module):
    """
    Class: WeightedHuberLoss
        - 피크 구간에 가중치를 부여한 Huber Loss 구현
        - 피크 가중치: 피크 구간에 대해 손실을 더 크게 반영
        - 피크 구간: 전체 부하 중 상위 몇 퍼센트로 정의
    Parameters:
        - delta: Huber Loss의 delta 값
        - peak_threshold: 전체 부하 중 상위 몇 퍼센트를 피크로 볼지
        - peak_weight: 피크 구간 가중 배수
    Returns:
        - weighted_loss: 피크 구간에 가중치가 적용된 Huber Loss 값
    """
    def __init__(self, delta=1.0, peak_threshold=0.9, peak_weight=3.0):
        super().__init__()
        self.delta = delta
        self.peak_threshold = peak_threshold  # 전체 부하 중 상위 몇 퍼센트를 피크로 볼지
        self.peak_weight = peak_weight # 피크 구간 가중 배수

    def forward(self, y_pred, y_true):
        # 절대 오차 계산
        error = torch.abs(y_true - y_pred)
        
        # 기본 huber 손실 계산
        huber_loss = torch.where(
            error <= self.delta,
            0.5 * error**2,
            self.delta * (error - 0.5 * self.delta)
        )
        
        # 피크 구간 마스크 생성
        normed = y_true / y_true.max()
        peak_mask = (normed > self.peak_threshold).float() # peak_threshold 초과면 1, 아니면 0

        # 피크에만 가중치 부여
        weights = torch.ones_like(y_true) + peak_mask * (self.peak_weight - 1) # 피크 구간은 peak_weight 배수

        # 최종 가중 손실
        weighted_loss = torch.mean(weights * huber_loss)
        return weighted_loss
    
# 모델 훈련 함수
def train_model(path, input_window, output_window, 
                criterion, optimizer_type, num_epochs, batch_size,
                model1, model2, device, threshold_value):
    """
    Function: train_model
        - 하이브리드 or 단일 시계열 예측 모델 훈련
    Parameters:
        - path: str, 데이터가 저장된 디렉토리 경로
        - input_window: int, 입력 시퀀스 길이
        - output_window: int, 출력 시퀀스 길이
        - criterion: 손실 함수
        - optimizer_type: str, 최적화 알고리즘 ("Adam", "SGD", "RMSprop" 등)
        - num_epochs: int, 학습 epoch 수
        - model1: str or None, "CNN" or None
        - model2: str, "LSTM" or "Transformer"
        - device: torch.device, 모델과 데이터를 올릴 디바이스 (cpu or cuda)
    Returns:
        - None
    """
    # 1. 데이터 로드
    _, _, _, train_preprocessed, valid_preprocessed, _, _, _, _, _, _, _, _ = load_data(path)

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
    train_loader, valid_loader, _ = build_dataloader(
        torch_train_x, torch_train_y, torch_valid_x, torch_valid_y, None, None, batch_size=batch_size
    )

    # 4. 모델 초기화
    model1, model2 = build_model(model1, model2, train_preprocessed, input_window, output_window, device)

    # train 데이터의 상위 5% 값을 피크 임계값으로 설정
    if threshold_value is not None:
        peak_threshold = torch.quantile(torch_train_y, threshold_value)
        criterion.peak_threshold = peak_threshold.item() # criterion 변경

    # 5. 옵티마이저 설정
    if model1 is not None: 
        # 하이브리드 모델의 경우
        optimizer = torch.optim.__dict__[optimizer_type](list(model1.parameters()) + list(model2.parameters()), lr=1e-4)
    else: # 단일 시계열 예측 모델의 경우
        optimizer = torch.optim.__dict__[optimizer_type](list(model2.parameters()), lr=1e-4)
    
    # 6. 모델 저장 경로 및 best loss 초기화
    path_name = os.path.basename(os.path.normpath(path))
    best_loss = float('inf')

    # input, output_window가 달라지면 가장 뒤에 _(input, output)_ 추가
    # default: input_window=10, output_window=1
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
    if threshold_value is not None:
        best_model2_path = best_model2_path.replace('.pth', f'_peak_weight.pth')
        if best_model1_path is not None:
            best_model1_path = best_model1_path.replace('.pth', f'_peak_weight.pth')

    # 7. 학습 루프
    print("\n======================================================================\n")
    if threshold_value is not None:
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
    "산업체명 변경할 경우 **path** 변수 수정 필요"
    path = "data/preprocessed/광명금속/"
    path_name = os.path.basename(os.path.normpath(path)) # 예: "메인텍 2공장"
    input_window = 36 # 15분 단위, 24 = 6시간
    output_window = 32 # 15분 단위, 24 = 6시간
    model1 = "CN" # "CNN" or None
    model2 = "Transformer" # "LSTM" or "Transformer_encoder" or "Transformer"
    num_epochs = 500
    batch_size = 512
    criterion = nn.HuberLoss(delta=1.0, reduction="mean") # Huber Loss
    optimizer_type = "Adam" # "Adam", "SGD", "RMSprop"

    train_model(path, input_window, output_window, 
                criterion, optimizer_type, num_epochs, batch_size,
                model1, model2, device, threshold_value=None)
