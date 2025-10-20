"""
==============================================================================
File: evaluate_metrics.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-10-14

Description: 모델 평가 및 지표 계산
    Functions:
        - evaluate: 검증 또는 테스트 데이터셋에 대한 손실 계산
        - metrics: MAE, RMSE, MAPE, R² 지표 계산
Note
    - train_model.py에서 실행
    - CNN + LSTM / CNN + Transformer / LSTM 단독 / Transformer 단독 (추가 가능)
    - 산업체마다 데이터 특성이 다를 수 있음. 수정 필요 - 현재는 광명금속의 포멧을 따름
==============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as skm
from utils.setup import _to_device

"""
모델 평가 함수
- valid 또는 test 데이터셋에 대한 손실 계산
"""
@torch.no_grad()
def evaluate(loader, model1, model2, criterion, device):
    """
    Function: evaluate
        - 하이브리드 혹은 단일 예측 모델을 검증 모드로 한 epoch 평가
    Parameters:
        - loader: DataLoader, 평가할 데이터 로더
        - model1: CNN 모델
        - model2: 시계열 데이터 처리 모델
        - criterion: 손실 함수
        - device: torch.device, 모델과 데이터를 올릴 디바이스 (cpu or cuda)
    Returns:
        - epoch_loss: float, 전체 배치에 대한(한 epoch 동안의) 평균 손실
    """
    # 모델을 평가 모드로 전환
    if model1 is not None:
        model1.eval()
    model2.eval()

    total_sum, total_cnt = 0.0, 0 # epoch 동안의 누적 손실과 샘플 수

    for batch in loader:
        xb, yb = _to_device(batch, device) # 배치를 장치에 맞게 변환

        if model1 is not None: 
            # CNN 입력 형태로 변환
            # [B, L, F] -> [B, F, L]
            if xb.dim() == 3:
                # L, F 위치 스위치
                xb = xb.permute(0, 2, 1) # [B, L, F] -> [B, F, L]
            elif xb.dim() == 2: # L=1인 경우 [B, F] 형태
                xb = xb.unsqueeze(-1) # [B, F] -> [B, F, 1]
            
            # 1. model1 forward
            md1_out = model1(xb) # [B, F, L] 형태

            # 2. model2 forward
            yhat = model2(md1_out) # [B, output_window] or [B,] 형태
        else:
            # 단일 모델: [B, L, F] 보장
            if xb.dim() == 3 and xb.shape[1] < xb.shape[2]: # [B, F, L]이면
                xb = xb.permute(0, 2, 1) # -> [B, L, F]
            elif xb.dim() == 2: # [B, L]이면
                xb = xb.unsqueeze(-1) # -> [B, L, 1]

            yhat = model2(xb)

        # 타깃 차원 보정 (1D -> 2D) | [B] vs [B,1] 정렬
        if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
            yb = yb.unsqueeze(-1) # (N,) -> (N, 1)

        # 3. 손실 계산
        loss = criterion(yhat, yb)

        # 역전파 생략
        
        bs = xb.size(0) # 배치 크기
        total_sum += loss.item() * bs # 배치 손실의 합
        total_cnt += bs # 배치 샘플 수 누적

    return total_sum / max(total_cnt, 1) # epoch 평균 손실


"""
모델 평가 지표 계산 함수
- MAE, RMSE, MAPE, R² 계산
"""
def metrics(best_model1_path, best_model2_path, model1, model2,
                          data_loader,
                          std_scaler, mm_scaler, 
                          device,
                          target_idx: int | None = None,
                          logged: bool = False):
    """
    Function: metrics
        - 하이브리드 or 단일 모델에 대한 예측값과 실제값을 비교하여
          MAE, RMSE, MAPE, R² 지표 계산
    Parameters:
        - best_model1_path: CNN 모델 가중치 파일 경로
        - best_model2_path: 시계열 데이터 처리 모델 가중치 파일 경로
        - model1: CNN 모델 인스턴스
        - model2: 시계열 데이터 처리 모델 인스턴스
        - data_loader: 평가에 사용할 DataLoader (valid, test)
        - std_scaler: StandardScaler 인스턴스 (표준화 역변환에 사용)
        - mm_scaler: MinMaxScaler 인스턴스 (정규화 역변환에 사용)
        - device: torch.device (CPU or cuda)
        - target_idx: 다변수 스케일러에서 타겟 변수 인덱스 (None: 단일 타겟)
        - logged: 학습 시 log1p 적용 여부 (True/False)
    Returns:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - mape: Mean Absolute Percentage Error
        - r2: R² Score
    """

    # best model 로드
    if model1 is not None:
        model1.load_state_dict(torch.load(best_model1_path, map_location=device))
        model1.to(device)
        model1.eval()
    model2.load_state_dict(torch.load(best_model2_path, map_location=device))
    model2.to(device)
    model2.eval()

    # 타겟 실제값, 예측값 저장용
    all_y, all_yhat = [], []

    with torch.no_grad():
        for batch in data_loader:
            xb, yb = _to_device(batch, device) # 배치를 장치에 맞게 변환

            if model1 is not None: 
                # CNN 입력 형태로 변환
                # [B, L, F] -> [B, F, L]
                if xb.dim() == 3:
                    # L, F 위치 스위치
                    xb = xb.permute(0, 2, 1) # [B, L, F] -> [B, F, L]
                elif xb.dim() == 2: # L=1인 경우 [B, F] 형태
                    xb = xb.unsqueeze(-1) # [B, F] -> [B, F, 1]
                
                # 1. model1 forward
                md1_out = model1(xb) # [B, F, L] 형태
                
                # 2. model2 forward
                yhat = model2(md1_out) # [B, output_window] or [B,] 형태
            else:
                # 단일 모델: [B, L, F] 보장
                if xb.dim() == 3 and xb.shape[1] < xb.shape[2]: # [B, F, L]이면
                    xb = xb.permute(0, 2, 1) # -> [B, L, F]
                elif xb.dim() == 2: # [B, L]이면
                    xb = xb.unsqueeze(-1) # -> [B, L, 1]

                yhat = model2(xb)

            # 타깃 차원 보정 (1D -> 2D) | [B] vs [B,1] 정렬
            if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
                yb = yb.unsqueeze(-1)

            # CPU로 이동 후 numpy 변환
            all_y.append(yb.detach().cpu().numpy())
            all_yhat.append(yhat.detach().cpu().numpy())

    # 배치 합치기 (배치 단위 -> 전체 샘플 단위)
    all_y = np.concatenate(all_y, axis=0) # (N, ) or (N, O)
    all_yhat = np.concatenate(all_yhat, axis=0) # (N, ) or (N, O)

    # 2D로 정렬 (스케일러: (N,1) 형태 기대)
    all_y = all_y.reshape(-1, 1)
    all_yhat = all_yhat.reshape(-1, 1)

    # 역변환: Standard -> MinMax -> (로그)
    if std_scaler is not None: 
        y_after_scaled  = std_scaler.inverse_transform(all_y)
        yhat_after_scaled = std_scaler.inverse_transform(all_yhat)

    if mm_scaler is not None:
        y_after_scaled  = mm_scaler.inverse_transform(y_after_scaled)
        yhat_after_scaled = mm_scaler.inverse_transform(yhat_after_scaled)

    # 최종 결과
    if logged:# 타겟 로그 변환 시: log1p -> expm1로 복원
        all_y = np.expm1(y_after_scaled).ravel()
        all_yhat = np.expm1(yhat_after_scaled).ravel()
    else:
        all_y = y_after_scaled.ravel()
        all_yhat = yhat_after_scaled.ravel()

    # 지표 계산
    mae  = skm.mean_absolute_error(all_y, all_yhat)
    rmse = float(np.sqrt(skm.mean_squared_error(all_y, all_yhat)))
    mape = skm.mean_absolute_percentage_error(all_y, all_yhat)
    r2   = skm.r2_score(all_y, all_yhat)

    return mae, rmse, mape, r2