"""
==============================================================================
File: metrics.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-11-19

Description: 모델 평가 지표 계산
    Functions:
        - metrics: MAE, RMSE, MAPE, R² 지표 계산
Note
    - n04_evaluate_and_visualize.py에서 실행
==============================================================================
"""
import numpy as np
import sklearn.metrics as skm
import torch
import torch.nn as nn
"""
추가 평가 지표 함수
"""
def symmetric_mape(y_true, y_pred):
    """
    Function: sMAPE (symmetric Mean Absolute Percentage Error)
        - 실제값과 예측값의 평균으로 나눔
    Parameters:
        - y_true: 실제값 배열
        - y_pred: 예측값 배열   
    Returns:
        - sMAPE 값 (백분율)
    """
    numerator = np.abs(y_pred - y_true) # 예측 오차의 절댓값
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 # 실제값과 예측값의 절댓값 평균
    
    # 분모가 0인 경우 제외
    mask = denominator > 1e-8
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(numerator[mask] / denominator[mask]) * 100

def compute_peak_metrics_per_cycle(y_true, y_pred, tolerance=1):
    """
    각 예측 사이클(한 output_window)에 대해 PAPE, HR, lag 계산
    """
    true_peak_idx = np.argmax(y_true)
    pred_peak_idx = np.argmax(y_pred)

    true_peak_val = y_true[true_peak_idx]
    pred_peak_val = y_pred[pred_peak_idx]

    pape = abs(pred_peak_val - true_peak_val) / (true_peak_val + 1e-9) * 100
    hr = 1 if abs(pred_peak_idx - true_peak_idx) <= tolerance else 0
    lag = pred_peak_idx - true_peak_idx
    return pape, hr, lag

"""
모델 평가 지표 계산 함수
- MAE, RMSE, MAPE, R² 계산
"""
@torch.no_grad()
def metrics(best_model1_path, best_model2_path, model1, model2,
            data_loader, scaler, device, logged: bool = False, output_window=32, threshold=0.1):
    """
    Function: metrics
        - 하이브리드 or 단일 시계열 예측 모델 평가 지표 계산
    Parameters:
        - best_model1_path: str or None, 저장된 CNN 모델 경로 (None 가능)
        - best_model2_path: str, 저장된 시계열 데이터 처리 모델 경로
        - model1: CNN 모델 (None 가능)
        - model2: 시계열 데이터 처리 모델 (LSTM or Transformer)
        - data_loader: DataLoader, 평가 데이터 로더
        - std_scaler: StandardScaler 객체, 표준화 스케일러
        - mm_scaler: MinMaxScaler 객체, 정규화 스케일러
        - device: torch.device, 모델과 데이터를 올릴 디바이스 (cpu or cuda)
        - target_idx: int or None, 역변환 시 사용할 타겟 피처 인덱스 (None이면 전체 피처 사용)
        - logged: bool, 로그 변환된 데이터인지 여부
        - threshold: float, MAPE 계산 시 0 나누기 방지를 위한 임계값
    Returns:
        - mae: float, Mean Absolute Error
        - rmse: float, Root Mean Squared Error
        - mape: float, Mean Absolute Percentage Error
        - mape_filtered: float, 임계값 기반 필터링된 MAPE
        - smape: float, symmetric Mean Absolute Percentage Error
        - r2: float, R² (결정 계수)
        - pape: float, Peak Absolute Percentage Error
        - hr: float, Hit Rate
        - lag: float, Lag Index
    """

    # 1. 모델 로드 및 평가 모드 전환
    if model1 is not None:
        model1.load_state_dict(torch.load(best_model1_path, map_location=device))
        model1.to(device)
        model1.eval()
    model2.load_state_dict(torch.load(best_model2_path, map_location=device))
    model2.to(device)
    model2.eval()

    # 2. 저장 리스트 초기화
    all_y, all_yhat = [], []
    pape_list, hr_list, lag_list = [], [], []

    # 3. 배치별 평가
    for batch in data_loader:
        xb, yb = batch
        xb = xb.to(device) # 입력 윈도우
        yb = yb.to(device) # 예측 구간

        # ====================================================
        # 1) CNN 모델이 있는 경우 (하이브리드) 
        # ====================================================

        #    CNN 입력 형태: [B, F, L] (배치 크기, 피처 수, 시퀀스 길이) -> CNN 출력 형태 [B, L, F]
        if model1 is not None: 
            # CNN 입력 형태로 변환 [B, L, F] -> [B, F, L]
            if xb.dim() == 3:
                xb = xb.permute(0, 2, 1) # L, F 위치 변환
            elif xb.dim() == 2: # L=1인 경우(입력 피처가 1개) [B, F] 형태
                xb = xb.unsqueeze(-1) # [B, F] -> [B, F, 1] # L 차원 추가
            
            # 1) model1 forward 
            md1_out = model1(xb) # md1_out 형태: [B, L, F]
            src = md1_out # CNN 출력값을 model2의 입력으로 사용
        else:
            src = xb # CNN이 없는 경우 원본 입력 사용

        # ====================================================
        # 2) model2 forward
        # ====================================================
        
        # 2-1) encoder-decoder 구조일 때
        if hasattr(model2, "decoder") or hasattr(model2, "dec_embedding"):
            # Autoregressive 방식으로 디코더 입력 생성

            #    디코더 초기 입력을 x_36 (src의 마지막 시점)으로 설정
            #    src: (B, L_enc, F). src의 마지막 시점의 output_dim 개수 피처를 사용한다고 가정
            #    모델의 output_dim에 맞게 src의 마지막 값 중 해당 차원만 선택
            output_dim = model2.output_dim if hasattr(model2, 'output_dim') else yb.size(-1) 
            
            # x_36 (src의 마지막 시점 값)을 가져옴: (B, 1, output_dim)
            # src의 마지막 시점의 첫 output_dim 피처를 사용
            dec_input = src[:, -1:, :output_dim].clone().to(device)

            outputs = []

            # autoregressive decoding loop 횟수를 output_window (32회)로 변경
            for _ in range(output_window): # 32번 반복 (t=37부터 t=68까지)
                
                y_pred = model2(src, dec_input) 
                next_pred = y_pred[:, -1:, :]  # 새로 예측된 마지막 시점 (t=37, t=38, ...)
                outputs.append(next_pred)
                dec_input = torch.cat([dec_input, next_pred], dim=1)  # 예측 누적

            yhat = torch.cat(outputs, dim=1) # [B, output_window, F] # 32개 시점의 예측: y_37부터 y_68

            tgt_target = yb if yb.dim() == 3 else yb.unsqueeze(-1)  # [B, output_window, F] # 32개의 실제값

        # 2-2) encoder-only 구조일 때
        elif isinstance(model2, torch.nn.TransformerEncoder) or hasattr(model2, "encoder"):
            yhat = model2(src)
            tgt_target = yb

        # 2-3) 일반 RNN류 (LSTM, GRU 등)
        else:
            yhat = model2(src)
            tgt_target = yb

        # -----------------------------------------------------
        # 차원 보정
        if yhat.dim() == 3 and yhat.size(-1) == 1:
            yhat = yhat.squeeze(-1)
        if tgt_target.dim() == 3 and tgt_target.size(-1) == 1:
            tgt_target = tgt_target.squeeze(-1)

        # -----------------------------------------------------
        # PAPE, HR, Lag (배치별 계산)
        tgt_np = tgt_target.detach().cpu().numpy()   # shape (B, 32)
        pred_np = yhat.detach().cpu().numpy()        # shape (B, 32)

        B = tgt_np.shape[0]

        for i in range(B):
            y_true_seq = tgt_np[i]    # (32,)
            y_pred_seq = pred_np[i]   # (32,)

            # true 피크가 0일 때는 계산 불가 → skip
            if np.max(y_true_seq) < 1e-6:
                continue

            pape, hr, lag = compute_peak_metrics_per_cycle(
                y_true_seq,
                y_pred_seq,
                tolerance=4
            )

            pape_list.append(pape)
            hr_list.append(hr)
            lag_list.append(lag)

        # -----------------------------------------------------
        # 예측값 및 실제값 저장
        all_y.append(tgt_target.detach().cpu().numpy())
        all_yhat.append(yhat.detach().cpu().numpy())

    # 4. 전체 배치 결합
    all_y = np.concatenate(all_y, axis=0)
    all_yhat = np.concatenate(all_yhat, axis=0)

    # 5. 2D 형태로 변환
    all_y = all_y.reshape(-1, 1)
    all_yhat = all_yhat.reshape(-1, 1)

    # 6. 역변환
    if scaler is not None:
        all_y = scaler.inverse_transform(all_y)
        all_yhat = scaler.inverse_transform(all_yhat)

    if logged:
        all_y = np.expm1(np.maximum(all_y, 0)).ravel()
        all_yhat = np.expm1(np.maximum(all_yhat, 0)).ravel()
    else:
        all_y = all_y.ravel()
        all_yhat = all_yhat.ravel()

    # 7. 평가 지표 계산
    mae  = skm.mean_absolute_error(all_y, all_yhat)
    rmse = np.sqrt(skm.mean_squared_error(all_y, all_yhat))
    mape = skm.mean_absolute_percentage_error(all_y, all_yhat)

    # 필터링된 MAPE (0 근처 제외)
    mask = all_y > threshold
    mape_filtered = skm.mean_absolute_percentage_error(all_y[mask], all_yhat[mask])
    smape = symmetric_mape(all_y, all_yhat)
    r2 = skm.r2_score(all_y, all_yhat)

    # 8. 피크 관련 지표 평균
    pape = np.mean(pape_list)
    hr   = np.mean(hr_list)
    lag  = np.mean(lag_list)

    return mae, rmse, mape, mape_filtered, smape, r2, pape, hr, lag

"""
longformer 모델 평가 지표 계산 함수
- MAE, RMSE, MAPE, R² 계산
"""
@torch.no_grad()
def metrics_longformer(best_model1_path, best_model2_path, model1, model2,
                    data_loader, scaler, device, logged: bool = False, threshold=0.1):
    """
    Function: metrics_longformer
        - longformer 모델 평가 지표 계산
    Parameters:
        - best_model1_path1: CNN 모델 가중치 경로
        - best_model2_path: longformer 모델 가중치 경로
        - model1: CNN 모델 객체
        - model2: longformer 모델 객체
        - data_loader: 평가용 데이터 로더
        - scaler: 스케일러 객체 (None 가능)
        - device: 연산 장치 (CPU/GPU)
        - logged: 로그 변환 여부
        - threshold: MAPE 계산 시 필터링 임계값
    Returns:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - mape: Mean Absolute Percentage Error
        - mape_filtered: 필터링된 MAPE
        - smape: symmetric Mean Absolute Percentage Error
        - r2: R² Score
        - pape: Peak Absolute Percentage Error
        - hr: Hit Rate
        - lag: Lag
    """
    # 1. 모델 로드 및 평가 모드 전환
    if model1 is not None:
        model1.load_state_dict(torch.load(best_model1_path, map_location=device))
        model1.to(device)
        model1.eval()
    model2.load_state_dict(torch.load(best_model2_path, map_location=device))
    model2.to(device)
    model2.eval()

    # 2. 저장 리스트 초기화
    all_y, all_yhat = [], []
    pape_list, hr_list, lag_list = [], [], []

    # 3. 배치별 평가
    for batch in data_loader:
        x_enc, x_dec, yb, x_mark_enc, x_mark_dec = batch

        x_enc = x_enc.to(device)
        x_dec = x_dec.to(device)
        yb = yb.to(device)
        x_mark_enc = x_mark_enc.to(device)
        x_mark_dec = x_mark_dec.to(device)

        # ====================================================
        # 1) CNN 모델이 있는 경우 (하이브리드) 
        # ====================================================

        #    CNN 입력 형태: [B, F, L] (배치 크기, 피처 수, 시퀀스 길이) -> CNN 출력 형태 [B, L, F]
        if model1 is not None: 
            # CNN 입력 형태로 변환 [B, L, F] -> [B, F, L]
            if x_enc.dim() == 3:
                x_enc = x_enc.permute(0, 2, 1) # L, F 위치 변환
            elif x_enc.dim() == 2: # L=1인 경우(입력 피처가 1개) [B, F] 형태
                x_enc = x_enc.unsqueeze(-1) # [B, F] -> [B, F, 1] # L 차원 추가
            
            # 1) model1 forward 
            md1_out = model1(x_enc) # md1_out 형태: [B, L, F]
            src = md1_out # CNN 출력값을 model2의 입력으로 사용
        else:
            src = x_enc # CNN이 없는 경우 원본 입력 사용
        
        
        # ====================================================
        # 2) model2 forward
        # ====================================================
        yhat = model2(src, x_mark_enc, x_dec, x_mark_dec)
        
        if model2.__class__.__name__ == 'Autoformer':
            yhat = yhat[:, :, -1:] # (B, pred_len, 1) # autoformer일 때만
        tgt_target = yb

        # 차원 보정 
        if yhat.dim() == 3 and yhat.size(-1) == 1:
            yhat = yhat.squeeze(-1)
        if tgt_target.dim() == 3 and tgt_target.size(-1) == 1:
            tgt_target = tgt_target.squeeze(-1)

        # PAPE, HR, Lag (배치별 계산)
        tgt_np = tgt_target.detach().cpu().numpy()   # shape (B, 32)
        pred_np = yhat.detach().cpu().numpy()        # shape (B, 32)

        B = tgt_np.shape[0]

        for i in range(B):
            y_true_seq = tgt_np[i]    # (32,)
            y_pred_seq = pred_np[i]   # (32,)

            # true 피크가 0일 때는 계산 불가 → skip
            if np.max(y_true_seq) < 1e-6:
                continue

            pape, hr, lag = compute_peak_metrics_per_cycle(
                y_true_seq,
                y_pred_seq,
                tolerance=4
            )

            pape_list.append(pape)
            hr_list.append(hr)
            lag_list.append(lag)

        # -----------------------------------------------------
        # 예측값 및 실제값 저장
        all_y.append(tgt_target.detach().cpu().numpy())
        all_yhat.append(yhat.detach().cpu().numpy())

    # 4. 전체 배치 결합
    all_y = np.concatenate(all_y, axis=0)
    all_yhat = np.concatenate(all_yhat, axis=0)

    # 5. 2D 형태로 변환
    all_y = all_y.reshape(-1, 1)
    all_yhat = all_yhat.reshape(-1, 1)

    # 6. 역변환
    if scaler is not None:
        all_y = scaler.inverse_transform(all_y)
        all_yhat = scaler.inverse_transform(all_yhat)

    if logged:
        all_y = np.expm1(np.maximum(all_y, 0)).ravel()
        all_yhat = np.expm1(np.maximum(all_yhat, 0)).ravel()
    else:
        all_y = all_y.ravel()
        all_yhat = all_yhat.ravel()

    # 7. 평가 지표 계산
    mae  = skm.mean_absolute_error(all_y, all_yhat)
    rmse = np.sqrt(skm.mean_squared_error(all_y, all_yhat))
    mape = skm.mean_absolute_percentage_error(all_y, all_yhat)

    # 필터링된 MAPE (0 근처 제외)
    mask = all_y > threshold
    mape_filtered = skm.mean_absolute_percentage_error(all_y[mask], all_yhat[mask])
    smape = symmetric_mape(all_y, all_yhat)
    r2 = skm.r2_score(all_y, all_yhat)

    # 8. 피크 관련 지표 평균
    if len(pape_list) == 0:
        pape = np.nan
        hr = np.nan
        lag  = np.nan
    else:
        pape = np.mean(pape_list)
        hr   = np.mean(hr_list)
        lag  = np.mean(lag_list)

    return mae, rmse, mape, mape_filtered, smape, r2, pape, hr, lag
