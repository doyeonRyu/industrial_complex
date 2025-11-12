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
    - 산업체마다 데이터 특성이 다를 수 있음. 수정 필요 - 현재는 메인텍 2공장의 포멧을 따름
==============================================================================
"""
import torch
import os
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from utils.setup import _to_device

# ==============================================================================
# 모델 평가 함수
# - valid 또는 test 데이터셋에 대한 손실 계산
# ==============================================================================
@torch.no_grad()
def evaluate(loader, model1, model2, criterion, device, output_window):
    """
    Function: evaluate
        - 하이브리드 혹은 단일 예측 모델을 검증 모드로 한 epoch 평가
    parameters:
        - loader: DataLoader, 평가 데이터 로더
        - model1: CNN 모델 (None 가능)
        - model2: 시계열 데이터 처리 모델 (LSTM or Transformer)
        - criterion: 손실 함수
        - device: torch.device, 모델과 데이터를 올릴 디바이스 (cpu or cuda)
        - output_window: int, 출력 시퀀스 길이
    Returns:
        - epoch_loss: float, 전체 배치에 대한(한 epoch 동안의) 평균 손실
    """
    # 1. 모델을 평가 모드로 전환
    if model1 is not None:
        model1.eval()
    model2.eval()

    # 2. 전체 배치에 대한 누적 손실과 샘플 수 초기화
    total_sum, total_cnt = 0.0, 0

    # 3. 데이터 로더에서 배치 단위로 반복
    for batch in loader:
        xb, yb = _to_device(batch, device) # 배치를 CUDA 장치로 변환

        # CNN 모델이 있는 경우 (하이브리드) 
        #    입력 형태: [B, F, L] (배치 크기, 피처 수, 시퀀스 길이) -> CNN 출력 형태 [B, L, F]
        if model1 is not None: 
            # CNN 입력 형태로 변환 [B, F, L]
            # [B(배치 사이즈), L(input_window 길이), F(feature 수)] -> [B, F, L]
            if xb.dim() == 3:
                # L, F 위치 변환
                xb = xb.permute(0, 2, 1) # [B, L, F] -> [B, F, L]
            elif xb.dim() == 2: # L=1인 경우(입력 피처가 1개) [B, F] 형태
                xb = xb.unsqueeze(-1) # [B, F] -> [B, F, 1] # L 차원 추가
            
            # 1) model1 forward 
            md1_out = model1(xb) # xb 형태: [B, F, L] -> md1_out 형태: [B, L, F]
            src = md1_out # CNN 출력값을 model2의 입력으로 사용 # 형태 [B, L, F]
        else:
            src = xb # CNN이 없는 경우 원본 입력 사용 # 형태 [B, L, F]

        # 2) model2 forward
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

            yhat = torch.cat(outputs, dim=1) # 32개 시점의 예측: y_37부터 y_68

            tgt_target = yb if yb.dim() == 3 else yb.unsqueeze(-1) # 32개의 실제값

            loss = criterion(yhat, tgt_target)

        # 2-2) encoder-only 구조일 때
        elif isinstance(model2, torch.nn.TransformerEncoder) or hasattr(model2, "encoder"):
            yhat = model2(src)
            # yb: 해당 시점의 실제값
            loss = criterion(yhat, yb)

        # 2-3) 일반 RNN류 (LSTM, GRU 등)
        else:
            yhat = model2(src)
            # yb: 해당 시점의 실제값
            loss = criterion(yhat, yb)

        # 3) 배치 손실 누적
        bs = xb.size(0)
        total_sum += loss.item() * bs
        total_cnt += bs

    return total_sum / max(total_cnt, 1)

# ==============================================================================
# 추가 평가 지표 함수
# ==============================================================================
def PAPE(y_true, y_pred):
    """
    Function: PAPE (Peak Absolute Percentage Error)
        - 실제값과 예측값의 피크 값 차이를 백분율로 계산
    Parameters:
        - y_true: 실제값 배열
        - y_pred: 예측값 배열
    Returns:
        - PAPE 값 (백분율)
    """
    peak_true = y_true.max()
    peak_pred = y_pred.max()
    return abs(peak_pred - peak_true) / peak_true * 100

def HR(y_true, y_pred, tolerance=1):
    """
    Function: HR (Hit Rate)
        - 실제값과 예측값의 피크 위치가 허용 오차 내에 있는지 확인
    Parameters:
        - y_true: 실제값 배열
        - y_pred: 예측값 배열
        - tolerance: 피크 위치 허용 오차 (타임스텝 단위; default=1)
    Returns:
        - HR 값 (0 또는 1)
    """
    true_peak_idx = np.argmax(y_true)
    pred_peak_idx = np.argmax(y_pred)
    return int(abs(pred_peak_idx - true_peak_idx) <= tolerance)

def lag_index(y_true, y_pred):
    '''
    Function:
        - 예측 시계열이 실제 시계열보다 얼마나 지연되었는지 계산
    Parameters:
        - y_true: 실제값 배열
        - y_pred: 예측값 배열
    Return:
        - lag (양수면 y_pred가 뒤로 밀림, 단위=타임스텝)
    '''
    y_true = y_true - np.mean(y_true)
    y_pred = y_pred - np.mean(y_pred)
    corr = np.correlate(y_true, y_pred, mode='full')
    lags = np.arange(-len(y_true)+1, len(y_true))
    lag = lags[np.argmax(corr)]
    return lag

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



def compute_peak_metrics_per_cycle(y_true_seq, y_pred_seq, tolerance=1):
    """
    각 예측 사이클(한 output_window)에 대해 PAPE, HR, lag 계산
    """
    true_peak_idx = np.argmax(y_true_seq)
    pred_peak_idx = np.argmax(y_pred_seq)

    true_peak_val = y_true_seq[true_peak_idx]
    pred_peak_val = y_pred_seq[pred_peak_idx]

    pape = abs(pred_peak_val - true_peak_val) / (true_peak_val + 1e-8) * 100
    hr = int(abs(pred_peak_idx - true_peak_idx) <= tolerance)
    lag = pred_peak_idx - true_peak_idx
    return pape, hr, lag




"""
모델 평가 지표 계산 함수
- MAE, RMSE, MAPE, R² 계산
"""
@torch.no_grad()
def metrics(best_model1_path, best_model2_path, model1, model2,
            data_loader,
            std_scaler, mm_scaler,
            device,
            target_idx: int | None = None,
            logged: bool = False,
            input_window=36, output_window=32, threshold=0.1):
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
        - input_window: int, 입력 시퀀스 길이
        - output_window: int, 출력 시퀀스 길이
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

    import numpy as np
    import torch
    import sklearn.metrics as skm

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
        xb, yb = _to_device(batch, device)  # xb: 입력윈도우, yb: 예측 구간

        # CNN 모델이 있는 경우 (하이브리드) 
        #    입력 형태: [B, F, L] (배치 크기, 피처 수, 시퀀스 길이) -> CNN 출력 형태 [B, L, F]
        if model1 is not None: 
            # CNN 입력 형태로 변환 [B, F, L]
            # [B(배치 사이즈), L(input_window 길이), F(feature 수)] -> [B, F, L]
            if xb.dim() == 3:
                # L, F 위치 변환
                xb = xb.permute(0, 2, 1) # [B, L, F] -> [B, F, L]
            elif xb.dim() == 2: # L=1인 경우(입력 피처가 1개) [B, F] 형태
                xb = xb.unsqueeze(-1) # [B, F] -> [B, F, 1] # L 차원 추가
            
            # 1) model1 forward 
            md1_out = model1(xb) # xb 형태: [B, F, L] -> md1_out 형태: [B, L, F]
            src = md1_out # CNN 출력값을 model2의 입력으로 사용 # 형태 [B, L, F]
        else:
            src = xb # CNN이 없는 경우 원본 입력 사용 # 형태 [B, L, F]

        # 2) model2 forward
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
        pape, hr, lag = compute_peak_metrics_per_cycle(
            tgt_target.detach().cpu().numpy().ravel(),
            yhat.detach().cpu().numpy().ravel(),
            tolerance=1
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
    if std_scaler is not None:
        all_y = std_scaler.inverse_transform(all_y)
        all_yhat = std_scaler.inverse_transform(all_yhat)

    if mm_scaler is not None:
        if target_idx is not None:
            n_features = mm_scaler.n_features_in_
            dummy_y = np.zeros((len(all_y), n_features))
            dummy_yhat = np.zeros((len(all_yhat), n_features))
            dummy_y[:, target_idx] = all_y.ravel()
            dummy_yhat[:, target_idx] = all_yhat.ravel()
            all_y = mm_scaler.inverse_transform(dummy_y)[:, target_idx:target_idx+1]
            all_yhat = mm_scaler.inverse_transform(dummy_yhat)[:, target_idx:target_idx+1]
        else:
            all_y = mm_scaler.inverse_transform(all_y)
            all_yhat = mm_scaler.inverse_transform(all_yhat)

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

@torch.no_grad()
def save_predictions_to_excel(
    best_model1_path, best_model2_path, model1, model2,
    save_first_path, save_full_path,
    data_loader, df, std_scaler, mm_scaler,
    device,
    input_window=36, output_window=32,
    target_idx: int | None = None,
    logged: bool = False
):
    """
    Function: save_predictions_to_excel
        - 모델 예측값을 엑셀 파일로 저장
    Parameters:
        - best_model1_path: CNN 모델 가중치 경로 (None 가능)
        - best_model2_path: LSTM/Transformer 모델 가중치 경로
        - model1: CNN 모델 객체 (None 가능)
        - model2: LSTM/Transformer 모델 객체
        - save_first_path: 첫 번째 윈도우 예측값 저장 경로
        - save_full_path: 전체 예측값 저장 경로
        - data_loader: 평가용 데이터 로더
        - df: 원본 데이터프레임 (인덱스용)
        - std_scaler: 표준화 스케일러 객체 (None 가능)
        - mm_scaler: Min-Max 스케일러 객체 (None 가능)
        - device: 연산 장치 (CPU/GPU)
        - input_window: 입력 윈도우 크기
        - output_window: 출력 윈도우 크기
        - target_idx: 타겟 피처 인덱스 (None 가능)
        - logged: 로그 변환 여부
    Returns:
        - None (엑셀 파일로 저장)
    """

    # 1. 디렉토리 생성
    os.makedirs(os.path.dirname(save_first_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_full_path), exist_ok=True)

    # 2. 모델 로드 및 평가 모드 전환
    if model1 is not None:
        model1.load_state_dict(torch.load(best_model1_path, map_location=device))
        model1.to(device)
        model1.eval()
    model2.load_state_dict(torch.load(best_model2_path, map_location=device))
    model2.to(device)
    model2.eval()

    # 3. 예측값 및 실제값 저장 리스트 초기화
    preds_firststep, trues_firststep, time_idxs = [], [], []
    first_window_pred = None
    first_window_true = None
    first_window_idx = None

    # 4. 배치별 예측 수행
    with torch.no_grad():
        for i, (xb, yb) in enumerate(data_loader):
            xb, yb = xb.to(device), yb.to(device)

            # CNN 모델이 있는 경우 (하이브리드) 
            #    입력 형태: [B, F, L] (배치 크기, 피처 수, 시퀀스 길이) -> CNN 출력 형태 [B, L, F]
            if model1 is not None: 
                # CNN 입력 형태로 변환 [B, F, L]
                # [B(배치 사이즈), L(input_window 길이), F(feature 수)] -> [B, F, L]
                if xb.dim() == 3:
                    # L, F 위치 변환
                    xb = xb.permute(0, 2, 1) # [B, L, F] -> [B, F, L]
                elif xb.dim() == 2: # L=1인 경우(입력 피처가 1개) [B, F] 형태
                    xb = xb.unsqueeze(-1) # [B, F] -> [B, F, 1] # L 차원 추가
                
                # 1) model1 forward 
                md1_out = model1(xb) # xb 형태: [B, F, L] -> md1_out 형태: [B, L, F]
                src = md1_out # CNN 출력값을 model2의 입력으로 사용 # 형태 [B, L, F]
            else:
                src = xb # CNN이 없는 경우 원본 입력 사용 # 형태 [B, L, F]

            # 2) model2 forward
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

            # 3) 차원 보정
            if yhat.dim() == 3 and yhat.size(-1) == 1:
                yhat = yhat.squeeze(-1)
            if tgt_target.dim() == 3 and tgt_target.size(-1) == 1:
                tgt_target = tgt_target.squeeze(-1)

            yhat_np = yhat.detach().cpu().numpy()
            y_true_np = tgt_target.detach().cpu().numpy()

            # 5. 첫 번째 윈도우의 전체 예측값 저장
            if first_window_pred is None:
                first_window_pred = yhat_np[0].copy()
                first_window_true = y_true_np[0].copy()
                first_window_idx = np.arange(input_window, input_window + len(first_window_pred))

            # 6. 각 배치의 첫 번째 시점 예측값 저장
            for b in range(yhat_np.shape[0]):
                time_idx = i + b + input_window
                if time_idx < len(df):
                    preds_firststep.append(yhat_np[b, 0])  # 첫 step 예측
                    trues_firststep.append(y_true_np[b, 0])
                    time_idxs.append(time_idx)

    # 7. 역변환 함수 정의
    def inverse_transform(arr):
        if std_scaler is not None:
            arr = std_scaler.inverse_transform(arr)
        if mm_scaler is not None:
            if target_idx is not None:
                n_features = mm_scaler.n_features_in_
                dummy = np.zeros((len(arr), n_features))
                dummy[:, target_idx] = arr.ravel()
                arr = mm_scaler.inverse_transform(dummy)[:, target_idx:target_idx + 1]
            else:
                arr = mm_scaler.inverse_transform(arr)
        if logged:
            arr = np.expm1(np.maximum(arr, 0))
        return arr

    # 8. 역변환 수행
    preds_inv = inverse_transform(np.array(preds_firststep).reshape(-1, 1))
    trues_inv = inverse_transform(np.array(trues_firststep).reshape(-1, 1))
    first_pred_inv = inverse_transform(first_window_pred.reshape(-1, 1)).ravel()
    first_true_inv = inverse_transform(first_window_true.reshape(-1, 1)).ravel()

    # 9. 데이터프레임 생성
    # 1) 첫 번째 윈도우 전체 결과
    first_df = pd.DataFrame({
        "datetime": pd.to_datetime(df["datetime"].iloc[first_window_idx].values),
        "actual(usage_kWh)": df["usage_kWh"].iloc[first_window_idx].values,
        "actual_inverse": first_true_inv,
        "predicted": first_pred_inv
    })

    # 2) 전체 시점 첫 예측값 결과
    full_df = pd.DataFrame({
        "datetime": pd.to_datetime(df["datetime"].iloc[time_idxs].values),
        "actual(usage_kWh)": df["usage_kWh"].iloc[time_idxs].values,
        "actual_inverse": trues_inv.ravel(),
        "predicted": preds_inv.ravel()
    })

    # 10. 엑셀 파일로 저장
    # 10. 엑셀 파일로 저장
    first_df["datetime"] = first_df["datetime"].dt.tz_localize(None)
    full_df["datetime"] = full_df["datetime"].dt.tz_localize(None)

    first_df.to_excel(save_first_path, index=False)
    full_df.to_excel(save_full_path, index=False)

    print(f"[excel 저장] 첫 번째 윈도우 전체 예측 결과 저장: {save_first_path}")
    print(f"[excel 저장] 전체 시점 저장: {save_full_path}")

    return first_df, full_df
