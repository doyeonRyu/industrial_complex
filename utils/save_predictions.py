"""
==============================================================================
File: evaluate_metrics.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-11-19

Description: 모델 평가 및 지표 계산
    Functions:
        - save_predictions_to_excel: LSTM, Transformer
        - save_predictions_to_excel_longformer: Informer, Autoformer
Note
    - n04_evaluate_and_visualize.py에서 실행
==============================================================================
"""

import torch
import os
import numpy as np
import pandas as pd

@torch.no_grad()
def save_predictions_to_excel(
    best_model1_path, best_model2_path, model1, model2,
    save_first_path, save_full_path,
    data_loader, df, scaler,
    device,
    input_window=36, output_window=32,
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
        - scaler: 스케일러 객체 (None 가능)
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
        arr_2d = arr.reshape(-1, 1)

        # Step 1: inverse scaling
        if scaler is not None:
            arr_2d = scaler.inverse_transform(arr_2d)

        # Step 2: inverse log transform
        if logged:
            arr_2d = np.expm1(np.maximum(arr_2d, 0))

        return arr_2d.reshape(-1)

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
    first_df["datetime"] = first_df["datetime"].dt.tz_localize(None)
    full_df["datetime"] = full_df["datetime"].dt.tz_localize(None)

    first_df.to_excel(save_first_path, index=False)
    full_df.to_excel(save_full_path, index=False)

    print(f"[excel 저장] 첫 번째 윈도우 전체 예측 결과 저장: {save_first_path}")
    print(f"[excel 저장] 전체 시점 저장: {save_full_path}")

    return first_df, full_df

@torch.no_grad()
def save_predictions_to_excel_longformer(
    best_model1_path, best_model2_path, model1, model2,
    save_first_path, save_full_path,
    data_loader, df, scaler,
    device,
    input_window=36, label_len=18, output_window=32,
    logged: bool = False
):
    """
    Function: save_predictions_to_excel
        - 모델 예측값을 엑셀 파일로 저장
    Parameters:
        - best_model1_path: CNN 모델 가중치 경로 (None 가능)
        - best_model2_path: longformer 모델 가중치 경로
        - model1: CNN 모델 객체 (None 가능)
        - model2: longformer 모델 객체
        - save_first_path: 첫 번째 윈도우 예측값 저장 경로
        - save_full_path: 전체 예측값 저장 경로
        - data_loader: 평가용 데이터 로더
        - df: 원본 데이터프레임 (인덱스용)
        - scaler: 스케일러 객체 (None 가능)
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
    for i, (x_enc, x_dec, yb, x_mark_enc, x_mark_dec) in enumerate(data_loader):
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

        if model2.__class__.__name__ == 'Autoformer': # Autoformer
            if yhat.dim() == 3:
                yhat = yhat[:, :, -1:]  # (B, pred_len, 1)

        # 차원 보정
        if yhat.dim() == 3 and yhat.size(-1) == 1:
            yhat = yhat.squeeze(-1)
        if yb.dim() == 3 and yb.size(-1) == 1:
            yb = yb.squeeze(-1)

        # numpy 변환
        yhat_np = yhat.detach().cpu().numpy()
        y_true_np = yb.detach().cpu().numpy()

        # 3) 첫번째 윈도우 저장
        if first_window_pred is None:
            first_window_pred = yhat_np[0].copy()
            first_window_true = y_true_np[0].copy() 

            if model2.__class__.__name__ == 'Informer': # Informer
                first_window_idx = np.arange(
                    input_window + label_len,
                    input_window + label_len + len(first_window_pred)
                )
            else: # Autoformer
                first_window_idx = np.arange(
                    input_window,
                    input_window + len(first_window_pred)
                )

        # 4) 각 배치의 첫 번째 시점 예측값 저장
        for b in range(yhat_np.shape[0]):
            time_idx = i + b + input_window + label_len
            if time_idx < len(df):
                preds_firststep.append(yhat_np[b, 0])  # 첫 step 예측
                trues_firststep.append(y_true_np[b, 0])
                time_idxs.append(time_idx)

    # 5. 역변환 함수 정의
    def inverse_transform(arr):
        arr_2d = arr.reshape(-1, 1)

        if scaler is not None:
            arr_2d = scaler.inverse_transform(arr_2d)

        if logged:
            arr_2d = np.expm1(np.maximum(arr_2d, 0))

        return arr_2d.reshape(-1)

    # 6. 역변환 수행
    preds_inv = inverse_transform(np.array(preds_firststep).reshape(-1, 1))
    trues_inv = inverse_transform(np.array(trues_firststep).reshape(-1, 1))
    first_pred_inv = inverse_transform(first_window_pred.reshape(-1, 1)).ravel()
    first_true_inv = inverse_transform(first_window_true.reshape(-1, 1)).ravel()

    # 7. 데이터프레임 생성
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

    # 8. 엑셀 파일로 저장
    first_df["datetime"] = first_df["datetime"].dt.tz_localize(None)
    full_df["datetime"] = full_df["datetime"].dt.tz_localize(None)

    first_df.to_excel(save_first_path, index=False)
    full_df.to_excel(save_full_path, index=False)

    print(f"[excel 저장] 첫 번째 윈도우 전체 예측 결과 저장: {save_first_path}")
    print(f"[excel 저장] 전체 시점 저장: {save_full_path}")

    return first_df, full_df
