"""
==============================================================================
File: evaluate_metrics.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-11-19

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
        xb, yb = batch
        xb = xb.to(device)  # 입력 윈도우
        yb = yb.to(device)  # 예측 구간

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

@torch.no_grad()
def evaluate_longformer(loader, model1, model2, criterion, device, output_window):
    # 1. 모델을 평가 모드로 전환
    if model1 is not None:
        model1.eval()  
    model2.eval()
    
    total_loss, total_cnt = 0.0, 0 # epoch 동안의 누적 손실과 샘플 수

    for batch in loader:
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
            yhat = yhat[:, :, -1:]  # (B, pred_len, 1) # autoformer일 때만
            if yb.dim() == 2:  
                yb = yb.unsqueeze(-1)

        loss = criterion(yhat, yb)

        bs = x_enc.size(0)
        total_loss += loss.item() * bs
        total_cnt += bs

    return total_loss / max(total_cnt, 1)