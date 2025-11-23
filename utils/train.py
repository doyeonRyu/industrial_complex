"""
==============================================================================
File: train.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-11-19

Description: 하이브리드 혹은 단일 모델 학습
    Functions:
        - train: 한 epoch 동안 모델 학습
Note
    - train_model.py에서 실행
==============================================================================
"""
import torch
import torch.nn as nn

def train(loader, model1, model2, criterion, optimizer, device):
    """
    Function: train_model
        if 하이브리드 모델 (CNN + 시계열 데이터 처리 모델)인 경우
            - model1: CNN 모델
            - model2: 시계열 데이터 처리 모델
        else (단일 시계열 데이터 처리 모델인 경우)
            - model1: None
            - model2: 시계열 데이터 처리 모델
        - 모델을 한 epoch 동안 학습
    Parameters:
        - loader: DataLoader, 학습 데이터 로더
        - model1: CNN 모델 (하이브리드 모델인 경우) or None (단일 모델인 경우)
        - model2: 시계열 데이터 처리 모델 (LSTM, Transformer 등)
        - criterion: 손실 함수
        - optimizer: 최적화 알고리즘
        - device: torch.device, 모델과 데이터를 올릴 디바이스 (cpu or cuda)
    Returns:
        - epoch_loss: float, 전체 배치에 대한(한 epoch 동안의) 평균 손실
    """
    # 1. 모델을 학습 모드로 전환
    if model1 is not None:
        model1.train()
    model2.train()
    
    total_loss, total_cnt = 0.0, 0 # epoch 동안의 누적 손실과 샘플 수

    # 2. DataLoader에서 배치 단위로 데이터 로드
    for batch in loader:
        xb, yb = batch # 입력 데이터와 타깃 데이터 분리
        xb = xb.to(device) # 입력 데이터
        yb = yb.to(device) # 타깃 데이터

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
            # teacher forcing 적용
            if yb.dim() == 3: # [B(배치 크기), T(출력 시퀀스 길이), F(출력 피처 수)]
                tgt_input = yb[:, :-1, :] # 디코더 입력 (마지막 타임스텝 제외) # 형태: [B, T-1, F]
                tgt_target = yb[:, 1:, :] # 디코더 타깃 (처음 타임스텝 제외) # 형태: [B, T-1, F]
            else: # yb.dim() == 2: [B, T] 형태 (출력 피처가 1개)
                # 디코더 입력/타깃 차원 보정 (2D -> 3D)
                yb = yb.unsqueeze(-1)
                tgt_input = yb[:, :-1, :] # 디코더 입력 (마지막 타임스텝 제외) # 형태: [B, T-1, 1]
                tgt_target = yb[:, 1:, :] # 디코더 타깃 (처음 타임스텝 제외) # 형태: [B, T-1, 1]
            
            # src: encoder 입력, tgt_input: decoder 입력
            # src 형태: [B, L, F], tgt_input 형태: [B, T-1, F]
            yhat = model2(src, tgt_input) # decoder 출력 형태: [B, T-1, F]

            # 타깃 차원 보정 (3D -> 2D) | [B, T-1, F] vs [B, T-1, 1] 정렬
            if yhat.dim() > tgt_target.dim() and yhat.size(-1) == 1:
                yhat = yhat.squeeze(-1) # [B, T, 1] -> [B, T]로 변환하여 tgt_target ([B, T])과 맞춤
            if tgt_target.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
                tgt_target = tgt_target.unsqueeze(-1) # (N,) -> (N, 1)

            loss = criterion(yhat, tgt_target)
        
        # 2-2) encoder-only 구조일 때
        elif isinstance(model2, torch.nn.TransformerEncoder) or hasattr(model2, "encoder"):
            # forward(src) 형태
            yhat = model2(src)
            # 타깃 차원 보정 (1D -> 2D) | [B] vs [B,1] 정렬
            if yhat.dim() > yb.dim() and yhat.size(-1) == 1:
                yhat = yhat.squeeze(-1) # [B, T, 1] -> [B, T]로 변환하여 yb ([B, T])와 맞춤
            if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
                yb = yb.unsqueeze(-1) # (N,) -> (N, 1)
            loss = criterion(yhat, yb)

        # 2-3) 일반 RNN류 (LSTM, GRU 등)
        else:
            yhat = model2(src)
            # 타깃 차원 보정 (1D -> 2D) | [B] vs [B,1] 정렬
            if yhat.dim() > yb.dim() and yhat.size(-1) == 1:
                yhat = yhat.squeeze(-1) # [B, T, 1] -> [B, T]로 변환하여 yb ([B, T])와 맞춤
            if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
                yb = yb.unsqueeze(-1) # (N,) -> (N, 1)
            loss = criterion(yhat, yb)

        # 3) 역전파
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # 4) 배치 손실 누적
        bs = xb.size(0) # 배치 크기
        total_loss += loss.item() * bs # 배치 손실의 합
        total_cnt += bs # 배치 샘플 수 누적
        
    return total_loss / max(total_cnt, 1) # epoch 평균 손실

def train_longformer(loader, model1, model2, criterion, optimizer, device):
    # 1. 모델을 학습 모드로 전환
    if model1 is not None:
        model1.train()  
    model2.train()
    
    total_loss, total_cnt = 0.0, 0 # epoch 동안의 누적 손실과 샘플 수

    # 2. DataLoader에서 배치 단위로 데이터 로드
    for batch in loader:
        x_enc, x_dec, yb, x_mark_enc, x_mark_dec = batch
        x_enc = x_enc.to(device)
        x_dec = x_dec.to(device)
        yb= yb.to(device)
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
        y_hat = model2(src, x_mark_enc, x_dec, x_mark_dec)
        if model2.__class__.__name__ == 'Autoformer':
            y_hat = y_hat[:, :, -1:]  # (B, pred_len, 1) # autoformer일 때만
            if yb.dim() == 2:  
                yb = yb.unsqueeze(-1)

        loss = criterion(y_hat, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x_enc.size(0)
        total_loss += loss.item() * bs
        total_cnt += bs

    return total_loss / max(total_cnt, 1)
