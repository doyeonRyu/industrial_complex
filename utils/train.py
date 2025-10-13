import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.setup import _to_device, SeqDataset, SeqDataset_single_model
        
def train(loader, model, criterion, optimizer, device):
    """
    Function: train
        - 모델을 한 epoch 동안 학습
    Parameters:
        - loader: DataLoader
            - 학습 데이터 로더
        - model: 학습할 모델
        - criterion: 손실 함수
        - optimizer: 옵티마이저
        - device: torch.device
    Returns:
        - float
            - 평균 학습 손실
    """
    model.train() # 모델 학습 모드로 전환
    total, n = 0.0, 0 # 손실 합계와 샘플 수 초기화

    # 배치 단위로 학습 
    for batch in loader: 
        xb, s_idx, meta, yb = _to_device(batch, device) # 배치를 장치에 맞게 변환

        # 모델의 foward 함수에서 입력 인자 개수에 따라 다르게 호출
        if hasattr(model, "forward") and model.forward.__code__.co_argcount >= 4:
            # 4개 이상이면 s_idx, meta도 전달 (4개 이상: self, x, station_idx, meta)
            yhat = model(xb, s_idx, meta)
        else: # 4개 미만이면 s_idx, meta는 None 
            yhat = model(xb)

        # 타깃 차원 보정 (1D -> 2D)
        if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
            yb = yb.unsqueeze(-1)

        loss = criterion(yhat, yb) # 손실 계산
        optimizer.zero_grad(set_to_none=True) # 옵티마이저 기울기 초기화
        loss.backward() # 역전파
        optimizer.step() # 파라미터 업데이트

        bs = xb.size(0) # 배치 크기
        total += loss.item() * bs # 손실 합계 갱신
        n += bs # 샘플 수 갱신
    return total / max(n, 1) # 평균 손실 반환

def train_randomSearch(loader, model, criterion, optimizer, device, grad_clip=None):
    """
    Function: train
        - 모델을 한 epoch 동안 학습
    Parameters:
        - loader: DataLoader
            - 학습 데이터 로더
        - model: 학습할 모델
        - criterion: 손실 함수
        - optimizer: 옵티마이저
        - device: torch.device
    Returns:
        - float
            - 평균 학습 손실
    """
    model.train() # 모델 학습 모드로 전환
    total, n = 0.0, 0 # 손실 합계와 샘플 수 초기화

    # 배치 단위로 학습 
    for batch in loader: 
        xb, s_idx, meta, yb = _to_device(batch, device) # 배치를 장치에 맞게 변환

        # 모델의 foward 함수에서 입력 인자 개수에 따라 다르게 호출
        if hasattr(model, "forward") and model.forward.__code__.co_argcount >= 4:
            # 4개 이상이면 s_idx, meta도 전달 (4개 이상: self, x, station_idx, meta)
            yhat = model(xb, s_idx, meta)
        else: # 4개 미만이면 s_idx, meta는 None 
            yhat = model(xb)

        # 타깃 차원 보정 (1D -> 2D)
        if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
            yb = yb.unsqueeze(-1)

        loss = criterion(yhat, yb) # 손실 계산
        optimizer.zero_grad(set_to_none=True) # 옵티마이저 기울기 초기화
        loss.backward() # 역전파

        # gradient clipping (너무 큰 기울기 방지)
        if (grad_clip is not None) and (grad_clip > 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step() # 파라미터 업데이트

        bs = xb.size(0) # 배치 크기
        total += loss.item() * bs # 손실 합계 갱신
        n += bs # 샘플 수 갱신
    return total / max(n, 1) # 평균 손실 반환

def train_cnn_lstm(loader, cnn, lstm, criterion, optimizer, device):
    """
    Function: train_cnn_lstm
        - CNN + LSTM 모델을 한 epoch 동안 학습 
    Parameters:
        - loader: DataLoader, 학습 데이터 로더
        - cnn: PVPlantCNN 모델
        - lstm: PVPlantLSTM 모델
        - criterion: 손실 함수
        - optimizer: 최적화 알고리즘
        - device: torch.device, 연산 장치 (CPU or GPU)
    Returns:
        - epoch_loss: float, epoch 동안의 평균 손실
    """
    cnn.train()
    lstm.train()

    total_sum, total_cnt = 0.0, 0 # epoch 동안의 누적 손실과 샘플 수

    for batch in loader:
        xb, s_idx, meta, yb = _to_device(batch, device) # 배치를 장치에 맞게 변환
        #  s_idx, meta는 단일 발전소이므로 사용하지 않음

        # CNN 입력 형태로 변환
        # [B, L, F] -> [B, F, L]
        if xb.dim() == 3:
            # L, F 위치 스위치
            xb = xb.permute(0, 2, 1) # [B, L, F] -> [B, F, L]
        elif xb.dim() == 2: # L=1인 경우 [B, F] 형태
            xb = xb.unsqueeze(-1) # [B, F] -> [B, F, 1]
        
        # 1. CNN forward
        cnn_out = cnn(xb) # [B, F, L] 형태
        
        # 2. LSTM forward
        yhat = lstm(cnn_out) # [B, output_window] or [B,] 형태

        # 타깃 차원 보정 (손실 계산을 위해)
        if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
            yb = yb.unsqueeze(-1) # (N,) -> (N, 1)

        # 3. 손실 계산
        loss = criterion(yhat, yb)

        # 4. 역전파
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = xb.size(0) # 배치 크기
        total_sum += loss.item() * bs # 배치 손실의 합
        total_cnt += bs # 배치 샘플 수 누적

    return total_sum / max(total_cnt, 1) # epoch 평균 손실

def train_single_model(loader, model, criterion, optimizer, device):
    """
    Function: train
        - 모델을 한 epoch 동안 학습
    Parameters:
        - loader: DataLoader
            - 학습 데이터 로더
        - model: 학습할 모델
        - criterion: 손실 함수
        - optimizer: 옵티마이저
        - device: torch.device
    Returns:
        - float
            - 평균 학습 손실
    """
    model.train() # 모델 학습 모드로 전환
    total, n = 0.0, 0 # 손실 합계와 샘플 수 초기화

    # 배치 단위로 학습 
    for batch in loader: 
        xb, s_idx, meta, yb = _to_device(batch, device) # 배치를 장치에 맞게 변환

        # # 모델의 foward 함수에서 입력 인자 개수에 따라 다르게 호출
        # if hasattr(model, "forward") and model.forward.__code__.co_argcount >= 4:
        #     # 4개 이상이면 s_idx, meta도 전달 (4개 이상: self, x, station_idx, meta)
        #     yhat = model(xb, s_idx, meta)
        # else: # 4개 미만이면 s_idx, meta는 None 
        yhat = model(xb)

        # 타깃 차원 보정 (1D -> 2D)
        if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
            yb = yb.unsqueeze(-1)

        loss = criterion(yhat, yb) # 손실 계산
        optimizer.zero_grad(set_to_none=True) # 옵티마이저 기울기 초기화
        loss.backward() # 역전파
        optimizer.step() # 파라미터 업데이트

        bs = xb.size(0) # 배치 크기
        total += loss.item() * bs # 손실 합계 갱신
        n += bs # 샘플 수 갱신
    return total / max(n, 1) # 평균 손실 반환