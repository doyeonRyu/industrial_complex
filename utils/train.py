import torch
import torch.nn as nn
from utils.setup import _to_device

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
        - cnn: PVPlantCNN 모델
        - lstm: PVPlantLSTM 모델
        - criterion: 손실 함수
        - optimizer: 최적화 알고리즘
        - device: torch.device, 모델과 데이터를 올릴 디바이스 (cpu or cuda)
    Returns:
        - epoch_loss: float, 전체 배치에 대한(한 epoch 동안의) 평균 손실
    """
    if model1 is not None:
        model1.train()
    model2.train()

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
            yhat = model2(xb)

        # 타깃 차원 보정 (1D -> 2D) | [B] vs [B,1] 정렬
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