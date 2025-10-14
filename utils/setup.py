"""
==============================================================================
File: setup.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-10-14

Description: 모델 설정 및 데이터셋 클래스
    Functions:
        - _to_device: DataLoader에서 가져온 배치를 장치에 맞게 변환
        - SeqDataset: 시계열 데이터셋 클래스
Note
    - train_model.py에서 실행
    - CNN + LSTM / CNN + Transformer / LSTM 단독 / Transformer 단독 (추가 가능)
==============================================================================
"""
import torch

def _to_device(batch, device):
    """
    Function: _to_device
        - DataLoader에서 가져온 배치를 장치에 맞게 변환
        - 동일한 배치 형태 유지를 위해
    Parameters:
        - batch: tuple
            - (xb, s_idx, meta, yb) 형태의 배치 데이터
        - device: torch.device
            - 데이터를 이동시킬 장치 (cuda)
    Returns:
        - tuple
            - 장치로 이동된 (xb, s_idx, meta, yb)
    """
    xb, yb = batch
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    
    return xb, yb

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # 2개 반환 (단일 station 경우)
        return self.x[idx], self.y[idx]
