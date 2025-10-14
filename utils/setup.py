import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
