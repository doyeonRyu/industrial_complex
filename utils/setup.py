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
    if len(batch) == 2:
        xb, yb = batch
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        return xb, None, None, yb
    else: 
        xb, s_idx, meta, yb = batch
        xb = xb.to(device, non_blocking=True)
        s_idx = s_idx.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        return xb, s_idx, meta, yb

class SeqDataset(torch.utils.data.Dataset):
    """
    Class: SeqDataset
        - 시계열 입력(x), 타깃(y), 발전소 인덱스(s), 메타데이터(meta)를 포함하는 데이터셋
        - 하나의 샘플을 (x, s, meta, y) 형태의 묶음으로 반환
        - 이후 DataLoader로 감싸서 배치 단위로 모델에 공급
    Parameters:
        - x: 입력 시계열, shape=(N, L, F)
        - y: 타깃 값, shape=(N,) 또는 (N, T)
        - s: 발전소 인덱스, shape=(N,)
        - meta: 메타데이터, shape=(N, M) 또는 None
    Returns: None
    """
    def __init__(self, x, y, s, meta=None):
        self.x = x
        self.y = y
        self.s = s.long()
        self.meta = meta  # None 또는 (N, M)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        if self.meta is None:
            return self.x[i], self.s[i], None, self.y[i]
        else:
            return self.x[i], self.s[i], self.meta[i], self.y[i]

class SeqDataset_single_model(Dataset):
    def __init__(self, x, y, s=None, meta=None):
        # 필수 입력
        self.x = x
        self.y = y
        # 선택 입력
        self.s = s
        self.meta = meta

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 4개 반환
        if self.s is not None and self.meta is not None:
            return self.x[idx], self.y[idx], self.s[idx], self.meta[idx]
        # 3개 반환
        elif self.s is not None:
            return self.x[idx], self.y[idx], self.s[idx]
        # 2개 반환 (단일 station 경우)
        else:
            return self.x[idx], self.y[idx]
