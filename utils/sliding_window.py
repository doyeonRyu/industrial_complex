import numpy as np
import pandas as pd
import torch

# 슬라이딩 윈도우 생성 함수
def build_sliding_window(df, input_window, output_window,
                    target_col="usage_kWh", keep_target_in_x=True):
    """
    Function: sliding_window
        - input_window 길이 만큼의 슬라이딩 윈도우를 사용하여
          output_window 길이 만큼의 타겟 생성
    Parameters:
        - df: pandas DataFrame, 시계열 데이터(전처리 완료된 상태)
        - input_window: input_window 길이
        - output_window: output_window 길이
        - target_col: 타겟 컬럼명 (default: "usage_kWh")
        - keep_target_in_x: 입력 피처에 타겟 컬럼 포함 여부 (default: True)
    Returns:
        - x_arr: torch.Tensor, shape (N, input_window, F)
        - y_arr: torch.Tensor, shape (N, output_window) or (N,)
          (output_window=1일 때는 (N,)로 차원 축소)
        - N: 생성된 샘플 수
    Note:
        - N = T - input_window - output_window + 1 (T: 전체 타임스탬프 수)
        - L: input_window
        - F: feature 수
    """
    xs, ys = [], [] # 입력과 타겟 리스트

    # 인덱스 초기화
    sub = df.reset_index(drop=True) # sub: 인덱스 초기화된 DataFrame

    # 숫자형 feature 선택
    numeric_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = numeric_cols.copy()
    if not keep_target_in_x:
        feat_cols = [c for c in feat_cols if c != target_col]
    
    X = sub[feat_cols].to_numpy()
    y_all = pd.to_numeric(sub[target_col], errors="coerce").to_numpy()

    T = len(sub)
    N = T - input_window - output_window + 1
    if N <= 0:
        return None, None

    for i in range(N):
        xs.append(X[i:i+input_window]) 
        ys.append(y_all[i+input_window:i+input_window+output_window])

    x_arr = np.stack(xs).astype(np.float32) # (N, L, F)
    y_arr = np.stack(ys).astype(np.float32) # (N, output_window)

    # output_window = 1 일 때 차원 축소 (N, 1) -> (N,)
    if y_arr.shape[1] == 1:
        y_arr = y_arr[:, 0]

    # torch.Tensor로 변환 # (윈도우 수, 윈도우 길이, 피처 수), (윈도우 수, 출력 길이)
    return torch.from_numpy(x_arr), torch.from_numpy(y_arr)