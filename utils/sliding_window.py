import numpy as np
import pandas as pd
import torch

# 슬라이딩 윈도우 생성 함수
def build_sliding_window(
        df, input_window, output_window,
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
    x_list, y_list = [], [] # 입력과 타겟 리스트

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
        x_list.append(X[i:i+input_window]) 
        y_list.append(y_all[i+input_window:i+input_window+output_window])

    x_arr = np.stack(x_list).astype(np.float32) # (N, L, F)
    y_arr = np.stack(y_list).astype(np.float32) # (N, output_window)

    # output_window = 1 일 때 차원 축소 (N, 1) -> (N,)
    if y_arr.shape[1] == 1:
        y_arr = y_arr[:, 0]

    # torch.Tensor로 변환 # (윈도우 수, 윈도우 길이, 피처 수), (윈도우 수, 출력 길이)
    return torch.from_numpy(x_arr), torch.from_numpy(y_arr)



def build_sliding_window_longformer(
        model, 
        df_enc, df_mark, 
        seq_len, label_len, pred_len,
        target_col="usage_kWh",
    ):
    """
    Function: build_sliding_window_longformer
        - Longformer 모델용 슬라이딩 윈도우 생성
        - longformer
            - 일반 변수 + 날짜 변수 분리해서 사용
            - input_window -> seq_len
            - label_len 추가 (디코더 입력용)
            - output_window -> pred_len
    Parameters:
        - model: str, "Informer" or "Autoformer"
        - df_enc: pandas DataFrame, 인코더 입력 데이터프레임
        - df_mark: pandas DataFrame, 날짜 특성 데이터프레임
        - seq_len: int, 인코더 시퀀스 길이
        - label_len: int, 디코더 라벨 길이
        - pred_len: int, 디코더 예측 길이
        - target_col: str, 타겟 컬럼명 (default: "usage_kWh")
    Returns:
        - x_enc: torch.Tensor, 인코더 입력 텐서, shape (N, seq_len, F_enc)
        - x_mark_enc: torch.Tensor, 인코더 날짜 특성 텐서, shape (N, seq_len, T_dim)
        - x_dec: torch.Tensor, 디코더 입력 텐서, shape (N, label_len + pred_len, F_enc)
        - x_mark_dec: torch.Tensor, 디코더 날짜 특성 텐서, shape (N, label_len + pred_len, T_dim)
        - y: torch.Tensor, 타겟 텐서, shape (N, pred_len, 1)
    """

    # 데이터프레임 인덱스 재설정
    df_enc = df_enc.reset_index(drop=True)
    df_mark = df_mark.reset_index(drop=True)

    # Encoder
    # 1. 데이터프레임을 넘파이 배열로 변환
    X = df_enc.to_numpy().astype(np.float32) # 일반 변수
    M = df_mark.to_numpy().astype(np.float32) # 날짜 변수
    Y = df_enc[target_col].to_numpy().astype(np.float32).reshape(-1, 1) # 타겟 변수
    
    # 2. 시계열 길이 및 슬라이딩 윈도우 개수 계산
    T = len(df_enc) # 전체 시계열 길이
    F_enc = X.shape[1] # 인코더 feature 개수 (예: 15)
    N = T - (seq_len + label_len + pred_len) + 1 # 슬라이딩 윈도우 개수
    
    # 3. 결과를 담을 리스트 초기화
    x_enc_list, x_mark_enc_list = [], [] # Encoder 입력 리스트
    x_dec_list, x_mark_dec_list = [], [] # Decoder 입력 리스트
    y_list = []

    # Informer
    if model == "Informer":
        """
        Informer 슬라이딩 윈도우 생성 과정
            - seq_len의 직후 label_len + pred_len 만큼을 디코더의 입력으로 사용
        Encoder:
            - x_enc: 모든 feature (usage, lag, rolling 등)
            - x_mark_enc: 날짜 feature
        Decoder:
            - x_dec: 타겟 0-padding
            - x_mark_dec: 날짜 feature
        Target:
            - y_target: 예측 구간만
        """
        # 4. 슬라이딩 윈도우 생성
        for i in range(N):
            # Encoder
            x_enc = X[i : i + seq_len] # x_enc: 모든 feature (usage, lag, rolling 등)
            x_mark_enc = M[i : i + seq_len] # x_mark_enc: 날짜 feature
            
            # Decoder
            dec_start = i + seq_len # Decoder 시작 인덱스: 인코더 끝나는 시점 i + seq_len
            dec_end = dec_start + label_len + pred_len # Decoder 끝나는 인덱스: 인코더 끝나는 시점 + label_len + pred_len
            
            # Decoder 입력: 날짜 feature만 사용
            x_mark_dec = M[dec_start : dec_end]
            
            # Label 구간: 과거 타겟 실제값 
            dec_label_target = Y[dec_start : dec_start + label_len]

            # Pred 구간: 타겟 0-padding
            dec_pred_target = np.zeros((pred_len, 1), dtype=np.float32)
        
            # Decoder 입력 = label + pred
            x_dec = np.concatenate([dec_label_target, dec_pred_target], axis=0)
            
            # Target (예측 구간만 실제 타겟)
            y_target = Y[dec_start + label_len : dec_end]
            
            x_enc_list.append(x_enc)
            x_mark_enc_list.append(x_mark_enc)
            x_dec_list.append(x_dec)
            x_mark_dec_list.append(x_mark_dec)
            y_list.append(y_target)
    
    # Autoformer
    elif model == "Autoformer":
        """
        Autoformer 슬라이딩 윈도우 생성 과정
            - Autoformer는 seq_len의 tail 부분 label_len 만큼을 디코더의 label_len으로 사용
        Encoder:
            - x_enc: 모든 feature (usage, lag, rolling 등)
            - x_mark_enc: 날짜 feature
        Decoder:
            - x_dec: label_len 구간은 실제 타겟값, pred_len 구간은 타겟 0-padding
            - x_mark_dec: 날짜 feature
        Target:
            - y_target: 예측 구간만
        """
        # 4. 슬라이딩 윈도우 생성
        for i in range(N):
            # Encoder: 모든 feature 사용
            x_enc = X[i : i + seq_len]  # (seq_len, F_enc)
            x_mark_enc = M[i : i + seq_len]  # (seq_len, T_dim)

            # Decoder 시작/끝 인덱스
            dec_start = i + seq_len - label_len
            dec_end = i + seq_len + pred_len

            # Label 구간 (label_len): encoder feature 그대로 사용
            dec_label = X[dec_start : i + seq_len] # (label_len, F_enc)
            
            # Pred 구간 (pred_len): 타겟 0-padding + 날짜 
            F_enc = X.shape[-1]
            dec_pred = np.zeros((pred_len, F_enc), dtype=np.float32)

            # Decoder 입력 = Label + Pred 
            x_dec = np.concatenate([dec_label, dec_pred], axis=0) # (label_len+pred_len, F_enc)
            x_mark_dec = M[dec_start : dec_end] # (label_len+pred_len, T_dim)
            
            # Target (예측 구간)
            y_target = Y[i + seq_len : i + seq_len + pred_len]  # (pred_len, 1)

            x_enc_list.append(x_enc)
            x_mark_enc_list.append(x_mark_enc)
            x_dec_list.append(x_dec)
            x_mark_dec_list.append(x_mark_dec)
            y_list.append(y_target)

    # 최종 텐서 변환 및 반환
    return (
        torch.tensor(np.array(x_enc_list), dtype=torch.float32), # (N, seq_len, F_enc)
        torch.tensor(np.array(x_mark_enc_list), dtype=torch.float32), # (N, seq_len, T_dim)
        torch.tensor(np.array(x_dec_list), dtype=torch.float32), # (N, label_len+pred_len, F_enc)
        torch.tensor(np.array(x_mark_dec_list), dtype=torch.float32), # (N, label_len+pred_len, T_dim)
        torch.tensor(np.array(y_list), dtype=torch.float32), # (N, pred_len, 1)
    )