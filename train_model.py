"""
==============================================================================
File: train_model.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-10-14

Description: 하이브리드 혹은 단일 시계열 예측 모델 초기화 및 훈련 과정
    Functions:
        - load_data: 전처리 완료된 CSV 파일과 스케일러 로드
        - sliding_window: 슬라이딩 윈도우 생성
        - build_dataloader: DataLoader 생성
        - build_model: 모델 초기화
        - train_model: 모델 훈련
Note
    - main 모듈 변수 설정 후 실행
    - 모델: CNN + LSTM / CNN + Transformer / LSTM 단독 / Transformer 단독 (추가 가능)
    - 산업체마다 데이터 특성이 다를 수 있음. 수정 필요 - 현재는 광명금속의 포멧을 따름
    - 각 모델의 파라미터는 글로벌 변수로 설정
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import joblib
import pickle
import os
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.CNN import CNN
from models.LSTM import LSTM
from models.Transformer import Transformer
from torch.utils.data import DataLoader
from utils.setup import SeqDataset
from utils.train import train
from utils.evaluate_metrics import evaluate


# CNN 파라미터 설정
out_channels = 64 # CNN에서 추출할 feature map 수 (입력 feature를 통해 out_channels 수 만큼 새로운 특징 추출)
kernel_size = 3 # 커널 크기
stride = 1 # kernel 이동 간격
dilation = 1 # kernel 사이의 간격
padding = 1 # 입력 양 쪽에 채워 넣는 padding 크기
groups = 1 # 1: default
bias = True # bias 사용
padding_mode = 'zeros' # padding 채우는 값
cnn_dropout = 0.1 # dropout 비율
pool_kernel_size = 2 # MaxPooling kernel size
pool_stride = 2 # MaxPooling stride
cnn_dropout=0.1
next_in_features = 32 # LSTM 입력 피처 수 # out_channels에서 fully connected layer로 축소
use_bn = True # Batch Normalization 사용 여부

# LSTM 파라미터 설정
hidden_size = 128 # LSTM hidden state 크기
num_lstm_layers = 1 # LSTM 레이어 수
lstm_dropout = 0.1 # dropout 비율

# Transformer 파라미터 설정
d_model = 128
nhead = 8
dim_feedforward = 512
num_ts_layers = 3
ts_dropout = 0.1

# 전처리 완료된 데이터 로드 함수
def load_data(path):
    """
    Function: load_data
        - 지정된 경로에서 전처리 완료된 CSV 파일과 스케일러 로드
    Parameters:
        - path: str, 데이터가 저장된 디렉토리 경로
    Returns:
        - train_origin, valid_origin, test_origin: 원본 데이터
        - train_scaled, valid_scaled, test_scaled: 스케일링된 입력 데이터
        - train_scaled_y, valid_scaled_y, test_scaled_y: 스케일링된 타겟 데이터
        - minmax_scaler: 입력 피처용 MinMaxScaler 객체
        - standard_scaler: 입력 피처용 StandardScaler 객체
        - y_minmax_scaler: 타겟 피처용 MinMaxScaler 객체
        - y_standard_scaler: 타겟 피처용 StandardScaler 객체
    """
    path_name = os.path.basename(os.path.normpath(path))  # 예: "광명금속"
    print(f"Loading data from {path_name}...")

    # origin 파일은 시각화시에만 사용 
    train_origin = pd.read_csv(path + f"{path_name}_train_origin.csv")
    valid_origin = pd.read_csv(path + f"{path_name}_valid_origin.csv")
    test_origin = pd.read_csv(path + f"{path_name}_test_origin.csv")

    train_scaled = pd.read_csv(path + f"{path_name}_train_scaled.csv")
    valid_scaled = pd.read_csv(path + f"{path_name}_valid_scaled.csv")
    test_scaled = pd.read_csv(path + f"{path_name}_test_scaled.csv")

    train_scaled_y = pd.read_csv(path + f"{path_name}_train_scaled_y.csv")
    valid_scaled_y = pd.read_csv(path + f"{path_name}_valid_scaled_y.csv")
    test_scaled_y = pd.read_csv(path + f"{path_name}_test_scaled_y.csv")
    
    # sklearn 객체 (StandardScaler, MinMaxScaler 등) 로드 함수
    def safe_load_sklearn_obj(file_path):
        '''
        Function: safe_load_sklearn_obj
            - sklearn 객체 (StandardScaler, MinMaxScaler 등)를
            joblib 또는 pickle로 안전하게 로드
        Parameters:
            - file_path: str, 로드할 파일 경로
        Returns:
            - sklearn 객체
        '''
        # 1) joblib로 먼저 시도
        try:
            # joblib로 저장한 파일이면 여기서 정상 로드됨
            return joblib.load(file_path)
        except Exception:
            pass

        # 2) pickle로 재시도
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            # 어떤 방식으로도 로드가 안 되면 상세 메시지 제공
            raise RuntimeError(f"스케일러 로드 실패: {file_path}\n"
                            f"- 원인 후보: 저장 방식 불일치(joblib vs pickle), 파일 손상, "
                            f"파이썬/스킷런 버전 불일치, 다른 객체를 잘못 저장\n"
                            f"원본 에러: {type(e).__name__}: {e}")
        
    minmax_scaler = safe_load_sklearn_obj(path + f"{path_name}_minmax_scaler.pkl")
    standard_scaler = safe_load_sklearn_obj(path + f"{path_name}_standard_scaler.pkl")
    y_minmax_scaler = safe_load_sklearn_obj(path + f"{path_name}_y_minmax_scaler.pkl")
    y_standard_scaler = safe_load_sklearn_obj(path + f"{path_name}_y_standard_scaler.pkl")

    print(f"Data and scalers loaded successfully from {path_name}.\n")

    return (train_origin, valid_origin, test_origin,
            train_scaled, valid_scaled, test_scaled,
            train_scaled_y, valid_scaled_y, test_scaled_y,
            minmax_scaler, standard_scaler, y_minmax_scaler, y_standard_scaler)

# 슬라이딩 윈도우 생성 함수
def sliding_window(df, input_window, output_window,
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
        - F: feature 수
        - L: input_window
    """
    xs, ys = [], [] # 입력과 타겟 리스트

    # 인덱스 초기화
    sub = df.reset_index(drop=True)

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
        xs.append(X[i:i+input_window]) #
        ys.append(y_all[i+input_window:i+input_window+output_window])

    x_arr = np.stack(xs).astype(np.float32) # (N, L, F)
    y_arr = np.stack(ys).astype(np.float32) # (N, output_window)

    # output_window = 1 일 때 차원 축소 (N, 1) -> (N,)
    if y_arr.shape[1] == 1:
        y_arr = y_arr[:, 0]

    return torch.from_numpy(x_arr), torch.from_numpy(y_arr) # torch.Tensor로 변환

# DataLoader 생성 함수
def build_dataloader(torch_train_x, torch_train_y, 
                torch_valid_x, torch_valid_y, 
                torch_test_x, torch_test_y, batch_size
    ):
    """
    Function: build_dataloader
        - torch.Tensor로 변환된 (x, y) 데이터를 DataLoader로 변환
    Parameters:
        - torch_train_x, torch_train_y: 학습 데이터 (torch.Tensor)
        - torch_valid_x, torch_valid_y: 검증 데이터 (torch.Tensor)
        - torch_test_x, torch_test_y: 테스트 데이터 (torch.Tensor)
        - batch_size: 배치 크기 (int)
    Returns:
        - train_loader, valid_loader, test_loader: DataLoader 객체 
            (shape: (N, L, F), (N,) or (N, output_window))
    """
    # (x, y)만 생성
    train_ds = SeqDataset(torch_train_x, torch_train_y)
    valid_ds = SeqDataset(torch_valid_x, torch_valid_y)
    test_ds  = SeqDataset(torch_test_x,  torch_test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    return train_loader, valid_loader, test_loader

# 모델 초기화 함수
def build_model(model1, model2, train_scaled, input_window, output_window, device):
    """
    Function: build_model
        - 하이브리드 or 단일 시계열 예측 모델 초기화
        - model1: CNN 모델 (None 가능)
        - model2: 시계열 데이터 처리 모델 ("LSTM" or "Transformer")
    Parameters:
        - model1: str or None, "CNN" or None
        - model2: str, "LSTM" or "Transformer"
        - train_scaled: pandas DataFrame, 학습 데이터 (스케일링된 상태)
        - input_window: int, 입력 시퀀스 길이
        - output_window: int, 출력 시퀀스 길이
        - device: torch.device, 모델과 데이터를 올릴 디바이스 (cpu or cuda)
    Returns:
        - model1, model2: 초기화된 모델 (model1은 None 가능)
    """
    if model1 is not None: # CNN + 시계열 예측 모델 하이브리드 모델 (2-step hybrid)
        # CNN 모델 생성
        cnn = CNN(
            in_channels=train_scaled.shape[1], # 입력 피처 수
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            dropout=cnn_dropout,
            pool_kernel_size=pool_kernel_size,
            pool_stride=pool_stride,
            next_in_features=next_in_features,
            use_bn=use_bn
        ).to(device)
        print("CNN: \n", cnn)

        if model2 == "LSTM":
            # LSTM 모델 생성
            
            lstm = LSTM(
                input_size=next_in_features,
                hidden_size=hidden_size,
                output_size=output_window,
                num_layers=num_lstm_layers,
                dropout=lstm_dropout
            ).to(device)
            print("LSTM: \n", lstm)
            return cnn, lstm
        
        elif model2 == "Transformer":
            # Transformer 모델 생성
            transformer = Transformer(
                input_dim=next_in_features,
                d_model=d_model,
                nhead=nhead,
                num_enc_layers=num_ts_layers,
                dim_feedforward=dim_feedforward,
                output_window=output_window,
                dropout=ts_dropout,
                max_len = input_window,
                head_hidden=d_model//2
            ).to(device)
            print("Transformer: \n", transformer)
            return cnn, transformer
        
    else: # 시계열 예측 모델 단독
        if model2 == "LSTM":
            # LSTM 모델 생성
            # LSTM 파라미터 설정
            # LSTM 모델 생성
            lstm = LSTM(
                input_size = train_scaled.shape[1], # LSTM에 처음 data input이므로 변수 개수와 동일하게 들어가야 함
                hidden_size=hidden_size,
                output_size=output_window,
                num_layers=num_lstm_layers,
                dropout=lstm_dropout
            ).to(device)
            print("LSTM: \n", lstm)
            return None, lstm
        
        elif model2 == "Transformer":
            # Transformer 모델 생성
            # Transformer 파라미터 설정
            transformer = Transformer(
                input_dim=train_scaled.shape[1],
                d_model=d_model,
                nhead=nhead,
                num_enc_layers=num_ts_layers,
                dim_feedforward=dim_feedforward,
                output_window=output_window,
                dropout=ts_dropout,
                max_len = input_window,
                head_hidden=d_model//2
            ).to(device)
            print("Transformer: \n", transformer)
            return None, transformer

# 모델 훈련 함수
def train_model(path, input_window, output_window, 
                criterion, optimizer_type, num_epochs,
                model1, model2, device):
    """
    Function: train_model
        - 하이브리드 or 단일 시계열 예측 모델 훈련
    Parameters:
        - path: str, 데이터가 저장된 디렉토리 경로
        - input_window: int, 입력 시퀀스 길이
        - output_window: int, 출력 시퀀스 길이
        - criterion: 손실 함수
        - optimizer_type: str, 최적화 알고리즘 ("Adam", "SGD", "RMSprop" 등)
        - num_epochs: int, 학습 epoch 수
        - model1: str or None, "CNN" or None
        - model2: str, "LSTM" or "Transformer"
        - device: torch.device, 모델과 데이터를 올릴 디바이스 (cpu or cuda)
    Returns:
        - None
    """
    _, _, _, train_scaled, valid_scaled, _, _, _, _, _, _, _, _ = load_data(path)

    # sliding_window, torch.Tensor 변환
    torch_train_x, torch_train_y = sliding_window(
        train_scaled, input_window, output_window,
        target_col="usage_kWh", keep_target_in_x=True
    )
    torch_valid_x, torch_valid_y = sliding_window(
        valid_scaled, input_window, output_window,
        target_col="usage_kWh", keep_target_in_x=True
    )

    train_loader, valid_loader, _ = build_dataloader(
        torch_train_x, torch_train_y, torch_valid_x, torch_valid_y, None, None, batch_size=512
    )

    model1, model2 = build_model(model1, model2, train_scaled, input_window, output_window, device)

    # 스텝별 파라미터를 하나의 optimizer에 넣기
    if model1 is not None: 
        optimizer = torch.optim.__dict__[optimizer_type](list(model1.parameters()) + list(model2.parameters()), lr=1e-4)
    else:
        optimizer = torch.optim.__dict__[optimizer_type](list(model2.parameters()), lr=1e-4)
    path_name = os.path.basename(os.path.normpath(path))  # 예: "광명금속"

    best_loss = float('inf')

    # 모델 저장 경로
    #    input, output_window가 달라지면 가장 뒤에 _(input, output)_ 추가
    #    default: input_window=10, output_window=1
    if model1 is not None: 
        best_model1_path = f'results/{path_name}_hybrid_{model1.__class__.__name__}_with_{model2.__class__.__name__}.pth'
        best_model2_path = f'results/{path_name}_hybrid_{model2.__class__.__name__}.pth'
    else:
        best_model1_path = None
        best_model2_path = f'results/{path_name}_{model2.__class__.__name__}.pth'

    print("\n======================================================================\n")
    print(f"Training started for {path_name}...\n")
    for epoch in range(1, num_epochs + 1):
            # 1. 학습
            train_loss = train(train_loader, model1, model2, criterion, optimizer, device)

            # 2. 검증: 평균 손실
            overall_loss = evaluate(valid_loader, model1, model2, criterion, device)

            # 3. best model 저장 
            if overall_loss < best_loss:
                best_loss = overall_loss
                if model1 is not None:
                    torch.save(model1.state_dict(), best_model1_path)
                    torch.save(model2.state_dict(), best_model2_path)
                else:
                    torch.save(model2.state_dict(), best_model2_path)
                print(f"[{path_name}] ({epoch:03d}) Saved best | valid loss: {best_loss:.6f}")

            # 4. 출력: train / valid 평균 손실
            print(f"[{path_name}] ({epoch:03d}) train {train_loss:.6f} | valid {overall_loss:.6f}")

    print(f"\nTraining complete. Best valid loss: {best_loss:.6f}\n")
    print("======================================================================\n")

# main 실행 블록
if __name__ == "__main__":
    path = "data/광명금속/"
    path_name = os.path.basename(os.path.normpath(path)) # 예: "광명금속"
    input_window = 10 # 15분 단위, 24 = 6시간
    output_window = 1 # 15분 단위, 24 = 6시간
    model1 = None # "CNN" or None
    model2 = "Transformer" # "LSTM" or "Transformer"
    num_epochs = 500
    criterion = nn.HuberLoss(delta=1.0, reduction="mean") # Huber Loss
    optimizer_type = "Adam" # "Adam", "SGD", "RMSprop"

    train_model(path, input_window, output_window, 
                criterion, optimizer_type, num_epochs, 
                model1, model2, device)
