"""
==============================================================================
File: evaluate_and_visualize.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-15
Last Modified: 2025-10-15

Description: 하이브리드 혹은 단일 시계열 예측 모델 결과 평가 지수 및 시각화
    Functions:
        - run_evaluation_and_visualization: 모델 평가 및 시각화 실행
            - metrics, plot_predictions_chained 함수 사용
    1) 모델 평가 지표 출력
    2) 예측 결과 시각화 및 저장

Note
    - 
==============================================================================
"""

import os
import torch

from utils.load_data import load_data
from utils.sliding_window import build_sliding_window
from utils.dataloader import build_dataloader
from utils.model import build_model
from utils.evaluate_metrics import metrics, save_predictions_to_excel
from plots.plot import plot_predictions_chained, plot_single_prediction
device = "cuda" if torch.cuda.is_available() else "cpu"

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
num_ts_enc_layers = 2
num_ts_dec_layers = 1
ts_dropout = 0.1

def run_evaluation_and_visualization():
    path = "data/preprocessed/광명금속/"
    industry_name = os.path.basename(os.path.normpath(path))

    input_window = 36
    output_window = 32
    model1 = "CNN" # "CNN" or None
    model2 = "Transformer_encoder" # "LSTM" or "Transformer_encoder" or "Transformer"
    if model1 is not None:
        model1_path = f"results/{industry_name}/{industry_name}_hybrid_{model1}_with_{model2}_({input_window},{output_window}).pth"
        model2_path = f"results/{industry_name}/{industry_name}_hybrid_{model2}_with_{model1}_({input_window},{output_window}).pth"
        print(f"Evaluating Hybrid Model: {model1_path} + {model2_path}")
    else:
        model1_path = None
        model2_path = f"results/{industry_name}/{industry_name}_{model2}_({input_window},{output_window}).pth"
    batch_size = 512
    data_type = "valid"
    view_days = 5
    start_idx = 0 # 5시부터 

    print("=== Loading data and models ===")
    _, valid_origin, test_origin, train_preprocessed, valid_preprocessed, test_preprocessed, _, _, _, _, _, _, y_standard_scaler = load_data(path)

    # sliding_window, torch.Tensor 변환
    torch_train_x, torch_train_y = build_sliding_window(
        train_preprocessed, input_window, output_window,
        target_col="usage_kWh", keep_target_in_x=True
    )
    torch_valid_x, torch_valid_y = build_sliding_window(
        valid_preprocessed, input_window, output_window,
        target_col="usage_kWh", keep_target_in_x=True
    )

    torch_test_x, torch_test_y = build_sliding_window(
        test_preprocessed, input_window, output_window,
        target_col="usage_kWh", keep_target_in_x=True
    )

    _, valid_loader, test_loader = build_dataloader(
        torch_train_x, torch_train_y, torch_valid_x, torch_valid_y, torch_test_x, torch_test_y, batch_size=batch_size
    )

    model1, model2 = build_model(model1, model2, train_preprocessed, input_window, output_window, device)

    print(f"=== Evaluating [{model1.__class__.__name__} + {model2.__class__.__name__}] Performance ===")
    data_loader = valid_loader if data_type == "valid" else test_loader
    val_mae, val_rmse, val_mape, val_mape_filterd, val_smape, val_r2, val_pape, val_hr, val_lag = metrics(
        model1_path, model2_path, model1, model2,
        data_loader, y_standard_scaler, None, device
    )
    print(f"[{industry_name} | {data_type}] Performance:\nMAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.4f}, MAPE_filtered: {val_mape_filterd:.4f}, sMAPE: {val_smape:.4f}, R²: {val_r2:.4f}, PAPE: {val_pape:.4f}, HR: {val_hr:.4f}, Lag: {val_lag:.4f}")
    # 전체 예측 결과 엑셀 저장
    data_origin = valid_origin if data_type == "valid" else test_origin
    if model1_path is not None:
        save_first_path = f"results/{industry_name}/{industry_name}_{data_type}_hybrid_{model1.__class__.__name__}_with_{model2.__class__.__name__}_({input_window},{output_window})_first_window.xlsx"
        save_full_path = f"results/{industry_name}/{industry_name}_{data_type}_hybrid_{model1.__class__.__name__}_with_{model2.__class__.__name__}_({input_window},{output_window})_full_series.xlsx"
    else:
        save_first_path = f"results/{industry_name}/{industry_name}_{data_type}_{model2.__class__.__name__}_({input_window},{output_window})_first_window.xlsx"
        save_full_path = f"results/{industry_name}/{industry_name}_{data_type}_{model2.__class__.__name__}_({input_window},{output_window})_full_series.xlsx"

    save_predictions_to_excel(
        model1_path, model2_path, model1, model2,
        save_first_path, save_full_path,
        data_loader,
        data_origin,
        y_standard_scaler,
        None,
        device,
        input_window=input_window,
        output_window=output_window,
        target_idx=None,
        logged=False
    )
    print('\nSuccessfully evaluated the model.\n')
    print("======================================================================\n")
    
    print(f"=== [{model1.__class__.__name__} + {model2.__class__.__name__}] Plotting Evaluation Results ===")
    data_origin = valid_origin if data_type == "valid" else test_origin    
    plot_single_prediction(
        model1_path, model2_path,
        model1, model2,
        data_loader=data_loader,
        df=data_origin,
        device=device,
        input_window=input_window,
        output_window=output_window,
        start_idx=start_idx,
        std_scaler=y_standard_scaler,
        mm_scaler=None,
        logged=False,
        industry_name=industry_name,
        datatype=data_type
    )
    # plot_predictions_chained(
    #     model1, model2,
    #     data_loader=data_loader,
    #     df=data_origin,
    #     device=device,
    #     input_window=input_window,
    #     output_window=output_window,
    #     view_days=view_days,
    #     std_scaler=y_standard_scaler,
    #     mm_scaler=None,
    #     logged=False,
    #     industry_name=industry_name,
    #     datatype=data_type
    # )
    print(f'\nSuccessfully saved the plot [{industry_name} | {data_type}].\n')
    print('======================================================================\n')

# main 실행 블록
if __name__ == "__main__":
    """
    - 산업체명 변경할 경우 **path** 변수 수정 필요
    - 모델 종류, 입출력 윈도우 크기 등도 필요 시 수정
    """
    run_evaluation_and_visualization()