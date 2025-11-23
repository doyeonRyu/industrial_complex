"""
==============================================================================
File: n04_evaluate_and_visualize_longformer.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-15
Last Modified: 2025-11-19

Description: longformer(Informer, Autoformer)용 예측 모델 결과 평가 지수 및 시각화
    Functions:
        - run_evaluation_and_visualization: 모델 평가 및 시각화 실행
        - metrics, save_predictions_to_excel_longformer, plot_single_prediction_longformer 함수 사용
    1) 모델 평가 지표 출력
    2) 예측 결과 엑셀 저장
    3) 예측 결과 시각화 및 저장

Note
    - 코드 실행 시 {industry_name}, {model1}, {model2} 설정
        - 실행 예시: python n04_evaluate_and_visualize_longformer.py --industry_name 광명금속 (--model1 CNN) --model2 Informer
    - 모델: CNN + Informer / CNN + Autoformer / Informer 단독 / Autoformer 단독
==============================================================================
"""

import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset

from utils.load_data import load_data
from utils.sliding_window import build_sliding_window_longformer
from utils.model import initiate_model_longformer
from utils.metrics import metrics_longformer
from utils.save_predictions import save_predictions_to_excel_longformer
from plots.plot import plot_single_prediction_longformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# 명령행 인자 파서 설정 함수
def parse_args():
    parser = argparse.ArgumentParser(description="전력 사용량 예측 모델 훈련 스크립트")
    parser.add_argument("--industry_name", type=str, default=None, help="결과 파일명에 사용할 산업체 이름")
    parser.add_argument("--model1", type=str, default=None, help="모델1 유형 선택 (CNN 또는 None)")
    parser.add_argument("--model2", type=str, default=None, help="모델2 유형 선택 (LSTM 또는 Transformer)")
    return parser.parse_args()

# 모델 평가 및 시각화 실행 함수
def run_evaluation_and_visualization(industry_name, model1, model2):
    path = f"data/preprocessed/{industry_name}/"

    input_window = 36
    label_len = 18
    output_window = 32
    threshold_value = 0.1 # 0 근처 값들에 대한 MAPE, PAPE 계산 시 필터링 임계값
    if model1 is not None:
        model1_path = f"results/{industry_name}/{industry_name}_hybrid_{model1}_with_{model2}_({input_window},{output_window}).pth"
        model2_path = f"results/{industry_name}/{industry_name}_hybrid_{model2}_with_{model1}_({input_window},{output_window}).pth"
        print(f"Evaluating Hybrid Model: {model1_path} + {model2_path}")
    else:
        model1_path = None
        model2_path = f"results/{industry_name}/{industry_name}_{model2}_({input_window},{output_window}).pth"
        print(f"Evaluating {model2_path}")

    batch_size = 512
    data_type = "valid"
    view_days = 5
    start_idx = 0  

    print("\n" + "=" * 60)
    print("[Loading data and models] ...")
    print("=" * 60)

    _, valid_origin, test_origin, train_preprocessed, valid_preprocessed, test_preprocessed, train_x_mark, valid_x_mark, test_x_mark, _, _, _, _, y_scaler = load_data(path, model_type="longformer")

    # sliding_window, torch.Tensor 변환
    train_x_enc,train_x_mark_enc, train_x_dec, train_x_mark_dec, train_y_target = build_sliding_window_longformer(
        model2,
        train_preprocessed, # 일반 feature
        train_x_mark, # 날짜 시간 feature
        seq_len=input_window,
        label_len=label_len,
        pred_len=output_window,
        target_col="usage_kWh",
    )
    valid_x_enc, valid_x_mark_enc, valid_x_dec, valid_x_mark_dec, valid_y_target = build_sliding_window_longformer(
        model2,
        valid_preprocessed, # 일반 feature
        valid_x_mark, # 날짜 시간 feature
        seq_len=input_window,
        label_len=label_len,
        pred_len=output_window,
        target_col="usage_kWh",
    )

    test_x_enc, test_x_mark_enc, test_x_dec, test_x_mark_dec, test_y_target = build_sliding_window_longformer(
        model2,
        test_preprocessed, # 일반 feature
        test_x_mark, # 날짜 시간 feature
        seq_len=input_window,
        label_len=label_len,
        pred_len=output_window,
        target_col="usage_kWh",
    )

    valid_ds = TensorDataset(valid_x_enc, valid_x_dec, valid_y_target, valid_x_mark_enc, valid_x_mark_dec)
    test_ds = TensorDataset(test_x_enc, test_x_dec, test_y_target, test_x_mark_enc, test_x_mark_dec)
    
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 모델 초기화
    model1, model2 = initiate_model_longformer(model1, model2, train_x_enc, train_x_dec, input_window, label_len, output_window, device)

    print("\n" + "=" * 60)
    print(f"Evaluating [{model1.__class__.__name__} + {model2.__class__.__name__}] Performance ...")
    print("=" * 60)
    
    data_loader = valid_loader if data_type == "valid" else test_loader
    
    val_mae, val_rmse, val_mape, val_mape_filterd, val_smape, val_r2, val_pape, val_hr, val_lag = metrics_longformer(
        model1_path, model2_path, model1, model2,
        data_loader, y_scaler, device, logged=True, 
        threshold=threshold_value
    )
    print(f"[{industry_name} | {data_type}] Performance:\nMAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE_filtered: {val_mape_filterd:.4f}, sMAPE: {val_smape:.4f}, R²: {val_r2:.4f}, PAPE: {val_pape:.4f}, HR: {val_hr:.4f}, Lag: {val_lag:.4f}")
    
    # 전체 예측 결과 엑셀 저장
    data_origin = valid_origin if data_type == "valid" else test_origin
    if model1_path is not None:
        save_first_path = f"results/{industry_name}/{industry_name}_{data_type}_hybrid_{model1.__class__.__name__}_with_{model2.__class__.__name__}_({input_window},{output_window})_first_window.xlsx"
        save_full_path = f"results/{industry_name}/{industry_name}_{data_type}_hybrid_{model1.__class__.__name__}_with_{model2.__class__.__name__}_({input_window},{output_window})_full_series.xlsx"
    else:
        save_first_path = f"results/{industry_name}/{industry_name}_{data_type}_{model2.__class__.__name__}_({input_window},{output_window})_first_window.xlsx"
        save_full_path = f"results/{industry_name}/{industry_name}_{data_type}_{model2.__class__.__name__}_({input_window},{output_window})_full_series.xlsx"

    save_predictions_to_excel_longformer(
        model1_path, model2_path, model1, model2,
        save_first_path, save_full_path,
        data_loader,
        data_origin,
        y_scaler,
        device,
        input_window=input_window,
        label_len=label_len,
        output_window=output_window,
        logged=True
    )
    print('\nSuccessfully evaluated the model.')
    print("=" * 60)

    print("\n" + "=" * 60)
    print(f"[{model1.__class__.__name__} + {model2.__class__.__name__}] Plotting Evaluation Results ...")
    print("=" * 60)
    
    data_origin = valid_origin if data_type == "valid" else test_origin    
    
    plot_single_prediction_longformer(
        model1_path, model2_path, model1, model2,
        data_loader=data_loader,
        df=data_origin,
        device=device,
        input_window=input_window,
        output_window=output_window,
        start_idx=start_idx,
        scaler=y_scaler,
        logged=True,
        industry_name=industry_name,
        datatype=data_type,
        peak_weight=False
    )
    
    # 원하는 날짜 범위 시계열 예측 결과 플롯 저장
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

    print(f'\nSuccessfully saved the plot [{industry_name} | {data_type}].')
    print("=" * 60)

# main 실행 블록
if __name__ == "__main__":
    """
    - 코드 실행 시 {industry_name}, {model1}, {model2} 설정
    - 모델 종류, 입출력 윈도우 크기 등도 필요 시 수정
    """
    args = parse_args()

    industry_name = args.industry_name
    if industry_name is None:
        industry_name = "광명금속"
    model1 = args.model1 # "CNN" or None
    if model1 is None:
        model1 = None
    model2 = args.model2 # "Informer" or "Autoformer"

    run_evaluation_and_visualization(industry_name, model1, model2)