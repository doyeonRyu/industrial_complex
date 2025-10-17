"""
==============================================================================
File: plot.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-15
Last Modified: 2025-10-15

Description: 하이브리드 혹은 단일 시계열 예측 모델 결과 시각화
    Functions:
        - plot_predictions_chained: 모델의 예측 결과를 시각화
            - 원하는 날짜만큼 예측 결과 병렬 시각화 
Note
    - 
==============================================================================
"""

import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from utils.setup import _to_device

"""
산업 단지 예측 결과 시각화 함수
"""
def plot_predictions_chained(
    model1, model2, data_loader, df, device,
    input_window=10, output_window=1, start_idx=0,
    std_scaler=None,  
    mm_scaler=None,      
    logged: bool = False, 
    view_days = 2, 
    industry_name="industry_name",
    datatype = "Validation" # or "Test"
):
    # 역변환 (std_scaler -> mm_scaler -> (log))
    def _inv_transform_1d(arr_1d):
        arr = arr_1d.reshape(-1, 1) # (N,) -> (N, 1)
        if std_scaler is not None:
            arr = std_scaler.inverse_transform(arr)
        if mm_scaler is not None:
            arr = mm_scaler.inverse_transform(arr)
        arr = arr.reshape(-1) # (N, 1) -> (N,)
        if logged:
            arr = np.expm1(arr) # log 역변환
        return arr
    
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
    
    # 시작/종료 시점 계산
    if start_idx + input_window >= len(df):
        print(f"[warn] start_idx+input_window가 데이터 길이 초과")
        return

    start_time = df.loc[start_idx, "datetime"] # 시작 시점
    input_end = df.loc[start_idx + input_window, "datetime"] # 입력 구간 종료 시점
    view_end_time = input_end + pd.Timedelta(days=view_days) # 시각화 종료 시점

    # 입력 구간, 출력 구간 나누기
    # 입력: [start_time, start_time + input_window]
    input_start = start_time
    input_end = df.loc[start_idx + input_window - 1, "datetime"]
    input_mask = (df["datetime"] >= input_start) & (df["datetime"] <= input_end)
    input_part = df.loc[input_mask, ["datetime", "usage_kWh"]].copy()
    
    # 출력 실제값: [input_end, view_end_time] 범위
    out_mask = (df["datetime"] > input_end) & (df["datetime"] <= view_end_time)
    output_part = df.loc[out_mask, ["datetime", "usage_kWh"]].copy()

    # 예측을 view_days 단위 만큼 수행
    if model1 is not None:
        model1.eval()
    model2.eval()

    # 필요한 샘플 인덱스들을 수집
    # df.iloc[i+input_window : i+input_window+output_window]의 datetime
    needed_indices = []
    i = start_idx
    while True:
        out_start_i = i + input_window # 출력 시작 인덱스
        out_end_i = i + input_window + output_window # 출력 종료 인덱스 (미포함)
        if out_start_i >= len(df):
            break
        t0 = df.loc[out_start_i, "datetime"]
        if t0 > view_end_time:
            break
        needed_indices.append(i)
        i += 1

    if len(needed_indices) == 0:
        print("[warn] 예측에 사용할 샘플 인덱스가 없습니다.")
        return

    # 한 번의 loader 순회로 needed_indices에 해당하는 예측만 뽑아오기
    seen = 0 # 지금까지 본 샘플 수 (글로벌 오프셋)
    pred_map = {} # 예측 결과를 시간별로 누적(겹치면 최신 예측으로 덮어쓰기)

    with torch.no_grad():
        for batch in data_loader:
            xb = batch[0]
            bs = xb.size(0)

            g0 = seen # 글로벌 오프셋 시작
            g1 = seen + bs # 글로벌 오프셋 끝 (미포함)

            # 이번 배치에서 필요한 인덱스들만 선별
            to_pick = [idx for idx in needed_indices if g0 <= idx < g1]
            if to_pick:
                xb, yb = _to_device(batch, device)
                
                if model1 is not None:
                    # CNN이 원하는 형태로 변환
                    if xb.dim() == 3: 
                        xb = xb.permute(0, 2, 1) 
                    elif xb.dim() == 2: 
                        xb = xb.unsqueeze(1)
                    md1_out = model1(xb) # [B, F, L] 
                    yhat = model2(md1_out) # [B, output_window] or [B,]
                else:
                    # 단일 모델: [B, L, F] 보장
                    if xb.dim() == 3 and xb.shape[1] < xb.shape[2]: # [B, F, L] 이면
                        xb = xb.permute(0, 2, 1) # -> [B, L, F]
                    elif xb.dim() == 2: # [B, L] 이면
                        xb = xb.unsqueeze(-1) # -> [B, L, 1]

                    yhat = model2(xb) # [B, output_window] or [B,]

                # CPU로 이동 후 numpy 변환
                yhat_np = yhat.detach().cpu().numpy()  # (B, H) 또는 (B,)

                # 필요한 오프셋만 꺼내 시간축에 매핑
                for idx in to_pick:
                    off = idx - g0 # 배치 내 오프셋
                    pred_seq = yhat_np[off].reshape(-1)
                    pred_seq = _inv_transform_1d(pred_seq) # 역변환
                    
                    # pred_seq를 df의 시간축에 맞춰 매핑
                    s = idx + input_window
                    e = min(s + output_window, len(df))
                    times = df.loc[s:e-1, "datetime"].values

                    # view_end_time을 넘는 부분은 버림
                    for t, v in zip(times, pred_seq[:len(times)]):
                        if t <= view_end_time:
                            pred_map[pd.Timestamp(t)] = float(v)

            # seen 이동
            seen += bs
            
            # 모든 needed_indices를 소화했으면 종료
            if seen > max(needed_indices):
                break

    # pred_map -> 정렬된 시계열로 변환
    if len(pred_map) == 0:
        print("[warn] 예측 결과가 비어 있음.")
        return
    
    pred_series = pd.Series(pred_map).sort_index()
    pred_time = pred_series.index
    y_pred_rec = pred_series.values

    # ===================================================
    # 시각화
    plt.figure(figsize=(10, 4.8))
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    plt.plot(input_part["datetime"], input_part["usage_kWh"], label="Input (Actual)", linewidth=2)
    plt.plot(output_part["datetime"], output_part["usage_kWh"], label="Output (Actual)", linewidth=2)
    plt.plot(pred_time, y_pred_rec, linestyle="--", label="Predicted", linewidth=2, marker="o", markersize=3)
    plt.axvline(input_part["datetime"].iloc[-1], linestyle=":", alpha=0.7)
    if model1 is not None:
        title = f"[{industry_name} | {datatype}] 2days usage_kWh Forecast | {model1.__class__.__name__} + {model2.__class__.__name__} | Input={input_window}, Output={output_window}"
    else:
        title = f"[{industry_name} | {datatype}] 2days usage_kWh Forecast | {model2.__class__.__name__} | Input={input_window}, Output={output_window}"
    plt.title(title, fontweight='bold')
    plt.xlabel("Time (MM-DD HH)")
    plt.ylabel("Usage (kWh)")
    plt.xticks(rotation=30)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    margin = pd.Timedelta(minutes=0)  # 양쪽 분 단위 여백
    plt.xlim([input_start - margin, view_end_time + margin])

    plt.tight_layout()
    
    if not os.path.exists(f"plots/{industry_name}"):
        os.makedirs(f"plots/{industry_name}")
    if model1 is not None:
        plt.savefig(f"plots/{industry_name}/[{industry_name}] {view_days}days_{datatype}_forecast({model1.__class__.__name__}_with_{model2.__class__.__name__})_({input_window},{output_window}).png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(f"plots/{industry_name}/[{industry_name}] {view_days}days_{datatype}_forecast({model2.__class__.__name__})_({input_window},{output_window}).png", dpi=300, bbox_inches="tight")

    # plt.show() # 주석 처리 - 노트북에서 실행 시 사용