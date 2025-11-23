"""
==============================================================================
File: plot.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-15
Last Modified: 2025-11-19

Description: 하이브리드 혹은 단일 시계열 예측 모델 결과 시각화
    Functions:
        - plot_predictions_chained: 모델의 예측 결과를 시각화
            - 원하는 날짜만큼 예측 결과 병렬 시각화 
        -plot_single_predictions: 모델 예측 결과를 한 output_window만큼 시각화

==============================================================================
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

"""
산업 단지 예측 결과 시각화 함수
"""
def plot_predictions_chained(
    best_model1_path, best_model2_path,
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

    input_last_idx   = start_idx + input_window - 1 # 입력 구간의 마지막 시점 인덱스
    output_first_idx = start_idx + input_window # 예측(출력) 구간의 첫 시점 인덱스

    start_time   = df.loc[start_idx, "datetime"]
    input_end    = df.loc[input_last_idx, "datetime"] # 입력 종료 시점(마지막 입력)
    view_end_time = input_end + pd.Timedelta(days=view_days) # 시각화 종료 시점

    # 입력 구간: [start_time, input_end]
    input_mask = (df["datetime"] >= start_time) & (df["datetime"] <= input_end)
    input_part = df.loc[input_mask, ["datetime", "usage_kWh"]].copy()

    # 출력(실측) 구간: (input_end, view_end_time]  → 입력 다음 시점부터
    out_mask = (df["datetime"] > input_end) & (df["datetime"] <= view_end_time)
    output_part = df.loc[out_mask, ["datetime", "usage_kWh"]].copy()

    # 예측을 view_days 단위 만큼 수행
    if model1 is not None:
        model1.load_state_dict(torch.load(best_model1_path, map_location=device))
        model1.to(device)
        model1.eval()
    model2.load_state_dict(torch.load(best_model2_path, map_location=device))
    model2.to(device)
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

    seen = 0 
    pred_map = {} # 예측 결과를 시간별로 누적(겹치면 최신 예측으로 덮어쓰기)

    
    with torch.no_grad():
        for batch in data_loader:
            xb, yb = batch
            xb = batch[0]
            bs = xb.size(0)

            g0 = seen # 글로벌 오프셋 시작
            g1 = seen + bs # 글로벌 오프셋 끝 (미포함)

            # 이번 배치에서 필요한 인덱스들만 선별
            to_pick = [idx for idx in needed_indices if g0 <= idx < g1]
            if to_pick:
                xb = xb.to(device)
                yb = yb.to(device)
                
                if model1 is not None:
                    # CNN 입력 형태 변환
                    if xb.dim() == 3: 
                        xb = xb.permute(0, 2, 1)
                    elif xb.dim() == 2: 
                        xb = xb.unsqueeze(1)
                    md1_out = model1(xb)
                    src = md1_out
                else:
                    src = xb
                
                # 2) model2 forward
                # 2-1) encoder-decoder 구조일 때
                if hasattr(model2, "decoder") or hasattr(model2, "dec_embedding"):
                    # Autoregressive 방식으로 디코더 입력 생성

                    #    디코더 초기 입력을 x_36 (src의 마지막 시점)으로 설정
                    #    src: (B, L_enc, F). src의 마지막 시점의 output_dim 개수 피처를 사용한다고 가정
                    #    모델의 output_dim에 맞게 src의 마지막 값 중 해당 차원만 선택
                    output_dim = model2.output_dim if hasattr(model2, 'output_dim') else yb.size(-1) 
                    
                    # x_36 (src의 마지막 시점 값)을 가져옴: (B, 1, output_dim)
                    # src의 마지막 시점의 첫 output_dim 피처를 사용
                    dec_input = src[:, -1:, :output_dim].clone().to(device)

                    outputs = []

                    # autoregressive decoding loop 횟수를 output_window (32회)로 변경
                    for _ in range(output_window): # 32번 반복 (t=37부터 t=68까지)
                        
                        y_pred = model2(src, dec_input) 
                        next_pred = y_pred[:, -1:, :]  # 새로 예측된 마지막 시점 (t=37, t=38, ...)
                        outputs.append(next_pred)
                        dec_input = torch.cat([dec_input, next_pred], dim=1)  # 예측 누적

                    yhat = torch.cat(outputs, dim=1) # [B, output_window, F] # 32개 시점의 예측: y_37부터 y_68

                    tgt_target = yb if yb.dim() == 3 else yb.unsqueeze(-1)  # [B, output_window, F] # 32개의 실제값

                # 2-2) encoder-only 구조일 때
                elif isinstance(model2, torch.nn.TransformerEncoder) or hasattr(model2, "encoder"):
                    yhat = model2(src)
                    tgt_target = yb

                # 2-3) 일반 RNN류 (LSTM, GRU 등)
                else:
                    yhat = model2(src)
                    tgt_target = yb

                # 차원 보정
                if yhat.dim() == 3 and yhat.size(-1) == 1:
                    yhat = yhat.squeeze(-1)
                if tgt_target.dim() == 3 and tgt_target.size(-1) == 1:
                    tgt_target = tgt_target.squeeze(-1)

                # CPU로 이동 후 numpy 변환
                yhat_np = yhat.detach().cpu().numpy() # [B, H] 또는 [B,]

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
            
            # 모든 needed_indices를 순차했으면 종료
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
    weekday_fmt = mdates.DateFormatter('%m-%d (%a)\n%H:%M')

    plt.figure(figsize=(10, 4.8))
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    plt.plot(input_part["datetime"], input_part["usage_kWh"], label="Input (Actual)", linewidth=2)
    plt.plot(output_part["datetime"], output_part["usage_kWh"], label="Output (Actual)", linewidth=2)
    plt.plot(pred_time, y_pred_rec, linestyle="--", label="Predicted", linewidth=2, marker="o", markersize=3)
    plt.axvline(input_end, linestyle=":", alpha=0.7)
    if model1 is not None:
        title = f"[{industry_name} | {datatype}] {view_days}days usage_kWh Forecast | {model1.__class__.__name__} + {model2.__class__.__name__} | Input={input_window}, Output={output_window}"
    else:
        title = f"[{industry_name} | {datatype}] {view_days}days usage_kWh Forecast | {model2.__class__.__name__} | Input={input_window}, Output={output_window}"
    plt.title(title, fontweight='bold')
    plt.xlabel("Time (MM-DD HH)")
    plt.ylabel("Usage (kWh)")
    plt.xticks(rotation=30)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    # 날짜 포맷 지정
    plt.gca().xaxis.set_major_formatter(weekday_fmt)
    # 보기 범위 조정
    margin = pd.Timedelta(minutes=0) # 좌우 여백
    plt.xlim([start_time - margin, view_end_time + margin])
    plt.tight_layout()
    
    if not os.path.exists(f"plots/{industry_name}"):
        os.makedirs(f"plots/{industry_name}")
    if model1 is not None:
        plt.savefig(f"plots/{industry_name}/[{industry_name}] {view_days}days_{datatype}_forecast({model1.__class__.__name__}_with_{model2.__class__.__name__})_({input_window},{output_window}).png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(f"plots/{industry_name}/[{industry_name}] {view_days}days_{datatype}_forecast({model2.__class__.__name__})_({input_window},{output_window}).png", dpi=300, bbox_inches="tight")

    # plt.show() # 주석 처리 - 노트북에서 실행 시 사용

# input -> output 한 세트 결과만 출력
def plot_single_prediction(
    best_model1_path, best_model2_path,
    model1, model2, data_loader, df, device,
    input_window=10, output_window=1, start_idx=0,
    scaler=None,      
    logged: bool = False, 
    industry_name="industry_name",
    datatype="Validation", # or "Test",
    peak_weight=False
):
    # 역변환 함수
    def _inv_transform_1d(arr_1d):
        arr_2d = arr_1d.reshape(-1, 1)

        # 1) scaler 역변환
        if scaler is not None:
            arr_2d = scaler.inverse_transform(arr_2d)

        # 2) 로그 역변환
        if logged:
            arr_2d = np.expm1(np.maximum(arr_2d, 0))
            
        return arr_2d.reshape(-1)

    # 시간 정렬 및 범위 확인
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
        df["datetime"] = (
            pd.to_datetime(df["datetime"], utc=True)  
              .dt.tz_convert("Asia/Seoul")   
              .dt.tz_localize(None)  
        )

    if start_idx + input_window + output_window > len(df):
        print(f"[warn] start_idx({start_idx}) + input_window({input_window}) + output_window({output_window}) > len(df)({len(df)})")
        return

    # 인덱스 계산
    input_start_idx = start_idx
    input_end_idx = start_idx + input_window - 1
    output_start_idx = start_idx + input_window
    output_end_idx = start_idx + input_window + output_window - 1

    # Input 구간 실측값
    input_mask = (df.index >= input_start_idx) & (df.index <= input_end_idx)
    input_part = df.loc[input_mask, ["datetime", "usage_kWh"]].copy()

    # Output 구간 실측값
    output_mask = (df.index >= output_start_idx) & (df.index <= output_end_idx)
    output_part = df.loc[output_mask, ["datetime", "usage_kWh"]].copy()

    # 모델 평가 모드
    if model1 is not None:
        model1.load_state_dict(torch.load(best_model1_path, map_location=device))
        model1.to(device)
        model1.eval()
    model2.load_state_dict(torch.load(best_model2_path, map_location=device))
    model2.to(device)
    model2.eval()

    seen = 0
    pred_result = None

    with torch.no_grad():
        for batch in data_loader:
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)
            
            bs = xb.size(0)
            g0, g1 = seen, seen + bs

            # 현재 배치에 start_idx 포함 여부 확인
            if g0 <= start_idx < g1:
                batch_offset = start_idx - g0

                # ====================================================
                # 1) CNN 모델이 있는 경우 (하이브리드) 
                # ====================================================

                #    CNN 입력 형태: [B, F, L] (배치 크기, 피처 수, 시퀀스 길이) -> CNN 출력 형태 [B, L, F]
                if model1 is not None: 
                    # CNN 입력 형태로 변환 [B, L, F] -> [B, F, L]
                    if xb.dim() == 3:
                        xb = xb.permute(0, 2, 1) # L, F 위치 변환
                    elif xb.dim() == 2: # L=1인 경우(입력 피처가 1개) [B, F] 형태
                        xb = xb.unsqueeze(-1) # [B, F] -> [B, F, 1] # L 차원 추가
                    
                    # 1) model1 forward 
                    md1_out = model1(xb) # md1_out 형태: [B, L, F]
                    src = md1_out # CNN 출력값을 model2의 입력으로 사용
                else:
                    src = xb # CNN이 없는 경우 원본 입력 사용

                # ====================================================
                # 2) model2 forward
                # ====================================================
                
                # 2-1) encoder-decoder 구조일 때
                if hasattr(model2, "decoder") or hasattr(model2, "dec_embedding"):
                    # 디코더 입력 초기화 (첫 시점만 실제값으로 시작)
                    if yb.dim() == 1:
                        yb = yb.unsqueeze(-1)
                    elif yb.dim() == 2:
                        yb = yb.unsqueeze(-1)
                    
                    dec_input = yb[batch_offset:batch_offset+1, 0:1, :]  # (1,1,F)
                    preds = []

                    for _ in range(output_window - 1):
                        y_pred = model2(src[batch_offset:batch_offset+1], dec_input)
                        next_pred = y_pred[:, -1:, :]
                        preds.append(next_pred)
                        dec_input = torch.cat([dec_input, next_pred], dim=1)

                    yhat = torch.cat(preds, dim=1)
                    pred_seq = yhat.squeeze(0).detach().cpu().numpy().reshape(-1)
                    pred_seq = _inv_transform_1d(pred_seq)

                # 2-2) encoder-only 구조일 때
                elif isinstance(model2, torch.nn.TransformerEncoder) or hasattr(model2, "encoder"):
                    yhat = model2(src[batch_offset:batch_offset+1])
                    pred_seq = yhat.squeeze(0).detach().cpu().numpy().reshape(-1)
                    pred_seq = _inv_transform_1d(pred_seq[:output_window])
                
                # 2-3) 일반 RNN류 (LSTM, GRU 등)
                else:
                    yhat = model2(src[batch_offset:batch_offset+1])
                    pred_seq = yhat.squeeze(0).detach().cpu().numpy().reshape(-1)
                    pred_seq = _inv_transform_1d(pred_seq[:output_window])

                pred_result = pred_seq
                pred_times = df.loc[output_start_idx:output_end_idx, "datetime"].values
                break

            seen += bs
            if seen > start_idx:
                break

    if pred_result is None:
        print("[warn] 예측 결과를 찾을 수 없습니다.")
        return

    # 예측값과 시간 매핑
    min_len = min(len(pred_times), len(pred_result))
    pred_times = pred_times[:min_len]
    y_pred = pred_result[:min_len]

    # -------------------------------------------------------------
    # 시각화
    weekday_fmt = mdates.DateFormatter('%m-%d (%a)\n%H:%M')
    plt.figure(figsize=(10, 4.8))
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # Input 구간
    plt.plot(input_part["datetime"], input_part["usage_kWh"], label="Input (Actual)", linewidth=2)
    # Output 구간 실측값
    plt.plot(output_part["datetime"], output_part["usage_kWh"], label="Output (Actual)", linewidth=2)
    # Output 구간 예측값
    plt.plot(pred_times, y_pred, linestyle="--", label="Predicted", linewidth=2, marker="o", markersize=4)

    # 경계선 표시
    plt.axvline(input_part["datetime"].iloc[-1], linestyle=":", alpha=0.7)

    # 제목
    if model1 is not None:
        title = f"[{industry_name} | {datatype}] usage_kWh Forecast | {model1.__class__.__name__} + {model2.__class__.__name__}"
    else:
        title = f"[{industry_name} | {datatype}] usage_kWh Forecast | {model2.__class__.__name__}"

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=5)) 
    ax.xaxis.set_major_formatter(weekday_fmt) 
    plt.xticks(rotation=30)
    plt.title(title, fontweight='bold')
    plt.xlabel("Time (MM-DD HH)")
    plt.ylabel("Usage (kWh)")
    plt.xticks(rotation=30)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.xlim([input_part["datetime"].iloc[0] - pd.Timedelta(minutes=30),
              output_part["datetime"].iloc[-1] + pd.Timedelta(minutes=30)])
    plt.tight_layout()

    # 저장
    os.makedirs(f"plots/{industry_name}", exist_ok=True)
    if peak_weight == True:
        save_path = (f"plots/{industry_name}/[{industry_name}] {datatype}_forecast"
                        f"({model1.__class__.__name__+'_with_' if model1 else ''}{model2.__class__.__name__})"
                        f"_({input_window},{output_window})_peak_weight.png")
    else:
        save_path = (f"plots/{industry_name}/[{industry_name}] {datatype}_forecast"
                    f"({model1.__class__.__name__+'_with_' if model1 else ''}{model2.__class__.__name__})"
                    f"_({input_window},{output_window}).png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[완료] {save_path} 저장 완료.")

# longformer 
@torch.no_grad()
def plot_single_prediction_longformer(
    best_model1_path, best_model2_path, model1, model2, 
    data_loader, df, device,
    input_window=36, label_len = 18, output_window=32, start_idx=0,
    scaler=None,      
    logged: bool = False, 
    industry_name="industry_name",
    datatype="Validation",
    peak_weight=False
):
    # 역변환 함수
    def _inv_transform_1d(arr_1d):
        arr_2d = arr_1d.reshape(-1, 1)

        # 1) scaler 역변환
        if scaler is not None:
            arr_2d = scaler.inverse_transform(arr_2d)

        # 2) 로그 역변환
        if logged:
            arr_2d = np.expm1(np.maximum(arr_2d, 0))

        return arr_2d.reshape(-1)

    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
        df["datetime"] = (
            pd.to_datetime(df["datetime"], utc=True)
              .dt.tz_convert("Asia/Seoul")
              .dt.tz_localize(None)
        )

    if start_idx + input_window + output_window > len(df):
        print("Index out of range")
        return
    
    if model2.__class__.__name__ == "Informer":
        input_start_idx = start_idx
        input_end_idx = start_idx + input_window + label_len - 1
        output_start_idx = start_idx + input_window + label_len
        output_end_idx = start_idx + input_window + label_len + output_window - 1

    elif model2.__class__.__name__ == "Autoformer":
        input_start_idx = start_idx
        input_end_idx = start_idx + input_window - 1
        output_start_idx = start_idx + input_window
        output_end_idx = start_idx + input_window + output_window - 1
    
    input_part = df.loc[input_start_idx:input_end_idx, ["datetime", "usage_kWh"]]
    output_part = df.loc[output_start_idx:output_end_idx, ["datetime", "usage_kWh"]]

    if model1 is not None:
        model1.load_state_dict(torch.load(best_model1_path, map_location=device))
        model1.to(device)
        model1.eval()
    model2.load_state_dict(torch.load(best_model2_path, map_location=device))
    model2.to(device)
    model2.eval()

    seen = 0
    pred_result = None # 예측 결과 저장 변수

    for batch in data_loader:
        x_enc, x_dec, yb, x_mark_enc, x_mark_dec = [
            b.to(device) for b in batch
        ]

        bs = x_enc.size(0)
        g0, g1 = seen, seen + bs

        # 현재 배치에 start_idx 포함 여부 확인
        if g0 <= start_idx < g1:
            batch_offset = start_idx - g0

            # ====================================================
            # 1) CNN 모델이 있는 경우 (하이브리드) 
            # ====================================================
            if model1 is not None:
                # CNN 입력 형태로 변환 [B, L, F] -> [B, F, L]
                if x_enc.dim() == 3:
                    x_enc = x_enc.permute(0, 2, 1) # L, F 위치 변환
                elif x_enc.dim() == 2: # L=1인 경우(입력 피처가 1개) [B, F] 형태
                    x_enc = x_enc.unsqueeze(-1) # [B, F] -> [B, F, 1] # L 차원 추가
                
                # 1) model1 forward 
                md1_out = model1(x_enc) # md1_out 형태: [B, L, F]
                src = md1_out # CNN 출력값을 model2의 입력으로 사용
            else:
                src = x_enc # CNN이 없는 경우 원본 입력 사용
            
            # ====================================================
            # 2) model2 forward
            # ====================================================
            yhat = model2(
                src[batch_offset:batch_offset+1],
                x_mark_enc[batch_offset:batch_offset+1],
                x_dec[batch_offset:batch_offset+1],
                x_mark_dec[batch_offset:batch_offset+1]
            ) 

            pred_seq = yhat.squeeze(0).detach().cpu().numpy().reshape(-1)
            pred_seq = _inv_transform_1d(pred_seq[:output_window])
            pred_result = pred_seq
            break

        seen += bs

    if pred_result is None:
        print("[warn] 예측 결과를 찾을 수 없습니다.")
        return

    # 예측값과 시간 매핑
    pred_times = df.loc[output_start_idx:output_end_idx, "datetime"].values
    min_len = min(len(pred_times), len(pred_result))
    pred_times = pred_times[:min_len]
    y_pred = pred_result[:min_len]

    # -------------------------------------------------------------
    # 시각화
    weekday_fmt = mdates.DateFormatter('%m-%d (%a)\n%H:%M')

    plt.figure(figsize=(10, 4.8))
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    plt.plot(input_part["datetime"], input_part["usage_kWh"], label="Input (Actual)", linewidth=2)
    plt.plot(output_part["datetime"], output_part["usage_kWh"], label="Output (Actual)", linewidth=2)
    plt.plot(pred_times, y_pred, "--", label="Predicted", linewidth=2, marker="o", markersize=4)

    plt.axvline(input_part["datetime"].iloc[-1], linestyle=":", alpha=0.7)

    title = f"[{industry_name} | {datatype}] Forecast | {model2.__class__.__name__}"
    plt.title(title, fontweight='bold')
    plt.xlabel("Time (MM-DD HH:MM)")
    plt.ylabel("Usage (kWh)")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=5))
    ax.xaxis.set_major_formatter(weekday_fmt)
    plt.xticks(rotation=30)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    plt.xlim([
        input_part["datetime"].iloc[0] - pd.Timedelta(minutes=30),
        output_part["datetime"].iloc[-1] + pd.Timedelta(minutes=30)
    ])

    plt.tight_layout()

    # 저장
    os.makedirs(f"plots/{industry_name}", exist_ok=True)
    if peak_weight:
        save_path = f"plots/{industry_name}/[{industry_name}] {datatype}_forecast({model2.__class__.__name__})_({input_window},{output_window})_peak_weight.png"
    else:
        save_path = f"plots/{industry_name}/[{industry_name}] {datatype}_forecast({model2.__class__.__name__})_({input_window},{output_window}).png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[완료] {save_path} 저장 완료.")
