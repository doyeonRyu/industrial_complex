import torch
import numpy as np

@torch.no_grad()
def predict(x_input, best_model1_path, best_model2_path,
                 model1, model2, device,
                 input_window=36, output_window=32):
    """
    Function: predict
        - 하나의 입력 구간(input_window)을 사용하여 output_window 시점 예측
        - 학습된 모델을 불러와 실제 예측 수행
    
    Parameters:
        - x_input: torch.Tensor, shape [1, input_window, F]
            예측에 사용할 입력 시계열 (배치 1개)
        - best_model1_path: CNN 가중치 경로 (None 가능)
        - best_model2_path: LSTM/Transformer 가중치 경로
        - model1: CNN 모델 (None 가능)
        - model2: Transformer / RNN / LSTM 등 시계열 모델
        - device: torch.device
        - input_window: int, 입력 시점 길이 (예: 36)
        - output_window: int, 예측 시점 길이 (예: 32)
    
    Returns:
        - yhat_np: np.ndarray, shape [output_window, output_dim]
            예측된 시계열 결과 (예: 32개 시점)
    """

    # 1. 모델 로드 및 평가 모드 전환
    if model1 is not None:
        model1.load_state_dict(torch.load(best_model1_path, map_location=device))
        model1.to(device)
        model1.eval()
    model2.load_state_dict(torch.load(best_model2_path, map_location=device))
    model2.to(device)
    model2.eval()

    # 2. 입력 텐서를 장치로 이동
    xb = x_input.to(device)  # [1, input_window, F]

    # 3. CNN이 있는 경우
    #    입력 형태: [B, F, L] (배치 크기, 피처 수, 시퀀스 길이) -> CNN 출력 형태 [B, L, F]
    if model1 is not None: 
        # CNN 입력 형태로 변환 [B, F, L]
        # [B(배치 사이즈), L(input_window 길이), F(feature 수)] -> [B, F, L]
        if xb.dim() == 3:
            # L, F 위치 변환
            xb = xb.permute(0, 2, 1) # [B, L, F] -> [B, F, L]
        elif xb.dim() == 2: # L=1인 경우(입력 피처가 1개) [B, F] 형태
            xb = xb.unsqueeze(-1) # [B, F] -> [B, F, 1] # L 차원 추가
        
        # 1) model1 forward 
        md1_out = model1(xb) # xb 형태: [B, F, L] -> md1_out 형태: [B, L, F]
        src = md1_out # CNN 출력값을 model2의 입력으로 사용 # 형태 [B, L, F]
    else:
        src = xb # CNN이 없는 경우 원본 입력 사용 # 형태 [B, L, F]

    # 4. Transformer (Encoder-Decoder 구조)
    if hasattr(model2, "decoder") or hasattr(model2, "dec_embedding"):
        output_dim = model2.output_dim if hasattr(model2, 'output_dim') else src.size(-1)

        # 디코더 초기 입력: src의 마지막 시점 (예: x_36)
        dec_input = src[:, -1:, :output_dim].clone().to(device)

        outputs = []
        for _ in range(output_window):  # 32 step 예측
            y_pred = model2(src, dec_input)
            next_pred = y_pred[:, -1:, :]       # 마지막 예측 시점
            outputs.append(next_pred)
            dec_input = torch.cat([dec_input, next_pred], dim=1)  # 누적 입력

        yhat = torch.cat(outputs, dim=1)  # [1, 32, output_dim]

    # 5. Encoder-only 모델
    elif isinstance(model2, torch.nn.TransformerEncoder) or hasattr(model2, "encoder"):
        yhat = model2(src)

    # 6. RNN / LSTM / GRU 모델
    else:
        yhat = model2(src)

    # 7. 예측 결과 반환
    yhat_np = yhat.detach().cpu().numpy().squeeze()  # [32, F] 형태로 변환
    
    return yhat_np
