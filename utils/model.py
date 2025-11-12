from models.CNN import CNN
from models.LSTM import LSTM
from models.Transformer import Transformer_encoder
from models.Transformer import Transformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# 모델 초기화 함수
def build_model(model1, model2, train_preprocessed, input_window, output_window, device):
    """
    Function: build_model
        - 하이브리드 or 단일 시계열 예측 모델 초기화
        - model1: CNN 모델 (None 가능)
        - model2: 시계열 데이터 처리 모델 ("LSTM" or "Transformer")
    Parameters:
        - model1: str or None, "CNN" or None
        - model2: str, "LSTM" or "Transformer"
        - train_preprocessed: pandas DataFrame, 학습 데이터 (전처리된 상태)
        - input_window: int, 입력 시퀀스 길이
        - output_window: int, 출력 시퀀스 길이
        - device: torch.device, 모델과 데이터를 올릴 디바이스 (cpu or cuda)
    Returns:
        - model1, model2: 초기화된 모델 (model1은 None 가능)
    """
    if model1 is not None: # CNN + 시계열 예측 모델 하이브리드 모델 (2-step hybrid)
        # CNN 모델 생성
        cnn = CNN(
            in_channels=train_preprocessed.shape[1], # 입력 피처 수
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

        elif model2 == "Transformer_encoder":
            # Transformer 모델 생성
            # Transformer 파라미터 설정
            transformer_encoder = Transformer_encoder(
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
            print("Transformer (encoder-only): \n", transformer_encoder)
            return cnn, transformer_encoder
        elif model2 == "Transformer":
            # Transformer 모델 생성
            # Transformer 파라미터 설정
            transformer = Transformer(
                input_dim=next_in_features,
                d_model=d_model,
                nhead=nhead,
                num_enc_layers=num_ts_enc_layers,
                num_dec_layers=num_ts_dec_layers,
                dim_feedforward=dim_feedforward,
                output_window=output_window,
                output_dim=1,
                dropout=ts_dropout,
                max_len = input_window,
            ).to(device)
            print("Transformer: \n", transformer)
            return cnn, transformer
        
    else: # 시계열 예측 모델 단독
        if model2 == "LSTM":
            # LSTM 모델 생성
            # LSTM 파라미터 설정
            # LSTM 모델 생성
            lstm = LSTM(
                input_size = train_preprocessed.shape[1], # LSTM에 처음 data input이므로 변수 개수와 동일하게 들어가야 함
                hidden_size=hidden_size,
                output_size=output_window,
                num_layers=num_lstm_layers,
                dropout=lstm_dropout
            ).to(device)
            print("LSTM: \n", lstm)
            return None, lstm
        
        elif model2 == "Transformer_encoder":
            # Transformer 모델 생성
            # Transformer 파라미터 설정
            transformer_encoder = Transformer_encoder(
                input_dim=train_preprocessed.shape[1],
                d_model=d_model,
                nhead=nhead,
                num_enc_layers=num_ts_layers,
                dim_feedforward=dim_feedforward,
                output_window=output_window,
                dropout=ts_dropout,
                max_len = input_window,
                head_hidden=d_model//2
            ).to(device)
            print("Transformer (encoder-only): \n", transformer_encoder)
            return None, transformer_encoder
        elif model2 == "Transformer":
            # Transformer 모델 생성
            # Transformer 파라미터 설정
            transformer = Transformer(
                input_dim=train_preprocessed.shape[1],
                d_model=d_model,
                nhead=nhead,
                num_enc_layers=num_ts_enc_layers,
                num_dec_layers=num_ts_dec_layers,
                dim_feedforward=dim_feedforward,
                output_window=output_window,
                output_dim=1,
                dropout=ts_dropout,
                max_len = input_window,
            ).to(device)
            print("Transformer: \n", transformer)
            return None, transformer
