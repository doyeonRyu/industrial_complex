# LSTM model | Basic version | for time series forecasting
import torch
import torch.nn as nn

"""
==============================================================================
LSTM 모델
- 입력 형태 (최종 전처리 형태): 
    - (B, L, F)
        - B: 배치 크기 | L: 시퀀스 길이 | F: 입력 피처 수
- 출력 형태 (최종 예측 형태):
    - (B, output_size)
        - B: 배치 크기 | output_size: 예측 변수 수
==============================================================================
"""
class LSTM(nn.Module):
    """
    Class: LSTM
        - 기본적인 LSTM 모델 구현
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size, 
        num_layers,
        bidirectional=False,
        dropout=0.1,
    ):
        """
        Function: __init__
            - LSTM 모델 초기화
        Parameters:
            - input_size (int): LSTM 입력 피처 수 (torch_train_x.shape[-1])
            - hidden_size (int): LSTM 은닉 상태 크기
            - output_size (int): 예측 변수 수 (single-step = 1, multi-step = H)
            - num_layers (int): LSTM 레이어 수
            - bidirectional (bool): 양방향 LSTM 여부
            - dropout (float): 드롭아웃 비율
        Returns:
            - None
        """
        super().__init__()
        self.hidden_size = hidden_size # LSTM hidden state 크기
        self.num_layers = num_layers # LSTM 레이어 수
        self.num_directions = 2 if bidirectional else 1 # 양방향 여부 (1: 단방향, 2: 양방향)

        # 1) LSTM 정의
        self.lstm = nn.LSTM( # LSTM 레이어
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0 # LSTM dropout은 layer > 1에서만 적용
        )

        # 2) Fully Connected Head (LSTM 출력 -> 최종 출력)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size) # 출력 레이어 정의

    def forward(self, x):
        """
        Function: forward
            - LSTM 모델 순전파
        Parameters:
            - x (torch.Tensor): 입력 시퀀스 (B, L, input_size) 
                - B: 배치 크기, L: 시퀀스 길이, input_size: 입력 피처 수
        Returns:
            - y_hat (torch.Tensor): 예측 출력 (B, output_size)
        """
        # 3) 입력 x를 LSTM에 통과
        out, (h_n, c_n) = self.lstm(x) 
        #    x: (B, L, input_size)
        #    out: 전체 시퀀스 출력, (B, L, H * num_directions)
        #    h_n: 마지막 hidden states, (num_layers * num_directions, B, H)
        #    c_n: 마지막 cell states, (num_layers * num_directions, B, H)
        
        # 4) 마지막 레이어의 hidden state만 추출
        h_last = h_n[-1] # (B, H * num_directions)

        # 5) FC Head 통과 후 최종 출력
        y_hat = self.fc(h_last) # (B, output_size)

        return y_hat # (B, output_size) 