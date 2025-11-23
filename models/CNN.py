import torch.nn as nn

"""
==============================================================================
1-D CNN 모델
- 입력 형태 (최종 전처리 형태): 
    - [B, C_in, L_in]
        - B: 배치 크기 | C_in: 입력 채널 수 | L_in: 시퀀스 길이
- 출력 형태 (시계열 데이터 예측 모델에 전달되는 형태):
    - [B, L_out=1, F]   
        - B: 배치 크기 | L_out: 출력 시퀀스 길이 (1로 고정) | F: LSTM 입력 피처 수
==============================================================================
"""
class CNN(nn.Module):
    """
    Class: CNN
        - CNN + 예측 하이브리드 모델을 위한 1-D CNN 모델
        1) 1-D CNN으로 특징 추출
        2) 예측 모델 입력 형태에 맞게 차원 변환 [B, C_out, L_out] -> [B, L_out, C_out] 
            - 예측 모델 입력 형태: [B, L, F]
                - L_out = 시퀀스 길이 
                - C_out = 입력 피처 수 (F)
        3) 예측 모델에 전달
    """
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                dilation=1,
                padding=0,
                groups=1,
                bias=True,
                padding_mode='zeros',
                pool_kernel_size=2,
                pool_stride=2,
                dropout=0.0,
                next_in_features=32, 
                use_bn=True 
    ) -> None:
        """
        Function: __init__
            - CNN 모델 초기화
        Parameters:
            - in_channels (int): 입력 피처 수
            - out_channels (int): CNN에서 추출할 feature map 수 (입력 feature를 통해 out_channels 수 만큼 새로운 특징 추출)
            - kernel_size (int): 한 kernel당 timestamp 개수
            - stride (int): kernel 이동 크기
            - dilation (int): kernel 내부에서 얼마만큼 띄어서 kernel을 적용할 것인가 (default: 1)
            - padding (int): 한 쪽 방향으로 얼마만큼 padding할 것인가 (그 만큼 양방향으로 적용) (default: 0)
            - groups (int): kernel의 height를 조절
            - bias (int): bias term을 둘 것인가 
            - padding_mode (str): padding 채우는 방식, 'zero', 'reflect', 'replicate', 'circular' (default: 'zero')
            - pool_kernel_size (int): max pooling 커널 크기
            - pool_stride (int): max pooling 스트라이드 크기
            - dropout (float): 드롭아웃 비율
            - next_in_features (int): 예측 모델 입력 피처 수, out_channels에서 fully connected layer로 축소
            - use_bn (bool): 배치 정규화 사용 여부
        Return: 
            - None
        """
        super().__init__()

        # 1D CNN 레이어
        self.cnn = nn.Conv1d(in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            dilation,
                            padding,
                            groups,
                            bias,
                            padding_mode)
        
        # 배치 정규화 레이어 
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        # 활성화 함수 ReLU
        self.relu = nn.ReLU()
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        # max pooling 레이어 [B, C_out, L_conv] -> [B, C_out, L_pool]
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        # 평탄화 레이어 [B, C_out, L_pool] -> [B, C_out * L_pool]
        self.flatten = nn.Flatten()
        # 예측 모델 입력에 맞도록 Lazy Linear 레이어로 차원 맞추기
        self.fc = nn.LazyLinear(next_in_features)

    def forward(self, x):
        """
        Function: forward
            - CNN 모델 순전파
        Parameters:
            - x (torch.Tensor): 입력 시퀀스 (B, C_in, L_in) 
                - B: 배치 크기 | C_in: 입력 채널 수 | L_in: 시퀀스 길이
        Return:
            - x (torch.Tensor): 예측 모델 입력에 맞게 변환된 시퀀스 (B, L_out=1, C_out=F)
                - B: 배치 크기, L_out: 출력 시퀀스 길이, C_out: 출력 채널 수, F: 예측 모델 입력 피처 수
        """
        
        # 1) Conv 블록 
        x = self.cnn(x) # [B, C_out, L_out]
        x = self.bn(x) # 배치 정규화 [B, C_out, L_out]
        x = self.relu(x) # 활성화 함수 [B, C_out, L_out]
        x = self.dropout(x) # 드롭아웃 [B, C_out, L_out]

        # 2) Pooling + Flatten
        x = self.pool(x) # max pooling [B, C_out, L_pool]
        x = self.flatten(x) # 평탄화 [B, C_out * L_pool]

        # 3) 예측 모델 입력에 맞게 차원 변환
        x = self.fc(x) # [B, F(next_in_features)]
        x = x.unsqueeze(1) # 예측 모델 입력에 맞게 차원 추가 [B, 1, F]

        return x # [B, L_out=1, C_out=F]
