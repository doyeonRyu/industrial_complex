# Autoformer 인코더-디코더 모듈 구현
import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Class: my_Layernorm
        - 사용자 정의 LayerNorm 클래스
        - 입력 텐서의 각 채널에 대해 Layer Normalization을 수행한 후, 평균 bias를 제거
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__() 
        self.layernorm = nn.LayerNorm(channels) # LayerNorm 인스턴스 생성

    def forward(self, x):
        """
        Function: forward
            - 입력 텐서 x에 대해 Layer Normalization을 수행하고, 평균 bias를 제거
        Parameters:
            - x: 입력 텐서 
                - shape: [batch_size, seq_len, channels]
        Returns:
            - bias가 제거된 Layer Normalized 텐서
                - shape: [batch_size, seq_len, channels]
        """
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Class: moving_avg
        - 이동 평균을 계산하는 모듈
        - 이동 평균: CNN 모델처럼 kernel과 stride를 사용하여 시계열 데이터의 이동 평균을 계산
        - series_decomp 모듈에서 사용
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size # 이동 평균을 계산할 커널 크기 
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0) # 1D 평균 풀링 레이어 생성

    def forward(self, x):
        """
        Function: forward
            - 입력 시계열 데이터 x에 대해 이동 평균을 계산
        Parameters:
            - x: 입력 시계열 데이터 
                - shape: [batch_size, seq_len, channels]
        Returns:
            - 이동 평균이 적용된 시계열 데이터 
                - shape: [batch_size, seq_len, channels]
        """
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Class: series_decomp
        - 시계열 분해 모듈
        - 시계열 데이터를 추세(trend)와 잔차(residual)로 분해
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1) # 이동 평균 모듈 생성

    def forward(self, x):
        """
        Function: forward
            - 입력 시계열 데이터 x를 추세와 잔차로 분해
        Parameters:
            - x: 입력 시계열 데이터 
                - shape: [batch_size, seq_len, channels]
        Returns:
            - res: residual 시계열 데이터
                - shape: [batch_size, seq_len, channels]
            - moving_mean: trend 시계열 데이터
                - shape: [batch_size, seq_len, channels]
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Class: EncoderLayer
        - Autoformer 인코더 레이어
        - attention: self-attention 메커니즘
        - decomposition: trend와 residual로 시계열 분해
        - encoder 모듈에서 사용 
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        """
        Function: forward
            - 입력 텐서 x에 대해 self-attention과 시계열 분해를 수행
        Parameters:
            - x: 입력 텐서 
                - shape: [batch_size, seq_len, d_model]
            - attn_mask: attention 마스크
        Returns:
            - res: 분해된 residual 텐서
                - shape: [batch_size, seq_len, d_model]
            - attn: attention 가중치
                - shape: [batch_size, num_heads, seq_len, seq_len]
        """
        new_x, attn = self.attention( # Self-Attention 메커니즘 적용
            x, x, x, # Query, Key, Value 모두 동일한 입력 x 사용
            attn_mask=attn_mask # attention 마스크 적용
        )
        x = x + self.dropout(new_x) # 잔차 연결 및 드롭아웃 적용
        x, _ = self.decomp1(x) # 첫 번째 시계열 분해
        y = x # 잔차에 대한 feed-forward 네트워크
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) # 1D 컨볼루션 및 활성화 함수 적용
        y = self.dropout(self.conv2(y).transpose(-1, 1)) #
        res, _ = self.decomp2(x + y) # 두 번째 시계열 분해
        return res, attn


class Encoder(nn.Module):
    """
    Class: Encoder
        - Autoformer 인코더
        - 여러 개의 EncoderLayer로 구성
        - 기존 encoder 형식과 달리 convolutional layer, normalization layer 등을 포함할 수 있음
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        """
        Function: forward
            - 입력 텐서 x에 대해 여러 개의 EncoderLayer를 순차적으로 적용
        Parameters:
            - x: 입력 텐서 
                - shape: [batch_size, seq_len, d_model]
            - attn_mask: attention 마스크
        Returns:
            - x: 최종 출력 텐서
                - shape: [batch_size, seq_len, d_model]
            - attns: 각 레이어의 attention 가중치 리스트
                - shape: [batch_size, num_heads, seq_len, seq_len]
        """
        attns = []
        if self.conv_layers is not None: # convolutional 레이어가 있는 경우
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None: # 정규화 레이어가 있는 경우
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Class: DecoderLayer
        - Autoformer 디코더 레이어
        - self-attention과 cross-attention 메커니즘을 포함
        - 시계열 분해 및 feed-forward 네트워크 포함
        - decoder 모듈에서 사용
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Function: forward
            - 입력 텐서 x에 대해 self-attention, cross-attention, 시계열 분해를 수행
        Parameters:
            - x: 입력 텐서 
                - shape: [batch_size, seq_len, d_model]
            - cross: cross-attention을 위한 텐서
                - shape: [batch_size, seq_len, d_model]
            - x_mask: self-attention 마스크
            - cross_mask: cross-attention 마스크
        Returns:
            - x: 분해된 residual 텐서
                - shape: [batch_size, seq_len, d_model]
            - residual_trend: 추세 텐서
                - shape: [batch_size, seq_len, c_out]
        """
        x = x + self.dropout(self.self_attention( # Self-Attention 메커니즘 적용
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x) # 첫 번째 시계열 분해
        x = x + self.dropout(self.cross_attention( # Cross-Attention 메커니즘 적용
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x) # 두 번째 시계열 분해
        y = x # 잔차에 대한 feed-forward 네트워크
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y) # 세 번째 시계열 분해

        residual_trend = trend1 + trend2 + trend3 # 추세 성분 합산
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Class: Decoder
        - Autoformer 디코더
        - 여러 개의 DecoderLayer로 구성
        - 기존 decoder 형식과 달리 convolutional layer, normalization layer 등을 포함할 수 있음
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        """
        Function: forward
            - 입력 텐서 x에 대해 여러 개의 DecoderLayer를 순차적으로 적용
        Parameters:
            - x: 입력 텐서 
                - shape: [batch_size, seq_len, d_model]
            - cross: cross-attention을 위한 텐서
                - shape: [batch_size, seq_len, d_model]
            - x_mask: self-attention 마스크
            - cross_mask: cross-attention 마스크
            - trend: 초기 추세 텐서
                - shape: [batch_size, seq_len, c_out]
        Returns:
            - x: 최종 출력 텐서
                - shape: [batch_size, seq_len, d_model]
            - trend: 최종 추세 텐서
                - shape: [batch_size, seq_len, c_out]
        """
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask) # 각 디코더 레이어 적용
            trend = trend + residual_trend # 추세 성분 누적

        if self.norm is not None: # 정규화 레이어가 있는 경우
            x = self.norm(x)

        if self.projection is not None: # 프로젝션 레이어가 있는 경우
            x = self.projection(x)
        return x, trend
