import torch
import torch.nn as nn
from layers.embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(nn.Module):
    """
    Class: Autoformer 모델 클래스
        - Autoformer: transformer 기반의 시계열 예측 모델
        - 장기 의존성 문제 해결을 위해 Auto-Correlation 메커니즘 도입
        - 기존에 parser 인자로 받던 부분들을 __init__ 함수의 인자로 직접 받도록 수정
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, embed='fixed', freq='h', activation='gelu',
                 output_attention=False,
                 moving_avg=25,
                 device=torch.device('cuda:0')):
        """
        Function: __init__
            - Autoformer 모델 초기화
        Parameters:
            - enc_in: 인코더 입력 차원 (예: 피처 수)
            - dec_in: 디코더 입력 차원 (예: 피처 수)
            - c_out: 출력 차원 (예: 예측할 피처 수)
            - seq_len: 입력 시퀀스 길이
            - label_len: 디코더에서 참조하는 과거 시퀀스 길이
            - pred_len: 예측할 시퀀스 길이
            - factor: Auto-Correlation 메커니즘의 축소 인자
            - d_model: 모델 차원
            - n_heads: 멀티헤드 어텐션의 헤드 수
            - e_layers: 인코더 레이어 수
            - d_layers: 디코더 레이어 수
            - d_ff: 피드포워드 네트워크의 차원
            - dropout: 드롭아웃 비율
            - embed: 임베딩 유형 (예: 'fixed', 'learnable')
            - freq: 시계열 데이터의 주기성 빈도 (예: 'h' - 시간별)
            - activation: 활성화 함수 유형 (예: 'gelu', 'relu')
            - output_attention: 어텐션 가중치 출력 여부
            - moving_avg: 시계열 분해를 위한 이동 평균 윈도우 크기
            - device: 모델이 실행될 디바이스 (예: CPU 또는 GPU)
        Returns:
            - None
        """
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Decomposition
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # Autoformer는 positional embedding을 사용하지 않음
        # DataEmbedding_wo_pos 사용
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)

        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)
        
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
