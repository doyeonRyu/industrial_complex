import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from layers.Informer_EncDec import Encoder, EncoderLayer, ConvLayer, EncoderStack
from layers.Informer_EncDec import Decoder, DecoderLayer
from layers.attn import FullAttention, ProbAttention, AttentionLayer
from layers.embed import DataEmbedding

"""
==============================================================================
Informer 모델
- 입력 형태 (최종 전처리 형태):
    - [batch, seq_len, features]
- 출력 형태 (최종 예측 형태):
    - [batch, out_len, features]
==============================================================================
"""
class Informer(nn.Module):
    """
    Class: Informer
        - Informer 모델 정의
    Parameters:
        - enc_in: 인코더 입력 차원 # transformer: encoder 입력 feature 수
        - dec_in: 디코더 입력 차원 # transformer: decoder 입력 feature 수
        - c_out: 출력 차원 # transformer: 최종 출력 feature 수
        - seq_len: 입력 시퀀스 길이 # transformer: 입력 시퀀스 길이 (input_window)
        - label_len: 라벨 길이 # transformer: 인코더에서 디코더로 전달되는 시퀀스 길이
        - out_len: 출력 시퀀스 길이 # transformer: 출력 시퀀스 길이 (output_window)
        - factor: ProbSparse Attention의 샘플링 팩터 # ProbSparse Attention에서 사용하는 샘플링 팩터
        - d_model: 모델 차원 # transformer: d_model
        - n_heads: 어텐션 헤드 수 # transformer: n_heads
        - e_layers: 인코더 레이어 수 # transformer: num_enc_layers
        - d_layers: 디코더 레이어 수 # transformer: num_dec_layers
        - d_ff: 피드포워드 네트워크 차원 # transformer: d_ff
        - dropout: 드롭아웃 비율 # transformer: dropout
        - attn: 어텐션 타입 ('prob' 또는 'full') # 어텐션 메커니즘 선택
        - embed: 임베딩 타입 ('timeF' 또는 'fixed', 'learned') # 임베딩 방식 선택
        - freq: 시계열 주기 빈도 ('h', 't', 's', 'd', 'b', 'w', 'm', 'q', 'y') # 시계열 데이터의 주기 빈도
        - activation: 활성화 함수 ('relu' 또는 'gelu') # transformer: 활성화 함수(gelu)
        - output_attention: 어텐션 가중치 출력 여부 # 어텐션 가중치 출력 여부 
        - distil: 인코더 디스틸링 사용 여부# 인코더에서 디스틸링 사용 여부
        - mix: 디코더 믹스 어텐션 사용 여부 # 디코더에서 믹스 어텐션 사용 여부
        - device: 디바이스 설정 # transformer: device *(cuda)
    Returns:    
        - 모델 출력 텐서 및 (선택적) 어텐션 가중치
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device="cuda"):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        
        # 어텐션 메커니즘 선택
        #    ProbAttention vs FullAttention
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # 인코더 
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 디코더 
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if not hasattr(self, "_printed_input"):
            # print(f"[DEBUG] x_enc shape: {x_enc.shape}") [512, 36, 15]
            # print(f"[DEBUG] x_mark_enc shape: {x_mark_enc.shape}") [512, 36, 5]
            self._printed_input = True
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [batch size(B), out_len(L), feature dimension(D)]



"""
InformerStack 모델
- informer 모델의 스택 버전
"""
class InformerStack(nn.Module):
    """
    Class: InformerStack
        - InformerStack 모델 정의
    Parameters:
        - enc_in: 인코더 입력 차원
        - dec_in: 디코더 입력 차원
        - c_out: 출력 차원
        - seq_len: 입력 시퀀스 길이
        - label_len: 라벨 길이
        - out_len: 출력 시퀀스 길이
        - factor: ProbSparse Attention의 샘플링 팩터
        - d_model: 모델 차원
        - n_heads: 어텐션 헤드 수
        - e_layers: 인코더 레이어 수 리스트
        - d_layers: 디코더 레이어 수
        - d_ff: 피드포워드 네트워크 차원
        - dropout: 드롭아웃 비율
        - attn: 어텐션 타입 ('prob' 또는 'full')
        - embed: 임베딩 타입 ('timeF' 또는 'fixed', 'learned')
        - freq: 시계열 주기 빈도 ('h', 't', 's', 'd', 'b', 'w', 'm', 'q', 'y')
        - activation: 활성화 함수 ('relu' 또는 'gelu')
        - output_attention: 어텐션 가중치 출력 여부
        - distil: 인코더 디스틸링 사용 여부
        - mix: 디코더 믹스 어텐션 사용 여부
        - device: 디바이스 설정
    Returns:
        - 모델 출력 텐서 및 (선택적) 어텐션 가중치
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        
        # 어텐션 메커니즘 선택
        #    ProbAttention vs FullAttention
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # 인코더 
        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)

        # 디코더 
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [batch size(B), out_len(L), feature dimension(D)]