import torch
import torch.nn as nn
import math

"""
==============================================================================
위치 인코딩 모듈
==============================================================================
"""
class PositionalEncoding(nn.Module):
    """
    Class: PositionalEncoding
        - Transformer의 위치 인코딩 구현: 위치 정보를 임베딩에 추가
        - sin/cos 함수를 사용하여 각 위치에 대해 고유한 벡터 생성
        - 순차성을 부여하되 의미 정보가 변질되지 않도록 -1 ~ 1 사이의 값을 갖도록 설계
        - 먼 위치: 더 큰 값, 가까운 위치: 더 작은 값
        - PE(pos, i) = sin(pos / (10,000^(2k/d_model))) if i=2k (k=0,1,2,...) # 짝수 인덱스
            - pos: 입력 시퀀스 데이터에서의 embedding vector 위치 (0부터 시작)
            - i: 임베딩 차원 인덱스 (0부터 시작)
            - d_model: 임베딩 차원
        - TimeSeriesEmbedding에서 사용됨
    foward parameters:
        - x (torch.Tensor): 입력 텐서, shape = (batch, seq_len, d_model)
    Return values:
        - x (torch.Tensor): 위치 인코딩이 추가된 텐서, shape = 동일 (batch, seq_len, d_model)
    """
    def __init__(self, d_model, max_len) -> None:
        """
        Function: __init__
            - 위치 인코딩 초기화
        Parameters:
            - self: 객체
            - d_model (int): 임베딩 차원 수 
            - max_len (int): 최대 시퀀스 길이 (inpuut window보다 같거나 커야 함)
        Returns: None
        """
        super().__init__()
        
        # [max_len, d_model] 크기의 0으로 채워진 텐서 초기화
        pe = torch.zeros(max_len, d_model)  # 공식 내: PE(pos, i)
        # 위치 인덱스 생성
        #    position: [0..max_len-1] 사이의 인덱스, shape: (max_len, 1)
        
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # 공식 내: pos
        # 주기 스케일링 항, 짝수 인덱스만큼 생성
        #    div_term: 주기 조절을 위한 분모항
        #    d_model의 절반 개수만큼 생성, 지수 함수로 스케일링
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model) # 공식 내: 1/(10000^(2k/d_model))
        )
        # 짝수 인덱스에 대해 sin 함수 적용 
        pe[:, 0::2] = torch.sin(position * div_term) # 공식 내: sin(pos / (10000^(2k/d_model)))
        # 홀수 인덱스에 대해 cos 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term) # 공식 내: cos(pos / (10000^(2k/d_model)))
        # pe shape: (1, max_len, d_model)
        
        # 버퍼로 등록(학습 파라미터 아님, 디바이스 이동/저장 포함)
        #    register_buffer -> 학습 파라미터로는 취급하지 않지만, 모델 저장/로드 시 저장되는 버퍼로 등록
        #    쓰임새: 위치 인코딩을 모델의 일부로 취급하여 GPU/CPU 간 이동 가능
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x) -> torch.Tensor:
        """
        Function: forward
            - 입력 임베딩에 위치 인코딩 더하기
            - (B, L, d_model) 임베딩에 (1, L, d_model) 위치 인코딩 더함
        Parameters:
            - self: 객체
            - x (torch.Tensor): 입력 임베딩, shape=(batch, seq_len, d_model)
        Returns:
            - x (torch.Tensor): 위치 인코딩이 추가된 텐서, shape = 동일 (batch, seq_len, d_model)
        """
        # x: (batch, seq_len, d_model)
        #    위치 인코딩 중에서 seq_len 만큼만 잘라서 입력에 더함
        #    transformer 입력 전 전처리 과정
        return x + self.pe[:, :x.size(1)] # input embedding + positional encoding matrix



"""
==============================================================================
시계열 임베딩 모듈
==============================================================================
"""
class TimeSeriesEmbedding(nn.Module):
    """
    Class: TimeSeriesEmbedding
        - 시계열 실수 입력을 Transformer가 처리할 수 있는 벡터 임베딩 공간(d_model 차원)으로 변환하는 임베딩 모듈.
        - 입력 피처 수 F -> d_model로 선형 투영
        - 위치 인코딩 추가 (PositionalEncoding 사용)
        - 레이어 정규화 및 드롭아웃 적용
    """
    def __init__(self, input_dim, d_model, max_len, dropout, use_scale=True) -> None:
        """
        Function: __init__
            - 임베딩 모듈 초기화
        Parameters:
            - input_dim (int): 입력 시계열의 특성 수
            - d_model (int): Transformer 모델의 임베딩 차원 # 입력값(x) 임베딩 차원 (전처리 과정에서 word embedding과 동일 역할)
            - max_len (int): 최대 시퀀스 길이
            - dropout (float): 드롭아웃 비율
            - use_scale (bool): 스케일링 여부, True면 √d_model로 나누기
        Returns: None
        """
        super().__init__()

        # input_dim -> d_model 차원으로 선형 변환 (F -> d_model로 선형 투영)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 위치 인코딩 모듈 (PositionalEncoding) 초기화
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        # 시퀀스의 각 시점별 벡터를 기준으로 정규화함
        self.norm = nn.LayerNorm(d_model) # 각 시퀀스 위치의 벡터 분포를 평균 0, 분산 1로 정규화
        # 과적합 방지를 위한 드롭아웃
        self.dropout = nn.Dropout(dropout) # 일부 뉴런을 랜덤하게 0으로 설정하여 학습 안정화
        # 스케일링 옵션
        self.use_scale = use_scale
        self.scale = math.sqrt(d_model) if use_scale else 1.0  # √d_model
        # Transformer 논문에서 제안: 입력을 √d_model로 나누어 안정적인 학습 유도

    def forward(self, x) -> torch.Tensor:
        """
        Function: forward
            - (B, L, F) 실수 시계열을 (B, L, d_model) 임베딩으로 투영 + 위치 인코딩, 정규화, 드롭아웃
        Parameters:
            - self: 객체
            - x: 입력 임베딩, shape=(batch_size, seq_len, input_dim)
        Returns:
            - x (torch.Tensor): 위치 인코딩이 더해진 임베딩, shape: (batch_size, seq_len, d_model)
            - 위치 인코딩 + LayerNorm + Dropout이 적용 완료된 임베딩 
        """
        # x: [batch(B), seq_len(L), input_dim(F)]
        B, L, _ = x.shape # B: batch size, L: seq_len

        # 시퀀스 길이 L이 위치 인코딩의 max_len보다 길면 에러 발생
        # seq_len(L): 한 입력 시퀀스의 시간 길이 (= input_window)
        assert L <= self.pos_encoder.pe.size(1), "seq_len이 max_len을 초과함."

        # 1) 입력을 d_model 차원으로 투영 & 스케일링
        x = self.input_proj(x) / self.scale # shape: (B, L, d_model)
       
        # 2) 위치 인코딩 더하기 
        x = self.pos_encoder(x) # shape: (B, L, d_model)
        
        # 3)  Layer Normalization으로 안정화
        x = self.norm(x) # shape: (B,L, d_model)
        
        # 4) 드롭아웃 적용 
        x = self.dropout(x) # shape: (B, L, d_model)
        
        # 5) 완성된 시계열 임베딩 반환             
        return x # shape: (B, L, d_model)



"""
==============================================================================
Transformer_encoder 모델
- 입력 형태 (최종 전처리 형태):
    - [B, L, F]
        - B: 배치 크기 | L: 입력 시퀀스 길이 (input_window) | F: 입력 피처 수
- 출력 형태 (최종 예측 형태):
    - [B, output_window]
        - B: 배치 크기 | output_window: 예측 변수 수
==============================================================================
"""
class Transformer_encoder(nn.Module):
    """
    Class: Transformer_encoder
        - Transformer encoder 기반 시계열 데이터 예측 모델
        - 실수 시계열 입력(F)을 d_model 차원으로 임베딩(위치 인코딩 포함)한 뒤 
            Transformer Encoder를 통과시켜, 풀링된 대표 벡터로 다단계 에측을 수행하는 인코더 기반 모델
        - 디코더 없이, 마지막/평균 풀링된 인코더 출력 -> MLP 헤드로 여러 스텝을 한 번에 회귀(오차 누적 방지)
    Note:
        - 인코더만 사용하므로, 미래 정보 누출 방지를 위해 causal mask를 추가
        - 여러 타깃 채널 예측 시: 
            - 최종 linear 출력 차원을 output_window * target_dim으로 바꾸고 reshape 필요 
    """
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_enc_layers,
        dim_feedforward,
        output_window,
        dropout,
        max_len,
        head_hidden,    
        pool="last",   
        use_casual=True,
        use_pad_mask=False
    ) -> None:
        """
        Function: __init__
            - Transformer 모델 초기화
        Parameters:
            - input_dim (int): 입력 피처 수 F (torch_train_x.shape[-1])
            - d_model (int): Transformer 모델의 임베딩 차원, LSTM의 hidden_size 역할
            - nhead (int): 멀티헤드 어텐션의 헤드 수 
            - num_enc_layers (int): Transformer 인코더 레이어 수
            - dim_feedforward (int): 인코더 내부 FFN 차원
            - output_window (int): 예측 변수 수 (single-step = 1, multi-step = H)
            - dropout (float): 드롭아웃 비율
            - max_len (int): 최대 시퀀스 길이
            - head_hidden (int): 예측 헤드 내부 은닉 차원
            - pool (str): 풀링 방식 ("last" 또는 "mean") (기본값: "last")
            - use_casual (bool): 인과 마스크 사용 여부 (기본값: True)
            - use_pad_mask (bool): 패딩 마스크 사용 여부 (기본값: False)
        Returns:
            - None
        """
        super().__init__()
        self.d_model = d_model # 임베딩 차원
        self.output_window = output_window # 최종 예측 길이 (출력 변수가 1개: 1)
        self.pool = pool # last 또는 mean # 대표 벡터 추출 방식
        self.use_casual = use_casual # 인과 마스크 사용 여부
        self.use_pad_mask = use_pad_mask # 패딩 마스크 사용 여부

        # 1) 임베딩: 실수 시계열 -> d_model 차원 + 위치 인코딩
        self.embedding = TimeSeriesEmbedding(
            input_dim=input_dim, 
            d_model=d_model, 
            max_len=max_len, 
            dropout=dropout, 
            use_scale=True
        )
        #    입력 (B, L, F) -> Linear로 (B, L, d_model), 
        #    PositionalEncoding으로 위치 정보 추가
        #    LayerNorm으로 안정화, Dropout으로 과적합 방지

        # 2) Transformer Encoder 구성
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, # 각 토큰(시점) 벡터의 임베딩 차원
            nhead=nhead, # 멀티 헤드 수 
            dim_feedforward=dim_feedforward, # FFN 내부 차원
            dropout=dropout, # 드롭아웃 비율
            batch_first=True, # 입력/출력 텐서 shape을 (B, L, d) 유지
            activation="gelu", # FFN 활성화 함수
            norm_first=True # LayerNorm -> Self-Attention -> FFN 순서
        )
        # 스택된 인코더 레이어: self-attention + FFN 블록을 num_enc_layers 만큼 반복
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers, norm=nn.LayerNorm(d_model))

        # 4) 예측 헤드: (B, d_model) -> (B, output_window * 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden), # d_model -> hidden
            nn.ReLU(), # 비선형 활성화
            nn.Dropout(dropout), # 과적합 완화
            nn.Linear(head_hidden, output_window) # hidden -> output_window (단계별 예측)
        ) # output: (B, output_window) # 최종 출력

    def _causal_mask(self, L: int, device: torch.device):
        """
        Function: _causal_mask
            - 인과 마스크 생성
            - Transformer Encoder에서 미래 시점 정보 누출 방지를 위해 사용
        Parameters:
            - L (int): 시퀀스 길이
            - device (torch.device): 마스크 텐서가 생성될 디바이스
        Returns:
            - mask (torch.Tensor): 인과 마스크, shape: (L, L)
        """
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x, pad_mask=None) -> torch.Tensor:
        """
        Function: forward
            - 시계열 배치를 임베딩 -> 인코더 통과 -> 풀링 -> 미래 output_window 스텝 회귀
        Parameters:
            - x (torch.Tensor): 입력 텐서, shape: [batch(B), seq_len(L), input_dim(F)]
            - pad_mask (torch.Tensor, optional): 패딩 마스크, shape: [batch(B), seq_len(L)]
        Return values:
            - y_hat (torch.Tensor): 예측 결과, shape: [batch(B), output_window, 1]
            - 미래 스텝별 단일 타깃 회귀
        """
        # x: [B, L, F] 입력
        B, L, _ = x.shape # B: batch size, L: seq_len

        # 1) 시계열 임베딩
        h = self.embedding(x) # [B, L, d_model] 임베딩 + 위치 인코딩 + Norm/Dropout

        # 2) 마스크 생성
        src_mask = None

        if self.use_casual:
            src_mask = self._causal_mask(L, h.device) # (L, L) 인과 마스크 생성
        
        src_key_padding_mask = pad_mask if (self.use_pad_mask and pad_mask is not None) else None

        # 3) Transformer Encoder 통과
        h = self.encoder(h, mask=src_mask, src_key_padding_mask=src_key_padding_mask) # [B, L, d_model] 인코더 통과 (현재 마스크 미적용)

        # 4) 대표 벡터 추출
        if self.pool == "last":
            rep = h[:, -1, :] # [B, d_model] 마지막 시점 벡터 사용
        else:
            rep = h.mean(dim=1) # [B, d_model] 시간 평균 풀링

        # 4) 예측 헤드 통과
        y_hat = self.head(rep) # [B, output_window]

        # 채널(출력 변수) 차원 추가 -> [B, output_window, 1]
        # y_hat = y_hat.unsqueeze(-1)
        return y_hat



"""
==============================================================================
Transformer 모델
- encoder-decoder 구조
- 입력 형태 (최종 전처리 형태):
    - [B, L, F]
        - B: 배치 크기 | L: 입력 시퀀스 길이 (input_window) | F: 입력 피처 수
- 출력 형태 (최종 예측 형태):
    - [B, L_dec, output_dim]
        - B: 배치 크기 | L_dec: 출력 시퀀스 길이 (output_window) | output_dim: 출력 피처 수
==============================================================================
"""
class Transformer(nn.Module):
    """
    Class: Transformer
        - Transformer encoder-decoder 기반 시계열 데이터 예측 모델
        - Encoder: 과거 시계열(input_window)을 d_model 차원으로 임베딩(위치 인코딩 포함)한 뒤 
            Transformer Encoder를 통과시켜, 인코더 출력을 생성
        - Decoder: 미래 시계열(output_window)을 d_model 차원으로 임베딩(위치 인코딩 포함)한 뒤
            Transformer Decoder를 통과시켜, 디코더 출력을 생성
            - 미래 시계열을 auto-regressive하게 생성
        - Teacher forcing 기법 적용 가능: 훈련 시 실제 미래 시계열을 디코더 입력으로 사용
    구조:
        1. Encoder: 입력 시퀀스 [B, L_enc, F] -> [B, L_enc, d_model]
        2. Decoder: 타깃 시퀀스 [B, L_dec, F] -> 예측 [B, L_dec, output_dim]
        3. Cross-attention: 디코더가 인코더 출력을 참조하여 예측 수행
    Note:
        - Decoder는 causal mask를 사용하여 미래 정보 누출 방지
        - 학습 시: teacher forcing 적용 (실제 타겟값을 decoder 입력으로 사용)
        - 추론 시: 이전 예측값을 decoder 입력으로 사용하여 auto-regressive 생성
    """
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_enc_layers,
        num_dec_layers, # 디코더 레이어 수 추가
        dim_feedforward,
        output_window,
        output_dim, # 출력 피처 수 추가
        dropout,
        max_len,
        # head_hidden, # 삭제 
        pool="last",   
        use_casual=True,
        use_pad_mask=False
    ) -> None:
        """
        Function: __init__
            - Transformer Encoder - Decoder 모델 초기화
        Parameters:
            - input_dim (int): 입력 피처 수 F (torch_train_x.shape[-1])
            - d_model (int): Transformer 모델의 임베딩 차원, LSTM의 hidden_size 역할
            - nhead (int): 멀티헤드 어텐션의 헤드 수 
            - num_enc_layers (int): Transformer 인코더 레이어 수
            - num_dec_layers (int): Transformer 디코더 레이어 수
            - dim_feedforward (int): 인코더 내부 FFN 차원
            - output_window (int): 예측할 미래 스텝 수 (L_dec)
            - output_dim (int): 예측 변수 수 (default: 1 단일 변수 예측 시)
            - dropout (float): 드롭아웃 비율
            - max_len (int): 최대 시퀀스 길이
            - pool (str): 풀링 방식 ("last" 또는 "mean") (default: "last")
            - use_casual (bool): 인과 마스크 사용 여부 (default: True)
            - use_pad_mask (bool): 패딩 마스크 사용 여부 (default: False)
        Returns:
            - None
        """
        super().__init__()
        self.d_model = d_model # 임베딩 차원
        self.output_window = output_window # 예측할 미래 스텝 수 (L_dec)
        self.output_dim = output_dim # 출력 피처 수
        self.pool = pool # last 또는 mean # 대표 벡터 추출 방식
        self.use_casual = use_casual # 인과 마스크 사용 여부
        self.use_pad_mask = use_pad_mask # 패딩 마스크 사용 여부

        # 1) Encoder 임베딩: 입력 시계열 (과거) -> d_model 차원 + 위치 인코딩
        #    입력 [B, L, F] -> Linear로 [B, L, d_model], 
        #    PositionalEncoding으로 위치 정보 추가
        #    LayerNorm으로 안정화, Dropout으로 과적합 방지

        self.enc_embedding = TimeSeriesEmbedding(
            input_dim=input_dim, 
            d_model=d_model, 
            max_len=max_len, 
            dropout=dropout, 
            use_scale=True
        )

        # 2) Decoder 임베딩: 타깃 시계열 (미래) -> d_model 차원 + 위치 인코딩
        #    decoder 입력은 output_dim 차원 (예측할 변수들)
        self.dec_embedding = TimeSeriesEmbedding(
            input_dim=output_dim,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            use_scale=True
        )

        # 3) Transformer Encoder 구성
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, # 각 토큰(시점) 벡터의 임베딩 차원
            nhead=nhead, # 멀티 헤드 수 
            dim_feedforward=dim_feedforward, # FFN 내부 차원
            dropout=dropout, # 드롭아웃 비율
            batch_first=True, # 입력/출력 텐서 shape을 [B, L, d] 유지
            activation="gelu", # FFN 활성화 함수
            norm_first=True # LayerNorm -> Self-Attention -> FFN 순서
        )
        # 스택된 인코더 레이어: self-attention + FFN 블록을 num_enc_layers 만큼 반복
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_enc_layers, 
            norm=nn.LayerNorm(d_model)
        )

        # 4) Transformer Decoder 구성
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        # 스택된 디코더 레이어
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_dec_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 5) 출력 프로젝션: [B, L_dec, d_model] -> [B, L_dec, output_dim]
        self.output_projection = nn.Linear(d_model, output_dim) # d_model -> output_dim
        
    def _causal_mask(self, L: int, device: torch.device):
        """
        Function: _causal_mask
            - 인과 마스크 생성
        
        Parameters:
            - L (int): 시퀀스 길이
            - device (torch.device): 텐서 디바이스
        
        Returns:
            - mask (torch.Tensor): shape (L, L), 상삼각 부분이 True
        """
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None) -> torch.Tensor:
        """
        Function: forward
            - Encoder-Decoder 구조로 시계열 배치를 처리하여 미래 스텝 예측
        Parameters:
            - src (torch.Tensor): 입력 텐서, shape: [batch(B), seq_len(L_enc), input_dim(F)]
            - tgt (torch.Tensor): 타겟 텐서, shape: [batch(B), output_window(L_dec), output_dim]
                - 학습 시: teacher forcing을 위한 실제 타깃값
                - 추론 시: autoregressive 생성을 위한 초기값 + 이전 예측값
            - src_pad_mask (torch.Tensor, optional): 소스 패딩 마스크, shape: [batch(B), seq_len(L_enc)]
            - tgt_pad_mask (torch.Tensor, optional): 타겟 패딩 마스크, shape: [batch(B), output_window(L_dec)]
        Return values:
            - y_hat (torch.Tensor): 예측 결과, shape: [batch(B), output_window, 1]
            - 미래 스텝별 단일 타깃 회귀
        """
        # 1) 입력 크기 정의
        #    src: [B, L_enc, F] 입력
        B, L_enc, _ = src.shape # B: batch size, L: seq_len
        _, L_dec, _ = tgt.shape # L_dec: output_window
        device = src.device

        # 2) 시계열 임베딩 (인코더/디코더 각각)
        #    시계열 특징 임베딩 + 위치 인코딩
        src_emb = self.enc_embedding(src) # [B, L_enc, d_model]
        tgt_emb = self.dec_embedding(tgt) # [B, L_dec, d_model]
        
        # 3) 마스크 생성
        src_mask = None
        tgt_mask = None

        # 인과 마스크 (디코더에서 미래 정보 차단)
        if self.use_casual:
            tgt_mask = self._causal_mask(L_dec, device) # [L_dec, L_dec]

        # 패딩 마스크 (선택)
        src_key_padding_mask = src_pad_mask if (self.use_pad_mask and src_pad_mask is not None) else None
        tgt_key_padding_mask = tgt_pad_mask if (self.use_pad_mask and tgt_pad_mask is not None) else None

        # 4) Encoder 통과
        memory = self.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        ) # [B, L_enc, d_model]

        # 5) Decoder 통과
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        ) # [B, L_dec, d_model]

        # 6) 출력 헤드 통과 -> 최종 예측
        y_hat = self.output_projection(out)

        return y_hat # [B, L_dec, output_dim]