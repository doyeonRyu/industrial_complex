import torch
import torch.nn as nn

"""
==============================================================================
PatchTST 임베딩 모듈
==============================================================================
"""
class PatchEmbedding(nn.Module):
    """
    Class: PatchEmbedding
        - PatchTST 스타일 패치 임베딩
        - 채널 독립적(CI) 또는 채널 종속적(CD) 패치 임베딩 지원
    """
    def __init__(self, 
                input_window, 
                patch_size, 
                input_dim, 
                d_model,
                stride=None, 
                channel_independent=True, 
                dropout=0.0
            ) -> None:
        """
        Function: __init__
            - PatchEmbedding 모듈 초기화
        Parameters:
            - input_window (int): 입력 시퀀스 길이
            - patch_size (int): 패치 크기
            - input_dim (int): 입력 피처 수 (채널 수)
            - d_model (int): 임베딩 차원
            - stride (int, optional): 패치 추출 시 이동 간격 (default: patch_size (비겹침))
            - channel_independent (bool, optional): 채널 독립적 임베딩 여부 (default: True)
                - True: 각 채널별로 시간 패치만 펼침 (채널 독립적)
                - False: 패치 내 시간×채널을 한꺼번에 펼침 (채널 종속적, 간이형)
            - dropout (float, optional): 드롭아웃 비율 (default: 0.0)
        Return: None
        """
        super().__init__()

        self.input_window = input_window
        self.patch_size = patch_size
        self.stride = stride or patch_size  # 기본값: 비겹침
        self.channel_independent = channel_independent

        assert self.patch_size <= input_window, "patch_size가 input_window보다 클 수 없음."
        assert (input_window - patch_size) % self.stride == 0, "stride가 나누어 떨어지지 않음."

        self.num_patches = 1 + (input_window - patch_size) // self.stride

        if channel_independent:
            # 각 채널(변수)별로 시간패치만 펼침
            self.proj = nn.Linear(patch_size, d_model)
        else:
            # 패치 내 시간×채널을 한꺼번에 펼침(간이형)
            self.proj = nn.Linear(patch_size * input_dim, d_model)

        # 패치 위치 임베딩(학습형)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, L, D)  # L=input_window, D=input_dim
        return:
          - CI = True  -> (B*D, N, d_model)
          - CI = False -> (B,   N, d_model)
        """
        B, L, D = x.shape
        assert L == self.input_window, "입력 길이가 input_window와 다를 수 없음."

        # (B, N, D, P) : N=num_patches, P=patch_size
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)

        if self.channel_independent:
            # (B, D, N, P) -> (B*D, N, P)
            patches = patches.permute(0, 2, 1, 3).contiguous().view(B*D, self.num_patches, self.patch_size)
            z = self.proj(patches) + self.pos_emb          # (B*D, N, d_model)
            z = self.dropout(z)
            return z
        else:
            # (B, N, D*P)
            patches = patches.permute(0, 1, 2, 3).contiguous().view(B, self.num_patches, D*self.patch_size)
            z = self.proj(patches) + self.pos_emb          # (B, N, d_model)
            z = self.dropout(z)
            return z



"""
==============================================================================
PatchTST 모델
- 입력 형태 (최종 전처리 형태):
    - 
- 출력 형태 (최종 예측 형태):
    - 
==============================================================================
"""
class PatchTST(nn.Module):
    """
    Class: PatchTST
        - PatchTST 모델

    """
    def __init__(self, 
                input_window, 
                patch_size, 
                input_dim, 
                d_model, 
                output_window,
                num_layers, 
                nhead, 
                dropout, 
                channel_independent=False, 
                target_channel_idx=0, 
                stride=None
            ) -> None:
        """
        Function: __init__
            - PatchTST 모델 초기화
        Parameters:
            - input_window (int): 입력 시퀀스 길이
        """
        super().__init__()
        self.channel_independent = channel_independent
        self.target_channel_idx = target_channel_idx
        self.output_window = output_window
        self.input_dim = input_dim

        # CI/stride 지원하는 PatchEmbedding을 쓰는 경우
        self.embedding = PatchEmbedding(
            input_window=input_window,
            patch_size=patch_size,
            input_dim=input_dim,
            d_model=d_model,
            stride=stride or patch_size,
            channel_independent=channel_independent,
            dropout=dropout
        )

        # 패치 개수 (비겹침이면 input_window//patch_size)
        self.num_patches = self.embedding.num_patches

        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_window)

    def forward(self, x) -> torch.Tensor:
        # x: (B, L, D_in)
        B, _, D_in = x.shape
        z = self.embedding(x)             # CI=False: (B,N,d) / CI=True: (B*D_in,N,d)
        z = z + self.pos_emb
        z = self.emb_norm(z)
        z = self.emb_drop(z)

        z = self.encoder(z)               # same shape

        rep = z.mean(dim=1)               # CI=False: (B,d) / CI=True: (B*D_in,d)
        out = self.decoder(rep).unsqueeze(-1)  # CI=False: (B,H,1) / CI=True: (B*D_in,H,1)

        if self.channel_independent:
            # (B*D_in,H,1) -> (B,D_in,H,1) -> 대상 채널만 선택 -> (B,H,1)
            out = out.view(B, D_in, self.output_window, 1)
            out = out[:, self.target_channel_idx, :, :]

        return out # (B,H,1)
