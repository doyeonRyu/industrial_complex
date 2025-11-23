import torch

class TriangularCausalMask():
    """
    Class: TriangularCausalMask
        - self attention을 위한 마스크 생성
    Parameters:
        - B: batch size
        - L: sequence length
        - device: 디바이스 설정 (기본값: "cpu")
    Returns:
        - triangular causal 마스크 텐서
    """
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L] # 마스크 shape 설정 # [batch size, head num, query len, key len]
        with torch.no_grad(): # 마스크 텐서 생성
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device) # upper triangular part를 1로 설정하여 미래의 토큰을 마스킹

    @property
    def mask(self):
        return self._mask

class ProbMask():
    """
    Class: ProbMask
        - ProbSparse Attention을 위한 마스크 생성
    Parameters:
        - B: batch size
        - H: head num
        - L: sequence length
        - index: 상위 쿼리 인덱스
        - scores: 어텐션 스코어 텐서
        - device: 디바이스 설정 (기본값: "cpu")
    Returns:
        - ProbSparse attention 마스크 텐서
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # 마스크 텐서 생성
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1) # upper triangular part를 1로 설정하여 미래의 토큰을 마스킹
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1]) # batch size와 head num에 맞게 확장
        indicator = _mask_ex[torch.arange(B)[:, None, None], # batch size에 맞는 인덱스 생성
                             torch.arange(H)[None, :, None], # head num에 맞는 인덱스 생성
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device) # scores shape에 맞게 뷰 변환
    
    @property
    def mask(self):
        return self._mask