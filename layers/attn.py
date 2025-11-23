# Informer 용 attenion 모듈 구현
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask 
# TrianguleCausalMask: self attention을 위한 마스크 # ProbMask: robSparse Attention을 위한 마스크

class FullAttention(nn.Module):
    """
    Class: FullAttention
        - 일반적인 full attention 메커니즘 구현
    Parameters:
        - mask_flag (bool): 마스크 사용 여부
        - factor (int): 사용되지 않음
        - scale (float): 스케일링 팩터
        - attention_dropout (float): 어텐션 드롭아웃 비율
        - output_attention (bool): 어텐션 가중치 출력 여부
    Returns:
        - 어텐션 출력 텐서, 어텐션 가중치
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        """
        Function: forward
            - 어텐션 메커니즘의 순전파 구현
        Parameters:
            - queries (torch.Tensor): 쿼리 텐서
            - keys (torch.Tensor): 키 텐서
            - values (torch.Tensor): 값 텐서
            - attn_mask (torch.Tensor): 어텐션 마스크 텐서
        Returns:
            - 어텐션 출력 텐서, 어텐션 가중치
        """
        B, L, H, E = queries.shape # [batch_size, query_len, head_num, dim_per_head]
        _, S, _, D = values.shape # [batch_size, key_len, head_num, dim_per_head]
        scale = self.scale or 1./sqrt(E) # 스케일링 팩터 설정

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # 어텐션 스코어 계산 [B, H, L, S]
        if self.mask_flag: # 마스크 적용
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device) # 마스크 생성

            scores.masked_fill_(attn_mask.mask, -np.inf) # 마스크 적용

        A = self.dropout(torch.softmax(scale * scores, dim=-1)) # 어텐션 가중치 계산 및 드롭아웃 적용
        V = torch.einsum("bhls,bshd->blhd", A, values) # 어텐션 출력 계산
        
        if self.output_attention: # 어텐션 가중치 출력 여부 확인
            return (V.contiguous(), A) # 어텐션 출력과 가중치 반환
        else:
            return (V.contiguous(), None) # 어텐션 출력만 반환

class ProbAttention(nn.Module):
    """
    Class: ProbAttention
        - ProbSparse Attention 메커니즘 구현
    Parameters:
        - mask_flag (bool): 마스크 사용 여부
        - factor (int): 샘플링 팩터
        - scale (float): 스케일링 팩터
        - attention_dropout (float): 어텐션 드롭아웃 비율
        - output_attention (bool): 어텐션 가중치 출력 여부
    Returns:
        - 어텐션 출력 텐서, 어텐션 가중치
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        """
        Function: _prob_QK
            - ProbSparse Attention을 위한 쿼리-키 유사도 계산
        Parameters:
            - Q (torch.Tensor): 쿼리 텐서
            - K (torch.Tensor): 키 텐서
            - sample_k (int): 샘플링할 키의 개수
            - n_top (int): 선택할 상위 쿼리의 개수
        Returns:
            - Q_K (torch.Tensor): 쿼리-키 유사도 텐서
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape # [batch_size, head_num, key_len, dim_per_head]
        _, _, L_Q, _ = Q.shape # [batch_size, head_num, query_len, dim_per_head]

        # 샘플 키에 대한 쿼리-키 유사도 계산
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E) # [B, H, L_q, L_k, D]
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :] # [B, H, L_q, sample_k, D]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2) # [B, H, L_q, sample_k]

        # 최대값과 평균값을 사용하여 쿼리 중요도 측정
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # 상위 쿼리에 대한 쿼리-키 유사도 계산
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top # Q_K: 상위 쿼리-키 유사도, M_top: 상위 쿼리 인덱스

    def _get_initial_context(self, V, L_Q):
        """
        Function: _get_initial_context
            - 초기 컨텍스트 벡터 생성
        Parameters:
            - V (torch.Tensor): 값 텐서
            - L_Q (int): 쿼리 시퀀스 길이
        Returns:
            - contex (torch.Tensor): 초기 컨텍스트 벡터
        """
        B, H, L_V, D = V.shape # [batch_size, head_num, value_len, dim_per_head]
        if not self.mask_flag: # no mask
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2) # mean 대신 sum 사용
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone() # [B, H, L_Q, D]
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex # [B, H, L_Q, D]

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """
        Function: _update_context
            - 선택된 쿼리에 대해 컨텍스트 벡터 업데이트
        Parameters:
            - context_in (torch.Tensor): 초기 컨텍스트 벡터
            - V (torch.Tensor): 값 텐서
            - scores (torch.Tensor): 쿼리-키 유사도 텐서
            - index (torch.Tensor): 상위 쿼리 인덱스
            - L_Q (int): 쿼리 시퀀스 길이
            - attn_mask (torch.Tensor): 어텐션 마스크 텐서
        Returns:
            - context (torch.Tensor): 업데이트된 컨텍스트 벡터
            - attn (torch.Tensor): 어텐션 가중치 텐서
        """
        B, H, L_V, D = V.shape # [batch_size, head_num, value_len, dim_per_head]

        if self.mask_flag: # 마스크 적용
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device) # 마스크 생성
            scores.masked_fill_(attn_mask.mask, -np.inf) # 마스크 적용

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores) # 어텐션 가중치 계산

        # 업데이트된 컨텍스트 벡터 계산
        context_in[torch.arange(B)[:, None, None], 
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        
        if self.output_attention: # 어텐션 가중치 출력 여부 확인
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns) # 업데이트된 컨텍스트와 가중치 반환
        else:
            return (context_in, None) # 업데이트된 컨텍스트만 반환

    def forward(self, queries, keys, values, attn_mask):
        """
        Function: forward
            - 어텐션 메커니즘의 순전파 구현
        Parameters:
            - queries (torch.Tensor): 쿼리 텐서
            - keys (torch.Tensor): 키 텐서
            - values (torch.Tensor): 값 텐서
            - attn_mask (torch.Tensor): 어텐션 마스크 텐서
        Returns:
            - 어텐션 출력 텐서와 (선택적으로) 어텐션 가중치
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # 스케일 팩터 적용
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # 초기 컨텍스트 벡터 생성
        context = self._get_initial_context(values, L_Q)
        # 선택된 top_k 쿼리로 컨텍스트 업데이트
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn # [B, L_Q, H, D] # 어텐션 출력과 가중치 반환


class AttentionLayer(nn.Module):
    """
    Class: AttentionLayer
        - 어텐션 레이어 구현
    Parameters:
        - attention: 어텐션 메커니즘
        - d_model: 모델 차원
        - n_heads: 헤드 수
        - d_keys: 키 차원 (기본값: d_model/n_heads)
        - d_values: 값 차원 (기본값: d_model/n_heads)   
        - mix: 쿼리, 키, 값의 혼합 여부 (기본값: False)
    Returns:
        - 어텐션 레이어 출력 텐서와 어텐션 가중치
    """
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn # 어텐션 레이어 출력과 가중치 반환