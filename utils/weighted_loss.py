import torch
import torch.nn as nn

# 피크 구간 가중치 부여한 Huber Loss 클래스
class WeightedHuberLoss(nn.Module):
    """
    Class: WeightedHuberLoss
        - 피크 구간에 가중치를 부여한 Huber Loss 구현
        - 피크 가중치: 피크 구간에 대해 손실을 더 크게 반영
        - 피크 구간: 전체 부하 중 상위 몇 퍼센트로 정의
    Parameters:
        - delta: Huber Loss의 delta 값
        - peak_threshold: 전체 부하 중 상위 몇 퍼센트를 피크로 볼지
        - peak_weight: 피크 구간 가중 배수
    Returns:
        - weighted_loss: 피크 구간에 가중치가 적용된 Huber Loss 값
    """
    def __init__(self, delta=1.0, peak_threshold=0.9, peak_weight=10.0):
        super().__init__()
        self.delta = delta
        self.peak_threshold = peak_threshold  # 전체 부하 중 상위 몇 퍼센트를 피크로 볼지
        self.peak_weight = peak_weight # 피크 구간 가중 배수

    def forward(self, y_pred, y_true):
        # 절대 오차 계산
        error = torch.abs(y_true - y_pred)
        
        # 기본 huber 손실 계산
        huber_loss = torch.where(
            error <= self.delta,
            0.5 * error**2,
            self.delta * (error - 0.5 * self.delta)
        )
        
        # 피크 구간 마스크 생성
        normed = y_true / y_true.max()
        peak_mask = (normed > self.peak_threshold).float() # peak_threshold 초과면 1, 아니면 0

        # 피크에만 가중치 부여
        weights = torch.ones_like(y_true) + peak_mask * (self.peak_weight - 1) # 피크 구간은 peak_weight 배수

        # 최종 가중 손실
        weighted_loss = torch.mean(weights * huber_loss)
        return weighted_loss