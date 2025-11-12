from utils.setup import SeqDataset
from torch.utils.data import DataLoader
import torch

# DataLoader 생성 함수
def build_dataloader(torch_train_x, torch_train_y, 
                torch_valid_x, torch_valid_y, 
                torch_test_x, torch_test_y, batch_size
    ):
    """
    Function: build_dataloader
        - torch.Tensor로 변환된 (x, y) 데이터를 DataLoader로 변환
    Parameters:
        - torch_train_x, torch_train_y: 학습 데이터 (torch.Tensor)
        - torch_valid_x, torch_valid_y: 검증 데이터 (torch.Tensor)
        - torch_test_x, torch_test_y: 테스트 데이터 (torch.Tensor)
        - batch_size: 배치 크기 (int)
    Returns:
        - train_loader, valid_loader, test_loader: DataLoader 객체 
            (shape: (N, L, F), (N,) or (N, output_window))
    """
    # (x, y)만 생성
    train_ds = SeqDataset(torch_train_x, torch_train_y)
    valid_ds = SeqDataset(torch_valid_x, torch_valid_y)
    test_ds  = SeqDataset(torch_test_x,  torch_test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    
    return train_loader, valid_loader, test_loader