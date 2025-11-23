import os
import pandas as pd
import joblib
import pickle

# 전처리 완료된 데이터 로드 함수
def load_data(path, model_type=None):
    """
    Function: load_data
        - 지정된 경로에서 전처리 완료된 CSV 파일과 스케일러 로드
    Parameters:
        - path: str (예: "data/preprocessed/광명금속/")
            - 데이터가 저장된 디렉토리 경로
            - path에 longformer가 포함된 경우 longformer 전처리 데이터 로드
    Returns:
        - train_origin, valid_origin, test_origin: 원본 데이터
        - train_preprocessed, valid_preprocessed, test_preprocessed: 전처리 완료된 입력 데이터
        - train_preprocessed_y, valid_preprocessed_y, test_preprocessed_y: 전처리 완료된 타겟 데이터
        - minmax_scaler: 입력 피처용 MinMaxScaler 객체
        - standard_scaler: 입력 피처용 StandardScaler 객체
        - y_minmax_scaler: 타겟 피처용 MinMaxScaler 객체
        - y_standard_scaler: 타겟 피처용 StandardScaler 객체
        if model_type == "longformer":
        - train_x_mark, valid_x_mark, test_x_mark: longformer 날씨 시간 관련 추출된 데이터
    """
    path_name = os.path.basename(os.path.normpath(path)) # 디렉토리 이름 추출 (예: "광명금속")

    print("\n" + "=" * 60)
    print(f"Loading data from {path_name}...")
    print("\n" + "=" * 60)

    # origin 파일은 시각화시에만 사용 
    if model_type == "longformer": # longformer 전처리 데이터 로드
        suffix = "_longformer"
        train_x_mark = pd.read_csv(path + f"{path_name}{suffix}_train_x_mark.csv")
        valid_x_mark = pd.read_csv(path + f"{path_name}{suffix}_valid_x_mark.csv")
        test_x_mark = pd.read_csv(path + f"{path_name}{suffix}_test_x_mark.csv")

    else:
        suffix = ""
        
    train_origin = pd.read_csv(path + f"{path_name}{suffix}_train_origin.csv")
    valid_origin = pd.read_csv(path + f"{path_name}{suffix}_valid_origin.csv")
    test_origin = pd.read_csv(path + f"{path_name}{suffix}_test_origin.csv")

    train_preprocessed = pd.read_csv(path + f"{path_name}{suffix}_train_preprocessed.csv")
    valid_preprocessed = pd.read_csv(path + f"{path_name}{suffix}_valid_preprocessed.csv")
    test_preprocessed = pd.read_csv(path + f"{path_name}{suffix}_test_preprocessed.csv")

    train_preprocessed_y = pd.read_csv(path + f"{path_name}{suffix}_train_preprocessed_y.csv")
    valid_preprocessed_y = pd.read_csv(path + f"{path_name}{suffix}_valid_preprocessed_y.csv")
    test_preprocessed_y = pd.read_csv(path + f"{path_name}{suffix}_test_preprocessed_y.csv")
    
    print("Train Origin Data Shape:", train_origin.shape)
    print("Validation Origin Data Shape:", valid_origin.shape)
    print("Test Origin Data Shape:", test_origin.shape)
    print("\n")
    print("Train Preprocessed Data Shape:", train_preprocessed.shape)
    print("Validation Preprocessed Data Shape:", valid_preprocessed.shape)
    print("Test Preprocessed Data Shape:", test_preprocessed.shape)
    print("\n")
    print("Train Preprocessed Y Data Shape:", train_preprocessed_y.shape)
    print("Validation Preprocessed Y Data Shape:", valid_preprocessed_y.shape)
    print("Test Preprocessed Y Data Shape:", test_preprocessed_y.shape)
    print("\n")

    # sklearn 객체 (scaler)) 로드 함수
    def safe_load_sklearn_obj(file_path):
        '''
        Function: safe_load_sklearn_obj
            - sklearn 객체 (StandardScaler, MinMaxScaler 등) 로드
        Parameters:
            - file_path: str
                - 로드할 파일 경로
        Returns:
            - sklearn 객체
        '''
        # 1) joblib로 먼저 시도
        try:
            return joblib.load(file_path)
        except Exception:
            pass

        # 2) pickle로 재시도
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"스케일러 로드 실패: {file_path}\n" + str(e))
        
    # minmax_scaler = safe_load_sklearn_obj(path + f"{path_name}_minmax_scaler{suffix}.pkl")
    # standard_scaler = safe_load_sklearn_obj(path + f"{path_name}_standard_scaler{suffix}.pkl")
    scaler = safe_load_sklearn_obj(path + f"{path_name}{suffix}_robust_scaler.pkl")
    # y_minmax_scaler = safe_load_sklearn_obj(path + f"{path_name}_y_minmax_scaler{suffix}.pkl")
    # y_standard_scaler = safe_load_sklearn_obj(path + f"{path_name}_y_standard_scaler{suffix}.pkl")
    y_scaler = safe_load_sklearn_obj(path + f"{path_name}{suffix}_y_robust_scaler.pkl")

    print(f"Data and scalers loaded successfully from {path_name}.\n")

    if model_type == "longformer": # Informer, Autoformer
        return (train_origin, valid_origin, test_origin,
                train_preprocessed, valid_preprocessed, test_preprocessed,
                train_x_mark, valid_x_mark, test_x_mark,
                train_preprocessed_y, valid_preprocessed_y, test_preprocessed_y,
                scaler, y_scaler)
    else: # LSTM, Transformer
        return (train_origin, valid_origin, test_origin,
                train_preprocessed, valid_preprocessed, test_preprocessed,
                train_preprocessed_y, valid_preprocessed_y, test_preprocessed_y,
                scaler, y_scaler)
