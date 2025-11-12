import os
import pandas as pd
import joblib
import pickle

# 전처리 완료된 데이터 로드 함수
def load_data(path):
    """
    Function: load_data
        - 지정된 경로에서 전처리 완료된 CSV 파일과 스케일러 로드
    Parameters:
        - path: str, 데이터가 저장된 디렉토리 경로
    Returns:
        - train_origin, valid_origin, test_origin: 원본 데이터
        - train_preprocessed, valid_preprocessed, test_preprocessed: 전처리 완료된 입력 데이터
        - train_preprocessed_y, valid_preprocessed_y, test_preprocessed_y: 전처리 완료된 타겟 데이터
        - minmax_scaler: 입력 피처용 MinMaxScaler 객체
        - standard_scaler: 입력 피처용 StandardScaler 객체
        - y_minmax_scaler: 타겟 피처용 MinMaxScaler 객체
        - y_standard_scaler: 타겟 피처용 StandardScaler 객체
    """
    path_name = os.path.basename(os.path.normpath(path))  # 예: "메인텍 2공장"
    print(f"Loading data from {path_name}...")

    # origin 파일은 시각화시에만 사용 
    train_origin = pd.read_csv(path + f"{path_name}_train_origin.csv")
    valid_origin = pd.read_csv(path + f"{path_name}_valid_origin.csv")
    test_origin = pd.read_csv(path + f"{path_name}_test_origin.csv")

    train_preprocessed = pd.read_csv(path + f"{path_name}_train_preprocessed.csv")
    valid_preprocessed = pd.read_csv(path + f"{path_name}_valid_preprocessed.csv")
    test_preprocessed = pd.read_csv(path + f"{path_name}_test_preprocessed.csv")

    train_preprocessed_y = pd.read_csv(path + f"{path_name}_train_preprocessed_y.csv")
    valid_preprocessed_y = pd.read_csv(path + f"{path_name}_valid_preprocessed_y.csv")
    test_preprocessed_y = pd.read_csv(path + f"{path_name}_test_preprocessed_y.csv")
    
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

    # sklearn 객체 (StandardScaler, MinMaxScaler 등) 로드 함수
    def safe_load_sklearn_obj(file_path):
        '''
        Function: safe_load_sklearn_obj
            - sklearn 객체 (StandardScaler, MinMaxScaler 등)를
            joblib 또는 pickle로 안전하게 로드
        Parameters:
            - file_path: str, 로드할 파일 경로
        Returns:
            - sklearn 객체
        '''
        # 1) joblib로 먼저 시도
        try:
            # joblib로 저장한 파일이면 여기서 정상 로드됨
            return joblib.load(file_path)
        except Exception:
            pass

        # 2) pickle로 재시도
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            # 어떤 방식으로도 로드가 안 되면 상세 메시지 제공
            raise RuntimeError(f"스케일러 로드 실패: {file_path}\n"
                            f"- 원인 후보: 저장 방식 불일치(joblib vs pickle), 파일 손상, "
                            f"파이썬/스킷런 버전 불일치, 다른 객체를 잘못 저장\n"
                            f"원본 에러: {type(e).__name__}: {e}")
        
    # minmax_scaler = safe_load_sklearn_obj(path + f"{path_name}_minmax_scaler.pkl")
    standard_scaler = safe_load_sklearn_obj(path + f"{path_name}_standard_scaler.pkl")
    # y_minmax_scaler = safe_load_sklearn_obj(path + f"{path_name}_y_minmax_scaler.pkl")
    y_standard_scaler = safe_load_sklearn_obj(path + f"{path_name}_y_standard_scaler.pkl")

    print(f"Data and scalers loaded successfully from {path_name}.\n")

    return (train_origin, valid_origin, test_origin,
            train_preprocessed, valid_preprocessed, test_preprocessed,
            train_preprocessed_y, valid_preprocessed_y, test_preprocessed_y,
            None, standard_scaler, None, y_standard_scaler)
