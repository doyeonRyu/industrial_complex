"""
==============================================================================
File: data_preprocessing.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-10-14

Description: data inspection 및 preprocessing 과정
    1) 컬럼명 변경
    2) 변수별 시각화
    3) Train / Valid / Test 분할
    4) X, y 분리 
    5) 이상치, 결측치 탐색 및 처리 
    6) 정규화, 표준화

Note
    - 산업체명 변경할 경우 
        - **industry_name**, **data** 변수 수정 필요
        
    - 산업체마다 데이터 특성이 다를 수 있음. 수정 필요 - 현재는 광명금속의 포멧을 따름
    - 추가적인 전처리 과정이 필요할 수 있음
    - 이후의 과정은 **modeling.py**에서 처리 (예: 슬라이딩 윈도우, 데이터로더 생성 등)
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# 데이터 시각화 및 탐색 함수
def data_inspection(industry_name: str, data: pd.DataFrame):
    """
    Function: data_inspection
        1) 컬럼명 변경
        2) 변수별 시각화
            - 요일별 평균 사용량 시각화
            - 월별 평균 사용량 시각화
            - 각 변수별 사용량과의 상관관계 산점도
            - 각 변수별 사용량과의 상관관계 히트맵
        - 시각화된 그래프는 plots/폴더명/파일명.png 형태로 저장
    Parameters:
        - industry_name: 산업체 이름 (예: "광명금속")
        - data: 전처리할 데이터프레임
    Returns:
        - data: 전처리된 데이터프레임
    """
    # 데이터 불러오기 
    if not data.endswith('.csv'): # 만약 뒤에 .csv가 붙어있지 않으면 자동으로 붙여줌
        data = data + '.csv'
    data = pd.read_csv("processed/" + data)

    print(f"[{industry_name}] Data inspection process...\n")

    # head() 출력
    print("Data Head:")
    print(data.head(), "\n")

    # info() 출력
    print("Data Info:")
    print(data.info(), "\n")

    # describe() 출력
    print("Data Describe:")
    print(data.describe(), "\n")

    # 1) 컬럼명 변경
    #    한글 컬럼명 사용시 시각화 등 추후 작업에서 오류 발생 가능성 있음
    print(data.columns, "-> (컬럼명 변경) \n")
    #    컬럼 영어로 변경
    data.columns = ['datetime', 'usage_kWh', 'max_demand_kW', 'reactive_usage_kWh_kVarh_inductive', 'reactive_usage_kWh_kVarh_capacitive', 'CO2_tCO2', 'usage_kWh_factor_inductive', 'usage_kWh_factor_capacitive']
    print(data.columns, "\n")

    # 2) 변수별 시각화
    feature_cols = [col for col in data.columns if col != 'datetime' and col != 'usage_kWh']
    
    # df 이름 형태의 폴더 생성 
    if not os.path.exists('../plots'):
        os.makedirs('../plots')
    if not os.path.exists(f'../plots/{industry_name}'):
        os.makedirs(f'../plots/{industry_name}')

    # 2-1) 요일별 평균 사용량 시각화

    # 날짜별 요일 추출
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['day_of_week'] = data['datetime'].dt.day_name()

    # 요일별 평균 사용량 계산
    avg_usage_by_day = data.groupby('day_of_week')['usage_kWh'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_usage_by_day.index, y=avg_usage_by_day.values)
    plt.title('weekly mean usage (kWh)')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Usage (kWh)')
    # plt.show() # 주석 처리 - 노트북에서 실행 시 사용
    plt.savefig(f'../plots/{industry_name}/[{industry_name}] weekly_mean_usage.png')
    # day_of_week 열 삭제
    data.drop(columns=['day_of_week'], inplace=True)

    # 2-2) 월별 평균 사용량 시각화
    # 날짜별 월 추출
    data['month'] = data['datetime'].dt.month

    # 월별 평균 사용량 계산
    avg_usage_by_month = data.groupby('month')['usage_kWh'].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_usage_by_month.index, y=avg_usage_by_month.values)
    plt.title('monthly mean usage (kWh)')
    plt.xlabel('Month')
    plt.ylabel('Average Usage (kWh)')
    # plt.show() # 주석 처리 - 노트북에서 실행 시 사용
    plt.savefig(f'../plots/{industry_name}/[{industry_name}] monthly_mean_usage.png')
    # month 열 삭제
    data.drop(columns=['month'], inplace=True)

    # 2-3) 각 변수별 사용량과의 상관관계 산점도
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(feature_cols):
        plt.subplot(3, 2, i + 1)
        sns.scatterplot(data=data, x=col, y='usage_kWh')
        plt.title(f'Usage vs {col}')
        plt.xlabel(col)
        plt.ylabel('Usage (kWh)')
    plt.tight_layout()
    # plt.show() # 주석 처리 - 노트북에서 실행 시 사용
    plt.savefig(f'../plots/{industry_name}/[{industry_name}] features_usage_scatter_plot.png')

    # 2-4) 각 변수별 사용량과의 상관관계 히트맵
    plt.figure(figsize=(10, 10))
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    # plt.show() # 주석 처리 - 노트북에서 실행 시 사용
    plt.savefig(f'../plots/{industry_name}/[{industry_name}] correlation_matrix.png')

    print(f"[{industry_name}] Data inspection completed and plots saved.\n")
    return data

# 데이터 전처리 함수
def data_preprocessing(industry_name: str, data: pd.DataFrame):
    """
    Function: data_preprocessing
        3) Train / Valid / Test 분할
        4) X, y 분리
        5) 이상치, 결측치 탐색 및 처리
        6) 정규화, 표준화
    Parameters:
        - industry_name: 산업체 이름 (예: "광명금속")
        - data: 전처리할 데이터프레임
    Returns:
        - None (전처리된 데이터는 CSV 파일로 저장)
    """

    # df 이름 첫번쨰 _이전까지로 설정
    industry_name = industry_name.split('_')[0]

    print(f"[{industry_name}] Data preprocessing process...\n")
    
    # 3) Train / Valid / Test 분할
    def split_train_val_test(data=data, train_ratio=0.7, val_ratio=0.15):    
        train_data = pd.DataFrame()
        val_data = pd.DataFrame()
        test_data = pd.DataFrame()

        total_len = len(data)
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))

        train_data = pd.concat([train_data, data.iloc[:train_end]])
        val_data = pd.concat([val_data, data.iloc[train_end:val_end]])
        test_data = pd.concat([test_data, data.iloc[val_end:]])

        # 인덱스 재설정
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

        return train_data, val_data, test_data
    
    train, valid, test = split_train_val_test(data)

    # 추후 시각화를 위해 원본 복사
    train_origin = train.copy()
    valid_origin = valid.copy()
    test_origin = test.copy()

    # 4) X, y 분리
    train_y = pd.DataFrame(train['usage_kWh'])
    valid_y = pd.DataFrame(valid['usage_kWh'])
    test_y = pd.DataFrame(test['usage_kWh'])

    # 분할된 시계열 데이터 범위 확인
    def get_date_range(df):
        date_ranges = {}
        min_date = pd.to_datetime(df['datetime']).min()
        max_date = pd.to_datetime(df['datetime']).max()
        date_ranges = {min_date, max_date}
        return date_ranges

    print("Train Date:\n", get_date_range(train))
    print("Validation Date:\n", get_date_range(valid))
    print("Test Date:\n", get_date_range(test))
    print("\n")
    
    # datetime 컬럼 제거
    train = train.drop(columns=["datetime"])
    valid = valid.drop(columns=["datetime"])
    test = test.drop(columns=["datetime"])

    # 5) 이상치, 결측치 탐색 및 처리
    # IQR 기법을 이용한 이상치 탐색
    def detect_outliers_iqr(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df < lower_bound) | (df > upper_bound))
        return outliers
    
    train_outliers = detect_outliers_iqr(train)
    valid_outliers = detect_outliers_iqr(valid)
    test_outliers = detect_outliers_iqr(test)
    print("Train Outliers:\n", train_outliers.sum())
    print("Validation Outliers:\n", valid_outliers.sum())   
    print("Test Outliers:\n", test_outliers.sum())
    print("\n")

    # 결측치 탐색
    print("Train Missing Values:\n", train.isnull().sum())
    print("Validation Missing Values:\n", valid.isnull().sum())
    print("Test Missing Values:\n", test.isnull().sum())
    print("\n")

    # 이상치 - 생략
    # 결측치 - 없음

    # 6) 정규화, 표준화
    MinMaxscaler = MinMaxScaler() # 0~1 사이로 정규화
    Standardscaler = StandardScaler() # 평균 0, 표준편차 1로 표준화

    # train_data
    # fit_transform 모두 
    def train_scaler_fit_transform(df, scaler):
        df = df.copy()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df, scaler

    # valid / test_data
    # transform only
    def valid_test_scaler_transform(df, scaler):
        df = df.copy()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        return df

    # Min-Max scaling
    train_minmax, minmax_scaler = train_scaler_fit_transform(train, MinMaxscaler)
    valid_minmax = valid_test_scaler_transform(valid, minmax_scaler)
    test_minmax = valid_test_scaler_transform(test,  minmax_scaler)

    # z-score standardization
    train_scaled, standard_scaler = train_scaler_fit_transform(train_minmax, Standardscaler)
    valid_scaled = valid_test_scaler_transform(valid_minmax, standard_scaler)
    test_scaled  = valid_test_scaler_transform(test_minmax, standard_scaler)

    # target usage_kWh에 대해서도 동일하게 진행
    # feature / target 분리해서 진행해야 함
    # y_xx_scaled: 추후 역변환 시 사용

    # Min-Max scaling
    train_minmax_y, y_minmax_scaler = train_scaler_fit_transform(train_y, MinMaxscaler)
    valid_minmax_y = valid_test_scaler_transform(valid_y, MinMaxscaler)
    test_minmax_y = valid_test_scaler_transform(test_y, MinMaxscaler)

    # z-score standardization
    train_scaled_y, y_standard_scaler = train_scaler_fit_transform(train_minmax_y, Standardscaler)
    valid_scaled_y = valid_test_scaler_transform(valid_minmax_y, Standardscaler)
    test_scaled_y  = valid_test_scaler_transform(test_minmax_y, Standardscaler)

    # 전처리 완료된 데이터 저장

    # df 이름 형태의 폴더 생성
    if not os.path.exists(f'{industry_name}'):
        os.makedirs(f'{industry_name}')

    # origin: 원본
    train_origin.to_csv(f'{industry_name}/{industry_name}_train_origin.csv', index=False)
    valid_origin.to_csv(f'{industry_name}/{industry_name}_valid_origin.csv', index=False)
    test_origin.to_csv(f'{industry_name}/{industry_name}_test_origin.csv', index=False)

    # scaled: 정규화, 표준화 완료
    train_scaled.to_csv(f'{industry_name}/{industry_name}_train_scaled.csv', index=False)
    valid_scaled.to_csv(f'{industry_name}/{industry_name}_valid_scaled.csv', index=False)
    test_scaled.to_csv(f'{industry_name}/{industry_name}_test_scaled.csv', index=False)

    # y_scaled: target usage_kWh에 대해서도 동일하게 진행
    train_scaled_y.to_csv(f'{industry_name}/{industry_name}_train_scaled_y.csv', index=False)
    valid_scaled_y.to_csv(f'{industry_name}/{industry_name}_valid_scaled_y.csv', index=False)
    test_scaled_y.to_csv(f'{industry_name}/{industry_name}_test_scaled_y.csv', index=False)

    # 스케일러 객체 저장
    import joblib
    joblib.dump(minmax_scaler, f'{industry_name}/{industry_name}_minmax_scaler.pkl')
    joblib.dump(standard_scaler, f'{industry_name}/{industry_name}_standard_scaler.pkl')
    joblib.dump(y_minmax_scaler, f'{industry_name}/{industry_name}_y_minmax_scaler.pkl')
    joblib.dump(y_standard_scaler, f'{industry_name}/{industry_name}_y_standard_scaler.pkl')

    print(f"[{industry_name}] Data preprocessing completed and saved.\n")

# main 실행 블록
if __name__ == "__main__":
    industry_name = "광명금속"
    data = "광명금속_시계열_데이터(2024.08_2025.09).csv"
    # data inspection
    data = data_inspection(industry_name, data)
    # data preprocessing
    data_preprocessing(industry_name, data)