"""
==============================================================================
File: n02_data_preprocessing.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-14
Last Modified: 2025-11-19

Description: data inspection 및 preprocessing 과정
    1) 컬럼명 변경
    2) 변수별 시각화
    3) 관련 변수 추가
    4) Train / Valid / Test 분할
    5) X, y 분리 
    6) 이상치, 결측치 탐색 및 처리
    7) 로그 변환
    8) 정규화, 표준화 
        - robust scaling만 진행

Note
    - 코드 실행시 {industry_name} 인자 설정 필요
    - 모델 유형 선택 시 {model_type} 인자 설정 필요 (longformer의 경우에만 사용)
        - 실행 예시: python n02_data_preprocessing.py --industry_name 광명금속 (--model_type longformer)
        
    - longformer 모델의 경우 전처리 과정이 일부 다름
    - 추가적인 전처리 과정이 필요할 수 있음
    - 이후의 과정은 {modeling.py}에서 처리 (예: 슬라이딩 윈도우, 데이터로더 생성 등)
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# 출력 결과 txt로 저장하는 클래스
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message) # 터미널에 출력
        self.log.write(message) # 파일에도 기록

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 명령행 인자 파서 설정 함수
def parse_args():
    parser = argparse.ArgumentParser(description="전력사용량 15분 데이터 전처리 스크립트")
    parser.add_argument("--industry_name", type=str, default=None, help="결과 파일명에 사용할 산업체 이름")
    parser.add_argument("--model_type", type=str, default=None, help="모델 유형 선택 (informer, autoformer의 경우에만 사용)")
    return parser.parse_args()

# 데이터 시각화 및 탐색 함수
def data_inspection(industry_name: str, data: pd.DataFrame) -> pd.DataFrame:
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
    data = pd.read_csv("data/preprocessed/" + data)

    print("=" * 60)
    print(f"[{industry_name}] Data inspection process...")
    print("=" * 60)
    
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
    
    # 상관관계가 0.35 이하인 컬럼 drop
    data = data.drop(columns=['reactive_usage_kWh_kVarh_capacitive', 'usage_kWh_factor_capacitive'])
    print(data.columns, "\n")

    # 2) 변수별 시각화
    feature_cols = [col for col in data.columns if col != 'datetime' and col != 'usage_kWh']
    
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False

    # df 이름 형태의 폴더 생성 
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists(f'plots/{industry_name}'):
        os.makedirs(f'plots/{industry_name}')

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
    plt.savefig(f'plots/{industry_name}/[{industry_name}] weekly_mean_usage.png')
    # day_of_week 열 삭제
    data.drop(columns=['day_of_week'], inplace=True)

    # 2-2) 월별 평균 사용량 시각화
    # 날짜별 월 추출
    data['month'] = data['datetime'].dt.month

    # 월별 평균 사용량 계산
    avg_usage_by_month = data.groupby('month')['usage_kWh'].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_usage_by_month.index, y=avg_usage_by_month.values)
    plt.title(f'[{industry_name}] monthly mean usage (kWh)')
    plt.xlabel('Month')
    plt.ylabel('Average Usage (kWh)')
    # plt.show() # 주석 처리 - 노트북에서 실행 시 사용
    plt.savefig(f'plots/{industry_name}/[{industry_name}] monthly_mean_usage.png')
    # month 열 삭제
    data.drop(columns=['month'], inplace=True)

    # 2-3) 시간대별 평균 사용량 시각화
    # 날짜별 시간 추출
    data['hour'] = data['datetime'].dt.hour
    # 시간대별 평균 사용량 계산
    avg_usage_by_hour = data.groupby('hour')['usage_kWh'].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_usage_by_hour.index, y=avg_usage_by_hour.values)
    plt.title(f'[{industry_name}] hourly mean usage (kWh)')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Usage (kWh)')
    # plt.show() # 주석 처리 - 노트북에서 실행 시 사용
    plt.savefig(f'plots/{industry_name}/[{industry_name}] hourly_mean_usage.png')
    # hour 열 삭제
    data.drop(columns=['hour'], inplace=True)

    # 2-4) 각 변수별 사용량과의 상관관계 산점도
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(feature_cols):
        plt.subplot(3, 2, i + 1)
        sns.scatterplot(data=data, x=col, y='usage_kWh')
        plt.title(f'[{industry_name}] Usage vs {col}')
        plt.xlabel(col)
        plt.ylabel('Usage (kWh)')
    plt.tight_layout()
    # plt.show() # 주석 처리 - 노트북에서 실행 시 사용
    plt.savefig(f'plots/{industry_name}/[{industry_name}] features_usage_scatter_plot.png')

    # 2-5) 각 변수별 사용량과의 상관관계 히트맵
    plt.figure(figsize=(10, 10))
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'[{industry_name}] Correlation Matrix')
    plt.tight_layout()
    # plt.show() # 주석 처리 - 노트북에서 실행 시 사용
    plt.savefig(f'plots/{industry_name}/[{industry_name}] correlation_matrix.png')
    
    print(f"[{industry_name}] Data inspection completed and plots saved.")
    return data



# 데이터 전처리 함수
def data_preprocessing(industry_name: str, data: pd.DataFrame, model_type: str = None) -> None:
    """
    Function: data_preprocessing
        3) 관련 변수 추가 - lag feature, rolling window feature, 날짜시간 파생변수
        4) Train / Valid / Test 분할
        5) X, y 분리
        6) 이상치, 결측치 탐색 및 처리
        7) 로그 변환
        8) 정규화, 표준화
            - robust scaling만 진행
    Parameters:
        - industry_name: 산업체 이름 (예: "광명금속")
        - data: 전처리할 데이터프레임
    Returns:
        - None (전처리된 데이터는 CSV 파일로 저장)
    """

    print("\n" + "=" * 60)
    print(f"[{industry_name}] Data preprocessing process...")
    print("=" * 60)

    # 3) 관련 변수 추가
    # lag feature 추가 
    def add_lag_features(df: pd.DataFrame, target_col: str, lags: list) -> pd.DataFrame:
        '''
        Function: add_lag_features
            - target_col에 대해 지정된 시차(lags)만큼의 lag feature를 추가
        Parameters:
            - df: pd.DataFrame
                - 시계열 데이터프레임
            - target_col: str
                - lag를 생성할 대상 컬럼명
            - lags: list
                - 생성할 lag 시차 리스트 (예: [1, 2, 3]은 1시점 전, 2시점 전, 3시점 전 lag 생성)
        Returns:
            - pd.DataFrame: lag feature가 추가된 데이터프레임
        '''
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        return df

    # rolling window feature 추가
    def add_rolling_features(df: pd.DataFrame, target_col: str, windows: list) -> pd.DataFrame:
        '''
        Function: add_rolling_features
            - windows 동안의 평균 사용량 변화
        Parameters:
            - df: pd.DataFrame
                - 시계열 데이터프레임
            - target_col: str
                - rolling mean을 생성할 대상 컬럼명
            - windows: list
                - 생성할 rolling 윈도우 크기 리스트 (예: [4, 96]은 1시간, 1일 rolling mean 생성)
        Returns:
            - pd.DataFrame: rolling mean, std, max feature가 추가된 데이터프레임
        '''
        df = df.copy()
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean() # 이전 시점까지의 평균
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std() # 이전 시점까지의 표준편차
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window).max() # 이전 시점까지의 최대값
        return df
    
    lag_steps = [1, 4, 96, 672] # 15분, 1시간, 1일, 1주
    rolling_windows = [4, 96] # 1시간, 1일

    data = add_lag_features(data, target_col='usage_kWh', lags=lag_steps)
    data = add_rolling_features(data, target_col='usage_kWh', windows=rolling_windows)

    data = data.dropna().reset_index(drop=True) # lag, rolling feature 생성 후 생긴 결측치 제거 # 기존 데이터에서는 Nan 없음 - 확인 완료
    print(f"[lag and rolling 변수 추가 완료]: {data.shape}\n")

    try:
        import holidays
        HAS_HOLIDAYS = True
    except Exception:
        HAS_HOLIDAYS = False

    # 날짜시간 파생변수 생성 함수
    def build_date_related_features(df: pd.DataFrame,
                                    datetime_col: str = "datetime",
                                    tz: str = "Asia/Seoul",
                                    drop_raw_parts: bool = True) -> pd.DataFrame:
        '''
        Function: build_date_related_features
            - datetime 기반 달력/시간 파생변수를 생성하고, 주기적 패턴을 sin/cos로 인코딩
            - 한국 공휴일(있으면) 및 주말 플래그 추가
        Parameters:
            - df: pd.DataFrame
                - datetime_col을 포함하는 데이터프레임
            - datetime_col: str
                - 날짜시간 컬럼명 (기본 "datetime")
            - tz: str
                - 타임존, naive면 로컬라이즈 / aware면 변환 (기본 "Asia/Seoul")
            - drop_raw_parts: bool
                - year/month/day/hour 등 원시 분해 컬럼을 드랍할지 여부 (기본 True)
        Returns:
            - out: pd.DataFrame
                - 파생변수가 추가된 데이터프레임 (copy 반환)
        '''
        out = df.copy()  # 원본 보존

        # 1) datetime 정리
        out[datetime_col] = pd.to_datetime(out[datetime_col])
        if out[datetime_col].dt.tz is None:
            out[datetime_col] = out[datetime_col].dt.tz_localize(tz)
        else:
            out[datetime_col] = out[datetime_col].dt.tz_convert(tz)

        dt = out[datetime_col]

        # 2) 기본 분해
        out['year'] = dt.dt.year # 연도
        out['month'] = dt.dt.month # 월(1~12)
        out['day'] = dt.dt.day # 일(1~31)
        out['hour'] = dt.dt.hour # 시(0~23)
        out['minute'] = dt.dt.minute # 분(0~59)
        out['dayofweek'] = dt.dt.weekday # 요일(월=0, 일=6)
        out['dayofyear'] = dt.dt.dayofyear # 연중 몇번째 날(1~366)
        out['weekofyear'] = dt.dt.isocalendar().week.astype(int) # 연중 몇번째 주(1~52/53)
        out['quarter'] = dt.dt.quarter # 분기(1~4)

        # 3) 주말/공휴일 플래그
        out['is_weekend'] = out['dayofweek'].isin([5, 6]).astype(int) # 토 or 일=1

        if HAS_HOLIDAYS:
            y0, y1 = out['year'].min(), out['year'].max()
            kr_holiday = holidays.country_holidays('KR', years=list(range(y0, y1 + 1)))
            holiday_dates = pd.to_datetime(list(kr_holiday.keys())).date # date 배열
            holiday_set = set(holiday_dates) # membership 테스트용 set
            out['is_holiday'] = dt.dt.date.isin(holiday_set).astype(int) # 벡터화된 isin
        else:
            out['is_holiday'] = 0

        # 4) 계절 라벨(겨울=0, 봄=1, 여름=2, 가을=3) 
        def _season(m: int) -> int:
            # 12,1,2=겨울(0) / 3,4,5=봄(1) / 6,7,8=여름(2) / 9,10,11=가을(3)
            if m in (12, 1, 2):
                return 0
            elif m in (3, 4, 5):
                return 1
            elif m in (6, 7, 8):
                return 2
            else:
                return 3
        out['season'] = out['month'].apply(_season).astype(int)

        # 5) 사이클릭 인코딩
        # 요일(7), 시(24), 월(12), 연중일(365/366)
        out['dow_sin']  = np.sin(2 * np.pi * out['dayofweek'] / 7.0)
        out['dow_cos']  = np.cos(2 * np.pi * out['dayofweek'] / 7.0)
        out['hour_sin'] = np.sin(2 * np.pi * out['hour'] / 24.0)
        out['hour_cos'] = np.cos(2 * np.pi * out['hour'] / 24.0)
        out['mon_sin']  = np.sin(2 * np.pi * (out['month'] - 1) / 12.0)
        out['mon_cos']  = np.cos(2 * np.pi * (out['month'] - 1) / 12.0)
        out['doy_sin']  = np.sin(2 * np.pi * (out['dayofyear'] - 1) / 366.0)
        out['doy_cos']  = np.cos(2 * np.pi * (out['dayofyear'] - 1) / 366.0)

        # 6) 원시 파트 드랍
        if drop_raw_parts:
            out = out.drop(columns=[
                'year', 'month', 'day', 'hour', 'minute',
                'dayofweek', 'dayofyear', 'weekofyear', 'quarter'
            ])
        return out
    
    def build_date_related_features_for_longformer(df: pd.DataFrame, datetime_col: str = "datetime", tz: str = "Asia/Seoul") -> pd.DataFrame:
        """
        Function: build_date_related_features_for_longformer
            - Longformer 모델용 날짜시간 파생변수 생성
            - month, day, weekday, hour, minute 컬럼 추가
        Parameters:
            - df: pd.DataFrame
                - datetime_col을 포함하는 데이터프레임
            - datetime_col: str
                - 날짜시간 컬럼명 (기본 "datetime")
            - tz: str
                - 타임존, naive면 로컬라이즈 / aware면 변환 (기본 "Asia/Seoul")
        Returns:
            - out: pd.DataFrame
                - 파생변수가 추가된 데이터프레임 (copy 반환)
        """
        out = df.copy()
        out[datetime_col] = pd.to_datetime(out[datetime_col])
        # timezone 처리
        if out[datetime_col].dt.tz is None:
            out[datetime_col] = out[datetime_col].dt.tz_localize(tz)
        else:
            out[datetime_col] = out[datetime_col].dt.tz_convert(tz)

        dt = out[datetime_col]

        # Informer TimeFeatureEmbedding 't' 입력형식 5개
        out["month"] = dt.dt.month # 1~12
        out["day"] = dt.dt.day # 1~31
        out["weekday"] = dt.dt.weekday # 0~6
        out["hour"] = dt.dt.hour # 0~23
        out["minute"] = dt.dt.minute # 0~59
        out["minute"] = (out["minute"] // 15).astype(int) # 15분 단위로 변환: 0,15,30,45 -> 0,1,2,3

        # 이외 나머지 feature는 longformer embedding에서 사용되지 않음
        return out

    # 날짜 관련 파생변수 추가
    if model_type is None:
        data = build_date_related_features(data, datetime_col="datetime", tz="Asia/Seoul", drop_raw_parts=True)
    elif model_type == "longformer":
        data = build_date_related_features_for_longformer(data, datetime_col="datetime", tz="Asia/Seoul")
    print(f"[날짜 관련 파생변수 생성 완료]: {data.shape}\n")
    
    # 4) Train / Valid / Test 분할
    def split_train_val_test(data: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.15, gap_weeks: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Function: split_train_val_test
            - 시계열 데이터를 "주 단위(월 00:00 ~ 일 23:45)"로 정렬하여 train/valid/test 세 구간으로 분할하는 함수
            - 각 구간 사이에 gap 주만큼의 비어 있는 구간을 둠 - 우선 생략
            - 분할 비율은 "주 개수"를 기준으로 계산함
        Parameters:
            - data: pd.DataFrame
                - 'datetime' 컬럼을 포함한 시계열 데이터프레임
            - train_ratio: float
                - 학습 구간 비율 (기본 0.7)
            - valid_ratio: float
                - 검증 구간 비율 (기본 0.15)
            - gap_weeks: int
                - train -> valid, valid -> test 사이에 비울 주(week) 수 (기본 0, 0으로 두면 gap 없이 연속 분할)
        Returns:
            - (train_df, valid_df, test_df): tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
                - 각 구간에 해당하는 데이터프레임 3개를 순서대로 반환
        '''

        # 1. 기본 정리: 시간 변환 및 정렬
        df = data.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        # 2. 보조 함수: 주의 시작(월 00:00)과 주의 끝(일 23:45) 계산
        def next_monday_0000(ts: pd.Timestamp) -> pd.Timestamp:
            # 주 시작을 월요일 00:00으로 정렬
            base = ts.normalize()
            return base if base.weekday() == 0 else (base + pd.offsets.Week(weekday=0))

        def this_week_sun_2345(ts: pd.Timestamp) -> pd.Timestamp:
            # 같은 주의 일요일 23:45 반환 (15분 간격 가정)
            return ts.normalize() + pd.Timedelta(days=(6 - ts.weekday())) + pd.Timedelta(hours=23, minutes=45)

        # 3. 전체 기간을 주 단위로 정렬 (월 00:00 ~ 일 23:45)
        global_start = next_monday_0000(df['datetime'].min())
        global_end = this_week_sun_2345(df['datetime'].max())

        # 월요일 00:00 기준으로 주 경계 생성 
        mondays = pd.date_range(start=global_start, end=global_end, freq='W-MON', inclusive='both')

        # mondays가 최소 1개는 있어야 함. (데이터가 매우 짧을 때 보정)
        if len(mondays) == 0:
            mondays = pd.DatetimeIndex([global_start])

        # 각 주의 [start, end] 경계를 리스트로 구성
        # 주 시작: 해당 Monday 00:00
        # 주 끝: 해당 주의 Sunday 23:45
        weeks = []
        for m in mondays:
            start_w = m
            end_w = this_week_sun_2345(m) # m가 월요일이므로 같은 주 일요일 23:45
            # 마지막 주가 global_end를 넘어가면 global_end로 조정
            end_w = min(end_w, global_end)
            weeks.append((start_w, end_w))

        # 유효한 주만 남기기 (시작 <= 끝)
        weeks = [(s, e) for s, e in weeks if s <= e]
        n_weeks = len(weeks)
        if n_weeks == 0:
            # 주 단위로 자를 수 없는 경우: 전체를 test로 반환 (혹은 예외 처리)
            return df.iloc[0:0], df.iloc[0:0], df

        # 4. 비율을 "주 개수"에 적용하여 주 단위로 분할 개수 결정
        n_train = int(np.floor(n_weeks * train_ratio))
        n_valid = int(np.floor(n_weeks * valid_ratio))

        # 남은 주 중에서 test 주 개수
        n_test = n_weeks - n_train - n_valid - (gap_weeks * 2)

        # 5. 주 인덱스 슬라이싱 (train- > gap -> valid -> gap -> test)
        idx = 0
        train_weeks = weeks[idx : idx + n_train]
        idx += n_train

        idx += gap_weeks # train과 valid 사이 gap
        valid_weeks = weeks[idx : idx + n_valid]
        idx += n_valid

        idx += gap_weeks # valid와 test 사이 gap
        test_weeks = weeks[idx : idx + n_test]

        # 6. 각 구간의 실제 시간 경계 계산
        def segment_bounds(week_list):
            if not week_list:
                return None, None
            seg_start = week_list[0][0]
            seg_end = week_list[-1][1]
            return seg_start, seg_end

        train_start, train_end = segment_bounds(train_weeks)
        valid_start, valid_end = segment_bounds(valid_weeks)
        test_start, test_end = segment_bounds(test_weeks)

        # 7. 데이터프레임 필터링
        def cut_df_by_time(df_, start_t, end_t):
            if start_t is None or end_t is None:
                return df_.iloc[0:0]  # 빈 df
            m = (df_['datetime'] >= start_t) & (df_['datetime'] <= end_t)
            return df_.loc[m].copy()

        train_df = cut_df_by_time(df, train_start, train_end)
        valid_df = cut_df_by_time(df, valid_start, valid_end)
        test_df  = cut_df_by_time(df, test_start,  test_end)

        # 8. 분할된 시계열 데이터 범위 확인
        print("[Train / Valid / Test Split]\n")
        print(f"Train: {len(train_weeks)} | {train_start} - {train_end}")
        print(f" - Train/Valid 사이의 gap: {gap_weeks}")
        print(f"Valid: {len(valid_weeks)} | {valid_start} - {valid_end}")
        print(f" - Valid/Test 사이의 gap: {gap_weeks}")
        print(f"Test: {len(test_weeks)} | {test_start} - {test_end}")
        print(f" - 데이터 개수: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}\n")

        return train_df, valid_df, test_df

    train, valid, test = split_train_val_test(data)

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    # datetime을 첫번째로, usage_kWh를 맨 마지막으로 이동
    cols = train.columns.tolist()
    cols.remove('datetime')
    cols.remove('usage_kWh')
    cols = ['datetime'] + cols + ['usage_kWh']
    train = train[cols]
    valid = valid[cols]
    test = test[cols]
    print("[Train / valid / test 데이터 분할 및 컬럼 순서 재배치 완료]\n")
    
    # 추후 시각화를 위해 원본 복사
    train_origin = train.copy()
    valid_origin = valid.copy()
    test_origin = test.copy()

    # datetime 컬럼 제거
    train = train.drop(columns=["datetime"])
    valid = valid.drop(columns=["datetime"])
    test = test.drop(columns=["datetime"])

    # 5) X, y 분리
    train_y = pd.DataFrame(train['usage_kWh'])
    valid_y = pd.DataFrame(valid['usage_kWh'])
    test_y = pd.DataFrame(test['usage_kWh'])

    train_x = train.drop(columns=["usage_kWh"])
    valid_x = valid.drop(columns=["usage_kWh"]) 
    test_x = test.drop(columns=["usage_kWh"])

    # longformer 일 때 일반 / 날짜 변수 분할
    if model_type == "longformer":
        train_x_mark = train_x[['month', 'day', 'weekday', 'hour', 'minute']]
        train_x_mark = pd.DataFrame(train_x_mark)
        train_x = train_x.drop(columns=['month', 'day', 'weekday', 'hour', 'minute'])
        valid_x_mark = valid_x[['month', 'day', 'weekday', 'hour', 'minute']]
        valid_x_mark = pd.DataFrame(valid_x_mark)
        valid_x = valid_x.drop(columns=['month', 'day', 'weekday', 'hour', 'minute'])
        test_x_mark = test_x[['month', 'day', 'weekday', 'hour', 'minute']]
        test_x_mark = pd.DataFrame(test_x_mark)
        test_x = test_x.drop(columns=['month', 'day', 'weekday', 'hour', 'minute'])
        
    # 6) 이상치, 결측치 탐색 및 처리
    print("[이상치 및 결측치 탐색 및 처리]\n")

    # 결측치 탐색
    print("이상치 처리 전 결측치 개수 \n")
    print("Train Missing Values:\n", train.isnull().sum())
    print("Validation Missing Values:\n", valid.isnull().sum())
    print("Test Missing Values:\n", test.isnull().sum())

    # IQR 이상치 - NaN 대체 
    def handle_outliers_iqr(df, target_cols, factor=1.5, df_name="df"):
        """
        Function: handle_outliers_iqr
            1. df의 수치형 컬럼별로 IQR 기반 이상치를 NaN으로 대체
            2. 이상치 개수를 컬럼별로 출력
        Parameters:
            - df: 처리할 DataFrame
            - factor: IQR 배수 (기본값: 1.5)
            - df_name: DataFrame 이름 (기본값: "df")
        Returns:
            - pd.DataFrame: 이상치가 NaN으로 대체된 DataFrame
        """
        df = df.copy()
        
        outliers = {} # 전체 컬럼별 이상치 기록
        cols_to_process = [col for col in target_cols if col in df.columns]
        
        for col in cols_to_process:
            Q1 = df[col].quantile(0.25) # 1사분위수
            Q3 = df[col].quantile(0.75) # 3사분위수
            IQR = Q3 - Q1 # 이상치
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # 이상치 인덱스
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = df.loc[mask, col]
            
            # 이상치를 NaN으로 처리
            df.loc[mask, col] = np.nan

        # 이상치 개수 출력
        print(f"{df_name}의 이상치 개수:")
        for col, series in outliers.items():
            print(f"{col:<15}: {series.shape[0]}")
        print("\n")
        return df
    
    target_colums = [
        "max_demand_kW",
        "reactive_usage_kWh_kVarh_inductive",
        "CO2_tCO2",
        "usage_kWh_factor_inductive",
    ]
    train_x = handle_outliers_iqr(train_x, target_colums, factor=3, df_name="Train_x")
    valid_x = handle_outliers_iqr(valid_x, target_colums, factor=3, df_name="Validation_x")
    test_x = handle_outliers_iqr(test_x, target_colums, factor=3, df_name="Test_x")

    # 타겟 변수 이상치 처리 x

    # 결측치 처리 
    # train: 선형 보간 -> 앞뒤 값으로 채우기 -> 남은 결측치는 0으로 채우기
    train_x = train_x.interpolate(method='linear')
    train_x = train_x.ffill().bfill()
    train_x = train_x.fillna(0)

    # train 평균 계산 (결측치 처리 후)
    train_mean_x = train_x.mean()

    # valid, test: 앞의 값으로만 채우기 -> 남은 결측치는 train 평균값으로 채우기
    valid_x = valid_x.ffill()
    valid_x = valid_x.fillna(train_mean_x)

    test_x = test_x.ffill()
    test_x = test_x.fillna(train_mean_x)

    # y의 결측 처리(없음)
    train_y = train_y.ffill()
    valid_y = valid_y.ffill()
    test_y  = test_y.ffill()

    print("[결측치 보정 작업 후]\n")
    print("Train Missing Values:\n", train_x.isnull().sum())
    print("Validation Missing Values:\n", valid_x.isnull().sum())
    print("Test Missing Values:\n", test_x.isnull().sum())
    print("\n")

    # 7) 로그 변환
    train_y_log = np.log1p(train_y.clip(lower=0))
    valid_y_log = np.log1p(valid_y.clip(lower=0))
    test_y_log = np.log1p(test_y.clip(lower=0))

    train_y = train_y_log
    valid_y = valid_y_log
    test_y  = test_y_log
    print("[로그 변환 완료]\n")

    # 8) 정규화, 표준화
    x_minmax_scaler = MinMaxScaler() # 0-1 사이로 정규화
    x_standard_scaler = StandardScaler() # 평균 0, 표준편차 1로 표준화
    x_robust_scaler = RobustScaler() # 중앙값 0, IQR 1로 강건한 표준화

    train_x_scaled = pd.DataFrame(
        x_robust_scaler.fit_transform(train_x),
        columns=train_x.columns, index=train_x.index
    )
    valid_x_scaled = pd.DataFrame(
        x_robust_scaler.transform(valid_x),
        columns=valid_x.columns, index=valid_x.index
    )
    test_x_scaled = pd.DataFrame(
        x_robust_scaler.transform(test_x),
        columns=test_x.columns, index=test_x.index
    )

    y_robust_scaler = RobustScaler()
    
    train_y_scaled = pd.DataFrame(
        y_robust_scaler.fit_transform(train_y),
        columns=train_y.columns, index=train_y.index
    )
    valid_y_scaled = pd.DataFrame(
        y_robust_scaler.transform(valid_y),
        columns=valid_y.columns, index=valid_y.index
    )
    test_y_scaled = pd.DataFrame(
        y_robust_scaler.transform(test_y),
        columns=test_y.columns, index=test_y.index
    )

    train_preprocessed = train_x_scaled
    valid_preprocessed = valid_x_scaled
    test_preprocessed  = test_x_scaled

    train_preprocessed_y = train_y_scaled
    valid_preprocessed_y = valid_y_scaled
    test_preprocessed_y  = test_y_scaled

    train_preprocessed['usage_kWh'] = train_preprocessed_y['usage_kWh'] 
    valid_preprocessed['usage_kWh'] = valid_preprocessed_y['usage_kWh']
    test_preprocessed['usage_kWh'] = test_preprocessed_y['usage_kWh']

    # 전처리 완료된 데이터 저장

    # 데이터 형태 출력
    print("[최종 데이터 형태 출력]\n")
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

    # df 이름 형태의 폴더 생성
    if not os.path.exists(f'data/preprocessed/{industry_name}'):
        os.makedirs(f'data/preprocessed/{industry_name}')

    # 파일명 suffix 설정
    suffix = f"_longformer" if model_type == "longformer" else ""

    # origin: 원본
    train_origin.to_csv(f"data/preprocessed/{industry_name}/{industry_name}{suffix}_train_origin.csv", index=False)
    valid_origin.to_csv(f"data/preprocessed/{industry_name}/{industry_name}{suffix}_valid_origin.csv", index=False)
    test_origin.to_csv(f"data/preprocessed/{industry_name}/{industry_name}{suffix}_test_origin.csv", index=False)

    # scaled: 정규화, 표준화 완료
    train_preprocessed.to_csv(f"data/preprocessed/{industry_name}/{industry_name}{suffix}_train_preprocessed.csv", index=False)
    valid_preprocessed.to_csv(f"data/preprocessed/{industry_name}/{industry_name}{suffix}_valid_preprocessed.csv", index=False)
    test_preprocessed.to_csv(f"data/preprocessed/{industry_name}/{industry_name}{suffix}_test_preprocessed.csv", index=False)

    # y_scaled: target usage_kWh 처리
    train_preprocessed_y.to_csv(f"data/preprocessed/{industry_name}/{industry_name}{suffix}_train_preprocessed_y.csv", index=False)
    valid_preprocessed_y.to_csv(f"data/preprocessed/{industry_name}/{industry_name}{suffix}_valid_preprocessed_y.csv", index=False)
    test_preprocessed_y.to_csv(f"data/preprocessed/{industry_name}/{industry_name}{suffix}_test_preprocessed_y.csv", index=False)

    # longformer 일 때 
    if model_type == "longformer":
        # mark: x_mark 분리본
        train_x_mark.to_csv(f'data/preprocessed/{industry_name}/{industry_name}{suffix}_train_x_mark.csv', index=False)
        valid_x_mark.to_csv(f'data/preprocessed/{industry_name}/{industry_name}{suffix}_valid_x_mark.csv', index=False)
        test_x_mark.to_csv(f'data/preprocessed/{industry_name}/{industry_name}{suffix}_test_x_mark.csv', index=False)

    # 스케일러 저장
    # joblib.dump(x_minmax_scaler, f'data/preprocessed/{industry_name}/{industry_name}{suffix}_minmax_scaler.pkl')
    # joblib.dump(x_standard_scaler, f'data/preprocessed/{industry_name}/{industry_name}{suffix}_standard_scaler.pkl')
    joblib.dump(x_robust_scaler, f"data/preprocessed/{industry_name}/{industry_name}{suffix}_robust_scaler.pkl")
    # joblib.dump(y_minmax_scaler, f'data/preprocessed/{industry_name}/{industry_name}{suffix}_y_minmax_scaler.pkl')
    # joblib.dump(y_standard_scaler, f'data/preprocessed/{industry_name}/{industry_name}{suffix}_y_standard_scaler.pkl')
    joblib.dump(y_robust_scaler, f"data/preprocessed/{industry_name}/{industry_name}{suffix}_y_robust_scaler.pkl")

    print(f"[{industry_name}] Data preprocessing completed and saved.\n")

    
# main 실행 블록
if __name__ == "__main__":
    """
    - 코드 실행 시 산업체 이름을 인자로 받음
        - 예: python n02_data_preprocessing.py --industry_name 광명금속 --model_type longformer
    """
    args = parse_args()
    industry_name = args.industry_name
    if industry_name is None:
        industry_name = "광명금속" # 기본값 설정
    
    model_type = args.model_type
    data = f"{industry_name}_시계열_데이터(2024.08_2025.09).csv"

    if not os.path.exists(f"data/preprocessed/{industry_name}"):
        os.makedirs(f"data/preprocessed/{industry_name}")

    if model_type is None:
        txt_path = f"data/preprocessed/{industry_name}/{industry_name}_데이터_전처리_log.txt"
    elif model_type == "longformer":
        txt_path = f"data/preprocessed/{industry_name}/{industry_name}_longformer_데이터_전처리_log.txt"
    sys.stdout = DualLogger(txt_path) # 로그 파일 경로 설정
    
    # data inspection
    data = data_inspection(industry_name, data)
    print("======================================================================\n")
    # data preprocessing
    data_preprocessing(industry_name, data, model_type=model_type)
    print("======================================================================\n")