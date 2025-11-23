"""
==============================================================================
File: n01_load_data.py
Project: 산업 단지 전력 사용량 예측 모델
Author: 유도연
Created Date: 2025-10-13
Last Modified: 2025-11-19

Description: 데이터 로드 및 병합 스크립트
    - 한 파일의 여러 테이블에서 15분 단위 시계열 데이터를 추출해 병합
    - 월 단위 파일을 순회하며 하나의 데이터로 통합하고 CSV로 저장
    - 한 산업체씩 실행

Note: 산업체명 변경할 경우
    - 코드 실행 시 --industry name {산업체명} 형태로 인자 전달해야 함
        - 실행 예시: python n01_load_data.py --industry_name 광명금속
    - 추가로 기간/키워드/확장자/정규식 등의 파라미터도 필요 시 조정 가능
==============================================================================
"""

import os
import re
from datetime import datetime, timedelta
import argparse
import pandas as pd

# 명령행 인자 파서 설정 함수
def parse_args():
    parser = argparse.ArgumentParser(description="전력사용량 15분 데이터 수집")
    parser.add_argument("--industry_name", type=str, default=None, help="결과 파일명에 사용할 산업체 이름")
    return parser.parse_args()
    
# 컬럼명 조정 함수
def make_new_cols_from_pairs(pairs):
    """
    Function: make_new_cols_from_pairs
        - 입력된 열 이름 쌍 리스트를 바탕으로 최종 열 이름 리스트 생성
        - 컬럼명에서 두 행을 쓰는 경우가 존재해 a_b 형태로 병합해주기 위함
    Parameters:
        - pairs: List[Tuple[str,str]]
            - 열 이름 쌍 리스트 (예: [('시','시'), ('사용량 (kWh)','사용량 (kWh)'), ...])
    Returns:
        - new_cols: List[str]
            - 최종 열 이름 리스트 (예: ['시', '사용량 (kWh)', ..., '역률 (%)_진상'])
    """
    # 컬럼명을 한 행만 쓰는 경우 하나로 병합 / 두 행을 쓰는 경우 a_b 형태로 병합
    new_cols = [a if a == b else f"{a}_{b}" for a, b in pairs]

    return new_cols

# 월 리스트 생성 함수
def build_month_list(start_ym, end_ym):
    """
    Function: build_month_list
        - 시작 "YYYY.MM"과 종료 "YYYY.MM" 사이의 모든 월을 포함하는 리스트 생성
        - 현재까지 존재하는 데이터: 2024.08 ~ 2025.09을 기본값으로 설정
    Parameters:
        - start_ym: str
            - 시작 "YYYY.MM" (예: "2024.08")
        - end_ym: str
            - 종료 "YYYY.MM" (예: "2025.09")
    Returns:
        - months: List[str]
            - "YYYY.MM" 형식의 월 리스트 (예: ["2024.08", "2024.09", ..., "2025.09"])
    """
    # 시작 연,월 분해
    sy, sm = map(int, start_ym.split('.')) # "2024.08" -> 2024, 8
    # 종료 연,월 분해
    ey, em = map(int, end_ym.split('.')) # "2025.09" -> 2025, 9

    # 현재 연,월 초기화
    y, m = sy, sm
    # 결과 리스트 초기화
    months = []

    # while 문: (y,m)이 종료 지점을 지날 때까지
    while (y < ey) or (y == ey and m <= em):

        # "YYYY.MM" 형식으로 추가
        months.append(f"{y}.{m:02d}")
        
        # 다음 달로 이동
        m += 1
        
        # 13월이면 다음 해로 넘어감
        if m == 13:
            m = 1
            y += 1
    
    # 최종 리스트 반환
    return months

# 단일 파일에서 15분 단위 시계열 데이터 추출 함수
def load_data(file_path, new_cols, date_str):
    """
    Function: load_data
        - 단일 파일에서 15분 단위 시계열 데이터를 추출해 데이터프레임으로 반환
            - 여러 테이블 중 가장 큰 테이블 선택
            - 좌우 2블록 구조 -> 한 블록 오름차순으로 정렬 후 병합 
        - 파일명에서 추출한 날짜 문자열(YYYYMMDD)을 바탕으로 datetime 열 생성
    Parameters:
        - file_path: str
            - 읽을 파일 경로 (예: "C:/.../data/raw/9.광명금속/2024.08/전기사용량_시간대별(20240801).xls")
        - new_cols: List[str]
            - 최종 열 이름 리스트 (예: ['시', '사용량 (kWh)', ..., '역률 (%)_진상'])
        - date_str: str
            - 파일명에서 추출한 날짜 문자열 (예: "20240801")
    Returns:
        - time_data: pd.DataFrame
            - 추출된 15분 단위 시계열 데이터 (datetime 열 포함)
    """

    # xls 파일, HTML 테이블로 읽기
    tables = pd.read_html(file_path, encoding='utf-8') 

    # 여러 테이블 중 가장 큰(행 수가 많은) 테이블 선택
    #    시계열 데이터이므로 가장 클 것으로 가정 (실제 파일 확인 후 진행함)
    df = max(tables, key=lambda x: len(x))

    # "HH:MM" 형식 판별 정규식
    time_pattern = r'^\d{2}:\d{2}$'

    # 값이 시간 형식인지 확인하는 내부 함수
    def is_time_format(val):
        # NaN이면 False
        if pd.isna(val):
            return False
        # 문자열로 바꿔 정규식 매칭
        return bool(re.match(time_pattern, str(val)))

    # 시간 값이 포함된 열 인덱스 찾기 (좌/우 블록 형태로 존재)
    time_cols = [i for i in range(df.shape[1]) if df.iloc[:, i].apply(is_time_format).any()]

    # 좌우 2블록 구조인 경우
    if len(time_cols) >= 2:
        # 왼쪽 블록 시작/끝 인덱스
        left_start = time_cols[0]
        left_end = time_cols[1]
        # 왼쪽 블록 슬라이싱
        left_block = df.iloc[:, left_start:left_end].copy()
        # 왼쪽 블록에서 첫 열이 시간 형식인 행만 필터
        left_time_mask = left_block.iloc[:, 0].apply(is_time_format)
        left_data = left_block[left_time_mask]

        # 오른쪽 블록 슬라이싱
        right_block = df.iloc[:, left_end:].copy()
        # 오른쪽 블록에서 첫 열이 시간 형식인 행만 필터
        right_time_mask = right_block.iloc[:, 0].apply(is_time_format)
        right_data = right_block[right_time_mask]

        # 오른쪽 블록의 컬럼을 왼쪽과 동일하게 맞춤 (열 개수가 일치)
        right_data.columns = left_data.columns
        # 위아래로 결합
        time_data = pd.concat([left_data, right_data], ignore_index=True)
    else:
        # 단일 블록 구조라면 첫 열이 시간 형태인 행만 사용
        time_mask = df.iloc[:, 0].apply(is_time_format)
        time_data = df[time_mask].copy()

    # 00:00 행 제외 (00:00 x 다음 24:00을 가짐)
    time_data = time_data[time_data.iloc[:, 0] != '00:00']

    time_data.columns = new_cols

    # 인덱스 정렬
    time_data = time_data.reset_index(drop=True)

    # 정렬용 시각 컬럼 시리즈 생성
    sort_series = pd.to_datetime(time_data['시'], format='%H:%M', errors='coerce')
    sort_series = sort_series.mask(time_data['시'] == '24:00', pd.to_datetime('1900-01-02 00:00:00'))
    time_data['_sort_time'] = sort_series

    # 시간 오름차순 정렬
    time_data = time_data.sort_values('_sort_time')
    time_data = time_data.drop(columns=['_sort_time'])

    # 파일명에서 추출한 날짜 문자열을 date로 변환
    date_base = datetime.strptime(date_str, "%Y%m%d").date()
    
    # datetime 결과를 담을 리스트
    datetimes = []

    # 각 행의 '시' 값에 대해 datetime 생성
    for t in time_data['시']:
        # '24:00'은 다음날 00:00으로 처리
        if t == '24:00':
            dt = datetime.combine(date_base, datetime.strptime('00:00', '%H:%M').time()) + timedelta(days=1)
        else:
            # 일반 "HH:MM" 시각은 같은 날짜에 결합
            dt = datetime.combine(date_base, datetime.strptime(t, '%H:%M').time())
        datetimes.append(dt)

    # 가장 앞에 datetime 열 삽입
    time_data.insert(0, 'datetime', datetimes)
    # '시' 열 제거
    time_data = time_data.drop(columns=['시'])

    # 완성된 데이터프레임 반환
    return time_data

# 여러 파일을 순회하며 통합하고 CSV로 저장하는 함수
def collect_quarterhour_data(
    base_dir,
    start_ym="2024.08",
    end_ym="2025.09",
    file_keyword="전기사용량_시간대별",
    file_ext=".xls",
    date_regex=r"\((\d{8})\)",
    cols_pairs=None,
    industry_name=None,
    output_csv="통합_15분단위_데이터.csv",
    output_encoding="utf-8-sig"
) -> pd.DataFrame:
    """
    Function: collect_quarterhour_data
        - 특정 산업체 폴더(base_dir) 아래에서 월별 하위폴더(YYYY.MM)를 순회하며,
            파일명에 날짜가 포함된 원시 파일을 읽어 15분 단위 시계열로 통합하고 CSV로 저장
    Parameters:
        - base_dir: str
            - 산업체 루트 경로 (예: "C:/.../data/raw/9.메인텍 2공장")
        - start_ym: str
            - 시작 "YYYY.MM" (예: "2024.08")
        - end_ym: str
            - 종료 "YYYY.MM" (예: "2025.09")
        - file_keyword: str
            - 파일명에 포함될 키워드 (예: "전기사용량_시간대별")
        - file_ext: str
            - 파일 확장자 (예: ".xls")
        - date_regex: str
            - 파일명에서 날짜(YYYYMMDD) 추출 정규식 
        - cols_pairs: List[Tuple[str,str]] or None
            - 열 이름 쌍 (None이면 기본값 사용)
        - output_csv: str
            - 저장할 CSV 파일명
        - output_encoding: str
            - CSV 인코딩 (예: "utf-8-sig")
    Returns:
        - final_df: pd.DataFrame or None
            - 통합된 전체 15분 단위 데이터 (없으면 None)
    """
    # 기본 열 이름 쌍이 없는 경우, 표준 쌍 사용
    if cols_pairs is None:
        cols_pairs = [
            ('시', '시'),
            ('사용량 (kWh)', '사용량 (kWh)'),
            ('최대수요 (kW)', '최대수요 (kW)'),
            ('무효전력 (kVarh)', '지상'),
            ('무효전력 (kVarh)', '진상'),
            ('CO2 (tCO2)', 'CO2 (tCO2)'),
            ('역률 (%)', '지상'),
            ('역률 (%)', '진상')
        ]
    # 열 이름 변환 리스트 생성
    new_cols = make_new_cols_from_pairs(cols_pairs)

    # 순회할 월 폴더 리스트 생성
    folders = build_month_list(start_ym, end_ym)

    # 누적 리스트 및 카운터 초기화
    all_data = []
    success_count = 0
    error_count = 0

    # 각 월 폴더 순회
    for folder in folders:
        # 폴더 경로 생성
        folder_path = os.path.join(base_dir, folder)
        # 폴더가 없으면 스킵
        if not os.path.exists(folder_path):
            print(f"폴더 없음: {folder_path}")
            continue

        # 폴더 내 파일 순회
        for file_name in os.listdir(folder_path):
            # 확장자와 키워드로 필터
            if file_name.endswith(file_ext) and file_keyword in file_name:
                # 파일명에서 (YYYYMMDD) 형식 날짜 추출
                match = re.search(date_regex, file_name)
                if match:
                    # 날짜 문자열 가져오기
                    date_str = match.group(1)
                    # 전체 경로 생성
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        # 개별 파일 파싱
                        df_part = load_data(file_path, new_cols, date_str)
                        # 누적 리스트에 추가
                        all_data.append(df_part)
                        # 성공 카운트 증가
                        success_count += 1
                    except Exception as e:
                        # 에러 로그 출력
                        print(f"오류 발생 [{file_name}]: {e}")
                        # 실패 카운트 증가
                        error_count += 1

    # 누적된 데이터가 있으면 통합
    if all_data:
        # 세로 방향 결합
        final_df = pd.concat(all_data, ignore_index=True)
        # datetime 기준 정렬
        final_df = final_df.sort_values('datetime').reset_index(drop=True)

        # 진행 결과 출력
        print("\n" + "=" * 60)
        print(f"{industry_name} 데이터 통합 완료")
        print("=" * 60)
        print(f"성공: {success_count}개 파일")
        print(f"오류: {error_count}개 파일")
        print(f"총 {len(final_df):,}개 행")
        print(f"\n기간: {final_df['datetime'].min()} ~ {final_df['datetime'].max()}")

        # CSV 저장
        # base_dir의 상위(= raw의 상위 -> data)
        parent_dir = os.path.dirname(os.path.dirname(base_dir))

        # processed 폴더 경로
        processed_dir = os.path.join(parent_dir, "preprocessed")

        # 폴더가 없으면 생성
        os.makedirs(processed_dir, exist_ok=True)
        out_path = os.path.join(processed_dir, output_csv)
        final_df.to_csv(out_path, index=False, encoding=output_encoding)
        print(f"\n저장 완료: {out_path}")

        # 최종 데이터프레임 반환
        return final_df
    else:
        # 누적 데이터가 없을 때 메시지 출력
        print("통합할 데이터가 없습니다.")
        # None 반환
        return None

# 메인 실행 블록
if __name__ == "__main__":
    """
    - 코드 실행 시 --industry name {산업체명} 형태로 인자 전달해야 함 (실행 예시: python n01_load_data.py --industry_name 광명금속)
    - 기간/키워드/확장자/정규식 등의 파라미터도 필요 시 조정 가능
    """
    args = parse_args()

    industry_name = args.industry_name
    if industry_name is None:
        industry_name = "광명금속" # 기본값 설정

    # 산업체 루트 경로
    base_dir = f"C:/Users/ryudo/Desktop/forecasting_models/industrial_complex/data/raw/9.{industry_name}"
    start_ym = "2024.08"
    end_ym = "2025.09"

    # 실행
    _ = collect_quarterhour_data(
        base_dir=base_dir,
        start_ym=start_ym,
        end_ym=end_ym,
        file_keyword="전기사용량_시간대별",
        file_ext=".xls",
        date_regex=r"\((\d{8})\)",
        cols_pairs=[
            ('시', '시'),
            ('사용량 (kWh)', '사용량 (kWh)'),
            ('최대수요 (kW)', '최대수요 (kW)'),
            ('무효전력 (kVarh)', '지상'),
            ('무효전력 (kVarh)', '진상'),
            ('CO2 (tCO2)', 'CO2 (tCO2)'),
            ('역률 (%)', '지상'),
            ('역률 (%)', '진상')
        ],
        industry_name=industry_name,
        output_csv=f"{industry_name}_시계열_데이터({start_ym}_{end_ym}).csv",
        output_encoding="utf-8-sig"
    )