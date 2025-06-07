import FinanceDataReader as fdr
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm


class DataCollector:
    """
    KOSPI 전체 종목의 과거 주가 데이터를 수집하여 로컬에 저장하는 클래스.
    """

    def __init__(self, data_dir="./quantmo_data/stock_data", years=12):
        self.data_dir = data_dir
        self.years = years
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=self.years * 365)

        # 데이터 저장 폴더 생성
        os.makedirs(self.data_dir, exist_ok=True)

        print("DataCollector가 초기화되었습니다.")
        print(f" - 데이터 저장 경로: {self.data_dir}")
        print(f" - 데이터 수집 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")

    def get_kospi_stock_list(self):
        """KOSPI 시장의 전체 종목 리스트를 가져옵니다."""
        print("KOSPI 전체 종목 리스트를 수집합니다...")
        try:
            stocks = fdr.StockListing('KOSPI')
            # 필수 컬럼(Code, Name) 확인
            if 'Code' not in stocks.columns or 'Name' not in stocks.columns:
                raise ValueError("FinanceDataReader에서 반환된 DataFrame에 'Code' 또는 'Name' 컬럼이 없습니다.")
            print(f"✅ 총 {len(stocks)}개의 KOSPI 종목을 확인했습니다.")
            return stocks
        except Exception as e:
            print(f"❌ KOSPI 종목 리스트 수집 중 오류 발생: {e}")
            return pd.DataFrame()

    def collect_and_save_stock_data(self, stock_code, stock_name):
        """단일 종목의 데이터를 수집하고 CSV 파일로 저장합니다."""
        save_path = os.path.join(self.data_dir, f"{stock_code}_{stock_name}.csv")

        # 이미 파일이 존재하면 건너뛰기
        if os.path.exists(save_path):
            return True, "이미 수집됨"

        try:
            # FinanceDataReader를 통해 데이터 로드
            df = fdr.DataReader(stock_code, self.start_date, self.end_date)

            # 데이터가 비어있는 경우 건너뛰기
            if df.empty:
                return False, "데이터 없음"

            # CSV로 저장
            df.to_csv(save_path)
            return True, "성공"

        except Exception as e:
            # 오류 발생 시 로그 남기고 건너뛰기
            return False, str(e)

    def run_collection(self):
        """전체 데이터 수집 파이프라인을 실행합니다."""
        kospi_stocks = self.get_kospi_stock_list()

        if kospi_stocks.empty:
            print("데이터 수집을 진행할 수 없습니다.")
            return

        success_count = 0
        fail_count = 0

        print("\n전체 종목 데이터 수집을 시작합니다. (시간이 오래 소요될 수 있습니다)")

        # tqdm을 사용하여 진행 상황 표시
        for _, row in tqdm(kospi_stocks.iterrows(), total=kospi_stocks.shape[0]):
            code, name = row['Code'], row['Name']

            # 특수문자 제거 (파일 이름 오류 방지)
            clean_name = "".join([c for c in name if c.isalnum() or c in (' ', '-')]).rstrip()

            success, reason = self.collect_and_save_stock_data(code, clean_name)

            if success:
                success_count += 1
            else:
                fail_count += 1
                # 실패 사유 로깅 (너무 많아지면 주석 처리 가능)
                # print(f" - 실패: {code} {clean_name} (사유: {reason})")

            # 서버 부하를 줄이기 위한 약간의 지연 시간
            time.sleep(0.1)

        print("\n" + "=" * 50)
        print("✅ 전체 데이터 수집 완료!")
        print(f" - 성공: {success_count}개 종목")
        print(f" - 실패/건너뜀: {fail_count}개 종목")
        print(f"데이터가 저장된 폴더: {self.data_dir}")
        print("=" * 50)


if __name__ == '__main__':
    collector = DataCollector()
    collector.run_collection()