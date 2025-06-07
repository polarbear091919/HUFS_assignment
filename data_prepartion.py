# data_preparation.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import parmap  # ❗ [추가된 부분] 멀티프로세싱 라이브러리

try:
    import pandas_ta as ta
except ImportError:
    print("pandas_ta 라이브러리가 필요합니다. 'pip install pandas_ta'를 실행해주세요.")
    exit()


# ❗ [추가된 부분] 단일 파일을 처리하는 함수를 클래스 외부로 분리
def process_single_file(file_path):
    """단일 주식 데이터 파일을 읽어 Feature와 Label을 계산합니다."""
    try:
        # 이 함수 안에서 사용할 Feature 파라미터들을 다시 정의해줍니다.
        feature_params = {
            "rsi": 14, "stoch_k": 14, "stoch_d": 3, "macd_fast": 12, "macd_slow": 26,
            "macd_signal": 9, "adx": 14, "sma_long": 200, "bb_len": 20, "bb_std": 2, "mfi": 14
        }
        holding_period, upper_pct, lower_pct = 15, 0.05, -0.05

        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

        # Feature 계산
        df['RSI'] = ta.rsi(df['Close'], length=feature_params['rsi'])
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=feature_params['stoch_k'], d=feature_params['stoch_d'])
        df['Stoch_D'] = stoch[
            f'STOCHd_{feature_params["stoch_k"]}_{feature_params["stoch_d"]}_{feature_params["stoch_d"]}']
        macd = ta.macd(df['Close'], fast=feature_params['macd_fast'], slow=feature_params['macd_slow'],
                       signal=feature_params['macd_signal'])
        df['MACD_Hist'] = macd[
            f'MACDh_{feature_params["macd_fast"]}_{feature_params["macd_slow"]}_{feature_params["macd_signal"]}']
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=feature_params['adx'])
        df['ADX'] = adx[f'ADX_{feature_params["adx"]}']
        df['SMA200'] = ta.sma(df['Close'], length=feature_params['sma_long'])
        df['Price_SMA200_Ratio'] = df['Close'] / df['SMA200']
        bbands = ta.bbands(df['Close'], length=feature_params['bb_len'], std=feature_params['bb_std'])
        df['BB_Percent'] = bbands[f'BBP_{feature_params["bb_len"]}_{feature_params["bb_std"]:.1f}']
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=feature_params['mfi'])
        df['OBV'] = ta.obv(df['Close'], df['Volume'])

        # Label 생성
        df['Label'] = 0
        for i in range(len(df) - holding_period):
            entry_price = df['Close'].iloc[i]
            upper_barrier, lower_barrier = entry_price * (1 + upper_pct), entry_price * (1 + lower_pct)
            for j in range(1, holding_period + 1):
                future_high, future_low = df['High'].iloc[i + j], df['Low'].iloc[i + j]
                if future_high >= upper_barrier:
                    df.loc[df.index[i], 'Label'] = 1;
                    break
                if future_low <= lower_barrier:
                    df.loc[df.index[i], 'Label'] = 2;
                    break

        return df

    except Exception as e:
        # print(f"오류 발생 ({os.path.basename(file_path)}): {e}")
        return None


class QuantMLDataProcessor:
    def __init__(self, data_dir="./quantmo_data"):
        self.data_dir = data_dir
        self.raw_data_path = os.path.join(data_dir, "stock_data")
        self.output_path = data_dir
        print("QuantMLDataProcessor가 초기화되었습니다.")

    def run_processing(self):
        stock_files = [os.path.join(self.raw_data_path, f) for f in os.listdir(self.raw_data_path) if
                       f.endswith('.csv')]

        if not stock_files:
            print(f"{self.raw_data_path}에 처리할 파일이 없습니다.")
            return

        print(f"총 {len(stock_files)}개 종목에 대한 데이터 전처리를 시작합니다... (멀티프로세싱 사용)")

        # ❗ [수정된 부분] parmap을 사용하여 병렬 처리
        # tqdm 대신 parmap의 pm_pbar=True 옵션으로 진행 상황 표시
        results = parmap.map(process_single_file, stock_files, pm_pbar=True)

        # None 값을 제거하고 유효한 DataFrame만 필터링
        all_processed_data = [res for res in results if res is not None]

        if not all_processed_data:
            print("처리된 데이터가 없습니다.")
            return

        print("\n모든 종목 처리가 완료되었습니다. 데이터 통합 및 저장을 시작합니다...")
        final_dataset = pd.concat(all_processed_data).dropna()

        output_file = os.path.join(self.output_path, "ml_training_data_raw.parquet")
        final_dataset.to_parquet(output_file)

        print(f"\n✅ 전처리 완료! Raw 데이터 저장: {output_file}")
        print(f" - 총 데이터 행 수: {len(final_dataset)}")


if __name__ == '__main__':
    processor = QuantMLDataProcessor()
    processor.run_processing()