import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import FinanceDataReader as fdr
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup
import time
import random


# --- Helper Class: FeatureCalculator ---
class FeatureCalculator:
    def __init__(self):
        self.feature_params = {
            "rsi": 14, "stoch_k": 14, "stoch_d": 3, "macd_fast": 12,
            "macd_slow": 26, "macd_signal": 9, "adx": 14, "sma_long": 200,
            "bb_len": 20, "bb_std": 2, "mfi": 14
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            import pandas_ta as ta
        except ImportError:
            print("pandas_ta 라이브러리가 필요합니다. 'pip install pandas_ta'를 실행해주세요.")
            return pd.DataFrame()

        df['RSI'] = ta.rsi(df['Close'], length=self.feature_params['rsi'])
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=self.feature_params['stoch_k'],
                         d=self.feature_params['stoch_d'])
        df['Stoch_D'] = stoch[
            f'STOCHd_{self.feature_params["stoch_k"]}_{self.feature_params["stoch_d"]}_{self.feature_params["stoch_d"]}']
        macd = ta.macd(df['Close'], fast=self.feature_params['macd_fast'], slow=self.feature_params['macd_slow'],
                       signal=self.feature_params['macd_signal'])
        df['MACD_Hist'] = macd[
            f'MACDh_{self.feature_params["macd_fast"]}_{self.feature_params["macd_slow"]}_{self.feature_params["macd_signal"]}']
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=self.feature_params['adx'])
        df['ADX'] = adx[f'ADX_{self.feature_params["adx"]}']
        df['SMA200'] = ta.sma(df['Close'], length=self.feature_params['sma_long'])
        df['Price_SMA200_Ratio'] = df['Close'] / df['SMA200']
        bbands = ta.bbands(df['Close'], length=self.feature_params['bb_len'], std=self.feature_params['bb_std'])
        df['BB_Percent'] = bbands[f'BBP_{self.feature_params["bb_len"]}_{self.feature_params["bb_std"]:.1f}']
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=self.feature_params['mfi'])
        df['OBV'] = ta.obv(df['Close'], df['Volume'])

        return df


# --- Helper Class: QuantMOThemeLookup (최종 수정 버전) ---
class QuantMOThemeLookup:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
        }

    def get_themes(self, stock_code, stock_name):
        print(f"\n'{stock_name}({stock_code})' 종목의 테마 정보를 조회합니다...")
        url = f"https://finance.finup.co.kr/Stock/{stock_code}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code != 200:
                print(f" -> 오류: 서버 응답 코드({response.status_code}). 접속이 차단되었을 수 있습니다.")
                return

            soup = BeautifulSoup(response.text, "html.parser")

            themes_li = soup.select("div.box_list > ul > li")

            if not themes_li:
                print(" -> 관련된 테마 정보가 없습니다.")
                return

            print("--- 테마 목록 ---")
            found_themes = 0
            for item in themes_li:
                # ❗ [수정된 부분] 클래스 이름 대신, span 태그의 순서로 이름과 등락률을 찾습니다.
                spans = item.select("span")

                # <li> 태그 안에 <span>이 2개 이상 있는지 확인
                if len(spans) >= 2:
                    theme_name = spans[0].text.strip()
                    theme_rate = spans[1].text.strip()
                    print(f" - {theme_name} (등락률: {theme_rate})")
                    found_themes += 1

            if found_themes == 0:
                print(" -> 테마 목록은 찾았으나, 세부 정보(이름, 등락률)를 가져오지 못했습니다.")
            print("-----------------")

        except Exception as e:
            print(f" -> 테마 정보 조회 중 예외가 발생했습니다: {e}")


# --- Main Class: QuantMLPredictor ---
class QuantMLPredictor:
    def __init__(self, model_path="./quantmo_data/xgb_model.json", scaler_path="./quantmo_data/scaler.joblib"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_columns = [
            'RSI', 'Stoch_D', 'MACD_Hist', 'ADX',
            'Price_SMA200_Ratio', 'BB_Percent', 'MFI', 'OBV'
        ]
        self.model, self.scaler, self.stock_list = None, None, None
        self.feature_calculator = FeatureCalculator()
        self.theme_lookup = QuantMOThemeLookup()

    def load_model_and_scaler(self):
        print("저장된 모델과 스케일러를 불러옵니다...")
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError("모델 또는 스케일러 파일이 없습니다. train_model.py를 먼저 실행하여 생성해주세요.")

        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        print("✅ 모델과 스케일러 로드 완료.")

    def predict_score(self, stock_code):
        """단일 종목의 스코어를 예측합니다. (클린 버전)"""
        try:
            df = fdr.DataReader(stock_code, start=pd.Timestamp.now() - pd.DateOffset(years=1))
            if len(df) < 200: return None, None

            df_featured = self.feature_calculator.calculate(df)
            latest_features_raw = df_featured[self.feature_columns].iloc[-2:-1]

            if latest_features_raw.isnull().values.any(): return None, None

            scaled_features = self.scaler.transform(latest_features_raw)
            pred_proba = self.model.predict_proba(scaled_features)[0]
            score = pred_proba[1] * 100

            return score, df.iloc[-1]['Close']
        except Exception:
            return None, None

    # 이하 run_daily_scoring, run_interactive_theme_lookup 함수는 동일합니다.
    # ... (생략) ...
    def run_daily_scoring(self):
        """코스피 전 종목에 대해 스코어링을 실행하고 랭킹을 보여줍니다."""
        self.load_model_and_scaler()

        self.stock_list = fdr.StockListing('KOSPI')[['Code', 'Name']]

        results = []

        target_stocks = self.stock_list

        print(f"\n총 {len(target_stocks)}개 KOSPI 종목에 대한 스코어링을 시작합니다...")
        for _, row in tqdm(target_stocks.iterrows(), total=len(target_stocks)):
            score, last_price = self.predict_score(row['Code'])
            if score is not None:
                results.append({
                    "코드": row['Code'], "종목명": row['Name'],
                    "스코어": score, "현재가": last_price
                })

        if not results:
            print("스코어링 결과가 없습니다.")
            return

        df_result = pd.DataFrame(results).sort_values(by="스코어", ascending=False).reset_index(drop=True)

        print("\n" + "=" * 60)
        print("📈 오늘의 주식 스코어 랭킹")
        pd.options.display.float_format = '{:.2f}'.format
        print(df_result.head(20))
        print("=" * 60)

    def run_interactive_theme_lookup(self):
        """사용자 입력을 받아 테마 정보를 조회합니다."""
        print("\n관심 있는 종목의 테마를 확인해보세요.")
        while True:
            stock_input = input("종목 코드 또는 종목명을 입력하세요 (종료: q 또는 exit): ")
            if stock_input.lower() in ['q', 'exit']:
                print("프로그램을 종료합니다.")
                break

            if self.stock_list is None:
                self.stock_list = fdr.StockListing('KOSPI')[['Code', 'Name']]

            if stock_input.isdigit() and len(stock_input) == 6:
                target_stock = self.stock_list[self.stock_list['Code'] == stock_input]
            else:
                target_stock = self.stock_list[self.stock_list['Name'] == stock_input]

            if not target_stock.empty:
                code = target_stock.iloc[0]['Code']
                name = target_stock.iloc[0]['Name']
                self.theme_lookup.get_themes(code, name)
            else:
                print(" -> 해당하는 종목을 찾을 수 없습니다. 정확한 종목 코드 또는 종목명을 입력해주세요.")


if __name__ == '__main__':
    predictor = QuantMLPredictor()
    predictor.run_daily_scoring()
    predictor.run_interactive_theme_lookup()