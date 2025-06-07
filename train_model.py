import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


class QuantMLModelTrainer:
    # ❗ [수정된 부분] data_path의 파일명을 최종 버전에 맞게 수정
    def __init__(self, data_path="./quantmo_data/ml_training_data_raw.parquet", model_save_path="./quantmo_data"):
        self.data_path = data_path
        self.model_save_path = model_save_path
        # ❗ [수정된 부분] 'OBV_Scaled'를 'OBV'로 변경하여 일관성 유지
        self.feature_columns = [
            'RSI', 'Stoch_D', 'MACD_Hist', 'ADX',
            'Price_SMA200_Ratio', 'BB_Percent', 'MFI', 'OBV'
        ]
        self.model = None
        self.scaler = None
        print("QuantMLModelTrainer가 초기화되었습니다.")

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} 파일을 찾을 수 없습니다.")
        df = pd.read_parquet(self.data_path)
        df = df.sort_index()
        print(f"데이터 로드 완료. 총 {len(df)}개의 데이터.")
        return df

    def split_and_scale_data(self, df):
        """데이터를 분할하고, 훈련 세트 기준으로 Scaler를 학습 및 적용합니다."""
        train_size = int(len(df) * 0.6)
        validation_size = int(len(df) * 0.2)
        train_df = df.iloc[:train_size]
        validation_df = df.iloc[train_size: train_size + validation_size]
        test_df = df.iloc[train_size + validation_size:]

        X_train_raw = train_df[self.feature_columns]
        y_train = train_df['Label']

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw)

        X_val = self.scaler.transform(validation_df[self.feature_columns])
        y_val = validation_df['Label']
        X_test = self.scaler.transform(test_df[self.feature_columns])
        y_test = test_df['Label']

        print("데이터 분할 및 스케일링 완료 (60:20:20):")
        print(f" - 훈련용: {len(X_train)}개, 검증용: {len(X_val)}개, 테스트용: {len(X_test)}개")
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_model(self, X_train, y_train, X_val, y_val):
        """XGBoost 모델을 훈련합니다."""
        print("\n모델 훈련을 시작합니다...")
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        self.model = xgb.XGBClassifier(
            objective='multi:softmax', num_class=3, n_estimators=1000,
            learning_rate=0.1, max_depth=5, use_label_encoder=False,
            eval_metric='mlogloss', random_state=42, tree_method='gpu_hist'
        )
        early_stopping_callback = xgb.callback.EarlyStopping(rounds=50, save_best=True)
        self.model.fit(X_train, y_train, sample_weight=sample_weights,
                       eval_set=[(X_val, y_val)], callbacks=[early_stopping_callback],
                       verbose=False)
        print("✅ 모델 훈련이 완료되었습니다.")

    def evaluate_model(self, X_test, y_test):
        """훈련된 모델을 테스트 세트로 평가합니다."""
        print("\n모델 성능 평가를 시작합니다...")
        y_pred = self.model.predict(X_test)

        precision_label_1 = precision_score(y_test, y_pred, labels=[1], average='micro')
        print("\n" + "=" * 50)
        print(f"🎯 핵심 성과: '매수 성공(Label 1)' 예측 정밀도 = {precision_label_1:.2%}")
        print("=" * 50)

        print("\n[Classification Report]")
        print(classification_report(y_test, y_pred, target_names=['보합(0)', '매수성공(1)', '매수실패(2)']))

        print("\n[Confusion Matrix]")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['보합(0)', '매수성공(1)', '매수실패(2)'],
                    yticklabels=['보합(0)', '매수성공(1)', '매수실패(2)'])
        plt.xlabel('Predicted Label');
        plt.ylabel('True Label');
        plt.title('Confusion Matrix')
        plt.show()

    def save_model_and_scaler(self):
        """훈련된 모델과 Scaler를 파일로 저장합니다."""
        if self.model is None or self.scaler is None:
            print("저장할 모델 또는 스케일러가 없습니다.")
            return

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        model_path = os.path.join(self.model_save_path, 'xgb_model.json')
        self.model.save_model(model_path)
        print(f"\n✅ 훈련된 모델이 저장되었습니다: {model_path}")

        scaler_path = os.path.join(self.model_save_path, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler가 저장되었습니다: {scaler_path}")

    def run_training_pipeline(self):
        """전체 훈련 파이프라인을 실행합니다."""
        try:
            df = self.load_data()
            X_train, y_train, X_val, y_val, X_test, y_test = self.split_and_scale_data(df)
            self.train_model(X_train, y_train, X_val, y_val)
            self.evaluate_model(X_test, y_test)
            self.save_model_and_scaler()
        except Exception as e:
            print(f"오류 발생: {e}")


if __name__ == '__main__':
    trainer = QuantMLModelTrainer()
    trainer.run_training_pipeline()