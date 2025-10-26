"""
MPOCryptoML 데이터 전처리 스크립트
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class DataPreprocessor:
    """데이터 전처리를 위한 클래스"""
    
    def __init__(self, data_dir="data", processed_dir="processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.scaler = StandardScaler()
        
        # 처리된 데이터 저장 디렉토리 생성
        self.processed_dir.mkdir(exist_ok=True)
    
    def load_data(self, file_path):
        """데이터 로드"""
        print(f"Loading data from {file_path}...")
        data = pd.read_csv(file_path)
        print(f"Data shape: {data.shape}")
        return data
    
    def explore_data(self, data):
        """데이터 탐색 (기본 통계)"""
        print("\n=== Data Info ===")
        print(data.info())
        print("\n=== Missing Values ===")
        print(data.isnull().sum())
        print("\n=== Basic Statistics ===")
        print(data.describe())
        return data
    
    def preprocess(self, data, target_column=None):
        """
        전처리 함수
        TODO: 논문에 맞게 전처리 로직 구현 필요
        """
        print("\nStarting preprocessing...")
        
        # 예시: 결측치 처리
        if data.isnull().sum().sum() > 0:
            print("Handling missing values...")
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        # 예시: 범주형 변수 처리
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"Encoding categorical variables: {list(categorical_cols)}")
            data = pd.get_dummies(data, columns=categorical_cols)
        
        # 타겟 변수 분리 (있다면)
        if target_column and target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            print(f"Target column '{target_column}' separated")
            return X, y
        
        return data
    
    def save_processed_data(self, data, filename="processed_data.csv"):
        """전처리된 데이터 저장"""
        save_path = self.processed_dir / filename
        data.to_csv(save_path, index=False)
        print(f"Processed data saved to {save_path}")
        return save_path


if __name__ == "__main__":
    # 전처리 인스턴스 생성
    preprocessor = DataPreprocessor()
    
    # 여기에 데이터 로드 및 전처리 코드 추가
    # 예시:
    # train_data = preprocessor.load_data("data/train.csv")
    # preprocessor.explore_data(train_data)
    # processed_data = preprocessor.preprocess(train_data)
    # preprocessor.save_processed_data(processed_data)
    
    print("Preprocessing pipeline ready!")
    print("Please customize the preprocessing steps according to the paper.")
