"""
실제 암호화폐 거래 데이터 로딩 모듈

1. Kaggle Ethereum Fraud Detection 데이터셋
2. Elliptic++ Bitcoin 데이터셋  
3. Etherscan API로 수집한 실제 거래 데이터
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
from graph import CryptoTransactionGraph


def load_ethereum_fraud_detection(data_dir: str = "./notebooks/ethereum-frauddetection-dataset") -> Optional[CryptoTransactionGraph]:
    """
    Kaggle Ethereum Fraud Detection 데이터셋 로드
    
    Args:
        data_dir: 데이터셋 디렉토리 경로
    
    Returns:
        CryptoTransactionGraph 객체 (timestamp 없음)
    """
    try:
        data_path = Path(data_dir) / "transaction_dataset.csv"
        
        if not data_path.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {data_path}")
            print("다음 명령으로 데이터를 다운로드하세요:")
            print("  python notebooks/01_exploration.ipynb")
            return None
        
        print(f"✓ 데이터 로딩 중: {data_path}")
        df = pd.read_csv(data_path)
        
        print(f"데이터 셰이프: {df.shape}")
        print(f"컬럼: {list(df.columns)[:10]}...")
        
        # 필요한 컬럼 매핑
        if 'Address' in df.columns:
            address_col = 'Address'
        else:
            print("❌ 'Address' 컬럼을 찾을 수 없습니다")
            return None
        
        if 'FLAG' in df.columns:
            flag_col = 'FLAG'
        else:
            print("⚠️ 'FLAG' 컬럼이 없습니다. 모든 노드를 정상(0)으로 설정합니다")
            flag_col = None
        
        # 그래프 생성
        graph = CryptoTransactionGraph()
        
        # 주소별로 라벨 설정
        for idx, row in df.iterrows():
            address = str(row[address_col])
            label = int(row[flag_col]) if flag_col else 0
            graph.node_labels[address] = label
        
        print(f"✓ {len(graph.node_labels)}개의 노드 로드됨")
        
        # ⚠️ 주의: 이 데이터셋에는 timestamp 정보가 없습니다
        print("⚠️ 이 데이터셋에는 timestamp 정보가 없습니다")
        print("   더미 timestamp를 생성하거나 Etherscan API를 사용해야 합니다")
        
        return graph
        
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None


def load_from_csv(csv_path: str, 
                   from_col: str = 'from_address',
                   to_col: str = 'to_address',
                   value_col: str = 'value',
                   timestamp_col: str = 'timestamp',
                   label_col: Optional[str] = None) -> Optional[CryptoTransactionGraph]:
    """
    CSV 파일에서 실제 거래 데이터 로드
    
    Args:
        csv_path: CSV 파일 경로
        from_col: 송신 주소 컬럼명
        to_col: 수신 주소 컬럼명
        value_col: 거래 금액 컬럼명
        timestamp_col: 타임스탬프 컬럼명
        label_col: 라벨 컬럼명 (optional)
    
    Returns:
        CryptoTransactionGraph 객체
    """
    try:
        print(f"✓ CSV 로딩 중: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"데이터 셰이프: {df.shape}")
        
        graph = CryptoTransactionGraph()
        
        # 엣지 추가
        for idx, row in df.iterrows():
            from_addr = str(row[from_col])
            to_addr = str(row[to_col])
            value = float(row[value_col])
            timestamp = float(row[timestamp_col])
            
            graph.add_edge(from_addr, to_addr, value, timestamp)
        
        # 라벨 설정 (있는 경우)
        if label_col and label_col in df.columns:
            for idx, row in df.iterrows():
                address = str(row[from_col])
                label = int(row[label_col])
                graph.node_labels[address] = label
        
        print(f"✓ {len(graph.nodes)} 노드, {len(graph.edges)} 엣지 로드됨")
        
        return graph
        
    except Exception as e:
        print(f"❌ CSV 로딩 실패: {e}")
        return None


def load_with_timestamp_simulation(graph_obj: CryptoTransactionGraph, 
                                   days_back: int = 30) -> CryptoTransactionGraph:
    """
    타임스탬프가 없는 데이터에 가짜 타임스탬프 추가
    
    Args:
        graph_obj: 그래프 객체 (엣지에 timestamp가 없음)
        days_back: 과거 며칠까지
    
    Returns:
        timestamp가 추가된 그래프
    """
    import random
    from datetime import datetime, timedelta
    
    # 기존 엣지를 timestamp 없이 생성했으므로, 새로 timestamp 추가해야 함
    start_time = datetime.now() - timedelta(days=days_back)
    
    edges_with_ts = []
    for from_addr, to_addr, value, _ in graph_obj.edges:
        days_ago = random.uniform(0, days_back)
        timestamp = (start_time + timedelta(days=days_ago)).timestamp()
        edges_with_ts.append((from_addr, to_addr, value, timestamp))
    
    graph_obj.edges = edges_with_ts
    
    print(f"✓ Simulated timestamp added for {len(edges_with_ts)} edges")
    
    return graph_obj


if __name__ == "__main__":
    print("="*60)
    print("실제 데이터 로딩 테스트")
    print("="*60)
    
    # 방법 1: Kaggle Ethereum Fraud Detection (timestamp 없음)
    print("\n[방법 1] Kaggle Ethereum Fraud Detection...")
    graph1 = load_ethereum_fraud_detection()
    
    if graph1:
        print(f"  ✓ {len(graph1.node_labels)} nodes loaded")
        print(f"  ❌ Timestamp 없음 - 시뮬레이션 필요")
    
    # 방법 2: CSV 파일 (timestamp 포함)
    print("\n[방법 2] CSV 파일 로딩 예제...")
    print("  사용법: load_from_csv('your_data.csv', ...)")
    
    print("\n" + "="*60)
    print("더미 데이터로 테스트를 계속 진행합니다")
    print("="*60)
