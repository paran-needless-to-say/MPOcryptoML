"""
Kaggle 집계 데이터를 그래프로 변환

전략:
- Kaggle의 집계 통계를 사용하여 시뮬레이션 그래프 생성
- "Time Diff between first and last"를 전체 기간으로 사용
- "Unique Sent To Addresses", "Unique Received From Addresses" 사용
- 개별 거래는 없지만 구조적으로 유사한 그래프 생성
"""
import pandas as pd
import numpy as np
from graph import CryptoTransactionGraph
import random
from datetime import datetime, timedelta


def kaggle_to_graph_realistic(csv_path: str, n_addresses: int = 50, seed: int = 42):
    """
    Kaggle 집계 데이터를 논문의 그래프 구조로 변환
    
    ⚠️ 중요: Kaggle에는 개별 거래가 없으므로 시뮬레이션
    
    전략:
    1. Kaggle의 "Time Diff"를 전체 기간으로 사용
    2. "Unique Sent To" 주소 개수만큼 out-edge 생성
    3. "Unique Received From" 주소 개수만큼 in-edge 생성
    4. 거래를 시간 범위에 균등하게 분포
    
    Args:
        csv_path: Kaggle CSV 경로
        n_addresses: 사용할 주소 개수
        seed: 랜덤 시드
    
    Returns:
        CryptoTransactionGraph
    """
    print("="*70)
    print("Kaggle 집계 데이터 → 그래프 변환 (시뮬레이션)")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    # Anomaly 주소를 포함하도록 샘플링
    anomaly_df = df[df['FLAG'] == 1]
    normal_df = df[df['FLAG'] == 0]
    
    n_anomalies = min(20, len(anomaly_df), n_addresses // 5)
    n_normal = n_addresses - n_anomalies
    
    sample_anomalies = anomaly_df.sample(n=n_anomalies, random_state=seed)
    sample_normal = normal_df.sample(n=n_normal, random_state=seed)
    
    sample_df = pd.concat([sample_anomalies, sample_normal]).sample(frac=1, random_state=seed)
    sample_df = sample_df.head(n_addresses)
    
    graph = CryptoTransactionGraph()
    random.seed(seed)
    np.random.seed(seed)
    
    # Kaggle 주소 리스트
    kaggle_addresses = sample_df['Address'].tolist()
    
    print(f"\n✓ Using {len(kaggle_addresses)} Kaggle addresses")
    print(f"  Anomalies: {sum(sample_df['FLAG'])}")
    
    # 1. 라벨 설정
    for _, row in sample_df.iterrows():
        address = row['Address']
        label = row['FLAG']
        graph.node_labels[address] = label
    
    print(f"  Labels: {sum(graph.node_labels.values())} anomalies")
    
    # 2. 각 주소마다 집계 정보를 사용하여 그래프 생성
    print("\nGenerating graph from aggregate statistics...")
    
    for _, row in sample_df.iterrows():
        address = row['Address']
        
        # 집계 정보
        sent_to_count = int(row['Unique Sent To Addresses'])  # 보낸 주소 개수
        recv_from_count = int(row['Unique Received From Addresses'])  # 받은 주소 개수
        total_diff_mins = row['Time Diff between first and last (Mins)']  # 전체 기간
        
        # 시작/끝 시간
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=total_diff_mins)
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()
        
        # In-degree edges 생성 (받은 거래들)
        # "Unique Received From Addresses" 수만큼
        for i in range(min(recv_from_count, len(kaggle_addresses))):
            # 랜덤 주소에서 받음
            from_addr = random.choice(kaggle_addresses)
            if from_addr == address:
                continue
            
            # 거래 시간을 전체 기간에 균등 분포
            timestamp = random.uniform(start_ts, end_ts)
            
            # 거래 금액 (Kaggle의 평균 값 사용)
            value = row['avg val received'] if not np.isnan(row['avg val received']) else 1.0
            
            graph.add_edge(from_addr, address, value, timestamp)
        
        # Out-degree edges 생성 (보낸 거래들)
        # "Unique Sent To Addresses" 수만큼
        for i in range(min(sent_to_count, len(kaggle_addresses))):
            # 랜덤 주소에게 보냄
            to_addr = random.choice(kaggle_addresses)
            if to_addr == address:
                continue
            
            timestamp = random.uniform(start_ts, end_ts)
            
            value = row['avg val sent'] if not np.isnan(row['avg val sent']) else 1.0
            
            graph.add_edge(address, to_addr, value, timestamp)
    
    print(f"\n✓ Graph created:")
    print(f"  Nodes (V): {len(graph.nodes)}")
    print(f"  Edges (E): {len(graph.edges)}")
    print(f"  Labels: {sum(graph.node_labels.values())} anomalies")
    
    print("\n⚠️ 이것은 시뮬레이션입니다!")
    print("   Kaggle 집계 정보를 사용하여 구조를 추정한 것입니다.")
    print("   정확한 논문 재현이 아닙니다.")
    
    return graph


if __name__ == "__main__":
    # 테스트
    graph = kaggle_to_graph_realistic(
        "./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv",
        n_addresses=100
    )
    
    print("\n" + "="*70)
    print("✅ Kaggle 집계 데이터 → 그래프 변환 완료")
    print("="*70)
    
    print("\n이 그래프로 알고리즘을 실행할 수 있습니다:")
    print("  python src/main.py")
    print("  python examples/quick_start.py")

