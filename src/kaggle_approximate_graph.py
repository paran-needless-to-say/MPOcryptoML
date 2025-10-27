"""
Kaggle 집계 통계로부터 근사 그래프 생성

⚠️ 경고: 정확한 논문 재현은 아니지만
구조적으로 알고리즘을 테스트할 수 있습니다.

전략:
- Kaggle의 "Time Diff"를 사용
- in-degree와 out-degree에 분배
- 시뮬레이션 그래프 생성
"""
import pandas as pd
import numpy as np
from graph import CryptoTransactionGraph
import random
from datetime import datetime, timedelta


def kaggle_to_graph_with_simulation(csv_path: str, 
                                   n_addresses: int = 100,
                                   seed: int = 42):
    """
    Kaggle 집계 통계를 활용한 근사 그래프
    
    Args:
        csv_path: Kaggle CSV
        n_addresses: 사용할 주소 개수
        seed: 랜덤 시드
    
    Returns:
        CryptoTransactionGraph
    """
    print("="*70)
    print("Kaggle 집계 통계 → 근사 그래프")
    print("⚠️ 정확한 논문 재현이 아님 (시뮬레이션)")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    sample_df = df.head(n_addresses)
    
    graph = CryptoTransactionGraph()
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. 라벨 설정
    for _, row in sample_df.iterrows():
        address = row['Address']
        label = row['FLAG']
        graph.node_labels[address] = label
    
    print(f"\n✓ Loaded {len(graph.node_labels)} labels")
    print(f"  Anomalies: {sum(graph.node_labels.values())}")
    
    # 2. 근사 그래프 생성
    # 집계 정보 사용:
    # - Unique Sent To Addresses: 보낸 주소 개수
    # - Unique Received From Addresses: 받은 주소 개수
    # - Time Diff between first and last: 전체 기간
    
    # 시뮬레이션: 각 주소마다 거래 생성
    for _, row in sample_df.iterrows():
        address = row['Address']
        sent_to = int(row['Unique Sent To Addresses'])  # 보낸 주소 개수
        recv_from = int(row['Unique Received From Addresses'])  # 받은 주소 개수
        total_diff = row['Time Diff between first and last (Mins)']  # 전체 기간
        
        # 전체 기간의 시작/끝
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=total_diff)
        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()
        
        # In-degree 거래 생성 (받은 거래들)
        for i in range(min(recv_from, 100)):  # 너무 많으면 제한
            # 랜덤 주소에서 받음
            from_addr = random.choice(sample_df['Address'].tolist())
            timestamp = random.uniform(start_timestamp, end_timestamp)
            value = np.random.exponential(1.0)  # 균등 분포
            
            graph.add_edge(from_addr, address, value, timestamp)
        
        # Out-degree 거래 생성 (보낸 거래들)
        for i in range(min(sent_to, 100)):
            # 랜덤 주소에게 보냄
            to_addr = random.choice(sample_df['Address'].tolist())
            timestamp = random.uniform(start_timestamp, end_timestamp)
            value = np.random.exponential(1.0)
            
            graph.add_edge(address, to_addr, value, timestamp)
    
    print(f"\n✓ Simulated graph created:")
    print(f"  Nodes (V): {len(graph.nodes)}")
    print(f"  Edges (E): {len(graph.edges)}")
    print(f"  Labels: {sum(graph.node_labels.values())} anomalies")
    
    print("\n⚠️ 이것은 시뮬레이션입니다!")
    print("   정확한 논문 재현이 아닙니다.")
    
    return graph


if __name__ == "__main__":
    # 테스트
    graph = kaggle_to_graph_with_simulation(
        "./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv",
        n_addresses=50
    )
    
    print("\n" + "="*70)
    print("✅ 시뮬레이션 그래프 생성 완료")
    print("="*70)
    
    print("\n💡 이 그래프로도 알고리즘은 실행 가능합니다.")
    print("   단, 정확도는 떨어질 수 있습니다.")

