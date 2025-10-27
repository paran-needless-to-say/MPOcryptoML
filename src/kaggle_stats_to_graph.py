"""
Kaggle 집계 통계를 활용한 그래프 생성 (대안)

⚠️ 중요: 개별 거래가 아닌 집계 통계만 있으므로
정확한 timestamp 복원이 불가능합니다.

하지만 "구조적으로" 알고리즘을 테스트하는 것은 가능합니다.
"""
import pandas as pd
import numpy as np
from graph import CryptoTransactionGraph
from typing import List
import random


def kaggle_to_graph_approximate(csv_path: str, 
                                 n_addresses: int = 100,
                                 seed: int = 42) -> CryptoTransactionGraph:
    """
    Kaggle 집계 통계를 활용한 근사 그래프 생성
    
    ⚠️ 경고: Kaggle에는 개별 거래가 없으므로:
    - 정확한 timestamp는 생성 불가
    - 대략적인 그래프 구조만 생성
    - 알고리즘 테스트용으로만 사용
    
    Args:
        csv_path: Kaggle CSV 경로
        n_addresses: 사용할 주소 개수
        seed: 랜덤 시드
    
    Returns:
        CryptoTransactionGraph (timestamp는 근사치)
    """
    print("="*70)
    print("Kaggle 집계 통계 → 근사 그래프")
    print("⚠️ 경고: 정확한 timestamp 없음")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    # n_addresses개만 샘플링
    sample_df = df.head(n_addresses)
    
    graph = CryptoTransactionGraph()
    random.seed(seed)
    np.random.seed(seed)
    
    # 각 주소별로 라벨 설정
    for _, row in sample_df.iterrows():
        address = row['Address']
        label = row['FLAG']
        graph.node_labels[address] = label
    
    print(f"\n✓ Loaded {len(graph.node_labels)} addresses with labels")
    print(f"  Anomalies: {sum(graph.node_labels.values())}")
    
    # ⚠️ 문제: 개별 거래가 없으므로 그래프 구조를 만들 수 없음
    # Kaggle 데이터에는:
    # - Unique Sent To Addresses: 118
    # - Unique Received From Addresses: 40
    # 만 있음
    
    # 해결책: 집계 정보로부터 보간
    # (실제 거래가 아니므로 정확하지 않음)
    
    print("\n❌ 개별 거래 데이터가 없어 그래프 구조 생성 불가")
    print("   Kaggle은 집계 통계만 제공합니다.")
    print("\n✅ 하지만 라벨(FLAG)은 사용 가능합니다!")
    
    return graph


def use_kaggle_labels_only(csv_path: str, 
                          etherscan_graph: CryptoTransactionGraph,
                          n_labels: int = 100):
    """
    Kaggle 라벨만 사용 + Etherscan 그래프 결합
    
    실제 사용 권장 방법:
    1. Etherscan에서 실제 거래로 그래프 생성
    2. Kaggle에서 라벨(FLAG) 가져오기
    3. 결합
    
    Args:
        csv_path: Kaggle CSV
        etherscan_graph: Etherscan으로 생성한 그래프
        n_labels: 사용할 라벨 개수
    
    Returns:
        CryptoTransactionGraph with labels
    """
    print("="*70)
    print("Kaggle 라벨 + Etherscan 그래프 결합")
    print("✅ 최적의 방법")
    print("="*70)
    
    # Kaggle에서 라벨만 가져오기
    df = pd.read_csv(csv_path)
    sample_df = df.head(n_labels)
    
    # Etherscan 그래프에 라벨 추가
    kaggle_labels = dict(zip(sample_df['Address'], sample_df['FLAG']))
    
    # 그래프의 노드에 라벨 매칭
    matched_labels = {}
    for node in etherscan_graph.nodes:
        if node in kaggle_labels:
            matched_labels[node] = kaggle_labels[node]
        else:
            matched_labels[node] = 0  # 기본값
    
    etherscan_graph.set_labels(matched_labels)
    
    print(f"\n✓ Matched {len([l for l in matched_labels.values() if l == 1])} labels")
    print(f"  Total nodes: {len(etherscan_graph.nodes)}")
    
    return etherscan_graph


if __name__ == "__main__":
    print("\n[테스트 1] Kaggle 집계 통계로 그래프 생성 시도")
    print("-" * 70)
    
    try:
        # 시도
        graph1 = kaggle_to_graph_approximate(
            "./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv",
            n_addresses=20
        )
        print("\n⚠️ 그래프 구조는 생성되지 않음 (개별 거래 없음)")
        print("   라벨만 있음")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n\n[결론]")
    print("="*70)
    print("Kaggle 집계 통계만으로는 논문 구현 불가능합니다.")
    print("이유: 개별 거래의 timestamp가 필요함")
    print("\n✅ 최선의 방법: Etherscan API 사용")
    print("   - 실제 거래 + 정확한 timestamp")
    print("   - Kaggle: 라벨(FLAG)만 사용")
    print("="*70)

