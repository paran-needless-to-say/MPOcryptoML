"""
MPOCryptoML: 그래프 구조 정의 및 더미 데이터 생성
논문의 G = (V, E, W, T) 구조를 구현
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import networkx as nx
from datetime import datetime, timedelta
import random


class CryptoTransactionGraph:
    """
    암호화폐 거래 그래프 구조
    G = (V, E, W, T) where:
    - V: nodes (addresses)
    - E: edges (transactions: sender → receiver)
    - W: weights (transaction amount/value)
    - T: timestamps (transaction time)
    """
    
    def __init__(self):
        self.nodes: List[str] = []
        self.edges: List[Tuple[str, str, float, float]] = []  # (from, to, value, timestamp)
        self.graph: Optional[nx.DiGraph] = None
        self.node_labels: Dict[str, int] = {}  # 0=정상, 1=사기
        self.node_features: Dict[str, Dict] = {}
    
    def add_edge(self, from_address: str, to_address: str, 
                 value: float, timestamp: float):
        """엣지 추가"""
        self.edges.append((from_address, to_address, value, timestamp))
        
        # 노드 추가
        if from_address not in self.nodes:
            self.nodes.append(from_address)
            if from_address not in self.node_labels:
                self.node_labels[from_address] = 0  # 기본값: 정상
            self.node_features[from_address] = {}
        
        if to_address not in self.nodes:
            self.nodes.append(to_address)
            if to_address not in self.node_labels:
                self.node_labels[to_address] = 0
            self.node_features[to_address] = {}
    
    def build_graph(self):
        """NetworkX 그래프 객체 생성"""
        self.graph = nx.DiGraph()
        
        for from_addr, to_addr, value, timestamp in self.edges:
            self.graph.add_edge(
                from_addr, 
                to_addr,
                weight=value,
                timestamp=timestamp
            )
        
        return self.graph
    
    def set_labels(self, labels: Dict[str, int]):
        """노드 라벨 설정 (0=정상, 1=사기)"""
        self.node_labels.update(labels)
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """인접 행렬 반환"""
        if self.graph is None:
            self.build_graph()
        
        # 노드 인덱스 매핑
        node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        n = len(self.nodes)
        adj_matrix = np.zeros((n, n))
        
        for from_addr, to_addr, value, _ in self.edges:
            i, j = node_to_idx[from_addr], node_to_idx[to_addr]
            adj_matrix[i][j] = value
        
        return adj_matrix
    
    def get_transactions_df(self) -> pd.DataFrame:
        """거래 데이터를 DataFrame으로 반환"""
        df = pd.DataFrame(
            self.edges,
            columns=['from_address', 'to_address', 'value', 'timestamp']
        )
        return df
    
    def get_node_info_df(self) -> pd.DataFrame:
        """노드 정보 DataFrame"""
        df = pd.DataFrame({
            'address': self.nodes,
            'label': [self.node_labels.get(node, 0) for node in self.nodes]
        })
        return df
    
    def save(self, filepath: str):
        """그래프를 JSON 파일로 저장"""
        import json
        data = {
            'nodes': self.nodes,
            'edges': [
                {
                    'from': from_addr,
                    'to': to_addr,
                    'value': value,
                    'timestamp': timestamp
                }
                for from_addr, to_addr, value, timestamp in self.edges
            ],
            'labels': self.node_labels
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def generate_dummy_data(
    n_nodes: int = 100,
    n_transactions: int = 500,
    anomaly_ratio: float = 0.1,
    seed: int = 42
) -> CryptoTransactionGraph:
    """
    더미 데이터 생성
    논문의 파이프라인 테스트를 위한 합성 그래프
    
    Args:
        n_nodes: 노드 개수
        n_transactions: 거래 개수
        anomaly_ratio: 사기 노드 비율
        seed: 랜덤 시드
    
    Returns:
        CryptoTransactionGraph 객체
    """
    np.random.seed(seed)
    random.seed(seed)
    
    graph = CryptoTransactionGraph()
    
    # 노드 생성
    nodes = [f"address_{i}" for i in range(n_nodes)]
    
    # 시작 시간 설정 (최근 30일)
    start_time = datetime.now() - timedelta(days=30)
    
    # 라벨 설정 (사기 노드 임의 선택)
    n_anomalies = max(1, int(n_nodes * anomaly_ratio))  # 최소 1개는 보장
    anomaly_nodes = random.sample(nodes, n_anomalies)
    labels = {node: 1 if node in anomaly_nodes else 0 for node in nodes}
    graph.set_labels(labels)
    
    # 거래 생성
    for _ in range(n_transactions):
        from_addr = random.choice(nodes)
        to_addr = random.choice(nodes)
        
        # 거래 금액 (ETH 단위)
        value = np.random.exponential(0.1)  # 지수 분포
        
        # 타임스탬프 생성 (지난 30일 동안 균등 분포)
        days_ago = random.uniform(0, 30)
        timestamp = (start_time + timedelta(days=days_ago)).timestamp()
        
        graph.add_edge(from_addr, to_addr, value, timestamp)
    
    # 사기 노드는 특정 패턴의 거래 추가
    for anomaly_node in anomaly_nodes:
        # fan-in 패턴 (많은 사람이 한 주소로 송금)
        if random.random() < 0.3:
            for _ in range(random.randint(5, 15)):
                from_addr = random.choice(nodes)
                value = np.random.exponential(0.05)
                days_ago = random.uniform(0, 7)
                timestamp = (start_time + timedelta(days=days_ago)).timestamp()
                graph.add_edge(from_addr, anomaly_node, value, timestamp)
        
        # fan-out 패턴 (한 주소에서 많은 사람에게 송금)
        if random.random() < 0.3:
            for _ in range(random.randint(5, 15)):
                to_addr = random.choice(nodes)
                value = np.random.exponential(0.05)
                days_ago = random.uniform(0, 7)
                timestamp = (start_time + timedelta(days=days_ago)).timestamp()
                graph.add_edge(anomaly_node, to_addr, value, timestamp)
    
    return graph


if __name__ == "__main__":
    # 테스트 코드
    print("Creating dummy graph...")
    graph = generate_dummy_data(n_nodes=50, n_transactions=200)
    
    print(f"\nGraph Statistics:")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print(f"Anomalies: {sum(graph.node_labels.values())}")
    
    # DataFrame으로 변환
    tx_df = graph.get_transactions_df()
    node_df = graph.get_node_info_df()
    
    print(f"\nTransaction DataFrame:\n{tx_df.head()}")
    print(f"\nNode DataFrame:\n{node_df.head()}")
