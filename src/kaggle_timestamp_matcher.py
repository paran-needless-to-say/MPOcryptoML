"""
Kaggle Ethereum Fraud Detection + Etherscan API 결합

전략:
1. Kaggle 데이터에서 주소와 라벨 추출 (FLAG)
2. Etherscan API로 각 주소의 실제 거래 timestamp 수집
3. 라벨 + timestamp 결합하여 완전한 그래프 생성
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time
import requests
from graph import CryptoTransactionGraph


def load_kaggle_addresses_with_labels(csv_path: str) -> pd.DataFrame:
    """
    Kaggle 데이터에서 주소와 라벨 추출
    
    Args:
        csv_path: Kaggle transaction_dataset.csv 경로
    
    Returns:
        DataFrame with columns: [address, label]
    """
    df = pd.read_csv(csv_path)
    
    if 'Address' not in df.columns:
        raise ValueError("'Address' column not found")
    
    if 'FLAG' not in df.columns:
        raise ValueError("'FLAG' column not found")
    
    # 주소와 라벨만 추출
    result = df[['Address', 'FLAG']].copy()
    result.columns = ['address', 'label']
    
    print(f"✓ Loaded {len(result)} addresses with labels")
    print(f"  Anomalies: {sum(result['label'])}")
    
    return result


def get_timestamp_from_etherscan(address: str, api_key: str) -> List[Tuple[str, str, float, int]]:
    """
    Etherscan API V2에서 특정 주소의 거래를 가져와서 (from, to, value, timestamp) 리스트 반환
    
    Args:
        address: 이더리움 주소
        api_key: Etherscan API key
    
    Returns:
        List of (from, to, value, timestamp)
    """
    # Etherscan API V2 사용 (chainid 필수)
    # Ethereum mainnet = 1
    url = "https://api.etherscan.io/v2/api"
    transactions = []
    
    params = {
        'chainid': '1',  # Ethereum mainnet
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'page': 1,
        'offset': 100,  # 최대 100개
        'sort': 'desc',
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        print(f"  API Response for {address[:10]}...: {data.get('message', 'No message')}")
        
        if data.get('status') == '1' and data.get('result'):
            for tx in data['result']:
                from_addr = tx.get('from', '')
                to_addr = tx.get('to', '')
                value = float(tx.get('value', '0')) / 1e18  # Wei to ETH
                timestamp = int(tx.get('timeStamp', '0'))
                
                if from_addr and to_addr and value > 0:
                    transactions.append((from_addr, to_addr, value, timestamp))
            
            print(f"  ✓ Got {len(transactions)} valid transactions")
        
        time.sleep(0.2)  # Rate limit
    
    except Exception as e:
        print(f"  ⚠️ Error fetching {address[:10]}...: {e}")
    
    return transactions


def match_kaggle_with_etherscan(
    kaggle_df: pd.DataFrame,
    api_key: str,
    limit: int = 100
) -> CryptoTransactionGraph:
    """
    Kaggle 데이터의 주소들에 Etherscan timestamp 매칭
    
    ⚠️ 중요: 
    Kaggle은 집계 데이터이므로 timestamp가 없음!
    대신 Etherscan API로 주소의 실제 거래를 가져와서 
    그래프를 재구성합니다.
    
    전략:
    1. Kaggle: 주소 + 라벨 (FLAG)만 사용
    2. Etherscan: 실제 거래 데이터로 그래프 재구성
    3. 결합: 라벨 + 실제 거래 그래프
    
    Args:
        kaggle_df: DataFrame with [address, label]
        api_key: Etherscan API key
        limit: 처리할 주소 개수
    
    Returns:
        CryptoTransactionGraph with labels and timestamps
    """
    graph = CryptoTransactionGraph()
    
    addresses = kaggle_df['address'].head(limit).tolist()
    labels = dict(zip(kaggle_df['address'], kaggle_df['label']))
    
    print(f"\nProcessing {len(addresses)} addresses from Kaggle data...")
    print(f"This will take approximately {len(addresses) * 0.2:.1f} seconds due to API rate limits")
    
    all_transactions = []
    
    for i, address in enumerate(addresses):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(addresses)}")
        
        txs = get_timestamp_from_etherscan(address, api_key)
        
        if txs:
            all_transactions.extend(txs)
        
        # Rate limit
        time.sleep(0.2)
    
    print(f"\n✓ Collected {len(all_transactions)} transactions from Etherscan")
    
    # 중복 제거
    unique_txs = list(set(all_transactions))
    print(f"✓ Unique transactions: {len(unique_txs)}")
    
    # 그래프 생성
    for from_addr, to_addr, value, timestamp in unique_txs:
        graph.add_edge(from_addr, to_addr, value, timestamp)
    
    # 라벨 설정 (Kaggle의 FLAG 사용)
    # ⚠️ 중요: 실제 그래프에 있는 노드만 라벨 설정
    actual_labels = {}
    for node in graph.nodes:
        if node in labels:
            actual_labels[node] = labels[node]
        else:
            actual_labels[node] = 0  # 라벨 없는 노드는 정상으로 가정
    
    graph.set_labels(actual_labels)
    
    print(f"\n✓ Graph created:")
    print(f"  Nodes (V): {len(graph.nodes)}")
    print(f"  Edges (E): {len(graph.edges)}")
    print(f"  Labels: {sum(actual_labels.values())} anomalies (from {sum(labels.values())} in Kaggle)")
    
    return graph


def create_hybrid_graph(
    kaggle_csv_path: str = "./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv",
    api_key: str = None,
    n_addresses: int = 50
) -> CryptoTransactionGraph:
    """
    Kaggle + Etherscan 하이브리드 그래프 생성
    
    전략:
    1. Kaggle: 주소 + 라벨 가져오기
    2. Etherscan: 각 주소의 실제 거래 timestamp 가져오기
    3. 결합: 라벨 + timestamp 있는 완전한 그래프
    
    Args:
        kaggle_csv_path: Kaggle 데이터 경로
        api_key: Etherscan API key (None이면 더미 timestamp 사용)
        n_addresses: 처리할 주소 개수
    
    Returns:
        Complete graph with labels and timestamps
    """
    print("="*70)
    print("Kaggle + Etherscan Hybrid Graph Creation")
    print("="*70)
    
    # 1. Kaggle 데이터에서 주소와 라벨 추출
    print("\n[Step 1] Loading Kaggle data...")
    kaggle_df = load_kaggle_addresses_with_labels(kaggle_csv_path)
    
    if api_key and api_key != "YOUR_API_KEY_HERE":
        # 2. Etherscan에서 timestamp 수집
        print(f"\n[Step 2] Fetching timestamps from Etherscan for {n_addresses} addresses...")
        graph = match_kaggle_with_etherscan(kaggle_df, api_key, limit=n_addresses)
        
        print("\n✅ Hybrid graph created successfully!")
        print("   - Labels from Kaggle ✅")
        print("   - Timestamps from Etherscan ✅")
        
    else:
        # API Key 없으면 더미 timestamp 사용
        print("\n⚠️ No Etherscan API key provided")
        print("   Using dummy timestamps (for algorithm testing only)")
        
        graph = CryptoTransactionGraph()
        
        # 라벨만 설정
        for _, row in kaggle_df.head(n_addresses).iterrows():
            address = row['address']
            label = row['label']
            graph.node_labels[address] = label
        
        print(f"✓ Labels set for {len(graph.node_labels)} addresses")
        print("⚠️ No transactions - this is for label testing only")
    
    return graph


if __name__ == "__main__":
    # 테스트 실행
    print("Testing hybrid graph creation...")
    
    # 방법 1: API Key 없이 (더미)
    graph1 = create_hybrid_graph(api_key=None, n_addresses=20)
    
    # 방법 2: API Key 있으면 실제 데이터
    # API_KEY = "YourAPIKeyHere"
    # graph2 = create_hybrid_graph(api_key=API_KEY, n_addresses=50)
    
    print("\n" + "="*70)
    print("Test completed")
    print("="*70)

