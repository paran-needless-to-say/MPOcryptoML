"""
Etherscan API 데이터를 그래프로 변환하는 모듈

이더스캔에서 받은 raw transaction 데이터를
논문의 G=(V, E, W, T) 그래프 구조로 변환
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import requests
import time
from graph import CryptoTransactionGraph


def parse_etherscan_txlist(address: str, api_key: str, limit: int = 10000):
    """
    Etherscan API의 txlist를 파싱하여 거래 리스트 반환
    
    Args:
        address: 이더리움 주소
        api_key: Etherscan API key
        limit: 최대 거래 개수
    
    Returns:
        List of transactions: (from, to, value, timestamp)
    """
    url = "https://api.etherscan.io/api"
    
    transactions = []
    
    page = 1
    offset = 1000  # Etherscan API limit per page
    
    while len(transactions) < limit:
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'page': page,
            'offset': offset,
            'sort': 'asc',  # 가장 오래된 거래부터
            'apikey': api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if data['status'] == '0':
                break  # 더 이상 데이터 없음
            
            txs = data['result']
            
            for tx in txs:
                # 필요한 필드만 추출
                from_addr = tx.get('from', '')
                to_addr = tx.get('to', '')
                value = float(tx.get('value', '0')) / 1e18  # Wei to ETH
                timestamp = int(tx.get('timeStamp', '0'))
                
                if from_addr and to_addr and value > 0:
                    transactions.append((from_addr, to_addr, value, timestamp))
            
            if len(txs) < offset:
                break  # 마지막 페이지
            
            page += 1
            time.sleep(0.2)  # API rate limit 방지
            
        except Exception as e:
            print(f"API 오류: {e}")
            break
    
    return transactions[:limit]


def convert_to_graph(transactions: List[Tuple], 
                     labels: Dict[str, int] = None) -> CryptoTransactionGraph:
    """
    거래 리스트를 그래프로 변환
    
    이 함수가 핵심입니다:
    - Address → Node (V)
    - Transaction → Edge (E) 
    - Value → Weight (W)
    - Timestamp → Time (T)
    
    Args:
        transactions: [(from, to, value, timestamp), ...]
        labels: {address: label} 딕셔너리 (optional)
    
    Returns:
        CryptoTransactionGraph 객체
    """
    graph = CryptoTransactionGraph()
    
    # 1. Transactions → Edges 변환
    for from_addr, to_addr, value, timestamp in transactions:
        # Address를 Node로 변환 (자동으로 add_edge에서 처리됨)
        graph.add_edge(from_addr, to_addr, value, timestamp)
    
    # 2. Labels 설정 (있는 경우)
    if labels:
        graph.set_labels(labels)
    
    print(f"✓ 그래프 변환 완료:")
    print(f"  Nodes (V): {len(graph.nodes)}")
    print(f"  Edges (E): {len(graph.edges)}")
    print(f"  Labels: {sum(graph.node_labels.values())} anomalies")
    
    return graph


def fetch_transactions_from_etherscan(addresses: List[str], api_key: str):
    """
    여러 주소의 거래를 수집하여 그래프 생성
    
    Args:
        addresses: 이더리움 주소 리스트
        api_key: Etherscan API key
    
    Returns:
        CryptoTransactionGraph 객체
    """
    all_transactions = []
    
    for address in addresses:
        print(f"Fetching transactions for {address[:10]}...")
        txs = parse_etherscan_txlist(address, api_key, limit=5000)
        all_transactions.extend(txs)
        print(f"  ✓ Got {len(txs)} transactions")
    
    # 중복 제거 (같은 거래가 여러 주소에서 나타날 수 있음)
    unique_txs = list(set(all_transactions))
    
    print(f"\n✓ Total unique transactions: {len(unique_txs)}")
    
    # 그래프로 변환
    graph = convert_to_graph(unique_txs)
    
    return graph


if __name__ == "__main__":
    # 테스트: 더미 주소로 변환 테스트
    print("="*60)
    print("Etherscan Data → Graph Conversion Test")
    print("="*60)
    
    # 1. 더미 거래 데이터 생성
    test_transactions = [
        ("0x1111111111111111111111111111111111111111", 
         "0x2222222222222222222222222222222222222222", 
         0.5, 1234567890.0),
        ("0x2222222222222222222222222222222222222222",
         "0x3333333333333333333333333333333333333333",
         1.0, 1234567900.0),
        ("0x3333333333333333333333333333333333333333",
         "0x4444444444444444444444444444444444444444",
         0.3, 1234568000.0),
    ]
    
    # 2. Labels (optional)
    test_labels = {
        "0x1111111111111111111111111111111111111111": 0,
        "0x2222222222222222222222222222222222222222": 1,  # anomaly
        "0x3333333333333333333333333333333333333333": 0,
        "0x4444444444444444444444444444444444444444": 0,
    }
    
    # 3. 그래프로 변환
    print("\n[Step 1] Converting transactions to graph...")
    graph = convert_to_graph(test_transactions, labels=test_labels)
    
    # 4. 결과 확인
    print("\n[Step 2] Graph structure:")
    print(f"  Nodes: {graph.nodes}")
    print(f"  Edges: {graph.edges}")
    print(f"  Labels: {graph.node_labels}")
    
    # 5. NetworkX 그래프로 변환
    print("\n[Step 3] Building NetworkX graph...")
    nx_graph = graph.build_graph()
    print(f"  ✓ NetworkX graph created")
    
    print("\n" + "="*60)
    print("✅ Conversion test successful!")
    print("="*60)

