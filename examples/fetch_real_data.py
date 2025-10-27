"""
실제 이더스캔 데이터를 불러와서 그래프 생성하는 예제

이더스캔 API 사용법:
1. https://etherscan.io/apis 에서 API Key 발급
2. 환경 변수 설정 또는 직접 입력
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from etherscan_parser import fetch_transactions_from_etherscan, convert_to_graph, parse_etherscan_txlist
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer
from anomaly_detector import MPOCryptoMLDetector
import pandas as pd


def main():
    print("="*70)
    print("Etherscan API로 실제 데이터 수집 및 그래프 생성")
    print("="*70)
    
    # ⚠️ 실제 API Key는 여기에 입력하거나 환경 변수 사용
    API_KEY = "YOUR_API_KEY_HERE"  # Etherscan API key
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️ API Key를 설정해주세요!")
        print("1. https://etherscan.io/apis 에서 API Key 발급")
        print("2. API_KEY 변수를 수정하거나 환경 변수 사용")
        print("\n테스트를 위해 더미 데이터를 사용합니다...")
        
        # 더미 예제
        from graph import generate_dummy_data
        graph_obj = generate_dummy_data(n_nodes=50, n_transactions=200)
        return graph_obj
    
    # 실제 주소 예제 (Wormhole exploit 등 유명 사기 케이스)
    suspicious_addresses = [
        # 여기에 실제 주소들을 입력
        "0x1234567890abcdef1234567890abcdef12345678",
    ]
    
    print(f"\n[Step 1] Fetching transactions from Etherscan...")
    print(f"  Addresses: {len(suspicious_addresses)}")
    
    try:
        # 실제 데이터 수집
        graph_obj = fetch_transactions_from_etherscan(
            addresses=suspicious_addresses, 
            api_key=API_KEY
        )
        
        print(f"\n[Step 2] Graph created:")
        print(f"  Nodes: {len(graph_obj.nodes)}")
        print(f"  Edges: {len(graph_obj.edges)}")
        
        # PPR 계산
        print(f"\n[Step 3] Computing PPR...")
        graph = graph_obj.build_graph()
        ppr = PersonalizedPageRank(graph, alpha=0.85, epsilon=0.01, p_f=0.1)
        
        source_nodes = ppr.get_source_nodes()
        print(f"  Found {len(source_nodes)} source nodes (in-degree=0)")
        
        # 샘플 노드에 대해 PPR 계산
        sample_nodes = list(source_nodes)[:min(20, len(source_nodes))]
        
        ppr_results = {}
        for node in sample_nodes:
            _, svn = ppr.compute_single_source_ppr(node)
            ppr_results[node] = svn
        
        print(f"  ✓ PPR computed for {len(ppr_results)} nodes")
        
        return graph_obj
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("더미 데이터로 테스트를 계속합니다...")
        
        from graph import generate_dummy_data
        graph_obj = generate_dummy_data(n_nodes=50, n_transactions=200)
        return graph_obj


def fetch_and_analyze(address: str, api_key: str):
    """
    특정 주소의 거래를 분석
    
    Args:
        address: 이더리움 주소
        api_key: Etherscan API key
    """
    print(f"\nAnalyzing address: {address}")
    
    # 1. 거래 수집
    transactions = parse_etherscan_txlist(address, api_key, limit=1000)
    print(f"  ✓ Collected {len(transactions)} transactions")
    
    if len(transactions) == 0:
        print("  ⚠️ No transactions found")
        return None
    
    # 2. 그래프로 변환
    graph_obj = convert_to_graph(transactions)
    
    # 3. 분석
    print(f"\n  Graph Analysis:")
    print(f"    Nodes (V): {len(graph_obj.nodes)}")
    print(f"    Edges (E): {len(graph_obj.edges)}")
    
    # 4. PPR 실행
    graph = graph_obj.build_graph()
    ppr = PersonalizedPageRank(graph, alpha=0.85, epsilon=0.01, p_f=0.1)
    
    source_nodes = ppr.get_source_nodes()
    print(f"    Source nodes (in-degree=0): {len(source_nodes)}")
    
    return graph_obj


if __name__ == "__main__":
    # 예제 실행
    graph = main()
    
    print("\n" + "="*70)
    print("✅ Test completed")
    print("="*70)
    
    # 실제 주소 분석 예제 (API key 필요)
    print("\n\n실제 주소 분석 예제:")
    print("fetch_and_analyze('0xAddress', 'YourAPIKey')")

