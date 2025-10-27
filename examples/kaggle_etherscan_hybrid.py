"""
Kaggle + Etherscan 하이브리드 데이터 예제

Kaggle: 라벨 (FLAG)
Etherscan: Timestamp

→ 완전한 그래프 생성
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from kaggle_timestamp_matcher import create_hybrid_graph
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer
from anomaly_detector import MPOCryptoMLDetector
from main import run_mpocrypto_ml_pipeline
import pandas as pd


def main():
    print("="*70)
    print("Kaggle + Etherscan Hybrid Pipeline")
    print("="*70)
    
    # ⚠️ 실제 API Key가 필요합니다
    API_KEY = "YOUR_API_KEY_HERE"  # Etherscan API key
    
    # Step 1: 하이브리드 그래프 생성
    print("\n[Step 1] Creating hybrid graph (Kaggle labels + Etherscan timestamps)...")
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️ API Key를 설정해주세요!")
        print("   1. https://etherscan.io/apis 에서 API Key 발급")
        print("   2. 아래 코드에서 API_KEY 변수 수정")
        print("   3. 또는 더미 데이터로 테스트 계속")
        
        # API Key 없으면 더미 그래프 사용
        from graph import generate_dummy_data
        graph_obj = generate_dummy_data(n_nodes=50, n_transactions=200)
        print("\n  → Using dummy data for algorithm testing")
        
    else:
        # 실제 하이브리드 그래프 생성
        # n_addresses: 처리할 Kaggle 주소 개수 (너무 많으면 오래 걸림)
        graph_obj = create_hybrid_graph(
            kaggle_csv_path="./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv",
            api_key=API_KEY,
            n_addresses=50  # 50개 주소만 처리 (테스트용)
        )
        print("\n  ✅ Hybrid graph created")
    
    # Step 2: 전체 파이프라인 실행
    print("\n[Step 2] Running full pipeline...")
    
    graph = graph_obj.build_graph()
    
    # PPR
    ppr = PersonalizedPageRank(graph, alpha=0.85, epsilon=0.01, p_f=0.1)
    source_nodes = ppr.get_source_nodes()
    
    print(f"\n  Found {len(source_nodes)} source nodes (in-degree=0)")
    
    # 일부 노드만 샘플링 (테스트)
    sample_nodes = list(source_nodes)[:min(20, len(source_nodes))]
    
    if len(sample_nodes) == 0:
        sample_nodes = graph_obj.nodes[:20]
    
    ppr_results = {}
    for node in sample_nodes:
        _, svn = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
    
    print(f"  ✓ PPR computed")
    
    # NTS/NWS
    scorer = NormalizedScorer(graph_obj, ppr_results)
    feature_scores = scorer.compute_all_scores()
    print(f"  ✓ NTS/NWS computed")
    
    # Anomaly Detection
    import numpy as np
    ppr_scores_dict = {}
    for node in sample_nodes:
        sps, _ = ppr.compute_single_source_ppr(node)
        ppr_scores_dict[node] = sps
    
    full_ppr_scores = {}
    for node in graph_obj.nodes:
        if node in ppr_scores_dict:
            full_ppr_scores[node] = ppr_scores_dict[node]
        else:
            full_ppr_scores[node] = np.zeros(len(graph_obj.nodes))
    
    detector = MPOCryptoMLDetector(
        ppr_scores=full_ppr_scores,
        feature_scores=feature_scores,
        labels=graph_obj.node_labels
    )
    
    detector.train_logistic_regression()
    detector.compute_anomaly_scores()
    
    print(f"  ✓ Anomaly scores computed")
    
    # 평가
    print("\n[Step 3] Results:")
    results_df = detector.get_results_df()
    results_df_sorted = results_df.sort_values('anomaly_score', ascending=False)
    
    print(f"\nTop 10 Anomaly Scores:")
    print(results_df_sorted.head(10))
    
    # 실제 anomaly detection 성능
    top10 = results_df_sorted.head(10)
    detected_frauds = top10[top10['label'] == 1]
    print(f"\n  ✓ Detected {len(detected_frauds)} actual frauds in top 10")
    
    print("\n" + "="*70)
    print("✅ Hybrid pipeline completed!")
    print("="*70)


if __name__ == "__main__":
    main()

