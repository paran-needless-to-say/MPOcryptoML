"""
실제 Etherscan API를 사용한 데이터 수집 및 실행

API Key: TZ66JXC2M8WST154TM3111MBRRX7X7UAF9
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from kaggle_timestamp_matcher import create_hybrid_graph, load_kaggle_addresses_with_labels
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer
from anomaly_detector import MPOCryptoMLDetector
import numpy as np
import pandas as pd


def main():
    print("="*70)
    print("🎯 Etherscan API로 실제 데이터 수집 및 실행")
    print("="*70)
    
    API_KEY = "TZ66JXC2M8WST154TM3111MBRRX7X7UAF9"
    
    # Step 1: Kaggle에서 주소와 라벨 가져오기
    print("\n[Step 1] Loading Kaggle data...")
    kaggle_path = "./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv"
    
    try:
        kaggle_df = load_kaggle_addresses_with_labels(kaggle_path)
        print(f"  ✓ Loaded {len(kaggle_df)} addresses from Kaggle")
        print(f"  ✓ Anomalies: {sum(kaggle_df['label'])}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return
    
    # Step 2: Etherscan에서 실제 거래 수집 (라벨이 있는 주소들)
    print("\n[Step 2] Fetching transactions from Etherscan...")
    print("  ⏳ This will take a few minutes...")
    print("  Processing 50 addresses with API calls...")
    
    # 라벨이 있는 주소들 우선 (anomaly가 있는 주소)
    anomaly_addresses = kaggle_df[kaggle_df['label'] == 1]['address'].tolist()
    normal_addresses = kaggle_df[kaggle_df['label'] == 0]['address'].tolist()
    
    # 샘플링: anomaly 10개 + normal 40개
    sample_addresses = anomaly_addresses[:10] + normal_addresses[:40]
    
    try:
        graph_obj = create_hybrid_graph(
            kaggle_csv_path=kaggle_path,
            api_key=API_KEY,
            n_addresses=50  # 전체 주소 중 50개만
        )
        
        print(f"\n✓ Graph created:")
        print(f"  Nodes (V): {len(graph_obj.nodes)}")
        print(f"  Edges (E): {len(graph_obj.edges)}")
        print(f"  Labels: {sum(graph_obj.node_labels.values())} anomalies")
        
    except Exception as e:
        print(f"\n❌ Error creating graph: {e}")
        print("\nFallback: Using dummy data for testing...")
        from graph import generate_dummy_data
        graph_obj = generate_dummy_data(n_nodes=100, n_transactions=500)
    
    # Step 3: PPR 실행
    print("\n[Step 3] Running Multi-source PPR...")
    graph = graph_obj.build_graph()
    ppr = PersonalizedPageRank(graph, alpha=0.85, epsilon=0.01, p_f=0.1)
    
    source_nodes = ppr.get_source_nodes()
    
    if len(source_nodes) == 0:
        source_nodes = set(graph_obj.nodes[:20])
    
    print(f"  Found {len(source_nodes)} source nodes")
    
    # 샘플링
    sample_nodes = list(source_nodes)[:min(20, len(source_nodes))]
    
    ppr_results = {}
    ppr_scores = {}
    
    for node in sample_nodes:
        sps, svn = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
        ppr_scores[node] = sps
    
    print(f"  ✓ PPR computed")
    
    # Step 4: NTS/NWS
    print("\n[Step 4] Computing NTS & NWS...")
    scorer = NormalizedScorer(graph_obj, ppr_results)
    feature_scores = scorer.compute_all_scores()
    print(f"  ✓ Features computed")
    
    # Step 5: Anomaly Detection
    print("\n[Step 5] Training and evaluating...")
    
    # Full PPR scores
    full_ppr_scores = {}
    for node in graph_obj.nodes:
        if node in ppr_scores:
            full_ppr_scores[node] = ppr_scores[node]
        else:
            full_ppr_scores[node] = np.zeros(len(graph_obj.nodes))
    
    detector = MPOCryptoMLDetector(
        ppr_scores=full_ppr_scores,
        feature_scores=feature_scores,
        labels=graph_obj.node_labels
    )
    
    detector.train_logistic_regression()
    detector.compute_anomaly_scores()
    
    # 평가
    print("\n[Step 6] Results:")
    results_df = detector.get_results_df()
    results_df_sorted = results_df.sort_values('anomaly_score', ascending=False)
    
    print(f"\nTop 10 Anomaly Scores:")
    print(results_df_sorted.head(10)[['node', 'label', 'anomaly_score']])
    
    # 성능
    for k in [5, 10, 20]:
        eval_results = detector.evaluate_precision_at_k(k=k)
        print(f"\nK={k}:")
        for metric, score in eval_results.items():
            print(f"  {metric}: {score:.4f}")
    
    print("\n" + "="*70)
    print("✅ Completed successfully!")
    print("="*70)
    
    return detector, graph_obj, results_df_sorted


if __name__ == "__main__":
    main()

