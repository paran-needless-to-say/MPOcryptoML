"""
Kaggle 데이터셋을 직접 그래프로 변환 (timestamp 시뮬레이션)

사용자가 원하는 것:
- 이미 있는 Kaggle Ethereum Fraud Detection 데이터
- 여기에 timestamp 붙이기
- 그래프로 만들기
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from kaggle_to_graph_realistic import kaggle_to_graph_realistic
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer
from anomaly_detector import MPOCryptoMLDetector
import numpy as np


def main():
    print("="*70)
    print("Kaggle 집계 데이터 → 그래프 (timestamp 시뮬레이션)")
    print("="*70)
    
    # Kaggle CSV 경로
    csv_path = "./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv"
    
    # 그래프 생성 (100개 주소)
    print("\n[Step 1] Converting Kaggle data to graph...")
    graph_obj = kaggle_to_graph_realistic(csv_path, n_addresses=100, seed=42)
    
    # PPR 실행
    print("\n[Step 2] Running algorithms...")
    graph = graph_obj.build_graph()
    ppr = PersonalizedPageRank(graph, alpha=0.85, epsilon=0.01, p_f=0.1)
    
    source_nodes = ppr.get_source_nodes()
    if len(source_nodes) == 0:
        source_nodes = set(graph_obj.nodes[:15])
    
    print(f"  Found {len(source_nodes)} source nodes")
    
    # 샘플링
    sample_nodes = list(source_nodes)[:min(15, len(source_nodes))]
    
    ppr_results = {}
    ppr_scores = {}
    
    for node in sample_nodes:
        sps, svn = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
        ppr_scores[node] = sps
    
    print(f"  ✓ PPR computed")
    
    # NTS/NWS
    scorer = NormalizedScorer(graph_obj, ppr_results)
    feature_scores = scorer.compute_all_scores()
    print(f"  ✓ NTS/NWS computed")
    
    # Anomaly Detection
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
    
    print(f"  ✓ Anomaly scores computed")
    
    # 결과
    print("\n[Step 3] Results:")
    results_df = detector.get_results_df()
    results_df_sorted = results_df.sort_values('anomaly_score', ascending=False)
    
    print(f"\nTop 10 Anomaly Scores:")
    print(results_df_sorted.head(10)[['node', 'label', 'anomaly_score']])
    
    print("\n" + "="*70)
    print("✅ Completed!")
    print("="*70)


if __name__ == "__main__":
    main()

