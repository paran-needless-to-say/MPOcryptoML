"""
200개 주소로 전체 파이프라인 실행

Kaggle 데이터셋 → 그래프 → 알고리즘 실행
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
    print("MPOCryptoML: 200개 주소로 전체 파이프라인 실행")
    print("="*70)
    
    # 1. Kaggle 데이터 → 그래프
    print("\n[Step 1] Converting Kaggle data to graph (200 addresses)...")
    graph_obj = kaggle_to_graph_realistic(
        csv_path="./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv",
        n_addresses=200,
        seed=42
    )
    
    print(f"\n  Final graph: {len(graph_obj.nodes)} nodes, {len(graph_obj.edges)} edges")
    
    # 2. PPR
    print("\n[Step 2] Computing Multi-source PPR...")
    graph = graph_obj.build_graph()
    ppr = PersonalizedPageRank(graph, alpha=0.5, epsilon=0.01, p_f=1.0)
    
    source_nodes = ppr.get_source_nodes()
    if len(source_nodes) == 0:
        source_nodes = set(graph_obj.nodes[:20])
    
    print(f"  Found {len(source_nodes)} source nodes")
    
    # 샘플링
    sample_nodes = list(source_nodes)[:min(25, len(source_nodes))]
    
    ppr_results = {}
    ppr_scores = {}
    
    for node in sample_nodes:
        sps, svn, nodes_list = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
        ppr_scores[node] = sps
    
    print(f"  ✓ PPR computed")
    
    # 3. NTS/NWS
    print("\n[Step 3] Computing NTS & NWS...")
    scorer = NormalizedScorer(graph_obj, ppr_results)
    feature_scores = scorer.compute_all_scores()
    print(f"  ✓ Features computed")
    
    # 4. Anomaly Detection
    print("\n[Step 4] Training and evaluating...")
    
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
    
    print(f"  ✓ Anomaly detection completed")
    
    # 5. 결과
    print("\n[Step 5] Results:")
    results_df = detector.get_results_df()
    results_df_sorted = results_df.sort_values('anomaly_score', ascending=False)
    
    print(f"\nTop 10 Anomaly Scores:")
    top10 = results_df_sorted.head(10)
    for idx, row in top10.iterrows():
        print(f"  {row['node'][:20]}... : label={row['label']}, score={row['anomaly_score']:.4f}")
    
    # 성능
    detected = top10[top10['label'] == 1]
    print(f"\n✅ Detected {len(detected)} actual anomalies in top 10")
    
    # 평가
    for k in [5, 10, 20]:
        eval_results = detector.evaluate_precision_at_k(k=k)
        precision = eval_results.get(f'precision@{k}', 0)
        recall = eval_results.get(f'recall@{k}', 0)
        f1 = eval_results.get(f'f1@{k}', 0)
        print(f"\nK={k}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    print("\n" + "="*70)
    print("✅ Completed!")
    print("="*70)
    
    return detector, graph_obj, results_df_sorted


if __name__ == "__main__":
    main()

