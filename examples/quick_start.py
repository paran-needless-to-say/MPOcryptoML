"""
MPOCryptoML 빠른 시작 가이드

논문의 4단계 파이프라인을 단계별로 실행하는 예제
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
from graph import generate_dummy_data
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer
from anomaly_detector import MPOCryptoMLDetector

def main():
    print("="*70)
    print("MPOCryptoML: Multi-Pattern Cryptocurrency Anomaly Detection")
    print("="*70)
    
    # ==========================================
    # Step 1: 그래프 생성
    # ==========================================
    print("\n[Step 1] Creating transaction graph...")
    graph_obj = generate_dummy_data(
        n_nodes=50,
        n_transactions=200,
        anomaly_ratio=0.15,
        seed=42
    )
    graph = graph_obj.build_graph()
    print(f"  ✓ Nodes: {len(graph_obj.nodes)}")
    print(f"  ✓ Edges: {len(graph_obj.edges)}")
    print(f"  ✓ Anomalies: {sum(graph_obj.node_labels.values())}")
    
    # ==========================================
    # Step 2: Multi-source PPR
    # ==========================================
    print("\n[Step 2] Computing Multi-source Personalized PageRank...")
    ppr = PersonalizedPageRank(graph, alpha=0.85, epsilon=0.01, p_f=0.1)
    
    # 샘플 노드에 대해 PPR 계산
    sample_nodes = graph_obj.nodes[:15]
    print(f"  Computing PPR for {len(sample_nodes)} nodes...")
    
    ppr_results = {}
    ppr_scores = {}
    
    for node in sample_nodes:
        sps, svn, nodes_list = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
        ppr_scores[node] = sps
    
    print(f"  ✓ PPR computed")
    
    # ==========================================
    # Step 3: NTS & NWS 계산
    # ==========================================
    print("\n[Step 3] Computing Normalized Timestamp & Weight Scores...")
    scorer = NormalizedScorer(graph_obj, ppr_results)
    feature_scores = scorer.compute_all_scores()
    print(f"  ✓ NTS/NWS computed")
    print(f"\n  Sample features:")
    print(feature_scores.head())
    
    # ==========================================
    # Step 4: Logistic Regression & Anomaly Detection
    # ==========================================
    print("\n[Step 4] Training Logistic Regression and computing Anomaly Scores...")
    
    # 전체 노드용 PPR 스코어 생성
    n_total_nodes = len(graph_obj.nodes)
    full_ppr_scores = {}
    for node in graph_obj.nodes:
        if node in ppr_scores:
            full_ppr_scores[node] = ppr_scores[node]
        else:
            full_ppr_scores[node] = np.zeros(n_total_nodes)
    
    # Anomaly Detector 생성
    detector = MPOCryptoMLDetector(
        ppr_scores=full_ppr_scores,
        feature_scores=feature_scores,
        labels=graph_obj.node_labels
    )
    
    # 모델 학습
    detector.train_logistic_regression()
    
    # Anomaly Score 계산
    anomaly_scores = detector.compute_anomaly_scores()
    print(f"  ✓ Anomaly scores computed")
    
    # ==========================================
    # Step 5: 평가
    # ==========================================
    print("\n[Step 5] Evaluation...")
    
    results_df = detector.get_results_df()
    results_df_sorted = results_df.sort_values('anomaly_score', ascending=False)
    
    # Precision@K, Recall@K
    for k in [5, 10]:
        eval_results = detector.evaluate_precision_at_k(k=k)
        print(f"\n  K={k}:")
        for metric, score in eval_results.items():
            print(f"    {metric}: {score:.4f}")
    
    # 최종 결과
    print("\n" + "="*70)
    print("Final Results")
    print("="*70)
    
    print("\nTop 10 Anomaly Scores (Higher = More suspicious):")
    top10 = results_df_sorted.head(10)
    print(top10[['node', 'label', 'nts', 'nws', 'pattern_score', 'anomaly_score']])
    
    # 실제 사기 탐지 성능
    detected_frauds = top10[top10['label'] == 1]
    print(f"\n  Actual frauds detected in top 10: {len(detected_frauds)} / {len(top10)}")
    
    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
