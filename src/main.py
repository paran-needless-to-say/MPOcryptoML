"""
MPOCryptoML 메인 파이프라인
논문의 전체 워크플로우를 통합한 실행 스크립트
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from graph import CryptoTransactionGraph, generate_dummy_data
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer
from anomaly_detector import MPOCryptoMLDetector


def run_mpocrypto_ml_pipeline(
    n_nodes: int = 100,
    n_transactions: int = 500,
    anomaly_ratio: float = 0.15,
    test_mode: bool = True
):
    """
    MPOCryptoML 전체 파이프라인 실행
    
    Args:
        n_nodes: 노드 개수
        n_transactions: 거래 개수
        anomaly_ratio: 사기 노드 비율
        test_mode: 테스트 모드 (샘플링 여부)
    
    Returns:
        detector: MPOCryptoMLDetector 객체
    """
    print("="*60)
    print("MPOCryptoML: Multi-Pattern Crypto Anomaly Detection")
    print("="*60)
    
    # Step 1: 그래프 생성
    print("\n[Step 1] Creating transaction graph...")
    graph_obj = generate_dummy_data(
        n_nodes=n_nodes,
        n_transactions=n_transactions,
        anomaly_ratio=anomaly_ratio,
        seed=42
    )
    graph = graph_obj.build_graph()
    
    print(f"  ✓ Nodes: {len(graph_obj.nodes)}")
    print(f"  ✓ Edges: {len(graph_obj.edges)}")
    print(f"  ✓ Anomalies: {sum(graph_obj.node_labels.values())}")
    
    # Step 2: Multi-source PPR
    print("\n[Step 2] Computing Multi-source Personalized PageRank...")
    ppr = PersonalizedPageRank(graph, alpha=0.5, epsilon=0.01, p_f=1.0)
    
    # Line 2-4: in-degree가 0인 노드를 source nodes로 식별
    source_nodes_all = ppr.get_source_nodes()
    
    # in-degree=0인 노드가 없으면 모든 노드를 source로 사용 (더미 데이터의 경우)
    if len(source_nodes_all) == 0:
        print(f"  ⚠️ No source nodes (in-degree=0) found")
        print(f"  Using all nodes as sources (dummy data scenario)")
        sample_nodes = graph_obj.nodes[:min(30, len(graph_obj.nodes))] if test_mode else graph_obj.nodes
    else:
        if test_mode:
            sample_nodes = list(source_nodes_all)[:min(30, len(source_nodes_all))]
            print(f"  Found {len(source_nodes_all)} source nodes (in-degree=0)")
            print(f"  (Test mode: processing {len(sample_nodes)} source nodes)")
        else:
            sample_nodes = source_nodes_all
            print(f"  Found {len(source_nodes_all)} source nodes (in-degree=0)")
            print(f"  Processing all {len(sample_nodes)} source nodes")
    
    ppr_results = {}
    ppr_scores_dict = {}
    
    for node in sample_nodes:
        sps, svn, all_nodes_list = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
        # 전체 스코어와 노드 리스트 저장 (anomaly detection에 사용)
        ppr_scores_dict[node] = (sps, all_nodes_list)
    
    print(f"  ✓ PPR computed for {len(ppr_results)} nodes")
    
    # Step 3: NTS & NWS 계산
    print("\n[Step 3] Computing Normalized Timestamp & Weight Scores...")
    scorer = NormalizedScorer(graph_obj, ppr_results)
    feature_scores = scorer.compute_all_scores()
    
    print(f"  ✓ NTS/NWS computed for all nodes")
    print(f"\n  Sample scores:\n{feature_scores.head()}")
    
    # Step 4: Logistic Regression 학습
    print("\n[Step 4] Training Logistic Regression model...")
    
    # PPR 스코어 딕셔너리 생성 (전체 노드용)
    # ppr_scores_dict는 이미 (sps, all_nodes_list) 튜플 형태
    full_ppr_scores = ppr_scores_dict
    
    detector = MPOCryptoMLDetector(
        ppr_scores=full_ppr_scores,
        feature_scores=feature_scores,
        labels=graph_obj.node_labels
    )
    
    detector.train_logistic_regression()
    
    # Step 5: Anomaly Score 계산
    print("\n[Step 5] Computing Anomaly Scores...")
    detector.compute_pattern_scores()
    anomaly_scores = detector.compute_anomaly_scores()
    
    print(f"  ✓ Anomaly scores computed for all nodes")
    
    # Step 6: 평가
    print("\n[Step 6] Evaluation...")
    eval_results = detector.evaluate_precision_at_k(k=min(10, n_nodes // 10))
    
    for metric, score in eval_results.items():
        print(f"  {metric}: {score:.4f}")
    
    # 결과 요약
    print("\n" + "="*60)
    print("Pipeline Summary")
    print("="*60)
    
    results_df = detector.get_results_df()
    results_df_sorted = results_df.sort_values('anomaly_score', ascending=False)
    
    print(f"\nTop 10 Anomaly Scores:")
    print(results_df_sorted.head(10)[['node', 'label', 'anomaly_score']])
    
    print(f"\n\nResults saved in detector object")
    print("Use detector.get_results_df() to view full results")
    
    return detector, graph_obj, results_df_sorted


def visualize_results(detector: MPOCryptoMLDetector, save_dir: str = "results"):
    """
    결과 시각화
    
    Args:
        detector: MPOCryptoMLDetector 객체
        save_dir: 저장 디렉토리
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 1. ROC 곡선
    print("\nGenerating visualizations...")
    
    # 2. Score 분포 시각화
    results_df = detector.get_results_df()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Anomaly Score 분포
    ax1 = axes[0, 0]
    ax1.hist(results_df[results_df['label'] == 0]['anomaly_score'].values,
             bins=50, alpha=0.7, label='Normal', color='blue')
    ax1.hist(results_df[results_df['label'] == 1]['anomaly_score'].values,
             bins=50, alpha=0.7, label='Fraud', color='red')
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Anomaly Score Distribution')
    ax1.legend()
    
    # NTS vs NWS 산점도
    ax2 = axes[0, 1]
    normal_data = results_df[results_df['label'] == 0]
    fraud_data = results_df[results_df['label'] == 1]
    ax2.scatter(normal_data['nts'], normal_data['nws'], 
               alpha=0.6, label='Normal', s=20, c='blue')
    ax2.scatter(fraud_data['nts'], fraud_data['nws'], 
               alpha=0.6, label='Fraud', s=20, c='red')
    ax2.set_xlabel('NTS')
    ax2.set_ylabel('NWS')
    ax2.set_title('NTS vs NWS')
    ax2.legend()
    
    # Top K Precision
    ax3 = axes[1, 0]
    k_values = [5, 10, 15, 20]
    precision_scores = []
    for k in k_values:
        results = detector.evaluate_precision_at_k(k=k)
        precision_scores.append(results[f'precision@{k}'])
    ax3.plot(k_values, precision_scores, marker='o')
    ax3.set_xlabel('K')
    ax3.set_ylabel('Precision@K')
    ax3.set_title('Precision at K')
    ax3.grid(True)
    
    # Anomaly Score 순위 (노드별)
    ax4 = axes[1, 1]
    sorted_results = results_df.sort_values('anomaly_score', ascending=False)
    ax4.bar(range(len(sorted_results)), sorted_results['anomaly_score'])
    ax4.set_xlabel('Rank')
    ax4.set_ylabel('Anomaly Score')
    ax4.set_title('Anomaly Score Ranking')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'mpocrypto_ml_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualizations saved to {save_dir}/mpocrypto_ml_results.png")
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MPOCryptoML Pipeline')
    parser.add_argument('--n-nodes', type=int, default=100, help='Number of nodes')
    parser.add_argument('--n-transactions', type=int, default=500, help='Number of transactions')
    parser.add_argument('--anomaly-ratio', type=float, default=0.15, help='Anomaly ratio')
    parser.add_argument('--no-test-mode', action='store_true', help='Disable test mode')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    detector, graph_obj, results_df = run_mpocrypto_ml_pipeline(
        n_nodes=args.n_nodes,
        n_transactions=args.n_transactions,
        anomaly_ratio=args.anomaly_ratio,
        test_mode=not args.no_test_mode
    )
    
    # 시각화
    if args.visualize:
        visualize_results(detector)
