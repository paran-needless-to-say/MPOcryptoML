"""
Baseline 비교 실행 스크립트
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import json
from graph import CryptoTransactionGraph
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer
from anomaly_detector import MPOCryptoMLDetector
from baselines import BaselineDetector
from tqdm import tqdm
import pandas as pd


def compare_baselines():
    print("="*70)
    print("📊 Baseline 비교")
    print("="*70)
    
    # 1. 그래프 로드
    print("\n[Step 1] Loading graph...")
    filepath = "results/graph_200_etherscan_real.json"
    
    graph_obj = CryptoTransactionGraph()
    with open(filepath, 'r') as f:
        data = json.load(f)
        graph_obj.nodes = data['nodes']
        graph_obj.edges = [(e['from'], e['to'], e['value'], e['timestamp']) for e in data['edges']]
        graph_obj.node_labels = {k: int(v) for k, v in data['labels'].items()}
    
    print(f"  Nodes: {len(graph_obj.nodes)}")
    print(f"  Anomalies: {sum(graph_obj.node_labels.values())}")
    
    # 2. PPR & Features
    print("\n[Step 2] Computing PPR and features...")
    graph = graph_obj.build_graph()
    ppr = PersonalizedPageRank(graph, alpha=0.5, epsilon=0.01, p_f=1.0)
    
    source_nodes = ppr.get_source_nodes()
    if len(source_nodes) == 0:
        source_nodes = set(graph_obj.nodes[:20])
    
    sample_nodes = list(source_nodes)[:min(10, len(source_nodes))]
    
    ppr_results = {}
    
    for node in tqdm(sample_nodes, desc="  PPR", ncols=70):
        sps, svn, all_nodes_list = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
    
    ppr_by_source = {}
    for node in sample_nodes:
        sps, svn, all_nodes_list = ppr.compute_single_source_ppr(node)
        ppr_by_source[node] = {n: float(sps[i]) for i, n in enumerate(all_nodes_list)}
    
    global_pi = {}
    for d in ppr_by_source.values():
        for n, val in d.items():
            global_pi[n] = global_pi.get(n, 0.0) + val
    
    if global_pi:
        m = max(global_pi.values())
        if m > 0:
            for n in list(global_pi.keys()):
                global_pi[n] /= m
    
    scorer = NormalizedScorer(graph_obj, ppr_results)
    feature_scores = scorer.compute_all_scores()
    
    # 3. MPOCryptoML 모델
    print("\n[Step 3] Training MPOCryptoML...")
    
    ppr_scores_dict = {}
    for source_node in sample_nodes:
        sps_array = pd.Series([global_pi.get(n, 0.0) for n in ppr_results[source_node]])
        ppr_scores_dict[source_node] = (sps_array.values, list(ppr_results[source_node]))
    
    detector = MPOCryptoMLDetector(
        ppr_scores=ppr_scores_dict,
        feature_scores=feature_scores,
        labels=graph_obj.node_labels
    )
    
    detector.train_logistic_regression()
    detector.compute_pattern_scores()
    detector.compute_anomaly_scores()
    
    # 4. Baselines
    print("\n[Step 4] Training Baselines...")
    baseline = BaselineDetector(feature_scores, graph_obj.node_labels)
    baseline.train_all()
    
    # 5. 비교
    print("\n" + "="*70)
    print("📊 Performance Comparison")
    print("="*70)
    
    # MPOCryptoML 결과
    results_df = detector.get_results_df()
    results_df_sorted = results_df.sort_values('anomaly_score', ascending=False)
    
    top_k = results_df_sorted.head(20)
    mpocrypto_metrics = {}
    for k in [5, 10, 20]:
        eval_results = detector.evaluate_precision_at_k(k=k)
        mpocrypto_metrics[f'precision@{k}'] = eval_results.get(f'precision@{k}', 0)
        mpocrypto_metrics[f'recall@{k}'] = eval_results.get(f'recall@{k}', 0)
    
    print("\n🎯 MPOCryptoML (Our Method):")
    print(f"  Precision@10: {mpocrypto_metrics['precision@10']:.4f}")
    print(f"  Recall@10: {mpocrypto_metrics['recall@10']:.4f}")
    print(f"  Train AUC: {detector.scores['auc']:.4f}" if hasattr(detector, 'scores') else "  AUC: N/A")
    
    print("\n📊 Baselines:")
    baseline_df = baseline.get_comparison_df()
    print(baseline_df.to_string(index=False))
    
    # 전체 비교 테이블
    print("\n" + "="*70)
    print("📋 Summary Comparison")
    print("="*70)
    
    comparison = []
    
    # MPOCryptoML
    comparison.append({
        'Model': 'MPOCryptoML (Ours)',
        'Method': 'Graph-based + PPR + LR',
        'Test AUC': mpocrypto_metrics.get('precision@10', 0)
    })
    
    # Baselines
    for _, row in baseline_df.iterrows():
        comparison.append({
            'Model': row['Model'],
            'Method': 'Feature-based',
            'Test AUC': row['Test_AUC']
        })
    
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    
    return detector, baseline, comparison_df


if __name__ == "__main__":
    detector, baseline, comparison = compare_baselines()
    
    print("\n" + "="*70)
    print("✅ Baseline Comparison Complete!")
    print("="*70)
    
    # Save results
    Path("results").mkdir(parents=True, exist_ok=True)
    comparison.to_csv("results/baseline_comparison.csv", index=False)
    print("\n💾 Saved: results/baseline_comparison.csv")

