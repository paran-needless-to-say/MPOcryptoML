"""
PPR 점수 분포 분석 스크립트
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from graph import CryptoTransactionGraph
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer
import json

def analyze_ppr_scores():
    print("="*70)
    print("🔍 PPR 점수 분석")
    print("="*70)
    
    # 1. 저장된 그래프 로드
    print("\n[Step 1] Loading graph...")
    filepath = "results/graph_200_etherscan_real.json"
    
    graph_obj = CryptoTransactionGraph()
    with open(filepath, 'r') as f:
        data = json.load(f)
        graph_obj.nodes = data['nodes']
        graph_obj.edges = [(e['from'], e['to'], e['value'], e['timestamp']) for e in data['edges']]
        graph_obj.node_labels = {k: int(v) for k, v in data['labels'].items()}
    
    print(f"  Nodes: {len(graph_obj.nodes)}")
    print(f"  Edges: {len(graph_obj.edges)}")
    print(f"  Anomalies: {sum(graph_obj.node_labels.values())}")
    
    # 2. PPR 계산
    print("\n[Step 2] Computing PPR...")
    graph = graph_obj.build_graph()
    ppr = PersonalizedPageRank(graph, alpha=0.5, epsilon=0.01, p_f=1.0)
    
    source_nodes = ppr.get_source_nodes()
    print(f"  Source nodes: {len(source_nodes)}")
    
    sample_sources = list(source_nodes)[:5]  # 5개만 샘플링
    print(f"  Sampling {len(sample_sources)} sources...")
    
    ppr_scores_dict = {}
    for source in sample_sources:
        sps, svn, all_nodes_list = ppr.compute_single_source_ppr(source)
        ppr_scores_dict[source] = (sps, all_nodes_list)
        print(f"  Source {source[:20]}... - Max score: {sps.max():.6f}, Min score: {sps.min():.6f}, Mean: {sps.mean():.6f}")
    
    # 3. PPR 점수 분포 분석
    print("\n[Step 3] Analyzing PPR score distribution...")
    
    # 첫 번째 source의 PPR 점수
    first_source = sample_sources[0]
    ppr_array, all_nodes_list = ppr_scores_dict[first_source]
    
    # 노드별로 PPR 점수 확인
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes_list)}
    
    print(f"\n  Sampling 20 random nodes:")
    np.random.seed(42)
    sample_nodes = np.random.choice(all_nodes_list, min(20, len(all_nodes_list)), replace=False)
    
    for node in sample_nodes[:20]:
        if node in node_to_idx:
            idx = node_to_idx[node]
            score = ppr_array[idx]
            label = graph_obj.node_labels.get(node, -1)
            print(f"    {node[:30]}... (label={label}): score={score:.6f}")
    
    # 4. π(vi) 계산해보기 (모든 source 합산)
    print("\n[Step 4] Computing π(vi) for all sources...")
    
    # 모든 source의 PPR 점수 합산
    pi_scores = {}
    for node in all_nodes_list:
        pi_sum = 0.0
        for source, (ppr_array, all_nodes_list) in ppr_scores_dict.items():
            if node in node_to_idx:
                idx = node_to_idx[node]
                pi_sum += ppr_array[idx]
        pi_scores[node] = pi_sum
    
    # π(vi) 분포 분석
    pi_values = list(pi_scores.values())
    print(f"\n  π(vi) statistics:")
    print(f"    Max: {max(pi_values):.6f}")
    print(f"    Min: {min(pi_values):.6f}")
    print(f"    Mean: {np.mean(pi_values):.6f}")
    print(f"    Std: {np.std(pi_values):.6f}")
    print(f"    Range: {max(pi_values) - min(pi_values):.6f}")
    
    # Anomaly vs Normal의 π(vi) 비교
    print("\n[Step 5] Comparing π(vi) for anomalies vs normal...")
    
    anomaly_pi = []
    normal_pi = []
    
    for node, pi_score in pi_scores.items():
        label = graph_obj.node_labels.get(node, -1)
        if label == 1:
            anomaly_pi.append(pi_score)
        elif label == 0:
            normal_pi.append(pi_score)
    
    print(f"  Anomalies (label=1):")
    print(f"    Count: {len(anomaly_pi)}")
    print(f"    Mean π(vi): {np.mean(anomaly_pi):.6f}")
    print(f"    Std: {np.std(anomaly_pi):.6f}")
    print(f"\n  Normal (label=0):")
    print(f"    Count: {len(normal_pi)}")
    print(f"    Mean π(vi): {np.mean(normal_pi):.6f}")
    print(f"    Std: {np.std(normal_pi):.6f}")
    
    if len(anomaly_pi) > 0 and len(normal_pi) > 0:
        # Mann-Whitney U test (non-parametric)
        from scipy.stats import mannwhitneyu
        try:
            stat, p_value = mannwhitneyu(anomaly_pi, normal_pi, alternative='two-sided')
            print(f"\n  Mann-Whitney U test:")
            print(f"    U-statistic: {stat}")
            print(f"    p-value: {p_value}")
            print(f"    {'Statistically different' if p_value < 0.05 else 'NOT statistically different'}")
        except:
            print("\n  Statistical test skipped")
    
    print("\n" + "="*70)
    print("✅ Analysis completed!")
    print("="*70)
    
    return pi_scores, graph_obj.node_labels


if __name__ == "__main__":
    analyze_ppr_scores()

