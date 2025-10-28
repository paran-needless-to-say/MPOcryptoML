"""
수정된 Anomaly Score로 저장된 그래프 테스트
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import json
from graph import CryptoTransactionGraph
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer
from anomaly_detector import MPOCryptoMLDetector
import networkx as nx
import numpy as np
from tqdm import tqdm


def load_graph_from_json(json_path: str):
    """저장된 JSON에서 그래프 로드"""
    with open(json_path) as f:
        data = json.load(f)
    
    graph = CryptoTransactionGraph()
    
    # Nodes
    graph.nodes = data['nodes']
    
    # Edges
    for edge in data['edges']:
        graph.add_edge(
            edge['from'],
            edge['to'],
            edge['value'],
            edge['timestamp']
        )
    
    # Labels
    graph.node_labels = data['labels']
    
    return graph


def main():
    print("="*70)
    print("🔧 수정된 Anomaly Score로 테스트")
    print("="*70)
    
    # 1. 저장된 그래프 로드
    print("\n[Step 1] Loading saved graph...")
    graph = load_graph_from_json("results/graph_200_etherscan_real.json")
    
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Anomalies: {sum(graph.node_labels.values())}")
    
    # 2. PPR 계산
    print("\n[Step 2] Computing PPR...")
    nx_graph = graph.build_graph()
    ppr = PersonalizedPageRank(nx_graph, alpha=0.5, epsilon=0.01, p_f=1.0)
    
    source_nodes = ppr.get_source_nodes()
    if len(source_nodes) == 0:
        source_nodes = set(graph.nodes[:20])
    
    print(f"  Found {len(source_nodes)} source nodes")
    
    # 샘플링 (더 빠른 테스트를 위해 10개로 줄임)
    sample_nodes = list(source_nodes)[:min(10, len(source_nodes))]
    
    ppr_results = {}
    ppr_scores_dict = {}
    
    # 진행 상황 표시
    print(f"  Computing PPR for {len(sample_nodes)} source nodes...")
    
    for node in tqdm(sample_nodes, desc="  PPR", ncols=70):
        sps, svn, all_nodes_list = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
        ppr_scores_dict[node] = (sps, all_nodes_list)
    
    print(f"  ✓ PPR computed")
    
    # 3. NTS/NWS
    print("\n[Step 3] Computing NTS & NWS...")
    scorer = NormalizedScorer(graph, ppr_results)
    feature_scores = scorer.compute_all_scores()
    print(f"  ✓ Features computed")
    
    # 4. Anomaly Detection
    print("\n[Step 4] Training and evaluating...")
    
    # Full PPR scores with node mapping
    full_ppr_scores = {}
    all_nodes_list = list(graph.nodes)
    
    for source in ppr_scores_dict.keys():
        full_ppr_scores[source] = ppr_scores_dict[source]
    
    detector = MPOCryptoMLDetector(
        ppr_scores=full_ppr_scores,
        feature_scores=feature_scores,
        labels=graph.node_labels
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
        print(f"  {row['node'][:30]}... : label={row['label']}, score={row['anomaly_score']:.4f}")
    
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
    
    return detector, graph, results_df_sorted


if __name__ == "__main__":
    main()

