"""
최종 해결책: Etherscan에서 anomaly 주소들의 실제 거래 가져오기

전략:
1. Kaggle에서 anomaly 주소 리스트 가져오기
2. 해당 주소들의 Etherscan 거래 수집
3. 그래프 생성 + 정확한 라벨
4. 알고리즘 실행
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from graph import CryptoTransactionGraph
from kaggle_timestamp_matcher import get_timestamp_from_etherscan
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer  
from anomaly_detector import MPOCryptoMLDetector
import pandas as pd
import numpy as np
import time


def main():
    print("="*70)
    print("🎯 Etherscan 실제 데이터 수집")
    print("="*70)
    
    API_KEY = "TZ66JXC2M8WST154TM3111MBRRX7X7UAF9"
    
    # 1. Kaggle에서 anomaly 주소만 가져오기
    print("\n[Step 1] Loading Kaggle anomaly addresses...")
    kaggle_path = "./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv"
    df = pd.read_csv(kaggle_path)
    
    # FLAG=1인 주소들 (anomaly) - 1000개로 샘플링
    print("\n  📊 Sampling 1000 addresses...")
    n_total = 1000
    n_anomalies = int(n_total * 0.2)  # 20% anomaly
    n_normal = n_total - n_anomalies
    
    anomaly_addresses = df[df['FLAG'] == 1]['Address'].head(n_anomalies).tolist()
    normal_addresses = df[df['FLAG'] == 0]['Address'].head(n_normal).tolist()
    
    target_addresses = anomaly_addresses + normal_addresses
    
    print(f"  ✓ Anomaly addresses: {len(anomaly_addresses)}")
    print(f"  ✓ Normal addresses: {len(normal_addresses)}")
    print(f"  Total: {len(target_addresses)} addresses")
    
    # 2. Etherscan에서 거래 수집
    print("\n[Step 2] Fetching transactions from Etherscan...")
    print(f"  Processing {len(target_addresses)} addresses...")
    
    graph = CryptoTransactionGraph()
    
    labels = {}
    for i, address in enumerate(target_addresses):
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(target_addresses)}")
        
        txs = get_timestamp_from_etherscan(address, API_KEY)
        
        for from_addr, to_addr, value, timestamp in txs:
            graph.add_edge(from_addr, to_addr, value, timestamp)
        
        # 라벨 설정 (Known 노드만)
        if address in anomaly_addresses:
            labels[address] = 1  # Anomaly
        else:
            labels[address] = 0  # Normal (정상 Known)
        
        # Unknown 노드는 라벨 딕셔너리에 추가하지 않음!
        # → 거래 상대방 중 일부는 Unknown
    
    # ✅ Weak Supervision 구현:
    # - Known: 직접 수집한 주소들 (라벨 있음)
    # - Unknown: 거래 상대방 (라벨 없음)
    # - PPR은 전체 노드에 적용 (구조 정보 활용)
    # - LR은 Known만 학습 (라벨 있음)
    
    graph.set_labels(labels)
    
    print(f"\n✓ Graph created:")
    print(f"  Nodes (V): {len(graph.nodes)}")
    print(f"  Edges (E): {len(graph.edges)}")
    print(f"  Labels: {sum(labels.values())} anomalies")
    
    # 2.5. 그래프 저장
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    
    graph.save("results/graph_200_etherscan_real.json")
    print(f"\n  💾 Graph saved to: results/graph_200_etherscan_real.json")
    
    # 3. 알고리즘 실행
    print("\n[Step 3] Running algorithms...")
    
    nx_graph = graph.build_graph()
    ppr = PersonalizedPageRank(nx_graph, alpha=0.5, epsilon=0.01, p_f=1.0)
    
    source_nodes = ppr.get_source_nodes()
    if len(source_nodes) == 0:
        source_nodes = set(graph.nodes[:10])
    
    sample_nodes = list(source_nodes)[:min(10, len(source_nodes))]
    print(f"  Using {len(sample_nodes)} source nodes")
    
    ppr_results = {}
    ppr_scores = {}
    
    from tqdm import tqdm
    print(f"  Computing PPR for {len(sample_nodes)} source nodes...")
    
    for node in tqdm(sample_nodes, desc="  PPR", ncols=70):
        sps, svn, nodes_list = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
        ppr_scores[node] = sps
    
    # NTS/NWS
    scorer = NormalizedScorer(graph, ppr_results)
    feature_scores = scorer.compute_all_scores()
    
    # Anomaly Detection
    full_ppr_scores = {}
    for node in graph.nodes:
        if node in ppr_scores:
            full_ppr_scores[node] = ppr_scores[node]
        else:
            full_ppr_scores[node] = np.zeros(len(graph.nodes))
    
    detector = MPOCryptoMLDetector(
        ppr_scores=full_ppr_scores,
        feature_scores=feature_scores,
        labels=labels
    )
    
    detector.train_logistic_regression()
    detector.compute_anomaly_scores()
    
    # 결과
    print("\n[Step 4] Results:")
    results_df = detector.get_results_df()
    results_df_sorted = results_df.sort_values('anomaly_score', ascending=False)
    
    print(f"\nTop 10 Anomaly Scores:")
    top10 = results_df_sorted.head(10)
    print(top10[['node', 'label', 'anomaly_score']])
    
    detected = top10[top10['label'] == 1]
    print(f"\n✅ Detected {len(detected)} actual anomalies in top 10")
    
    # 평가
    for k in [5, 10]:
        eval_results = detector.evaluate_precision_at_k(k=k)
        print(f"\nPrecision@{k}: {eval_results.get(f'precision@{k}', 0):.4f}")
    
    print("\n" + "="*70)
    print("✅ Completed successfully!")
    print("="*70)
    
    return detector, graph, results_df_sorted


if __name__ == "__main__":
    main()

