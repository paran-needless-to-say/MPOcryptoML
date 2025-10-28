"""
상세 분석: 사기 지갑 판정 및 피쳐 분석
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import json
import pandas as pd
import numpy as np
from graph import CryptoTransactionGraph
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer
from anomaly_detector import MPOCryptoMLDetector
from tqdm import tqdm


def detailed_analysis():
    print("="*70)
    print("📊 상세 분석: 사기 지갑 판정 및 피쳐 분석")
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
    
    # 실제 사기 지갑 개수
    total_anomalies = sum(graph_obj.node_labels.values())
    total_nodes = len(graph_obj.node_labels)
    total_normal = total_nodes - total_anomalies
    
    print(f"  총 노드: {total_nodes}")
    print(f"  실제 사기 지갑: {total_anomalies}개")
    print(f"  정상 지갑: {total_normal}개")
    
    # 2. PPR 계산
    print("\n[Step 2] Computing PPR...")
    graph = graph_obj.build_graph()
    ppr = PersonalizedPageRank(graph, alpha=0.5, epsilon=0.01, p_f=1.0)
    
    source_nodes = ppr.get_source_nodes()
    if len(source_nodes) == 0:
        source_nodes = set(graph_obj.nodes[:20])
    
    sample_nodes = list(source_nodes)[:min(10, len(source_nodes))]
    
    ppr_results = {}
    ppr_scores_dict = {}
    
    print(f"  Computing PPR for {len(sample_nodes)} source nodes...")
    for node in tqdm(sample_nodes, desc="  PPR", ncols=70):
        sps, svn, all_nodes_list = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
        ppr_scores_dict[node] = (sps, all_nodes_list)
    
    # 3. Feature 계산
    print("\n[Step 3] Computing features...")
    scorer = NormalizedScorer(graph_obj, ppr_results)
    feature_scores = scorer.compute_all_scores()
    
    # 4. Anomaly Detection
    print("\n[Step 4] Training model and computing scores...")
    full_ppr_scores = {}
    for source in ppr_scores_dict.keys():
        full_ppr_scores[source] = ppr_scores_dict[source]
    
    detector = MPOCryptoMLDetector(
        ppr_scores=full_ppr_scores,
        feature_scores=feature_scores,
        labels=graph_obj.node_labels
    )
    
    detector.train_logistic_regression()
    detector.compute_anomaly_scores()
    detector.compute_pattern_scores()
    
    # 5. 상세 분석
    print("\n" + "="*70)
    print("📊 상세 분석 결과")
    print("="*70)
    
    results_df = detector.get_results_df()
    results_df_sorted = results_df.sort_values('anomaly_score', ascending=False)
    
    print(f"\n[1] 실제 사기 지갑 현황:")
    print(f"  총 사기 지갑: {total_anomalies}개 ({total_anomalies/total_nodes*100:.2f}%)")
    print(f"  총 정상 지갑: {total_normal}개 ({total_normal/total_nodes*100:.2f}%)")
    
    # Top K 분석
    print(f"\n[2] 모델 판정 결과:")
    for k in [5, 10, 20, 50]:
        top_k = results_df_sorted.head(k)
        detected_anomalies = top_k[top_k['label'] == 1]
        precision = len(detected_anomalies) / k
        recall = len(detected_anomalies) / total_anomalies if total_anomalies > 0 else 0
        
        print(f"  Top {k}:")
        print(f"    감지된 사기: {len(detected_anomalies)}개")
        print(f"    Precision@{k}: {precision:.4f}")
        print(f"    Recall@{k}: {recall:.4f}")
    
    # Top 10 상세 분석
    print(f"\n[3] Top 10 이상 지갑 상세 분석:")
    print("-"*70)
    top10 = results_df_sorted.head(10)
    
    for idx, (_, row) in enumerate(top10.iterrows(), 1):
        node = row['node']
        label = row['label']
        anomaly_score = row['anomaly_score']
        nts = row['nts']
        nws = row['nws']
        pattern_score = row['pattern_score']
        
        # PPR 점수 계산
        first_source = list(full_ppr_scores.keys())[0]
        ppr_array, all_nodes_list = full_ppr_scores[first_source]
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes_list)}
        
        ppr_sum = 0.0
        if node in node_to_idx:
            idx_pos = node_to_idx[node]
            for source, (ppr_arr, _) in full_ppr_scores.items():
                if idx_pos < len(ppr_arr):
                    ppr_sum += ppr_arr[idx_pos]
        
        label_str = "🔴 사기" if label == 1 else "🟢 정상"
        
        print(f"\n  [{idx}] {node[:40]}...")
        print(f"      라벨: {label_str}")
        print(f"      Anomaly Score: {anomaly_score:.4f}")
        print(f"      구성:")
        print(f"        - π(vi) (PPR 합계): {ppr_sum:.6f}")
        print(f"        - F(θ,ω)(vi) (패턴 점수): {pattern_score:.6f}")
        print(f"        - NTS: {nts:.4f}")
        print(f"        - NWS: {nws:.4f}")
        
        # 피쳐 기여도 분석
        print(f"      피쳐 기여도:")
        if pattern_score > 0:
            ppr_contribution = ppr_sum / pattern_score
            print(f"        - PPR 기여: {ppr_sum:.6f}")
            print(f"        - 패턴 점수 기여: {pattern_score:.6f}")
            print(f"        - NTS 기여: {nts:.4f}")
            print(f"        - NWS 기여: {nws:.4f}")
    
    # 피쳐 통계
    print(f"\n[4] 피쳐 통계 (Anomaly vs Normal):")
    print("-"*70)
    
    anomaly_df = results_df[results_df['label'] == 1]
    normal_df = results_df[results_df['label'] == 0]
    
    print(f"\n  Anomaly 지갑 (n={len(anomaly_df)}):")
    print(f"    평균 Anomaly Score: {anomaly_df['anomaly_score'].mean():.4f}")
    print(f"    평균 NTS: {anomaly_df['nts'].mean():.4f}")
    print(f"    평균 NWS: {anomaly_df['nws'].mean():.4f}")
    print(f"    평균 Pattern Score: {anomaly_df['pattern_score'].mean():.4f}")
    
    print(f"\n  Normal 지갑 (n={len(normal_df)}):")
    print(f"    평균 Anomaly Score: {normal_df['anomaly_score'].mean():.4f}")
    print(f"    평균 NTS: {normal_df['nts'].mean():.4f}")
    print(f"    평균 NWS: {normal_df['nws'].mean():.4f}")
    print(f"    평균 Pattern Score: {normal_df['pattern_score'].mean():.4f}")
    
    print("\n" + "="*70)
    print("✅ 상세 분석 완료!")
    print("="*70)
    
    return detector, graph_obj, results_df_sorted


if __name__ == "__main__":
    detailed_analysis()

