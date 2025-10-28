"""
상세 탐지 리포트: 각 탐지된 사기 지갑에 대한 상세 정보
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
from pattern_analyzer import PatternAnalyzer
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.metrics import average_precision_score, roc_auc_score


def create_comprehensive_report():
    print("="*70)
    print("상세 탐지 리포트 생성")
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
    
    # 실제 학습/평가에 사용되는 사기 지갑 개수
    actual_anomalies = sum([1 for n in graph_obj.nodes if n in graph_obj.node_labels and graph_obj.node_labels[n] == 1])
    
    print(f"  Nodes: {len(graph_obj.nodes)}")
    print(f"  Edges: {len(graph_obj.edges)}")
    print(f"  Anomalies (실제 그래프에 포함): {actual_anomalies}개")
    print(f"  라벨 딕셔너리 총량: {sum(graph_obj.node_labels.values())}개 (200개 중 {sum(graph_obj.node_labels.values())-actual_anomalies}개는 그래프 외부)")
    
    # 2. PPR 계산
    print("\n[Step 2] Computing PPR...")
    graph = graph_obj.build_graph()
    ppr = PersonalizedPageRank(graph, alpha=0.5, epsilon=0.01, p_f=1.0)
    
    source_nodes = ppr.get_source_nodes()
    if len(source_nodes) == 0:
        source_nodes = set(graph_obj.nodes[:20])
    
    print(f"  Found {len(source_nodes)} source nodes (in-degree=0)")
    
    # ⚠️ 성능 문제로 샘플링 (필요시 전체 사용 가능)
    # 전체 사용: sample_nodes = list(source_nodes)
    # 실험: 50 → 100으로 증가
    sample_nodes = list(source_nodes)[:min(100, len(source_nodes))]  # 50 → 100으로 증가
    print(f"  Sampling {len(sample_nodes)} source nodes for PPR computation")
    
    ppr_results = {}
    
    print(f"  Computing PPR for {len(sample_nodes)} source nodes...")
    
    # Fix 3: 전역 pi aggregation (인덱스 불일치 방지)
    ppr_by_source = {}
    for node in tqdm(sample_nodes, desc="  PPR", ncols=70):
        sps, svn, all_nodes_list = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
        # dict 형태로 저장 (node -> score)
        ppr_by_source[node] = {n: float(sps[i]) for i, n in enumerate(all_nodes_list)}
    
    # 전역 pi 합산
    global_pi = {}
    for d in ppr_by_source.values():
        for n, val in d.items():
            global_pi[n] = global_pi.get(n, 0.0) + val
    
    # 정규화 (선택)
    if global_pi:
        m = max(global_pi.values())
        if m > 0:
            for n in list(global_pi.keys()):
                global_pi[n] /= m
    
    # detector는 튜플 형태 필요 (호환성)
    ppr_scores_dict = {}
    for source_node in sample_nodes:
        # 전역 pi를 활용하되, detector 호환성 위해 유지
        sps_array = np.array([global_pi.get(n, 0.0) for n in ppr_results[source_node]])
        ppr_scores_dict[source_node] = (sps_array, list(ppr_results[source_node]))
    
    # 3. Features
    print("\n[Step 3] Computing features...")
    scorer = NormalizedScorer(graph_obj, ppr_results)
    feature_scores = scorer.compute_all_scores()
    
    # 4. Anomaly Detection
    print("\n[Step 4] Training model...")
    full_ppr_scores = {}
    for source in ppr_scores_dict.keys():
        full_ppr_scores[source] = ppr_scores_dict[source]
    
    detector = MPOCryptoMLDetector(
        ppr_scores=full_ppr_scores,
        feature_scores=feature_scores,
        labels=graph_obj.node_labels
    )
    
    detector.train_logistic_regression()
    detector.compute_pattern_scores()
    detector.compute_anomaly_scores()
    
    # 5. Pattern Analysis
    print("\n[Step 5] Analyzing patterns...")
    pattern_analyzer = PatternAnalyzer(graph_obj)
    
    # 6. 상세 리포트 생성
    print("\n" + "="*70)
    print("상세 탐지 리포트")
    print("="*70)
    
    results_df = detector.get_results_df()
    results_df_sorted = results_df.sort_values('anomaly_score', ascending=False)
    
    # Top K 이상 지갑 분석
    k = 20
    top_k = results_df_sorted.head(k)
    
    # Fix 1: Top-K count (boolean -> int)
    detected_in_topk = int(top_k['label'].sum())
    
    print(f"\n[탐지 현황]")
    print(f"  전체 사기 지갑: {sum(graph_obj.node_labels.values())}개 (라벨 딕셔너리)")
    print(f"  그래프에 포함된 사기: {actual_anomalies}개 (실제 학습/평가)")
    print(f"  모델 탐지 (Top {k}): {detected_in_topk}개")
    print(f"  Precision@{k}: {detected_in_topk / k:.4f}")
    
    # 추가 지표
    y_true = top_k['label'].values.astype(int)
    y_score = top_k['anomaly_score'].values
    try:
        ap = average_precision_score(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        print(f"  Average Precision: {ap:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
    except:
        pass
    
    print(f"\n[Anomaly Score 상위 {k}개 지갑 상세 정보]")
    print("="*70)
    print("※ 이 지갑들은 모델이 '가장 이상하다'고 판단한 지갑들입니다.")
    print("※ 정상 지갑이더라도 모델이 이상하게 판단할 수 있습니다.")
    print("="*70)
    
    # PPR 점수 계산 (π(vi))
    first_source = list(full_ppr_scores.keys())[0]
    ppr_array, all_nodes_list = full_ppr_scores[first_source]
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes_list)}
    
    reports = []
    
    for idx, (_, row) in enumerate(top_k.iterrows(), 1):
        node = row['node']
        label = row['label']
        sigma = row['anomaly_score']
        nts = row['nts']
        nws = row['nws']
        pattern_score = row['pattern_score']
        
        # π(vi) 계산 (Fix 3: global_pi 사용)
        pi_sum = float(global_pi.get(node, 0.0))
        
        # Fix 2: 패턴 분석 (기본값 먼저 설정)
        pattern_type = "Unknown"
        pattern_details = {"in_degree": 0, "out_degree": 0, "omega_in": 0.0, "omega_out": 0.0}
        detected_patterns = []
        
        try:
            pattern_type, pattern_details = pattern_analyzer.analyze_pattern(node)
            if pattern_type and pattern_type != "Balanced":
                detected_patterns.append(pattern_type)
            
            # Fix 5: 구체적인 라벨 사용
            if pattern_details.get('in_degree', 0) > pattern_details.get('out_degree', 0) * 3:
                detected_patterns.append("Rapid-layering")
            if pattern_details.get('omega_in', 0) > pattern_details.get('omega_out', 0) * 2:
                detected_patterns.append("Value-mismatch")
        except Exception as e:
            pass
        
        # Evidence summary 생성
        in_count = pattern_details.get('in_degree', 0)
        out_count = pattern_details.get('out_degree', 0)
        in_amount = pattern_details.get('omega_in', 0)
        
        # Fix 6: 문장 톤 개선
        if label == 1:
            evidence = f"🔴 실제 사기 지갑. {in_count}회 입금, {out_count}회 송금. 총 입금: {in_amount:.2f} ETH. 패턴: {', '.join(detected_patterns) if detected_patterns else 'Unknown'}"
        else:
            evidence = f"🟢 비사기 추정(모델 기준). {in_count}회 입금, {out_count}회 송금. 총 입금: {in_amount:.2f} ETH. NTS={nts:.2f}, NWS={nws:.2f}"
        
        # Fix 8: Top paths (incoming + outgoing)
        top_paths = []
        try:
            # Incoming 거래
            incoming = [(e['from'], e['to'], e['value']) for e in data['edges'] if e['to'] == node]
            incoming_sorted = sorted(incoming, key=lambda x: x[2], reverse=True)[:3]
            
            for from_addr, to_addr, value in incoming_sorted:
                top_paths.append(f"{from_addr[:20]}... → {value:.2f} ETH")
            
            # Outgoing 거래
            outgoing = [(e['from'], e['to'], e['value']) for e in data['edges'] if e['from'] == node]
            outgoing_sorted = sorted(outgoing, key=lambda x: x[2], reverse=True)[:3]
            
            for from_addr, to_addr, value in outgoing_sorted:
                top_paths.append(f"→ {to_addr[:20]}... : {value:.2f} ETH")
        except:
            pass
        
        print(f"\n[{idx}] {node[:50]}...")
        print(f"  라벨: {'🔴 사기' if label == 1 else '🟢 정상'}")
        print(f"  σ(vi): {sigma:.4f}")
        print(f"  π(vi): {pi_sum:.6f}")
        print(f"  NTS: {nts:.4f}")
        print(f"  NWS: {nws:.4f}")
        print(f"  패턴: {', '.join(detected_patterns) if detected_patterns else 'None'}")
        print(f"  근거: {evidence}")
        if top_paths:
            print(f"  경로: {top_paths[0]}")
        
        reports.append({
            'rank': idx,
            'address': node,
            'label': label,
            'sigma': sigma,
            'pi': pi_sum,
            'pattern_score': pattern_score,
            'NTS': nts,
            'NWS': nws,
            'detected_patterns': detected_patterns,
            'evidence_summary': evidence,
            'top_paths': top_paths[:3]
        })
    
    print("\n" + "="*70)
    print("✅ 리포트 생성 완료")
    print("="*70)
    
    # Fix 4: 디렉토리 생성
    Path("results").mkdir(parents=True, exist_ok=True)
    
    # 리포트를 JSON으로 저장
    report_path = "results/detection_report.json"
    with open(report_path, 'w') as f:
        json.dump(reports, f, indent=2)
    
    print(f"\n💾 리포트 저장: {report_path}")
    
    return reports


if __name__ == "__main__":
    reports = create_comprehensive_report()

