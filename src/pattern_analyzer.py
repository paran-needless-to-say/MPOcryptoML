"""
패턴 분석: 각 지갑이 어떤 세탁 패턴을 보이는지 분석
논문의 θ(v_i), ω(v_i)를 이용한 패턴 식별
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from graph import CryptoTransactionGraph


class PatternAnalyzer:
    """
    패턴 분석: θ(v_i), ω(v_i)를 이용한 세탁 패턴 식별
    
    논문의 수학적 정의:
    - θ(v_i) = |θ_out(v_i) - θ_in(v_i)|: 시간 불균형
    - ω(v_i) = |ω_out(v_i) - ω_in(v_i)|: 금액 불균형
    
    패턴 해석:
    - θ_out >> θ_in → Fan-out (보내기 시간이 길음)
    - θ_out << θ_in → Fan-in (받기 시간이 짧음)
    - ω_out >> ω_in → Fan-out (보내기 금액이 큼)
    - ω_out << ω_in → Fan-in (받기 금액이 작음)
    """
    
    def __init__(self, graph: CryptoTransactionGraph):
        self.graph = graph
        self.tx_df = graph.get_transactions_df()
    
    def analyze_pattern(self, node: str) -> Tuple[str, Dict]:
        """
        노드의 패턴 분석
        
        Money Laundering Pattern Definitions:
        - Fan-in: 많은 입금 (θ_out << θ_in, ω_out << ω_in)
        - Fan-out: 많은 송금 (θ_out >> θ_in, ω_out >> ω_in)
        - Stack: 많은 입금 → 적은 출금 (in_count >> out_count, avg_in > avg_out)
        - Layering: 여러 거래로 복잡화 (many transactions)
        - Smurfing: 작은 금액 반복 입금 (many small in, avg_in < threshold)
        - Rapid-layering: 짧은 시간 내 빠른 거래 (high freq, short time span)
        
        Args:
            node: 분석할 노드 주소
        
        Returns:
            (pattern_type, pattern_details)
        """
        # NTS 계산 (상세 버전)
        in_timestamps = self.tx_df[self.tx_df['to_address'] == node]['timestamp'].values
        out_timestamps = self.tx_df[self.tx_df['from_address'] == node]['timestamp'].values
        
        theta_in = np.max(in_timestamps) - np.min(in_timestamps) if len(in_timestamps) > 0 else 0
        theta_out = np.max(out_timestamps) - np.min(out_timestamps) if len(out_timestamps) > 0 else 0
        
        # NWS 계산 (상세 버전)
        in_weights = self.tx_df[self.tx_df['to_address'] == node]['value'].values
        out_weights = self.tx_df[self.tx_df['from_address'] == node]['value'].values
        
        omega_in = np.sum(in_weights) if len(in_weights) > 0 else 0
        omega_out = np.sum(out_weights) if len(out_weights) > 0 else 0
        
        # 추가 지표
        in_count = len(in_timestamps)
        out_count = len(out_timestamps)
        avg_in = np.mean(in_weights) if len(in_weights) > 0 else 0
        avg_out = np.mean(out_weights) if len(out_weights) > 0 else 0
        
        # 시간 통계
        if len(in_timestamps) > 1:
            time_spans = np.diff(np.sort(in_timestamps))
            avg_time_between_in = np.mean(time_spans) if len(time_spans) > 0 else 0
        else:
            avg_time_between_in = 0
        
        # 패턴 판단
        patterns = []
        
        # 1. Fan-in/Fan-out
        if theta_out > theta_in * 1.5 and omega_out > omega_in * 1.5:
            patterns.append("Fan-out")
        elif theta_in > theta_out * 1.5 and omega_in > omega_out * 1.5:
            patterns.append("Fan-in")
        
        # 2. Stack: 많은 입금 → 적은 출금
        if in_count > out_count * 3 and avg_in > avg_out * 1.5:
            patterns.append("Stack")
        
        # 3. Smurfing: 작은 금액 반복 입금
        SMURF_THRESHOLD = 0.1  # 0.1 ETH 이하
        if in_count > 20 and avg_in < SMURF_THRESHOLD:
            patterns.append("Smurfing")
        
        # 4. Rapid-layering: 짧은 시간 내 빠른 거래
        # 24시간 기준
        DAY_IN_SECONDS = 86400
        if theta_in > 0 and theta_in < DAY_IN_SECONDS and in_count > 10:
            patterns.append("Rapid-layering")
        
        # 5. Layering: 여러 거래로 복잡화
        if in_count + out_count > 50:
            patterns.append("Layering")
        
        pattern_type = patterns[0] if patterns else "Balanced"
        
        details = {
            'theta_in': theta_in,
            'theta_out': theta_out,
            'omega_in': omega_in,
            'omega_out': omega_out,
            'in_degree': in_count,
            'out_degree': out_count,
            'in_amount': omega_in,
            'out_amount': omega_out,
            'avg_in': avg_in,
            'avg_out': avg_out,
            'patterns': patterns  # 추가: 모든 탐지된 패턴
        }
        
        return pattern_type, details
    
    # _classify_pattern 메서드 제거 (더 이상 사용 안 함)
    
    def analyze_all_patterns(self, nodes: list) -> pd.DataFrame:
        """모든 노드의 패턴 분석"""
        results = []
        
        for node in nodes:
            try:
                pattern_type, details = self.analyze_pattern(node)
                results.append({
                    'node': node,
                    'pattern_type': pattern_type,
                    **details
                })
            except:
                results.append({
                    'node': node,
                    'pattern_type': 'unknown',
                    'theta_in': 0,
                    'theta_out': 0,
                    'omega_in': 0,
                    'omega_out': 0
                })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # 테스트
    from graph import generate_dummy_data
    
    print("Testing Pattern Analyzer...")
    graph_obj = generate_dummy_data(n_nodes=50, n_transactions=200)
    analyzer = PatternAnalyzer(graph_obj)
    
    # Anomaly 노드 분석
    anomaly_nodes = [node for node, label in graph_obj.node_labels.items() if label == 1]
    
    print(f"\nAnalyzing {len(anomaly_nodes)} anomaly nodes...")
    for node in anomaly_nodes[:5]:
        pattern, details = analyzer.analyze_pattern(node)
        print(f"\n{node[:30]}...")
        print(f"  Pattern: {pattern}")
        print(f"  Time: {details['time_pattern']}")
        print(f"  Weight: {details['weight_pattern']}")

