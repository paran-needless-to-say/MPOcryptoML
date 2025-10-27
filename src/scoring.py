"""
Algorithm 2 & 3: Normalized Timestamp Score (NTS) & Normalized Weight Score (NWS)
논문 Algorithm 2, 3 정확한 구현
"""
import numpy as np
import pandas as pd
from typing import List, Set, Dict, Tuple
from datetime import datetime


class NormalizedScorer:
    """
    Algorithm 2 & 3 구현
    논문의 정확한 NTS/NWS 계산
    """
    
    def __init__(self, graph, ppr_visited_nodes: Dict[str, Set[str]]):
        """
        Args:
            graph: CryptoTransactionGraph 객체
            ppr_visited_nodes: {source_node: {visited_nodes}} 형태의 딕셔너리
        """
        self.graph = graph
        self.ppr_visited_nodes = ppr_visited_nodes
        self.tx_df = graph.get_transactions_df()
    
    def compute_algorithm2_nts(self, visited_nodes: Set[str]) -> Dict[str, float]:
        """
        Algorithm 2: Normalised Timestamp Score Algorithm
        논문 Algorithm 2 정확한 구현
        
        Args:
            visited_nodes: SVN (Set of visited nodes)
        
        Returns:
            {node: NTS_score} 딕셔너리
        """
        # Line 1: foreach v_i ∈ SVN do
        sts = {}  # Set of Timestamp Scores (unnormalized)
        
        for v_i in visited_nodes:
            # Line 2-6: Compute in-degree timestamp range
            in_timestamps = []
            for _, row in self.tx_df.iterrows():
                if row['to_address'] == v_i:  # in-degree
                    in_timestamps.append(row['timestamp'])
            
            if len(in_timestamps) > 0:
                max_ts_in = max(in_timestamps)  # Line 4
                min_ts_in = min(in_timestamps)  # Line 6
                theta_in = max_ts_in - min_ts_in  # Line 12: θ_in(v_i)
            else:
                theta_in = 0.0
            
            # Line 7-11: Compute out-degree timestamp range
            out_timestamps = []
            for _, row in self.tx_df.iterrows():
                if row['from_address'] == v_i:  # out-degree
                    out_timestamps.append(row['timestamp'])
            
            if len(out_timestamps) > 0:
                max_ts_out = max(out_timestamps)  # Line 9
                min_ts_out = min(out_timestamps)  # Line 11
                theta_out = max_ts_out - min_ts_out  # Line 13: θ_out(v_i)
            else:
                theta_out = 0.0
            
            # Line 14: θ(v_i) ← |θ_out(v_i) - θ_in(v_i)|
            theta_v_i = abs(theta_out - theta_in)
            
            # Line 15: Add θ(v_i) to STS
            sts[v_i] = theta_v_i
        
        # Line 16-17: Normalize all timestamp scores
        nts = {}
        if len(sts) > 0:
            min_theta = min(sts.values())
            max_theta = max(sts.values())
            
            if max_theta == min_theta:
                # 모든 값이 같으면 0으로 설정
                for v_i in sts:
                    nts[v_i] = 0.0
            else:
                # Min-max normalization
                for v_i in sts:
                    nts[v_i] = (sts[v_i] - min_theta) / (max_theta - min_theta)
        
        return nts
    
    def compute_algorithm3_nws(self, visited_nodes: Set[str]) -> Dict[str, float]:
        """
        Algorithm 3: Normalised Weight Score Algorithm
        논문 Algorithm 3 정확한 구현
        
        Args:
            visited_nodes: SVN (Set of visited nodes)
        
        Returns:
            {node: NWS_score} 딕셔너리
        """
        # Line 1: foreach v_i ∈ SVN do
        sws = {}  # Set of Weight Scores (unnormalized)
        
        for v_i in visited_nodes:
            # Line 2: Sum all in-degree weights
            omega_in_v_i = 0.0
            for _, row in self.tx_df.iterrows():
                if row['to_address'] == v_i:  # in-degree
                    omega_in_v_i += row['value']
            
            # Line 3: Sum all out-degree weights
            omega_out_v_i = 0.0
            for _, row in self.tx_df.iterrows():
                if row['from_address'] == v_i:  # out-degree
                    omega_out_v_i += row['value']
            
            # Line 4: ω(v_i) = |ω_in(v_i) - ω_out(v_i)|
            omega_v_i = abs(omega_out_v_i - omega_in_v_i)
            
            # Line 5: Add ω(v_i) to SWS
            sws[v_i] = omega_v_i
        
        # Line 6-7: Normalize all weight scores
        nws = {}
        if len(sws) > 0:
            min_omega = min(sws.values())
            max_omega = max(sws.values())
            
            if max_omega == min_omega:
                # 모든 값이 같으면 0으로 설정
                for v_i in sws:
                    nws[v_i] = 0.0
            else:
                # Min-max normalization
                for v_i in sws:
                    nws[v_i] = (sws[v_i] - min_omega) / (max_omega - min_omega)
        
        return nws
    
    def compute_nts(self, source_node: str, visited_nodes: Set[str]) -> float:
        """
        특정 노드의 NTS 점수 계산 (호환성 유지)
        """
        nts_dict = self.compute_algorithm2_nts(visited_nodes)
        return nts_dict.get(source_node, 0.0)
    
    def compute_nws(self, source_node: str, visited_nodes: Set[str]) -> float:
        """
        특정 노드의 NWS 점수 계산 (호환성 유지)
        """
        nws_dict = self.compute_algorithm3_nws(visited_nodes)
        return nws_dict.get(source_node, 0.0)
    
    def compute_scores(self, source_node: str) -> tuple:
        """
        특정 노드에 대해 NTS와 NWS 동시 계산
        
        Args:
            source_node: 시작 노드
        
        Returns:
            (NTS, NWS) 튜플
        """
        if source_node not in self.ppr_visited_nodes:
            return 0.0, 0.0
        
        visited_nodes = self.ppr_visited_nodes[source_node]
        nts = self.compute_nts(source_node, visited_nodes)
        nws = self.compute_nws(source_node, visited_nodes)
        
        return nts, nws
    
    def compute_all_scores(self) -> pd.DataFrame:
        """
        모든 노드에 대해 NTS와 NWS 계산
        
        Returns:
            DataFrame with columns: [node, nts, nws]
        """
        scores = []
        
        # 모든 visited nodes 수집
        all_visited_nodes = set()
        for visited_set in self.ppr_visited_nodes.values():
            all_visited_nodes.update(visited_set)
        
        if len(all_visited_nodes) == 0:
            # PPR 결과가 없으면 빈 DataFrame 반환
            return pd.DataFrame(columns=['node', 'nts', 'nws'])
        
        # Algorithm 2 & 3 실행
        nts_dict = self.compute_algorithm2_nts(all_visited_nodes)
        nws_dict = self.compute_algorithm3_nws(all_visited_nodes)
        
        # 모든 그래프 노드에 대해 점수 매핑
        for node in self.graph.nodes:
            nts = nts_dict.get(node, 0.0)
            nws = nws_dict.get(node, 0.0)
            
            scores.append({
                'node': node,
                'nts': nts,
                'nws': nws
            })
        
        return pd.DataFrame(scores)


if __name__ == "__main__":
    # 테스트 코드
    from graph import generate_dummy_data
    from ppr import PersonalizedPageRank
    
    print("Generating dummy graph...")
    graph_obj = generate_dummy_data(n_nodes=30, n_transactions=100)
    graph = graph_obj.build_graph()
    
    print("Computing PPR...")
    ppr = PersonalizedPageRank(graph)
    
    # 일부 노드에 대해 PPR 계산
    sample_nodes = graph_obj.nodes[:5]
    ppr_results = {}
    for node in sample_nodes:
        _, svn = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
    
    print("\nComputing NTS & NWS...")
    scorer = NormalizedScorer(graph_obj, ppr_results)
    
    scores_df = scorer.compute_all_scores()
    print(f"\nScores for sample nodes:\n{scores_df[scores_df['node'].isin(sample_nodes)]}")
