"""
Multi-source Personalized PageRank (PPR) 모듈
논문의 Algorithm 1을 정확히 구현
"""
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Set
from tqdm import tqdm
import random


class PersonalizedPageRank:
    """
    Multi-source Personalized PageRank 구현
    논문 Algorithm 1: Residual-based PPR + Random Walk
    """
    
    def __init__(self, graph: nx.DiGraph, alpha: float = 0.5, 
                 epsilon: float = 0.01, p_f: float = 1.0):
        """
        Args:
            graph: NetworkX 그래프 객체
            alpha: 랜덤 워크 가중치 인자, 논문의 α (논문: 0.5로 설정)
            epsilon: PPR 상대적 정확도 보장, 논문의 ε
            p_f: 실패 확률, 논문의 p_f (논문: 1.0으로 설정)
        
        Note: 논문 C장 Hyperparameter Tuning에서 명시:
            - α = 0.5 (balanced exploration/local reinforcement)
            - ρ = 1 (restart probability)
            - p_f = 1 (failure probability)
        """
        self.graph = graph
        self.alpha = alpha  # proportion allocation
        self.epsilon = epsilon  # PPR relative accuracy guarantee
        self.p_f = p_f  # failure probability
        
        # 노드 리스트 및 인덱스 매핑
        self.nodes = list(graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self.n = len(self.nodes)
        
        # 노드 차수 저장 (in-degree, out-degree)
        self._compute_degrees()
    
    def _compute_degrees(self):
        """각 노드의 in-degree와 out-degree 계산"""
        self.in_degrees = {}
        self.out_degrees = {}
        self.degrees = {}
        
        for node in self.nodes:
            self.in_degrees[node] = self.graph.in_degree(node)
            self.out_degrees[node] = self.graph.out_degree(node)
            self.degrees[node] = self.out_degrees[node]
    
    def get_source_nodes(self) -> Set[str]:
        """
        Line 2-4: in-degree가 0인 노드를 source nodes로 식별
        논문: foreach v_i ∈ V do
               if in-degree (v_i) = 0 then
               Add v_i to the list of source nodes
        
        Returns:
            Set of source nodes (in-degree = 0)
        """
        source_nodes = set()
        for node in self.nodes:
            if self.in_degrees[node] == 0:
                source_nodes.add(node)
        return source_nodes
    
    def _compute_work_count(self, source_node: str) -> float:
        """
        K(s) 계산: 논문 Line 9
        K(s) ← ((2/3 * ε + 2) * d(s) * log(2 / p_f)) / (ε^2 * α * (1 - α))
        
        Args:
            source_node: 시작 노드 s
        
        Returns:
            K(s): 워크 카운트 (예산)
        """
        d_s = self.degrees[source_node] + 1  # 차수 (자기 포함)
        
        numerator = (2/3 * self.epsilon + 2) * d_s * np.log(2 / self.p_f)
        denominator = self.epsilon ** 2 * self.alpha * (1 - self.alpha)
        
        K_s = numerator / denominator
        
        return max(1, int(K_s))  # 최소 1로 보장
    
    def compute_single_source_ppr(self, source_node: str, 
                                 verbose: bool = False) -> Tuple[np.ndarray, Set[str], List[str]]:
        """
        단일 소스 PPR 계산 - 논문 Algorithm 1
        
        Args:
            source_node: 시작 노드 s
            verbose: 진행 상황 출력 여부
        
        Returns:
            SPS: PPR 점수 배열
            SVN: 방문된 노드 집합
        """
        if source_node not in self.graph:
            raise ValueError(f"Source node {source_node} not in graph")
        
        # Line 9: 워크 카운트 계산
        K_s = self._compute_work_count(source_node)
        
        # 초기화: Line 5-7
        # r(s, u): 노드 u의 잔여 점수
        # π°(s, u): 노드 u의 임시 점수
        residual = {}  # r(s, v_i)
        temp_score = {}  # π°(s, v_i)
        final_score = {}  # π̂(s, v_i)
        
        # 초기화
        for node in self.nodes:
            residual[node] = 0.0
            temp_score[node] = 0.0
            final_score[node] = 0.0
        
        # Line 5: r(s, s) ← 1
        residual[source_node] = 1.0
        
        # N_s^m: 초기 이웃 노드 집합 (source_node 포함)
        neighbors = set([source_node])
        
        # Iterative Residual Pushing (Line 10-14)
        max_iterations = 100  # 무한 루프 방지
        iteration = 0
        
        while iteration < max_iterations:
            pushed = False
            
            for node in list(residual.keys()):
                d_i = max(1, self.degrees[node])  # 차수
                threshold = d_i / (self.alpha * K_s)
                
                # Line 10: while exists u_i such that r(s, u_i) > d(u_i) / (α * K(s))
                if residual[node] > threshold and self.degrees[node] > 0:
                    pushed = True
                    
                    # Line 11-14: 잔여 푸싱
                    out_neighbors = list(self.graph.successors(node))
                    
                    if len(out_neighbors) > 0:
                        push_amount = (1 - self.alpha) * residual[node] / d_i
                        
                        for neighbor in out_neighbors:
                            # Line 12: r(s, v_j) ← r(s, v_j) + (1 - α) * r(s, u_i) / d(u_i)
                            residual[neighbor] = residual.get(neighbor, 0.0) + push_amount
                            neighbors.add(neighbor)
                        
                        # Line 13: π°(s, u_i) ← π°(s, u_i) + α * r(s, u_i)
                        temp_score[node] += self.alpha * residual[node]
                    
                    # Line 14: r(s, u_i) ← 0
                    residual[node] = 0.0
            
            if not pushed:
                break
            
            iteration += 1
        
        # Line 15-18: 임시 점수를 최종 점수로 복사
        for node in neighbors:
            # Line 16: π̂(s, v_i) ← π°(s, v_i)
            final_score[node] = temp_score[node]
        
        # Line 19-26: Random Walk 시뮬레이션
        for node in list(neighbors):
            r_value = residual[node]
            if r_value > 0:
                num_walks = int(r_value * K_s)
                
                for _ in range(num_walks):
                    # 랜덤 워크 시뮬레이션
                    current = node
                    steps = 0
                    max_steps = 100  # 무한 루프 방지
                    
                    while steps < max_steps:
                        if random.random() < self.alpha:
                            # 텔레포트
                            break
                        
                        out_neighbors = list(self.graph.successors(current))
                        if len(out_neighbors) == 0:
                            break
                        
                        current = random.choice(out_neighbors)
                        steps += 1
                    
                    if current in neighbors:
                        # Line 24: π̂(s, v_i) ← π̂(s, v_i) + 1 / K(s)
                        final_score[current] += 1.0 / K_s
        
        # SPS와 SVN 생성
        # sps는 전체 노드에 대한 PPR 점수 배열
        sps = np.array([final_score.get(node, 0.0) for node in self.nodes])
        
        # SVN: 임계값 이상인 노드들
        threshold = np.max(sps) * 0.001  # 최대값의 0.1% 이상
        svn = {node for node in neighbors if final_score.get(node, 0.0) >= threshold}
        
        # 노드 정보와 함께 반환하여 매핑 가능하도록
        return sps, svn, self.nodes
    
    def compute_multi_source_ppr(self, source_nodes: List[str],
                                verbose: bool = False) -> Dict[str, Tuple[np.ndarray, Set[str]]]:
        """
        Multi-source PPR 계산
        여러 시작 노드에 대해 각각 PPR 계산
        
        Args:
            source_nodes: 시작 노드 리스트
            verbose: 진행 상황 출력 여부
        
        Returns:
            Dictionary: {source_node: (SPS, SVN)}
        """
        results = {}
        
        for source in tqdm(source_nodes, desc="Computing multi-source PPR"):
            sps, svn = self.compute_single_source_ppr(source, verbose=False)
            results[source] = (sps, svn)
        
        return results
    
    def compute_ppr_for_all_nodes(self, verbose: bool = False) -> Dict[str, np.ndarray]:
        """
        모든 노드를 시작점으로 PPR 계산
        (실제로는 각 노드의 PPR 점수만 저장)
        
        Args:
            verbose: 진행 상황 출력 여부
        
        Returns:
            Dictionary: {node: PPR_scores_array}
        """
        all_ppr_scores = {}
        
        for node in tqdm(self.nodes, desc="Computing PPR for all nodes"):
            sps, _ = self.compute_single_source_ppr(node, verbose=False)
            all_ppr_scores[node] = sps
        
        return all_ppr_scores


if __name__ == "__main__":
    # 테스트 코드
    from graph import generate_dummy_data
    
    print("Generating dummy graph...")
    graph_obj = generate_dummy_data(n_nodes=30, n_transactions=100)
    graph = graph_obj.build_graph()
    
    print("\nComputing PPR...")
    ppr = PersonalizedPageRank(graph, damping_factor=0.85)
    
    # 특정 노드에 대한 PPR 계산
    test_node = graph_obj.nodes[0]
    print(f"\nComputing PPR for node: {test_node}")
    sps, svn = ppr.compute_single_source_ppr(test_node, verbose=True)
    
    print(f"\nResults:")
    print(f"Number of visited nodes: {len(svn)}")
    print(f"Top 5 PPR scores:")
    top_indices = np.argsort(sps)[-5:][::-1]
    for idx in top_indices:
        print(f"  {ppr.nodes[idx]}: {sps[idx]:.6f}")
