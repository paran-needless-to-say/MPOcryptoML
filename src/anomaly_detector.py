"""
Anomaly Score 계산 및 평가 모듈
논문의 Step 3-4: Logistic Regression 및 Anomaly Score 계산
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, roc_curve, auc
)
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


class MPOCryptoMLDetector:
    """
    MPOCryptoML 논문의 전체 파이프라인 구현
    """
    
    def __init__(self, ppr_scores: Dict[str, np.ndarray], 
                 feature_scores: pd.DataFrame,
                 labels: Dict[str, int]):
        """
        Args:
            ppr_scores: {node: PPR_scores_array} 형태의 딕셔너리
            feature_scores: NTS와 NWS를 포함한 DataFrame
            labels: {node: label} 형태의 딕셔너리 (0=정상, 1=사기)
        """
        self.ppr_scores = ppr_scores
        self.feature_scores = feature_scores
        self.labels = labels
        
        # Features: [NTS, NWS]
        self.X = feature_scores[['nts', 'nws']].values
        self.y = np.array([labels[node] for node in feature_scores['node']])
        
        self.model = None
        self.pattern_scores = {}  # F(θ,ω)(vi)
        self.anomaly_scores = {}  # σ(vi)
    
    def train_logistic_regression(self, test_size: float = 0.2, 
                                  random_state: int = 42):
        """
        Logistic Regression 모델 학습
        논문의 Step 3
        
        Args:
            test_size: 테스트 세트 비율
            random_state: 랜덤 시드
        """
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, 
            random_state=random_state, stratify=self.y
        )
        
        # 모델 학습
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\n=== Logistic Regression Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        return self.model
    
    def compute_pattern_scores(self):
        """
        패턴 점수 F(θ,ω)(vi) 계산
        학습된 모델을 사용하여 각 노드의 패턴 점수 예측
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_logistic_regression() first.")
        
        # 각 노드의 패턴 점수 계산 (확률값)
        pattern_probabilities = self.model.predict_proba(self.X)[:, 1]
        
        for idx, node in enumerate(self.feature_scores['node']):
            self.pattern_scores[node] = pattern_probabilities[idx]
        
        return self.pattern_scores
    
    def compute_anomaly_scores(self):
        """
        Anomaly Score σ(vi) 계산
        논문의 Step 4: σ(vi) = π(vi) / F(θ,ω)(vi)
        
        π(vi)는 모든 source 노드들로부터 받은 PPR 점수의 합계
        
        Returns:
            Dict: {node: anomaly_score}
        """
        if not self.pattern_scores:
            self.compute_pattern_scores()
        
        # PPR은 {source: (sps, svn, all_nodes_list)} 형태로 저장되어야 함
        # 하지만 현재는 ppr_scores_dict에 sps만 저장
        # 따라서 PPR 클래스에서 반환된 all_nodes_list 정보가 필요함
        
        # 임시로 첫 번째 source의 ppr_array 길이를 확인
        first_source = list(self.ppr_scores.keys())[0]
        first_ppr_array = self.ppr_scores[first_source]
        
        # ppr_array는 전체 그래프 노드에 대한 점수
        # feature_scores의 노드들은 방문된 노드만 포함하므로
        # 노드 이름으로 직접 매핑 필요
        
        for node in self.feature_scores['node']:
            # 각 source로부터 받은 PPR 점수 합계
            # π(vi) = Σ_{s in sources} π̂(s, vi)
            
            # ppr_scores는 {source: sps_array} 형태
            # sps_array[i]는 전체 그래프의 특정 노드의 점수
            # 하지만 어떤 노드인지 알기 위해 노드 인덱스 정보 필요
            
            # 현재는 인덱스를 노드 이름으로 매핑 불가능
            # 임시로 feature_scores의 idx 사용
            idx = list(self.feature_scores['node']).index(node)
            
            ppr_sum = 0.0
            for source_node, ppr_array in self.ppr_scores.items():
                if idx < len(ppr_array):
                    ppr_sum += ppr_array[idx]
            
            # Anomaly Score = PPR 점수 합계 / 패턴 점수
            pattern_score = self.pattern_scores[node]
            if pattern_score < 1e-10:
                self.anomaly_scores[node] = float('inf')
            else:
                self.anomaly_scores[node] = ppr_sum / pattern_score
        
        return self.anomaly_scores
    
    def evaluate_precision_at_k(self, k: int = 10) -> Dict[str, float]:
        """
        Precision@K, Recall@K 계산
        논문의 평가 지표
        
        Args:
            k: 상위 k개
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.anomaly_scores:
            self.compute_anomaly_scores()
        
        # Anomaly score 기준 정렬
        sorted_nodes = sorted(
            self.anomaly_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 상위 k개
        top_k_nodes = [node for node, _ in sorted_nodes[:k]]
        
        # 실제 라벨 확인
        top_k_labels = [self.labels[node] for node in top_k_nodes]
        
        # Precision@K
        precision_k = sum(top_k_labels) / k
        
        # Recall@K
        total_anomalies = sum(self.labels.values())
        recall_k = sum(top_k_labels) / total_anomalies if total_anomalies > 0 else 0
        
        # F1@K
        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0
        
        results = {
            f'precision@{k}': precision_k,
            f'recall@{k}': recall_k,
            f'f1@{k}': f1_k
        }
        
        return results
    
    def plot_roc_curve(self, save_path: str = None):
        """ROC 곡선 시각화"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        y_pred_proba = self.model.predict_proba(self.X)[:, 1]
        fpr, tpr, _ = roc_curve(self.y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def get_results_df(self) -> pd.DataFrame:
        """
        모든 결과를 포함한 DataFrame 반환
        """
        if not self.anomaly_scores:
            self.compute_anomaly_scores()
        
        results = []
        for node in self.feature_scores['node']:
            results.append({
                'node': node,
                'label': self.labels[node],
                'nts': self.feature_scores[self.feature_scores['node'] == node]['nts'].values[0],
                'nws': self.feature_scores[self.feature_scores['node'] == node]['nws'].values[0],
                'pattern_score': self.pattern_scores[node],
                'anomaly_score': self.anomaly_scores[node]
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # 테스트 코드
    from graph import generate_dummy_data
    from ppr import PersonalizedPageRank
    from scoring import NormalizedScorer
    
    print("=== MPOCryptoML Pipeline Test ===")
    
    # 1. 데이터 생성
    print("\n1. Generating dummy data...")
    graph_obj = generate_dummy_data(n_nodes=50, n_transactions=200)
    graph = graph_obj.build_graph()
    
    # 2. PPR 계산
    print("\n2. Computing PPR...")
    ppr = PersonalizedPageRank(graph, alpha=0.5, epsilon=0.01, p_f=1.0)
    ppr_results = {}
    for node in graph_obj.nodes[:20]:  # 샘플링
        sps, svn, nodes_list = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
    
    # 3. Feature 계산
    print("\n3. Computing NTS & NWS...")
    scorer = NormalizedScorer(graph_obj, ppr_results)
    feature_scores = scorer.compute_all_scores()
    
    # 4. Anomaly Detection
    print("\n4. Training model and computing anomaly scores...")
    detector = MPOCryptoMLDetector(ppr_results, feature_scores, graph_obj.node_labels)
    detector.train_logistic_regression()
    
    # 5. 평가
    print("\n5. Evaluation...")
    results = detector.evaluate_precision_at_k(k=10)
    print(results)
    
    print("\n=== Test Completed ===")
