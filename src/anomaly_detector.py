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
    
    def __init__(self, ppr_scores: Dict[str, Tuple[np.ndarray, List[str]]], 
                 feature_scores: pd.DataFrame,
                 labels: Dict[str, int]):
        """
        Args:
            ppr_scores: {source: (sps_array, all_nodes_list)} 형태의 딕셔너리
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
        # 훈련/테스트 분할 (indices 저장)
        from sklearn.model_selection import train_test_split
        import pandas as pd
        
        # 노드 리스트와 함께 분할
        node_list = self.feature_scores['node'].values
        
        train_indices, test_indices, y_train, y_test = train_test_split(
            range(len(self.X)), self.y, test_size=test_size, 
            random_state=random_state, stratify=self.y
        )
        
        X_train = self.X[train_indices]
        X_test = self.X[test_indices]
        
        # train/test split 정보 저장
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.train_nodes = [node_list[i] for i in train_indices]
        self.test_nodes = [node_list[i] for i in test_indices]
        
        # 모델 학습 (클래스 불균형 해결)
        # class_weight='balanced': 자동으로 불균형 가중치 적용
        self.model = LogisticRegression(
            max_iter=1000, 
            random_state=random_state,
            class_weight='balanced',  # ✅ 불균형 해결 추가!
            solver='lbfgs'  # 안정적인 solver
        )
        self.model.fit(X_train, y_train)
        
        # 예측 및 평가 (test set만)
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\n=== Logistic Regression Evaluation ===")
        print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"F1-score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        return self.model
    
    def compute_pattern_scores(self):
        """
        패턴 점수 F(θ,ω)(vi) 계산
        학습된 모델을 사용하여 각 노드의 패턴 점수 예측
        
        수정: 전체 데이터가 아닌, Train 데이터만 사용하여 데이터 누수 방지
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_logistic_regression() first.")
        
        # Train 데이터만으로 패턴 점수 계산
        X_train = self.X[self.train_indices]
        pattern_probabilities = self.model.predict_proba(X_train)[:, 1]
        
        # Train 노드만 저장
        for idx, node in enumerate(self.train_nodes):
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
        
        # 첫 번째 source의 노드 리스트로 인덱스 매핑 생성
        first_source = list(self.ppr_scores.keys())[0]
        ppr_array, all_nodes_list = self.ppr_scores[first_source]
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes_list)}
        
        # Train 데이터만 Anomaly Score 계산
        for node in self.train_nodes:
            # 각 source로부터 받은 PPR 점수 합계
            # π(vi) = Σ_{s in sources} π̂(s, vi)
            
            ppr_sum = 0.0
            for source_node, (ppr_array, all_nodes_list) in self.ppr_scores.items():
                if node in node_to_idx:
                    idx = node_to_idx[node]
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
        Train 데이터만 포함 (데이터 누수 방지)
        """
        if not self.anomaly_scores:
            self.compute_anomaly_scores()
        
        results = []
        # Train 노드만 포함
        for node in self.train_nodes:
            row = self.feature_scores[self.feature_scores['node'] == node]
            if len(row) > 0:
                nts = row['nts'].values[0]
                nws = row['nws'].values[0]
                results.append({
                    'node': node,
                    'label': self.labels.get(node, 0),
                    'nts': nts,
                    'nws': nws,
                    'pattern_score': self.pattern_scores.get(node, 0.0),
                    'anomaly_score': self.anomaly_scores.get(node, 0.0)
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
