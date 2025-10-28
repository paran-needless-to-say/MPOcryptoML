"""
Baseline 모델 구현
논문과 비교하기 위한 기본 모델들

논문 baseline:
  - XGBoost (단순 feature 기반)
  - Isolation Forest
  - LOF (Local Outlier Factor)
  - Random Forest
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost not installed. XGBoost baseline will be skipped.")
from typing import Dict, List, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)


class BaselineDetector:
    """
    Baseline 모델들
    """
    
    def __init__(self, feature_scores: pd.DataFrame, labels: Dict[str, int]):
        """
        Args:
            feature_scores: NTS, NWS를 포함한 DataFrame
            labels: {node: label} 형태의 딕셔너리
        """
        self.feature_scores = feature_scores
        self.labels = labels
        
        # Train/Test split
        from sklearn.model_selection import train_test_split
        
        X = feature_scores[['nts', 'nws']].values
        y = np.array([labels[node] for node in feature_scores['node']])
        
        node_list = feature_scores['node'].values
        train_indices, test_indices, y_train, y_test = train_test_split(
            range(len(X)), y, test_size=0.2, 
            random_state=42, stratify=y
        )
        
        self.X_train = X[train_indices]
        self.X_test = X[test_indices]
        self.y_train = y_train
        self.y_test = y_test
        self.train_nodes = [node_list[i] for i in train_indices]
        self.test_nodes = [node_list[i] for i in test_indices]
        
        self.models = {}
        self.scores = {}
    
    def train_isolation_forest(self):
        """Isolation Forest"""
        print("\n[Baseline] Training Isolation Forest...")
        
        model = IsolationForest(
            contamination=0.1,  # 10% anomalies
            random_state=42,
            n_estimators=100
        )
        model.fit(self.X_train)
        
        pred_train = model.predict(self.X_train)
        pred_test = model.predict(self.X_test)
        
        # Convert -1/1 to 1/0
        pred_train_binary = (pred_train == 1).astype(int)
        pred_test_binary = (pred_test == 1).astype(int)
        
        self.models['IsolationForest'] = model
        
        # Evaluate
        train_auc = roc_auc_score(self.y_train, 1 - pred_train_binary)
        test_auc = roc_auc_score(self.y_test, 1 - pred_test_binary)
        
        self.scores['IsolationForest'] = {
            'train_auc': train_auc,
            'test_auc': test_auc
        }
        
        print(f"  Train AUC: {train_auc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        
        return model
    
    def train_lof(self):
        """Local Outlier Factor"""
        print("\n[Baseline] Training LOF...")
        
        model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )
        model.fit(self.X_train)
        
        pred_train = model.predict(self.X_train)
        pred_test = model.predict(self.X_test)
        
        pred_train_binary = (pred_train == 1).astype(int)
        pred_test_binary = (pred_test == 1).astype(int)
        
        self.models['LOF'] = model
        
        train_auc = roc_auc_score(self.y_train, 1 - pred_train_binary)
        test_auc = roc_auc_score(self.y_test, 1 - pred_test_binary)
        
        self.scores['LOF'] = {
            'train_auc': train_auc,
            'test_auc': test_auc
        }
        
        print(f"  Train AUC: {train_auc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        
        return model
    
    def train_xgboost(self):
        """XGBoost (논문 baseline)"""
        if not XGB_AVAILABLE:
            print("\n[Baseline] Skipping XGBoost (not installed)")
            return None
        
        print("\n[Baseline] Training XGBoost...")
        
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(self.X_train, self.y_train)
        
        pred_train = model.predict(self.X_train)
        pred_test = model.predict(self.X_test)
        proba_train = model.predict_proba(self.X_train)[:, 1]
        proba_test = model.predict_proba(self.X_test)[:, 1]
        
        self.models['XGBoost'] = model
        
        train_auc = roc_auc_score(self.y_train, proba_train)
        test_auc = roc_auc_score(self.y_test, proba_test)
        
        self.scores['XGBoost'] = {
            'train_auc': train_auc,
            'test_auc': test_auc
        }
        
        print(f"  Train AUC: {train_auc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        
        return model
    
    def train_random_forest(self):
        """Random Forest"""
        print("\n[Baseline] Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train)
        
        proba_train = model.predict_proba(self.X_train)[:, 1]
        proba_test = model.predict_proba(self.X_test)[:, 1]
        
        self.models['RandomForest'] = model
        
        train_auc = roc_auc_score(self.y_train, proba_train)
        test_auc = roc_auc_score(self.y_test, proba_test)
        
        self.scores['RandomForest'] = {
            'train_auc': train_auc,
            'test_auc': test_auc
        }
        
        print(f"  Train AUC: {train_auc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        
        return model
    
    def train_all(self):
        """모든 baseline 학습"""
        print("\n" + "="*70)
        print("🎯 Training All Baselines")
        print("="*70)
        
        self.train_isolation_forest()
        self.train_lof()
        self.train_random_forest()
        self.train_xgboost()
        
        print("\n" + "="*70)
        print("✅ All Baselines Trained")
        print("="*70)
        
        # 결과 요약
        print("\n📊 Results Summary:")
        for name, scores in self.scores.items():
            print(f"  {name:20s}: Test AUC = {scores['test_auc']:.4f}")
        
        return self.models
    
    def get_comparison_df(self) -> pd.DataFrame:
        """비교 결과를 DataFrame으로 반환"""
        results = []
        
        for name, scores in self.scores.items():
            results.append({
                'Model': name,
                'Train_AUC': scores['train_auc'],
                'Test_AUC': scores['test_auc']
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # 테스트
    from graph import generate_dummy_data
    from ppr import PersonalizedPageRank
    from scoring import NormalizedScorer
    
    print("Testing Baselines...")
    
    # 1. 데이터 생성
    graph_obj = generate_dummy_data(n_nodes=100, n_transactions=500)
    graph = graph_obj.build_graph()
    
    # 2. PPR
    ppr = PersonalizedPageRank(graph, alpha=0.5, epsilon=0.01, p_f=1.0)
    ppr_results = {}
    for node in list(graph_obj.nodes)[:20]:
        sps, svn, nodes_list = ppr.compute_single_source_ppr(node)
        ppr_results[node] = svn
    
    # 3. Features
    scorer = NormalizedScorer(graph_obj, ppr_results)
    feature_scores = scorer.compute_all_scores()
    
    # 4. Baseline
    baseline = BaselineDetector(feature_scores, graph_obj.node_labels)
    models = baseline.train_all()
    
    # 5. 비교
    print("\n" + baseline.get_comparison_df().to_string())
