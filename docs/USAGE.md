# MPOCryptoML 사용 가이드

## 프로젝트 개요

MPOCryptoML은 다중 패턴 암호화폐 이상거래 탐지를 위한 논문 기반 구현입니다.

## 빠른 시작

### 1. 기본 사용 (더미 데이터)

```bash
cd src
python main.py
```

### 2. 간단한 예제 실행

```bash
# 전체 파이프라인 테스트
python examples/test_pipeline.py

# 단계별 설명이 있는 간단 예제
python examples/quick_start.py
```

### 3. 고급 옵션

```bash
# 더 큰 데이터셋 사용
python src/main.py --n-nodes 200 --n-transactions 1000

# 시각화 생성
python src/main.py --visualize

# 전체 모드 (모든 노드 처리)
python src/main.py --no-test-mode
```

## 코드 구조

```
MPO_final/
├── src/
│   ├── graph.py              # 그래프 구조 및 더미 데이터 생성
│   ├── ppr.py                # Multi-source Personalized PageRank
│   ├── scoring.py            # NTS & NWS 계산
│   ├── anomaly_detector.py   # Anomaly Score 계산 및 평가
│   └── main.py              # 메인 파이프라인
├── examples/
│   ├── quick_start.py       # 단계별 예제
│   └── test_pipeline.py     # 간단 테스트
└── notebooks/
    └── 01_exploration.ipynb  # 데이터 탐색
```

## 논문 파이프라인

### Step 1: Multi-source Personalized PageRank (PPR)

```python
from src import generate_dummy_data, PersonalizedPageRank

graph_obj = generate_dummy_data(n_nodes=100)
graph = graph_obj.build_graph()

ppr = PersonalizedPageRank(graph)
sps, svn = ppr.compute_single_source_ppr(node)
# sps: PPR 점수 배열
# svn: 방문된 노드 집합
```

### Step 2: NTS & NWS 계산

```python
from src.scoring import NormalizedScorer

scorer = NormalizedScorer(graph_obj, ppr_results)
feature_scores = scorer.compute_all_scores()
# Returns: DataFrame with 'nts' and 'nws' columns
```

### Step 3: Logistic Regression

```python
from src.anomaly_detector import MPOCryptoMLDetector

detector = MPOCryptoMLDetector(
    ppr_scores=ppr_scores,
    feature_scores=feature_scores,
    labels=labels
)
detector.train_logistic_regression()
```

### Step 4: Anomaly Score 계산

```python
detector.compute_pattern_scores()
anomaly_scores = detector.compute_anomaly_scores()
# Anomaly Score: σ(vi) = π(vi) / F(θ,ω)(vi)
```

### Step 5: 평가

```python
# Precision@K, Recall@K
results = detector.evaluate_precision_at_k(k=10)

# 전체 결과 DataFrame
results_df = detector.get_results_df()
```

## 파라미터 설명

### generate_dummy_data

- `n_nodes`: 노드(주소) 개수
- `n_transactions`: 거래 개수
- `anomaly_ratio`: 사기 노드 비율 (0.0-1.0)
- `seed`: 랜덤 시드

### PersonalizedPageRank

- `damping_factor`: PageRank damping factor (기본값: 0.85)
- `max_iter`: 최대 반복 횟수
- `tol`: 수렴 허용 오차

## 데이터 형식

### 그래프 데이터 구조

```python
{
    "tx_hash": "0x...",
    "from_address": "0x...",
    "to_address": "0x...",
    "value": 0.5,  # ETH 단위
    "timestamp": 1234567890.0,  # Unix timestamp
    "label": 1  # 0=정상, 1=사기
}
```

### 더미 데이터 생성

현재 구현은 더미 데이터를 사용합니다. 실제 데이터를 사용하려면:

1. `CryptoTransactionGraph` 클래스를 사용하여 데이터 로드
2. 필요시 timestamp 필드 보완 (Etherscan API 사용)
3. 라벨 매핑 설정

## 평가 지표

- **Precision@K**: 상위 K개 중 실제 사기 비율
- **Recall@K**: 전체 사기 중 상위 K개에서 발견된 비율
- **F1-score**: Precision과 Recall의 조화평균
- **AUC**: ROC 곡선 아래 면적
- **Accuracy**: 전체 정확도

## 주의사항

1. 더미 데이터는 테스트 목적이므로 실제 성능 평가에는 적합하지 않음
2. 실제 데이터 사용 시 timestamp 필드 보완 필요
3. 대규모 그래프의 경우 PPR 계산에 시간이 오래 걸릴 수 있음
4. 현재는 테스트 모드로 일부 노드만 처리 (전체 처리 시 `--no-test-mode` 옵션 사용)

## 문제 해결

### Q: "Anomalies: 0"이 나오는 경우

A: 그래프 생성 시 라벨이 제대로 설정되지 않은 경우입니다. 수정되었습니다.

### Q: PPR 계산이 너무 느린 경우

A: 테스트 모드로 일부 노드만 처리하거나, 더 작은 데이터셋을 사용하세요.

### Q: 평가 점수가 낮은 경우

A: 더미 데이터는 의도적으로 간단하게 구성되어 있어 실제 성능과 다를 수 있습니다.

## 참고 자료

- 논문: MPOCryptoML (원본 논문)
- Ethereum Fraud Detection: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset
- Elliptic++: Bitcoin blockchain 데이터

## 라이선스

MIT License
