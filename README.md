# MPOCryptoML 논문 재현 프로젝트

**Multi-Pattern Cryptocurrency Anomaly Detection**

📄 **논문 기반 구현**: 암호화폐 거래 그래프에서 이상 거래를 탐지하는 모델

## 🚀 빠른 시작

```bash
# 기본 실행 (더미 데이터)
cd src && python main.py

# 예제 실행
python examples/quick_start.py
```

자세한 사용법은 [USAGE.md](USAGE.md)를 참고하세요.

## 논문 실험 환경

- **GPU:** 1 NVIDIA T4 Tensor Core GPU (AWS EC2 g4dn.2xlarge)
- **Memory:** 32 GiB
- **Storage:** 225 GB NVMe SSD
- **OS:** Ubuntu Linux
- **Python Version:** 3.8+
- **Main Libraries:** PyTorch, Scikit-learn

## 프로젝트 구조

```
MPO_final/
├── 📄 README.md                 # 프로젝트 메인 문서
├── 📄 requirements.txt           # Python 패키지 의존성
├── 📄 .gitignore                 # Git 무시 파일
│
├── 📁 docs/                      # 문서 폴더
│   ├── QUICKSTART.md             # 빠른 시작 가이드
│   ├── USAGE.md                  # 상세 사용법
│   └── DATA_GUIDE.md             # 데이터 가이드
│
├── 📁 src/                       # 소스 코드
│   ├── graph.py                 # 그래프 구조 + 더미 데이터
│   ├── ppr.py                   # Algorithm 1: Multi-source PPR
│   ├── scoring.py               # Algorithm 2, 3: NTS & NWS
│   ├── anomaly_detector.py      # Algorithm 4: Anomaly Detection
│   ├── main.py                  # 통합 파이프라인
│   └── load_real_data.py        # 실제 데이터 로드
│
├── 📁 examples/                  # 예제 스크립트
│   ├── quick_start.py            # 단계별 예제
│   └── test_pipeline.py         # 빠른 테스트
│
├── 📁 notebooks/                 # Jupyter 노트북
│   └── 01_exploration.ipynb     # 데이터 탐색
│
└── 📄 MPOCryptoML.pdf            # 논문 PDF
```

자세한 구조는 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 참고

## 설치 방법

### 1. Conda 환경 생성 (권장 - Python 3.11)

```bash
conda create -n mpo_env python=3.11 -y
conda activate mpo_env
pip install -r requirements.txt
```

### 2. 또는 venv 사용 (Python 3.11 이상 필요)

```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

**참고:** Python 3.13에서는 `opendatasets`가 작동하지 않습니다. Python 3.11을 사용하세요.

### 3. 캐글 API 설정

캐글에서 데이터를 다운로드하려면 API 토큰이 필요합니다:

1. https://www.kaggle.com/ 계정에 로그인
2. Settings > API > Create New Token
3. 다운로드된 `kaggle.json` 파일을 `~/.kaggle/` 디렉토리에 저장
4. 권한 설정: `chmod 600 ~/.kaggle/kaggle.json`

## 데이터 다운로드

캐글에서 데이터를 다운로드하는 방법:

```python
import opendatasets as od

# 캐글 데이터셋 URL
dataset_url = "https://www.kaggle.com/competitions/해당-데이터셋-URL"
od.download(dataset_url)
```

## 사용 방법

### 기본 사용 (더미 데이터)

```bash
cd src
python main.py
```

### 고급 옵션

```bash
# 더 많은 노드와 거래 사용
python main.py --n-nodes 200 --n-transactions 1000

# 시각화 생성
python main.py --visualize

# 전체 모드 (모든 노드 처리, 느림)
python main.py --no-test-mode
```

### 노트북에서 사용

```python
from src import generate_dummy_data, PersonalizedPageRank
from src.scoring import NormalizedScorer
from src.anomaly_detector import MPOCryptoMLDetector

# 1. 그래프 생성
graph_obj = generate_dummy_data(n_nodes=100, n_transactions=500)
graph = graph_obj.build_graph()

# 2. PPR 계산
ppr = PersonalizedPageRank(graph)
ppr_results = {}
for node in graph_obj.nodes[:20]:
    _, svn = ppr.compute_single_source_ppr(node)
    ppr_results[node] = svn

# 3. Feature 계산
scorer = NormalizedScorer(graph_obj, ppr_results)
feature_scores = scorer.compute_all_scores()

# 4. Anomaly Detection
detector = MPOCryptoMLDetector(
    ppr_scores={node: np.zeros(len(graph_obj.nodes)) for node in ppr_results},
    feature_scores=feature_scores,
    labels=graph_obj.node_labels
)
detector.train_logistic_regression()
detector.compute_anomaly_scores()

# 5. 결과 확인
results_df = detector.get_results_df()
print(results_df)
```

## 데이터 파이프라인

### Raw Data → Graph 변환

```python
from src.etherscan_parser import fetch_transactions_from_etherscan, convert_to_graph

# 1. Etherscan API에서 거래 수집
transactions = parse_etherscan_txlist(address, api_key)

# 2. Transaction 데이터를 그래프로 변환
# Address → Node (V)
# Transaction → Edge (E)
# Value → Weight (W)
# Timestamp → Time (T)
graph = convert_to_graph(transactions, labels)
```

### 알고리즘 구조

논문의 파이프라인:

1. **데이터 수집**: Etherscan API → Raw transactions
2. **그래프 변환**: Transactions → G=(V, E, W, T)
3. **Multi-source PPR**: SPS(PPR 점수)와 SVN(방문 노드 집합) 계산
4. **NTS & NWS**: Normalized Timestamp Score와 Weight Score 계산
5. **Logistic Regression**: 패턴 점수 F(θ,ω)(vi) 학습
6. **Anomaly Score**: σ(vi) = π(vi) / F(θ,ω)(vi) 계산

## 평가 지표

- Precision@K
- Recall@K
- F1-score
- Accuracy
- AUC

## 노트

- 현재는 더미 데이터로 구현되어 있음
- 실제 데이터 사용 시 timestamp 필드 보완 필요
