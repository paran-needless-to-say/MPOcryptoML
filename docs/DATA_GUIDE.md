# 📊 데이터 가이드

## 현재 상태

**현재 사용 중**: 더미 데이터 (합성 그래프)

- 논문 파이프라인 테스트를 위해 임의로 생성한 데이터
- `src/graph.py`의 `generate_dummy_data()` 함수로 생성
- 실제 암호화폐 거래가 아님

## 실제 데이터로 전환하기

### 옵션 1: Kaggle Ethereum Fraud Detection

**이미 다운로드됨**: `notebooks/ethereum-frauddetection-dataset/`

**문제점**:

- ❌ `timestamp` 필드 없음
- ✅ 주소별 집계된 특징만 있음
- ✅ 라벨(FLAG) 있음

**해결책**:

```python
from src.load_real_data import load_ethereum_fraud_detection, load_with_timestamp_simulation

# 1. 데이터 로드
graph = load_ethereum_fraud_detection()

# 2. Timestamp 시뮬레이션 추가
graph_with_ts = load_with_timestamp_simulation(graph, days_back=30)
```

### 옵션 2: Etherscan API로 실제 데이터 수집

**필요한 데이터 형식**:

```json
{
  "from_address": "0x...",
  "to_address": "0x...",
  "value": 0.5,
  "timestamp": 1234567890.0,
  "label": 1
}
```

**구현 예시**:

```python
import requests

def get_transactions(address, api_key):
    url = f"https://api.etherscan.io/api"
    params = {
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'page': 1,
        'offset': 100,
        'sort': 'asc',
        'apikey': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    # 여기서 거래 데이터 추출
    return data
```

### 옵션 3: Elliptic++ (Bitcoin)

**사용 가능한 데이터**:

- Bitcoin blockchain 데이터
- 노드 라벨 포함
- **확인 필요**: timestamp 포함 여부

### 옵션 4: CSV 파일 직접 제공

자신의 CSV 파일이 있다면:

```python
from src.load_real_data import load_from_csv

# CSV 컬럼 매핑
graph = load_from_csv(
    csv_path='your_data.csv',
    from_col='sender',
    to_col='receiver',
    value_col='amount',
    timestamp_col='block_timestamp',
    label_col='is_fraud'  # optional
)
```

## 더미 vs 실제 데이터

| 항목          | 더미 데이터          | 실제 데이터                  |
| ------------- | -------------------- | ---------------------------- |
| **Timestamp** | ✅ 있음 (시뮬레이션) | ❓ 없을 수 있음              |
| **거래 구조** | ✅ 올바름            | ✅ 실제 거래                 |
| **패턴**      | ✅ 복잡하지 않음     | ✅ 실제 패턴 (fan-in/out 등) |
| **성능 평가** | ⚠️ 의미 없음         | ✅ 실제 의미 있음            |
| **논문 재현** | ✅ 알고리즘 테스트   | ✅ 실제 평가                 |

## 현재 작업 환경

```python
# 현재 사용 중: 더미 데이터
from src.graph import generate_dummy_data

graph_obj = generate_dummy_data(
    n_nodes=100,
    n_transactions=500,
    anomaly_ratio=0.15,
    seed=42
)
```

## 실제 데이터로 전환 시 체크리스트

- [ ] Etherscan API 키 발급
- [ ] 거래 데이터 수집 (timestamp 포함)
- [ ] 라벨 매핑 (사기 주소 식별)
- [ ] 데이터 전처리
- [ ] CSV 또는 데이터베이스로 저장
- [ ] `load_from_csv()` 사용하여 로드

## 추천 작업 순서

1. ✅ **현재**: 더미 데이터로 알고리즘 구현 확인
2. 🔄 **다음**: 실제 데이터 수집/로드 모듈 작성
3. 🔄 **최종**: 실제 데이터로 전체 파이프라인 평가

## 참고 리소스

- [Etherscan API](https://docs.etherscan.io/)
- [QuickNode](https://www.quicknode.com/) - Blockchain data
- [Kaggle Ethereum Fraud Detection](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)
- [Elliptic++](https://github.com/elliptic-dataset/elliptic)

## 빠른 시작

### 더미 데이터 사용 (현재)

```bash
python examples/quick_start.py
```

### 실제 데이터 로드 (준비됨)

```bash
python -c "from src.load_real_data import load_ethereum_fraud_detection; graph = load_ethereum_fraud_detection(); print(f'Loaded {len(graph.node_labels)} nodes')"
```
