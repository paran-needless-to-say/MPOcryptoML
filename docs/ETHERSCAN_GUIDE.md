# Etherscan API 데이터 수집 가이드

## 논문의 그래프 정의 확인

### ✅ 현재 구현: G=(V, E, W, T)

논문에 따르면:

- **V**: Vertex set = wallets/accounts (주소)
- **E**: Edge set = transfers (거래)
- **W**: Weights = total amount of money (거래 금액)
- **T**: Timestamps = time of transfers

**현재 구현 검증** (`src/graph.py`):

```python
class CryptoTransactionGraph:
    """
    암호화폐 거래 그래프 구조
    G = (V, E, W, T) where:
    - V: nodes (addresses) ✅
    - E: edges (transactions: sender → receiver) ✅
    - W: weights (transaction amount/value) ✅
    - T: timestamps (transaction time) ✅
    """
```

**결론**: ✅ 논문 정의대로 정확히 구현됨!

## Etherscan API 사용법

### 1. API Key 발급

1. https://etherscan.io/apis 접속
2. 계정 생성 또는 로그인
3. "Create New API Key" 클릭
4. 무료 플랜 선택 (5 calls/sec)

### 2. 데이터 수집

```python
from src.etherscan_parser import fetch_transactions_from_etherscan
import os

API_KEY = os.getenv("ETHERSCAN_API_KEY")

# 여러 주소의 거래 수집
addresses = [
    "0x1234...",  # 주소 1
    "0x5678...",  # 주소 2
]

# 자동으로 그래프 생성
graph_obj = fetch_transactions_from_etherscan(
    addresses=addresses,
    api_key=API_KEY
)
```

### 3. 그래프 구조

```
Raw Etherscan Data
   ↓
Transactions: [(from, to, value, timestamp), ...]
   ↓
convert_to_graph()
   ↓
G=(V, E, W, T)
   ↓
Algorithm 1-4 실행
```

### 4. 실제 사용 예제

```python
# examples/fetch_real_data.py 실행
python examples/fetch_real_data.py

# 또는 직접 호출
from examples.fetch_real_data import fetch_and_analyze

graph = fetch_and_analyze('0xAddress', 'YourAPIKey')
```

## 주의사항

### Rate Limit

- Free tier: 5 calls/sec
- 코드에 `time.sleep(0.2)` 추가됨 (자동 처리)

### 데이터 양

- 한 주소당 최대 10,000개 거래
- 여러 주소 병렬 수집 가능

### 비용

- Etherscan API는 무료
- 방대한 데이터 수집 시 시간이 오래 걸릴 수 있음

## 논문 구현 체크리스트

- [x] G=(V, E, W, T) 구조 정의
- [x] Address → Node (V) 변환
- [x] Transaction → Edge (E) 변환
- [x] Value → Weight (W) 변환
- [x] Timestamp → Time (T) 변환
- [x] Etherscan API 연동
- [x] in-degree=0인 노드 식별
- [ ] 실제 데이터로 테스트

## 다음 단계

1. Etherscan API Key 발급
2. 실제 주소 데이터 수집
3. 그래프 생성 및 PPR 실행
4. 결과 분석
