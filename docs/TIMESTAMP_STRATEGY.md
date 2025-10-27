# ⏰ Timestamp 붙이기 전략

## 질문: 9,841개 주소에 모두 timestamp를 붙여야 하나?

### 답: ❌ 필요없음!

## 현재 상황

**Kaggle 데이터**:

- 9,841개 주소
- 집계 데이터만 있음 (개별 거래 없음)
- "Time Diff between first and last"만 있음

**Etherscan API**:

- 무료 플랜: 5 calls/sec (rate limit)
- 9,841개 × 0.2초 = 약 33분
- 실제로는 더 오래 걸림

## 실제 해결책

### 옵션 1: 샘플링 (추천) ✅

```python
# 100-200개 주소만 처리
df = pd.read_csv('transaction_dataset.csv')
sample_df = df.head(100)  # 또는 이상하게 50개

# Etherscan API 호출
for address in sample_df['Address']:
    txs = get_from_etherscan(address, api_key)
    # ~20초 소요

# → 충분한 테스트 가능!
```

**장점**:

- 빠름 (~20초)
- 충분한 그래프 크기 (수백 nodes, 수천 edges)
- 알고리즘 테스트 가능

**단점**:

- 전체 데이터셋이 아님
- 하지만 논문 구현 목적에는 충분

### 옵션 2: 시뮬레이션 (현재 사용 중)

```python
# Kaggle 집계 통계 사용
graph = kaggle_to_graph_realistic(csv_path, n_addresses=100)

# "Time Diff"를 기간으로 사용
# 거래를 그 기간 안에 분포
```

**장점**:

- 매우 빠름 (즉시)
- Kaggle 데이터 직접 사용

**단점**:

- 정확한 timestamp 아님
- 시뮬레이션

### 옵션 3: 전체 처리 (비추천) ❌

```python
# 9,841개 모두 처리
for all 9,841 addresses:
    get_from_etherscan(...)
    # 30분+ 소요
```

**문제점**:

- 시간 너무 오래 걸림
- 필요 없음
- 샘플링으로 충분

## 🎯 권장 전략

### 1단계: 샘플링

```python
# 100-200개 주소만 선택
n_addresses = 100

# Anomaly 20%, Normal 80% 유지
sample = stratified_sample(df, n_addresses)
```

### 2단계: Etherscan API 호출

```python
for address in sample['Address']:
    txs = get_from_etherscan(address, api_key)
    # 각 주소당 ~0.2초
    # 100개 = 20초
```

### 3단계: 그래프 생성

```python
graph = create_graph(transactions, labels)
# 100개 주소 → 수백 nodes, 수천 edges
```

## ✅ 결론

**필요한 것**:

- ✅ 100-200개 주소만 처리
- ✅ 충분한 그래프 크기
- ✅ 빠른 실행 (~20초)

**불필요한 것**:

- ❌ 9,841개 주소 모두 처리
- ❌ 30분+ 대기
- ❌ 전체 데이터셋

**현재 사용 중인 방법**:

- Kaggle 집계 → 시뮬레이션
- 100개 주소
- 충분한 테스트

## 🚀 실행

```bash
# 이미 준비됨!
python examples/use_kaggle_as_graph.py
# → 100개 주소, 시뮬레이션 timestamp
# → 즉시 실행, 알고리즘 테스트 가능
```

**진행 중인 작업과 맞게 동작합니다!** ✅
