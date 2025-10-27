# Kaggle 데이터 구조 분석

## ⚠️ 중요 발견

Kaggle Ethereum Fraud Detection 데이터는:

- ❌ **원시 거래 데이터가 아님**
- ✅ **이미 집계된 통계 데이터**

### 데이터 구조

```
Column: "Time Diff between first and last (Mins)"
→ 704785.63 minutes ≈ 489 days

→ 이 주소가 약 1.3년 동안 활동한 것
→ 하지만 개별 거래의 정확한 timestamp는 없음!
```

### 문제점

```
Kaggle 데이터:
  - Address: 0x0000...
  - FLAG: 1 (사기)
  - "Time Diff between first and last": 704785.63 mins
  - "Sent tnx": 721개
  - ❌ 각 거래의 정확한 timestamp 없음!

Etherscan API:
  - 주소의 "지금까지의 모든 거래"
  - ✅ Timestamp 있음
  - ❌ Kaggle이 계산한 집계 구간과 다름
```

## 🔧 해결 방안

### 방안 1: Etherscan으로 그래프 재구성 (추천)

**전략**: Kaggle은 라벨만 사용, 거래는 Etherscan에서 가져오기

```python
# 1. Kaggle에서 주소와 라벨만 가져오기
labels = {"0x...": FLAG}

# 2. Etherscan에서 해당 주소의 실제 거래 가져오기
transactions = get_from_etherscan("0x...")

# 3. 결합
graph = create_graph(transactions, labels)

# ✅ 완전한 그래프: 실제 거래 + 라벨
```

**장점**:

- 실제 거래 데이터 사용
- Timestamp 정확함
- 그래프 구조 실시간

**단점**:

- Etherscan의 거래 = 전체 기간
- Kaggle 집계 기간과 불일치 가능

### 방안 2: 보간법 (비추천)

Kaggle의 "Time Diff" 정보로 timestamp 시뮬레이션:

```python
# 불가능! 개별 거래가 없음
# 집계 데이터만 있으므로 정확한 timestamp 생성 불가
```

### 방안 3: 블록 번호 기반

Kaggle → Etherscan 시점 매칭:

- Kaggle 데이터가 언제 생성되었는지 확인
- 그 시점 이전의 Etherscan 거래만 사용

## ✅ 최종 추천

**하이브리드 방법** - Etherscan으로 그래프 재구성

```python
# 1. Kaggle: 주소 + 라벨
kaggle_df = pd.read_csv("transaction_dataset.csv")
labels = dict(zip(kaggle_df['Address'], kaggle_df['FLAG']))

# 2. Etherscan: 실제 거래 (최근 N개만)
for address in kaggle_df['Address'][:50]:
    txs = fetch_from_etherscan(address, api_key)
    # 실제 거래 그래프 구성

# 3. 결합
graph = create_graph(transactions, labels)
```

**이 방법이 최선입니다!**

## 🎯 핵심 포인트

1. **Kaggle은 집계 데이터**: 개별 거래 없음
2. **Etherscan은 원시 데이터**: 개별 거래 있음
3. **매칭 불가능**: 시간 구간이 다름
4. **해결책**: Etherscan으로 그래프 재구성 + Kaggle 라벨 사용

결론: timestamp는 Etherscan에서 가져오고, 라벨만 Kaggle에서 사용! ✅
