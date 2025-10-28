# Weak Supervision + Graph Reconstruction 구현

## 🎯 논문의 핵심

> "MPOCryptoML leverages weak supervision combined with graph reconstruction to support inductive learning and effectively mitigate label imbalance"

### 핵심 의미

1. **Weak supervision**: 일부 노드만 라벨 있음 (Known), 나머지는 Unknown
2. **Graph reconstruction**: 전체 그래프 구조 활용 (PPR)
3. **Inductive learning**: 학습 시 보지 못한 노드도 예측
4. **Mitigate imbalance**: Unknown 노드로 정보 활용하여 불균형 완화

---

## ⚠️ 현재 문제

### 현재 구조

```python
# final_solution.py 73-76번
for node in graph.nodes:
    if node not in labels:
        labels[node] = 0  # ❌ 모든 노드를 Known으로!

# 결과
Unknown: 0개
Known: 7,692개 (144개 사기 + 7,548개 정상)
```

### 논문 구조

```
Known 노드: 144개 (사기) + 일부 정상
Unknown 노드: 거래 상대방 중 일부
```

---

## 🔧 올바른 구현 방법

### 1. Known/Unknown 분리

```python
# 사기 지갑과 직접 연결된 정상 지갑만 Known
known_nodes = fraud_addresses + direct_contacts  # 소수
unknown_nodes = all_others  # 대다수

# 라벨 부여
labels = {}
for addr in known_nodes:
    if addr in fraud_addresses:
        labels[addr] = 1
    else:
        labels[addr] = 0

# Unknown은 라벨 없음
```

### 2. PPR은 전체 노드에 적용

```python
# PPR: 전체 그래프 (Known + Unknown)
π(s, v) for all v in G

# Unknown 노드도 PPR 점수 받음
# → 그래프 구조 정보 활용!
```

### 3. LR 학습은 Known만 사용

```python
# LR 학습: Known만 사용
X_train = known_nodes의 NTS, NWS
y_train = known_nodes의 라벨

# 예측: Unknown도 예측
X_all = all_nodes의 NTS, NWS
y_pred = model.predict(X_all)
```

---

## 💡 핵심 혜택

### 1. Label Imbalance 완화

**현재**: 사기 144개 (2%) vs 정상 7,548개 (98%)

- 불균형 매우 심함

**논문 방식**: 사기 144개 + 정상 일부 (Known) vs Unknown 대다수

- Unknown은 "학습에는 안 씀" (불균형 덜 심함)
- PPR은 전체 사용 (구조 정보)

### 2. Inductive Learning

Unknown 노드도 Anomaly Score 계산:

```python
σ(vi) = π(vi) / F(vi)  # Unknown에도 적용!

# Unknown도 탐지 가능
top_k = sorted(all_scores)[:k]  # Known+Unknown 모두 포함
```

### 3. Graph Reconstruction

PPR에서 전체 그래프 구조 활용:

- Unknown 노드의 연결 관계도 PPR 계산에 영향
- 더 풍부한 구조 정보

---

## 🚀 구현 방법

### Step 1: 그래프 구성 시 Unknown 분리

```python
# final_solution.py 수정
graph = CryptoTransactionGraph()

# Known만 라벨 부여
labels = {}
for address in target_addresses:
    if address in anomaly_addresses:
        labels[address] = 1
    else:
        labels[address] = 0

# 거래 상대방은 라벨 없음 (Unknown)
# → 라벨 딕셔너리에 추가하지 않음

graph.set_labels(labels)

# 결과
# Known: 200개 (사기 144 + 정상 일부)
# Unknown: 나머지
```

### Step 2: PPR은 전체 적용

```python
# 현재 구조 유지
ppr.compute_single_source_ppr(node)  # 모든 노드에 대해

# Unknown 노드도 PPR 점수 받음
```

### Step 3: LR은 Known만 학습

```python
# 현재처럼 동작함 (labels에 없는 노드는 자동 제외)
# 단, labels 딕셔너리에 Unknown 추가 안 하면 됨
```

---

## ✅ 결과 예상

### 현재 (모두 Known)

```
사기: 144개 (2%)
정상: 7,548개 (98%)
Unknown: 0개
```

### 수정 후 (Weak Supervision)

```
사기 Known: 144개
정상 Known: 1,000개 (예상)
Unknown: 6,548개

→ Known 비율: 사기 12.5% (불균형 완화!)
→ Unknown은 PPR용 (구조 정보)
```

---

## 🎯 다음 단계

1. **Unknown 노드 추가** (final_solution.py 수정)
2. **성능 비교** (Before vs After)
3. **논문 재현 정확도 향상**
