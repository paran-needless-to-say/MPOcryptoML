# 최종 요약: Weak Supervision 문제

## 🎯 논문의 핵심 문구

> "MPOCryptoML leverages weak supervision combined with graph reconstruction to support inductive learning and effectively mitigate label imbalance"

---

## 🔍 현재 상태 분석

### Weak Supervision 의미

**Known 노드** (라벨 있음, 학습에 사용):

- 소수: 144개 사기 + 일부 정상
- 비율: 논문에서는 5~10% 정도

**Unknown 노드** (라벨 없음, PPR에만 사용):

- 다수: 거래 상대방
- 비율: 90~95%

### 우리 현재 구조

**❌ 모두 Known으로 처리**:

```
Known: 7,692개 (100%)
Unknown: 0개 (0%)
```

**✅ 논문 구조**:

```
Known: 144개 사기 + 1,000개 정상 (15%)
Unknown: 6,548개 (85%)
```

---

## 💡 영향

### 1. Label Imbalance

**현재**:

- 사기 144개 (1.87%)
- 정상 7,548개 (98.13%)
- **불균형 매우 심함**

**논문 방식**:

- Known: 사기 144 + 정상 500 = 644개
- Unknown: 7,048개
- **Known 비율**: 사기 22% (불균형 덜 심함)

### 2. Graph Reconstruction

**두 방식 모두 동일**:

- PPR은 전체 노드에 적용
- 구조 정보는 활용함

### 3. Inductive Learning

**현재: 가능**

- Unknown=0이지만, PPR은 전체에 적용
- Unknown 노드도 예측 가능

**논문 방식: 더 나을 수 있음**

- Unknown 노드 분리로 불균형 완화

---

## ✅ 결론

### 현재 방식도 작동함

- **PPR**: 전체 노드 사용 ✅
- **LR**: Known만 학습 ✅
- **예측**: 모든 노드 가능 ✅

### 차이점

- **불균형**: 현재가 더 심함 (1.87%)
- **데이터**: 모든 노드가 Known
- **성능**: 논문 방식이 더 나을 가능성

### 권장사항

1. **현재**: 작동 중, 성능 개선 중
2. **이상적**: Unknown 노드 추가하여 불균형 완화
3. **타협안**: class_weight='balanced'로 대응 중

---

## 📊 핵심 포인트

### 우리가 한 것

✅ **Graph Reconstruction**: PPR 전체 노드 적용  
✅ **Inductive Learning**: Unknown 노드도 예측 가능  
⚠️ **Weak Supervision**: Known/Unknown 분리 안 함  
⚠️ **Imbalance Mitigation**: class_weight로만 해결

### 논문 vs 우리

| 항목                 | 논문         | 우리         |
| -------------------- | ------------ | ------------ |
| Graph Reconstruction | ✅           | ✅           |
| Inductive Learning   | ✅           | ✅           |
| Weak Supervision     | ✅           | ⚠️ (누락)    |
| Imbalance Mitigation | Unknown 활용 | class_weight |

**→ 구조는 비슷하나, 불균형 완화 방식이 다름**
