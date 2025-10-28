# 패턴 정의와 NTS/NWS의 관계

## 현재 구조

### 1. NTS/NWS 계산 (`src/scoring.py`)

논문의 Algorithm 2, 3을 정확히 구현:

```python
# Algorithm 2: NTS
θ(v_i) = |θ_out(v_i) - θ_in(v_i)|  # 절댓값 차이
NTS(v_i) = (θ(v_i) - min_θ) / (max_θ - min_θ)  # Min-max 정규화

# Algorithm 3: NWS
ω(v_i) = |ω_out(v_i) - ω_in(v_i)|  # 절댓값 차이
NWS(v_i) = (ω(v_i) - min_ω) / (max_ω - min_ω)  # Min-max 정규화
```

**특징**:

- `|θ_out - θ_in|`, `|ω_out - ω_in|` → **방향성 무시**
- 절댓값이므로 차이의 크기만 측정
- 정규화: 0~1 범위

### 2. 패턴 정의 (`src/pattern_analyzer.py`)

논문에는 **패턴 정의가 없음**. 우리가 도메인 지식으로 추가:

```python
# 패턴 판단
if theta_out > theta_in * 1.5 and omega_out > omega_in * 1.5:
    patterns.append("Fan-out")
elif theta_in > theta_out * 1.5 and omega_in > omega_out * 1.5:
    patterns.append("Fan-in")
```

**특징**:

- `θ_out > θ_in` 등으로 **방향성 고려**
- 비율 기준 (1.5배 등)
- 패턴 식별에 사용

## 연결 관계

### 문제점

현재는 **독립적으로 작동**:

1. `scoring.py`가 θ, ω를 계산하고 정규화 → NTS, NWS
2. `pattern_analyzer.py`가 θ, ω를 **다시 계산** → 패턴 분류
3. 두 결과가 **연결 안 됨**

### 개선 방향

패턴 분석에서 **이미 계산된 NTS/NWS를 재사용**:

```python
class PatternAnalyzer:
    def __init__(self, graph, feature_scores_df):
        self.graph = graph
        self.tx_df = graph.get_transactions_df()
        self.feature_scores = feature_scores_df  # NTS, NWS 포함

    def analyze_pattern(self, node: str) -> Tuple[str, Dict]:
        # NTS, NWS 재계산 대신 원본 θ, ω 사용
        nts = self.feature_scores.loc[node, 'nts']
        nws = self.feature_scores.loc[node, 'nws']

        # 원본 θ, ω 계산 (정규화 전)
        # ... (기존 로직)

        # 패턴 판단
        if theta_out > theta_in * 1.5 and omega_out > omega_in * 1.5:
            patterns.append("Fan-out")
```

또는:

```python
# NTS/NWS 계산 시 원본 값도 저장
feature_scores['nts'] = nts_normalized
feature_scores['theta_raw'] = theta_raw  # 정규화 전 원본 값
```

## 패턴의 수학적 의미

### 논문의 정의

- **θ(v_i) = |θ_out(v_i) - θ_in(v_i)|**: 시간 불균형 정도
- **ω(v_i) = |ω_out(v_i) - ω_in(v_i)|**: 금액 불균형 정도

**주석**: 절댓값이므로 방향성 무시!

### 패턴의 해석

논문의 θ, ω를 **비율**로 해석하면:

1. **Fan-in**:

   - θ_out << θ_in (보내기 시간 << 받기 시간)
   - ω_out << ω_in (보내기 금액 << 받기 금액)
   - → 받기만 많이 하는 패턴

2. **Fan-out**:

   - θ_out >> θ_in (보내기 시간 >> 받기 시간)
   - ω_out >> ω_in (보내기 금액 >> 받기 금액)
   - → 보내기만 많이 하는 패턴

3. **Rapid-layering**:

   - θ_in이 매우 작음 (짧은 시간 내 여러 거래)
   - in_count가 큼 (많은 입금)

4. **Value-mismatch**:
   - ω_out << ω_in (받은 금액 >> 보낸 금액)
   - → 돈이 쌓이는 패턴

## 결론

**NTS/NWS의 수학적 정의 vs 패턴 해석**:

| 항목   | NTS/NWS (논문) | 패턴 해석 (추가) |
| ------ | -------------- | ---------------- | --- | ------------------- |
| 계산   |                | θ_out - θ_in     |     | θ_out / θ_in (비율) |
| 의미   | 불균형 정도    | 방향성 포함      |
| 정규화 | O              | X                |
| 사용   | Anomaly Score  | 패턴 분류        |

**연결**:

- NTS/NWS: σ(vi) 계산에 사용 (Anomaly Score)
- 패턴: 설명 가능성(Explainability)에 사용

둘은 **보완 관계**:

- NTS/NWS → 얼마나 abnormal인가
- 패턴 → 어떤 패턴인가
