# 🎯 MPOCryptoML 프로젝트 요약

## ✅ 구현 완료

### 1. Algorithm 1: Multi-source PPR ✅

**파일**: `src/ppr.py`

논문의 정확한 구현:

- **Residual-based PPR** (Line 10-14)
- **Work Count K(s)** 계산 (Line 9)
- **Random Walk 시뮬레이션** (Line 19-26)
- 파라미터: `alpha=0.85`, `epsilon=0.01`, `p_f=0.1`

### 2. 데이터 생성 ✅

**파일**: `src/graph.py`

- `CryptoTransactionGraph` 클래스
- `generate_dummy_data()` - 더미 데이터 생성
- G=(V, E, W, T) 구조
- Timestamp 포함 거래 생성

### 3. NTS & NWS 계산 ✅

**파일**: `src/scoring.py`

- Normalized Timestamp Score (NTS)
- Normalized Weight Score (NWS)
- Feature 추출

### 4. Anomaly Detection ✅

**파일**: `src/anomaly_detector.py`

- Logistic Regression 학습
- Pattern Score 계산
- Anomaly Score: σ(vi) = π(vi) / F(θ,ω)(vi)
- 평가: Precision@K, Recall@K, F1

### 5. 통합 파이프라인 ✅

**파일**: `src/main.py`, `examples/quick_start.py`

전체 워크플로우 자동 실행

## 📊 현재 테스트 결과

```
Nodes: 50, Edges: 265, Anomalies: 7

Precision@5: 0.2000
Recall@5: 0.1429
F1@5: 0.1667

Top anomaly: address_14 (label=1, score=6.092910)
```

## 🎯 현재 상태

✅ **완료**:

1. PPR 알고리즘 구현 (논문 Algorithm 1)
2. 더미 데이터 생성
3. 전체 파이프라인 통합
4. 평가 메트릭

🔄 **다음 단계** (옵션 A 선택):

- 더미 데이터로 알고리즘 완성
- 실제 데이터 전환 준비
- 성능 튜닝

## 🚀 실행 방법

```bash
# 빠른 시작
cd /Users/yelim/Desktop/MPO_final
python examples/quick_start.py

# 전체 파이프라인
python src/main.py

# 테스트
python examples/test_pipeline.py
```

## 📁 파일 구조

```
MPO_final/
├── src/
│   ├── ppr.py              ✅ Algorithm 1: Multi-source PPR
│   ├── graph.py            ✅ 데이터 생성
│   ├── scoring.py          ✅ NTS/NWS
│   ├── anomaly_detector.py ✅ Anomaly detection
│   ├── main.py             ✅ 통합 파이프라인
│   └── load_real_data.py   ⏳ 실제 데이터 로드 (준비됨)
├── examples/
│   ├── quick_start.py      ✅ 단계별 예제
│   └── test_pipeline.py    ✅ 테스트
└── notebooks/
    └── 01_exploration.ipynb (Kaggle 데이터 탐색)
```

## 💡 선택 사항

### 옵션 A (현재 진행): 더미 데이터로 알고리즘 완성

- ✅ PPR 구현 완료
- ⏳ 나머지 알고리즘 확인 및 구현
- ⏳ 성능 튜닝
- ⏳ 실제 데이터 준비

### 옵션 B: 실제 데이터로 전환

- Etherscan API로 실제 거래 수집
- 또는 Kaggle 데이터 활용
- Timestamp 보완

어떤 방향으로 진행할까요?
