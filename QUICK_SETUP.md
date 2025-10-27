# 🚀 빠른 시작 가이드 (Kaggle + Etherscan 하이브리드)

## 최적 전략: Kaggle 라벨 + Etherscan Timestamp

### ✅ 준비된 것

1. **Kaggle 데이터**: `notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv`
   - 9,841개 주소
   - 2,179개 anomalies (FLAG=1)
   - ⚠️ Timestamp 없음

2. **구현된 코드**:
   - `src/kaggle_timestamp_matcher.py` - 하이브리드 그래프 생성
   - `examples/kaggle_etherscan_hybrid.py` - 전체 파이프라인

### 🔧 설정 (1분)

1. **Etherscan API Key 발급**:
   ```
   https://etherscan.io/apis → Create API Key (무료)
   ```

2. **코드 수정**:
   ```bash
   # examples/kaggle_etherscan_hybrid.py 열기
   # Line 16 수정
   API_KEY = "YourActualAPIKey"
   ```

3. **실행**:
   ```bash
   python examples/kaggle_etherscan_hybrid.py
   ```

### 📊 예상 시간

| 주소 개수 | 예상 시간 | 추천 여부 |
|-----------|----------|-----------|
| 20개 | ~4초 | 테스트용 |
| 50개 | ~10초 | **권장** |
| 100개 | ~20초 | 충분함 |
| 200개 | ~40초 | 대규모 |

## 💡 핵심 포인트

```python
# 이 코드 하나로 끝!
from src.kaggle_timestamp_matcher import create_hybrid_graph

graph = create_hybrid_graph(
    api_key="YourAPIKey",
    n_addresses=50  # 50개 주소만 처리
)

# 그 다음 알고리즘 실행 (기존 코드 그대로 사용)
from src.main import run_mpocrypto_ml_pipeline
results = run_mpocrypto_ml_pipeline(graph_obj=graph)
```

## ✨ 최종 결과

```
✓ 50개 주소에서:
  - Nodes (V): ~500-1000 nodes
  - Edges (E): ~1000-2000 edges  
  - Timestamps: 실제 Etherscan timestamps
  - Labels: Kaggle FLAG (2-3개 anomalies)
  - 평가 가능!
```

## 🎯 비교

| 방법 | 라벨 | Timestamp | 구현 난이도 |
|------|------|-----------|-------------|
| **더미 데이터** | ✅ | ✅ | ✅ 쉬움 |
| **Etherscan만** | ❌ | ✅ | ⚠️ 라벨 없음 |
| **Kaggle만** | ✅ | ❌ | ⚠️ Timestamp 없음 |
| **하이브리드** | ✅ | ✅ | ✅ **최적!** |

## 🚀 지금 실행

```bash
# 1. API Key 발급
# https://etherscan.io/apis

# 2. 코드 실행
python examples/kaggle_etherscan_hybrid.py

# 3. 완료! 🎉
```

이 방법이 **가장 완벽**합니다! ✅

