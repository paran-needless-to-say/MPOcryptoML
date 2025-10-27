# Kaggle + Etherscan 하이브리드 방법 가이드

## 📊 전략

### 문제:
1. **Kaggle 데이터**: 라벨(FLAG) ✅ 있음, Timestamp ❌ 없음
2. **Etherscan API**: Timestamp ✅ 있음, 라벨 ❌ 없음

### 해결책:
```
Kaggle 주소 리스트 + 라벨(FLAG)
         +
Etherscan API (Timestamp 수집)
         ↓
완전한 그래프: 라벨 + Timestamp ✅
```

## 🚀 사용 방법

### 1. API Key 발급

```
https://etherscan.io/apis → "Create New API Key" → Free Plan 선택
```

### 2. 코드 수정

`examples/kaggle_etherscan_hybrid.py` 열기:

```python
# Line 16 수정
API_KEY = "YourActualAPIKey"  # 발급받은 키 입력
```

### 3. 실행

```bash
python examples/kaggle_etherscan_hybrid.py
```

### 4. 처리 시간

- 50개 주소: 약 10초 (0.2초/주소)
- 100개 주소: 약 20초
- 1000개 주소: 약 3.3분

## 📋 데이터 흐름

```
1. Kaggle CSV 읽기
   ↓
   [Address, FLAG]
   - 9,841 addresses
   - 2,179 anomalies

2. Etherscan API 호출 (각 주소마다)
   ↓
   Timestamp 정보 수집
   - 실제 거래 timestamp
   - Block timestamp

3. 결합
   ↓
   Graph G=(V, E, W, T)
   ✓ V: Kaggle addresses
   ✓ E: Etherscan transactions
   ✓ W: Transaction values
   ✓ T: Etherscan timestamps
   ✓ Labels: Kaggle FLAG

4. 알고리즘 실행
   ↓
   Algorithm 1-4
```

## ⚙️ 설정 옵션

### 주소 개수 조절

```python
# 소량 테스트
graph = create_hybrid_graph(
    api_key=API_KEY,
    n_addresses=20  # 20개만 처리
)

# 중간 테스트
graph = create_hybrid_graph(
    api_key=API_KEY,
    n_addresses=100
)

# 대규모
graph = create_hybrid_graph(
    api_key=API_KEY,
    n_addresses=1000  # 오래 걸림!
)
```

### 에러 처리

API rate limit 발생 시:
- 자동으로 delay 추가 (0.2초)
- 너무 많은 주소는 시간이 오래 걸림
- 권장: 50-200개 주소로 테스트

## ✅ 장점

1. **완전한 데이터**: 라벨 + Timestamp 둘 다 있음
2. **실제 거래**: Etherscan 실제 거래 사용
3. **논문 호환**: Kaggle FLAG로 라벨링
4. **지속 가능**: API 계속 사용 가능

## ⚠️ 제한사항

1. **Rate limit**: 5 calls/sec (무료)
2. **처리 시간**: 주소당 0.2초 + API 응답 시간
3. **비용**: 무료지만 오래 걸림

## 📝 요약

| 항목 | Kaggle | Etherscan | 하이브리드 |
|------|--------|-----------|------------|
| **라벨** | ✅ 있음 | ❌ 없음 | ✅ 있음 |
| **Timestamp** | ❌ 없음 | ✅ 있음 | ✅ 있음 |
| **주소** | ✅ 9,841 | 필요시 | ✅ 모두 |
| **실행** | 즉시 | API 호출 필요 | 조금 느림 |

**결론**: 이 방법이 최적입니다! ✅

