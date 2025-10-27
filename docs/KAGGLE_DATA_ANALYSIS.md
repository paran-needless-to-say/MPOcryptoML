# 📊 Kaggle 데이터셋 상세 분석

## 기본 정보

- **총 주소**: 9,841개
- **정상 (FLAG=0)**: 7,662개 (77.86%)
- **사기 (FLAG=1)**: 2,179개 (22.14%)
- **불균형 데이터셋**: 3.5:1 비율

## 주요 발견

### 1. 시간 패턴 차이

```
Normal:
  - Avg min between sent tnx: 5,427분
  - Avg min between received tnx: 9,463분
  - Time Diff: 264,718분 (약 184일)

Anomaly:
  - Avg min between sent tnx: 3,888분 ⬇️ 더 자주 보냄
  - Avg min between received tnx: 2,875분 ⬇️⬇️ 훨씬 더 자주 받음!
  - Time Diff: 55,230분 (약 38일) ⬇️ 짧은 활동 기간
```

**해석**: 사기는 더 짧은 시간에 더 많은 거래!

### 2. 거래 패턴 차이

```
Normal:
  - Sent tnx: 147개
  - Received tnx: 203개
  - Total: 356개

Anomaly:
  - Sent tnx: 5개 ⬇️⬇️ 거의 안 보냄!
  - Received tnx: 24개
  - Total: 29개 ⬇️⬇️ 매우 적은 거래
```

**해석**: 사기는 거래가 매우 적음 (받기만 함?)

### 3. 네트워크 구조 차이

```
Normal:
  - Unique Sent To: 32.3개 주소
  - Unique Received From: 35.4개 주소

Anomaly:
  - Unique Sent To: 3.3개 주소 ⬇️⬇️ 거의 안 보냄
  - Unique Received From: 12.5개 주소
```

**해석**: 사기는 보내는 주소가 매우 적음 (fan-in 패턴?)

### 4. 금액 차이

```
Normal:
  - Avg val sent: 52.86 ETH
  - Total Ether sent: 13,025 ETH

Anomaly:
  - Avg val sent: 16.26 ETH
  - Total Ether sent: 87.37 ETH
```

**해석**: 사기는 금액이 적음

## 🎯 결론

### 사기 패턴:

1. ⚠️ 짧은 활동 기간 (38일 vs 184일)
2. ⚠️ 거의 거래 안 보냄 (5개 vs 147개)
3. ⚠️ 받기만 함 (fan-in 패턴)
4. ⚠️ 적은 금액 (16 ETH vs 52 ETH)
5. ⚠️ 빠른 수신 (2,875분 간격 vs 9,463분)

### 그래프 변환 전략:

```
Kaggle 데이터:
  ✅ Time Diff between first and last (Mins) = 전체 기간
  ✅ Unique Sent To Addresses = out-degree 개수
  ✅ Unique Received From Addresses = in-degree 개수
  ✅ Timestamp는 시간 범위 안에 분포 시뮬레이션

→ 시뮬레이션 그래프 생성 가능!
```

## 📝 다음 단계

1. ✅ 데이터 탐색 완료
2. ✅ 시뮬레이션 그래프 생성 완료
3. ⏳ 알고리즘 실행
4. ⏳ 결과 분석
