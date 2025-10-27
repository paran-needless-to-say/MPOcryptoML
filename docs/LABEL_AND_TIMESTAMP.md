# ✅ 라벨 + Timestamp 둘 다 있습니다!

## 🔄 Hybrid 전략

### 문제

- **Kaggle**: 라벨 있음 ✅, Timestamp 없음 ❌ (집계 데이터)
- **Etherscan**: Timestamp 있음 ✅, 라벨 없음 ❌ (실제 거래)

### 해결책: 둘 다 사용!

```python
# examples/final_solution.py

# Step 1: Kaggle에서 라벨 가져오기
df = pd.read_csv('transaction_dataset.csv')
anomaly_addresses = df[df['FLAG'] == 1]['Address']  # 라벨=1
normal_addresses = df[df['FLAG'] == 0]['Address']   # 라벨=0

# Step 2: Etherscan에서 Timestamp 가져오기
for address in addresses:
    txs = get_timestamp_from_etherscan(address, API_KEY)
    # → 실제 거래, 정확한 timestamp

# Step 3: 합치기
graph.add_edge(from_addr, to_addr, value, timestamp)  # Etherscan
labels[address] = 1 or 0                            # Kaggle FLAG
graph.set_labels(labels)
```

## ✅ 결과

| 항목        | 소스          | 상태 |
| ----------- | ------------- | ---- |
| 라벨        | Kaggle FLAG   | ✅   |
| Timestamp   | Etherscan API | ✅   |
| 거래 데이터 | Etherscan API | ✅   |

**논문 재현 완벽 가능!** 🎉

## 📝 실행

```bash
# 라벨 + Timestamp 둘 다 있는 그래프 생성
python examples/final_solution.py
```

결과:

- ✅ 라벨 있음 (Kaggle FLAG)
- ✅ Timestamp 있음 (Etherscan)
- ✅ 알고리즘 실행 가능
