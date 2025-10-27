"""
200개 주소의 실제 timestamp를 Etherscan에서 가져오기

전략:
1. Kaggle에서 200개 주소 선택 (20% anomaly)
2. Etherscan API로 실제 거래 가져오기
3. 정확한 timestamp로 그래프 생성
4. 알고리즘 실행
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from graph import CryptoTransactionGraph
from kaggle_timestamp_matcher import match_kaggle_with_etherscan
import pandas as pd
import time


def main():
    print("="*70)
    print("🔍 Etherscan에서 200개 주소의 실제 timestamp 가져오기")
    print("="*70)
    
    API_KEY = "TZ66JXC2M8WST154TM3111MBRRX7X7UAF9"
    
    # 1. Kaggle에서 200개 주소 샘플링
    print("\n[Step 1] Loading Kaggle addresses...")
    kaggle_path = "./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv"
    df = pd.read_csv(kaggle_path)
    
    # 20% anomaly, 80% normal 유지
    anomaly_df = df[df['FLAG'] == 1]
    normal_df = df[df['FLAG'] == 0]
    
    n_anomalies = 40  # 20%
    n_normal = 160    # 80%
    
    sample_anomalies = anomaly_df.head(n_anomalies)
    sample_normal = normal_df.head(n_normal)
    
    sample_df = pd.concat([sample_anomalies, sample_normal])
    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  ✓ Selected {len(sample_df)} addresses")
    print(f"  ✓ Anomalies: {sum(sample_df['FLAG'])}")
    print(f"  ✓ Normal: {len(sample_df) - sum(sample_df['FLAG'])}")
    
    # 2. Etherscan API 호출
    print("\n[Step 2] Fetching transactions from Etherscan...")
    print(f"  ⏳ This will take about {len(sample_df) * 0.2:.0f} seconds...")
    print(f"  (Rate limit: 5 calls/sec)")
    
    start_time = time.time()
    
    # match_kaggle_with_etherscan 함수는 이미 구현되어 있음
    graph = match_kaggle_with_etherscan(
        kaggle_df=sample_df,
        api_key=API_KEY,
        limit=len(sample_df)
    )
    
    elapsed = time.time() - start_time
    print(f"\n  ✓ Completed in {elapsed:.1f} seconds")
    
    # 3. 결과
    print("\n[Step 3] Results:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Anomalies: {sum(graph.node_labels.values())}")
    
    # 4. 그래프 저장
    print("\n[Step 4] Saving graph...")
    graph.save("results/graph_200_etherscan.json")
    print("  ✓ Saved to results/graph_200_etherscan.json")
    
    print("\n" + "="*70)
    print("✅ Real timestamp graph created!")
    print("="*70)
    
    print("\n📝 Next steps:")
    print("  1. Load this graph in your main pipeline")
    print("  2. Run PPR, NTS/NWS algorithms")
    print("  3. Compare with simulated data")
    
    return graph


if __name__ == "__main__":
    main()

