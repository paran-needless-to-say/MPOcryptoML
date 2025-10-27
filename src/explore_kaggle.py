"""
Kaggle Ethereum Fraud Detection 데이터셋 상세 탐색

목적:
1. FLAG 분포 확인
2. 주요 특징 분석
3. Anomaly vs Normal 비교
4. 데이터 품질 검증
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def explore_kaggle_dataset(csv_path: str = "./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv"):
    """
    Kaggle 데이터셋 상세 탐색
    """
    print("="*70)
    print("Kaggle Ethereum Fraud Detection 데이터셋 탐색")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n[Step 1] Loading data...")
    df = pd.read_csv(csv_path)
    
    print(f"  Total addresses: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    
    # 2. FLAG 분포
    print("\n[Step 2] FLAG distribution:")
    flag_dist = df['FLAG'].value_counts()
    print(f"  Normal (FLAG=0): {flag_dist.get(0, 0)} ({flag_dist.get(0,0)/len(df)*100:.2f}%)")
    print(f"  Anomaly (FLAG=1): {flag_dist.get(1, 0)} ({flag_dist.get(1,0)/len(df)*100:.2f}%)")
    
    # 3. 주소별 기본 통계
    print("\n[Step 3] Key statistics:")
    print(f"  Avg min between sent tnx: Normal={df[df['FLAG']==0]['Avg min between sent tnx'].mean():.2f}, Anomaly={df[df['FLAG']==1]['Avg min between sent tnx'].mean():.2f}")
    print(f"  Avg min between received tnx: Normal={df[df['FLAG']==0]['Avg min between received tnx'].mean():.2f}, Anomaly={df[df['FLAG']==1]['Avg min between received tnx'].mean():.2f}")
    print(f"  Time Diff (Mins): Normal={df[df['FLAG']==0]['Time Diff between first and last (Mins)'].mean():.0f}, Anomaly={df[df['FLAG']==1]['Time Diff between first and last (Mins)'].mean():.0f}")
    
    # 4. 거래 통계
    print("\n[Step 4] Transaction statistics:")
    print(f"  Sent tnx: Normal={df[df['FLAG']==0]['Sent tnx'].mean():.0f}, Anomaly={df[df['FLAG']==1]['Sent tnx'].mean():.0f}")
    print(f"  Received tnx: Normal={df[df['FLAG']==0]['Received Tnx'].mean():.0f}, Anomaly={df[df['FLAG']==1]['Received Tnx'].mean():.0f}")
    print(f"  Total transactions: Normal={df[df['FLAG']==0]['total transactions (including tnx to create contract'].mean():.0f}, Anomaly={df[df['FLAG']==1]['total transactions (including tnx to create contract'].mean():.0f}")
    
    # 5. 네트워크 구조
    print("\n[Step 5] Network structure:")
    print(f"  Unique Sent To: Normal={df[df['FLAG']==0]['Unique Sent To Addresses'].mean():.1f}, Anomaly={df[df['FLAG']==1]['Unique Sent To Addresses'].mean():.1f}")
    print(f"  Unique Received From: Normal={df[df['FLAG']==0]['Unique Received From Addresses'].mean():.1f}, Anomaly={df[df['FLAG']==1]['Unique Received From Addresses'].mean():.1f}")
    
    # 6. 금액 통계
    print("\n[Step 6] Value statistics:")
    print(f"  Avg val sent: Normal={df[df['FLAG']==0]['avg val sent'].mean():.2f}, Anomaly={df[df['FLAG']==1]['avg val sent'].mean():.2f}")
    print(f"  Total Ether sent: Normal={df[df['FLAG']==0]['total Ether sent'].mean():.2f}, Anomaly={df[df['FLAG']==1]['total Ether sent'].mean():.2f}")
    
    # 7. 시간 패턴
    print("\n[Step 7] Time patterns:")
    print(f"  Time Diff range (Normal): {df[df['FLAG']==0]['Time Diff between first and last (Mins)'].min():.0f} - {df[df['FLAG']==0]['Time Diff between first and last (Mins)'].max():.0f} mins")
    print(f"  Time Diff range (Anomaly): {df[df['FLAG']==1]['Time Diff between first and last (Mins)'].min():.0f} - {df[df['FLAG']==1]['Time Diff between first and last (Mins)'].max():.0f} mins")
    
    # 8. 샘플 주소
    print("\n[Step 8] Sample addresses:")
    print("\n  Normal addresses (first 5):")
    normal_addrs = df[df['FLAG']==0]['Address'].head(5)
    for i, addr in enumerate(normal_addrs):
        print(f"    {i+1}. {addr}")
    
    print("\n  Anomaly addresses (first 5):")
    anomaly_addrs = df[df['FLAG']==1]['Address'].head(5)
    for i, addr in enumerate(anomaly_addrs):
        print(f"    {i+1}. {addr}")
    
    # 9. 시각화 준비
    print("\n[Step 9] Feature distributions (for visualization)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # FLAG 분포
    axes[0, 0].bar(['Normal', 'Anomaly'], [flag_dist.get(0, 0), flag_dist.get(1, 0)])
    axes[0, 0].set_title('FLAG Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # Time Diff
    axes[0, 1].hist(df[df['FLAG']==0]['Time Diff between first and last (Mins)'], 
                    bins=30, alpha=0.7, label='Normal')
    axes[0, 1].hist(df[df['FLAG']==1]['Time Diff between first and last (Mins)'], 
                    bins=30, alpha=0.7, label='Anomaly')
    axes[0, 1].set_title('Time Difference Distribution')
    axes[0, 1].set_xlabel('Minutes')
    axes[0, 1].legend()
    
    # Transaction count
    axes[1, 0].hist(df[df['FLAG']==0]['Sent tnx'], bins=30, alpha=0.7, label='Normal')
    axes[1, 0].hist(df[df['FLAG']==1]['Sent tnx'], bins=30, alpha=0.7, label='Anomaly')
    axes[1, 0].set_title('Sent Transactions Distribution')
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].legend()
    
    # Unique connections
    axes[1, 1].scatter(df[df['FLAG']==0]['Unique Sent To Addresses'], 
                      df[df['FLAG']==0]['Unique Received From Addresses'],
                      alpha=0.5, label='Normal', s=10)
    axes[1, 1].scatter(df[df['FLAG']==1]['Unique Sent To Addresses'],
                      df[df['FLAG']==1]['Unique Received From Addresses'],
                      alpha=0.5, label='Anomaly', s=10)
    axes[1, 1].set_title('Network Connections')
    axes[1, 1].set_xlabel('Sent To')
    axes[1, 1].set_ylabel('Received From')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('results/kaggle_exploration.png', dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Visualization saved to results/kaggle_exploration.png")
    plt.close()
    
    # 10. 결론
    print("\n" + "="*70)
    print("Key Findings:")
    print("="*70)
    print(f"✅ Total: {len(df)} addresses")
    print(f"✅ Anomalies: {flag_dist.get(1, 0)} ({flag_dist.get(1,0)/len(df)*100:.2f}%)")
    print(f"✅ Normal: {flag_dist.get(0, 0)} ({flag_dist.get(0,0)/len(df)*100:.2f}%)")
    print(f"✅ Imbalanced dataset (needs sampling strategy)")
    print("\n✅ Ready for graph conversion!")
    print("="*70)
    
    return df


if __name__ == "__main__":
    df = explore_kaggle_dataset()

