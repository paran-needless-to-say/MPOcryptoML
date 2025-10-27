"""
MPOCryptoML 파이프라인 간단 테스트 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import run_mpocrypto_ml_pipeline

if __name__ == "__main__":
    print("="*60)
    print("MPOCryptoML Pipeline Test")
    print("="*60)
    
    # 파이프라인 실행
    detector, graph_obj, results_df = run_mpocrypto_ml_pipeline(
        n_nodes=50,
        n_transactions=200,
        anomaly_ratio=0.15,
        test_mode=True
    )
    
    print("\n\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    # 상위 이상 거래 탐지 결과
    print("\nTop 10 Anomaly Scores:")
    print(results_df.head(10)[['node', 'label', 'anomaly_score']])
    
    # 평가 지표
    print("\n\nEvaluation Metrics:")
    for k in [5, 10]:
        results = detector.evaluate_precision_at_k(k=k)
        print(f"\nK={k}:")
        for metric, score in results.items():
            print(f"  {metric}: {score:.4f}")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)
