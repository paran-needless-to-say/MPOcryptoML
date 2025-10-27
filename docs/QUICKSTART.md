# 🚀 MPOCryptoML 빠른 시작 가이드

## 올바른 실행 방법

**현재 디렉토리가 중요합니다!** 프로젝트 루트에서 실행해야 합니다.

```bash
# ✅ 올바른 방법 (프로젝트 루트에서)
cd /Users/yelim/Desktop/MPO_final
python examples/quick_start.py

# ❌ 잘못된 방법 (src 디렉토리에서)
cd src
python examples/quick_start.py  # 에러 발생!
```

## 1️⃣ 가장 간단한 실행

```bash
cd /Users/yelim/Desktop/MPO_final
python examples/test_pipeline.py
```

이 명령으로 전체 파이프라인을 간단히 테스트할 수 있습니다.

## 2️⃣ 단계별 실행 (학습용)

```bash
cd /Users/yelim/Desktop/MPO_final
python examples/quick_start.py
```

각 단계(PPR, NTS/NWS, Logistic Regression, Anomaly Score)의 자세한 출력을 볼 수 있습니다.

## 3️⃣ 고급 옵션

```bash
cd /Users/yelim/Desktop/MPO_final/src
python main.py --help  # 모든 옵션 보기

# 더 큰 데이터셋
python main.py --n-nodes 200 --n-transactions 1000

# 시각화 생성
python main.py --visualize

# 전체 노드 처리 (느림)
python main.py --no-test-mode
```

## 4️⃣ 노트북에서 사용

Jupyter 노트북에서 직접 사용하려면:

```python
import sys
sys.path.append('/Users/yelim/Desktop/MPO_final/src')

from graph import generate_dummy_data
from ppr import PersonalizedPageRank
from scoring import NormalizedScorer
from anomaly_detector import MPOCryptoMLDetector

# 사용 예제는 USAGE.md 참고
```

## 📁 프로젝트 구조

```
MPO_final/
├── src/              # 메인 소스 코드 (여기서 main.py 실행 가능)
├── examples/          # 예제 스크립트 (프로젝트 루트에서 실행)
├── notebooks/         # Jupyter 노트북
└── ...
```

## 🔧 주요 파라미터

- `--n-nodes`: 노드 개수 (기본: 100)
- `--n-transactions`: 거래 개수 (기본: 500)
- `--anomaly-ratio`: 사기 비율 (기본: 0.15)
- `--visualize`: 결과 시각화 생성
- `--no-test-mode`: 모든 노드 처리 (느림)

## ⚠️ 문제 해결

### 경로 에러 발생 시

```bash
# 현재 위치 확인
pwd

# 올바른 위치로 이동
cd /Users/yelim/Desktop/MPO_final

# 실행
python examples/quick_start.py
```

### Python 버전 문제

```bash
# Python 3.11 사용 (권장)
python3.11 examples/quick_start.py

# 가상환경 활성화
source venv/bin/activate
```

## 📖 더 알아보기

- 자세한 사용법: [USAGE.md](USAGE.md)
- 논문 구조: [README.md](README.md)
