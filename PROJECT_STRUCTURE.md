# 📁 MPOCryptoML 프로젝트 구조

```
MPO_final/
├── 📄 README.md                 # 프로젝트 메인 문서
├── 📄 requirements.txt           # Python 패키지 의존성
├── 📄 PROJECT_STRUCTURE.md       # 이 파일 (구조 설명)
├── 📄 .gitignore                 # Git 무시 파일
│
├── 📁 docs/                      # 문서 폴더
│   ├── QUICKSTART.md             # 빠른 시작 가이드
│   ├── USAGE.md                  # 상세 사용법
│   ├── DATA_GUIDE.md             # 데이터 가이드
│   ├── IMPLEMENTATION_STATUS.md # 구현 현황
│   └── PROJECT_SUMMARY.md        # 프로젝트 요약
│
├── 📁 src/                       # 소스 코드
│   ├── __init__.py              # 패키지 초기화
│   ├── graph.py                 # 그래프 구조 + 더미 데이터
│   ├── ppr.py                   # Algorithm 1: Multi-source PPR
│   ├── scoring.py               # Algorithm 2, 3: NTS & NWS
│   ├── anomaly_detector.py      # Algorithm 4: Anomaly Detection
│   ├── main.py                  # 통합 파이프라인
│   ├── load_real_data.py        # 실제 데이터 로드
│   └── preprocess.py            # 데이터 전처리 (레거시)
│
├── 📁 examples/                  # 예제 스크립트
│   ├── quick_start.py            # 단계별 예제
│   └── test_pipeline.py         # 빠른 테스트
│
├── 📁 notebooks/                 # Jupyter 노트북
│   ├── 01_exploration.ipynb     # 데이터 탐색
│   └── ethereum-frauddetection-dataset/  # Kaggle 데이터셋
│
├── 📁 data/                      # 원본 데이터 (git ignored)
│
├── 📁 processed/                 # 전처리된 데이터 (git ignored)
│
├── 📁 results/                   # 결과 파일 (git ignored)
│
├── 📁 logs/                      # 로그 파일 (git ignored)
│
└── 📄 MPOCryptoML.pdf            # 논문 PDF
```

## 📋 파일 역할

### Core Implementation
- `src/graph.py`: 그래프 구조 G=(V,E,W,T) 및 더미 데이터 생성
- `src/ppr.py`: Algorithm 1 - Multi-source Personalized PageRank
- `src/scoring.py`: Algorithm 2, 3 - NTS/NWS 계산
- `src/anomaly_detector.py`: Algorithm 4 - Anomaly Score 계산 및 평가
- `src/main.py`: 전체 파이프라인 실행

### Examples
- `examples/quick_start.py`: 단계별 상세 예제
- `examples/test_pipeline.py`: 빠른 통합 테스트

### Documentation
- `README.md`: 프로젝트 개요 및 설치 방법
- `docs/QUICKSTART.md`: 빠른 시작 가이드
- `docs/USAGE.md`: 상세 사용 설명서
- `docs/DATA_GUIDE.md`: 데이터 관련 가이드

## 🚀 사용 방법

```bash
# 빠른 시작
python examples/quick_start.py

# 전체 파이프라인
python src/main.py

# 예제 실행
python examples/test_pipeline.py
```

## 📊 구현 현황

✅ Algorithm 1: Multi-source PPR
✅ Algorithm 2: Normalised Timestamp Score
✅ Algorithm 3: Normalised Weight Score
✅ Algorithm 4: Anomaly Detection & Evaluation

