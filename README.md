# MPOCryptoML 논문 재현 프로젝트

## 논문 실험 환경

- **GPU:** 1 NVIDIA T4 Tensor Core GPU (AWS EC2 g4dn.2xlarge)
- **Memory:** 32 GiB
- **Storage:** 225 GB NVMe SSD
- **OS:** Ubuntu Linux
- **Python Version:** 3.8+
- **Main Libraries:** PyTorch, Scikit-learn

## 프로젝트 구조

```
MPO_final/
├── data/                  # 원본 데이터 (캐글에서 다운로드)
├── processed/             # 전처리된 데이터
├── notebooks/             # Jupyter notebooks
├── src/                   # 소스 코드
│   ├── preprocess.py     # 데이터 전처리
│   ├── model.py          # 모델 정의
│   └── train.py          # 학습 스크립트
├── requirements.txt       # 필요한 패키지
└── README.md
```

## 설치 방법

### 1. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 캐글 API 설정

캐글에서 데이터를 다운로드하려면 API 토큰이 필요합니다:

1. https://www.kaggle.com/ 계정에 로그인
2. Settings > API > Create New Token
3. 다운로드된 `kaggle.json` 파일을 `~/.kaggle/` 디렉토리에 저장
4. 권한 설정: `chmod 600 ~/.kaggle/kaggle.json`

## 데이터 다운로드

캐글에서 데이터를 다운로드하는 방법:

```python
import opendatasets as od

# 캐글 데이터셋 URL
dataset_url = "https://www.kaggle.com/competitions/해당-데이터셋-URL"
od.download(dataset_url)
```

## 다음 단계

1. 캐글에서 데이터 다운로드
2. 데이터 탐색 및 전처리 (`notebooks/exploration.ipynb`)
3. 모델 구현
4. 학습 및 평가
