# AI Insight Engine

CSV로 업로드한 자유 응답(의견) 텍스트를 embedding + clustering으로 자동 분류하고,
주제별 keyword·대표 의견·감성 비율을 요약하며, 자연어로 검색까지 할 수 있는 웹 앱입니다.

## 기능

- CSV 업로드 (UTF-8-SIG / UTF-8 / CP949 인코딩 자동 감지)
- 문장 embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- KMeans clustering (Number of Topics 직접 지정 가능)
- Silhouette score 기반 추천 k 표시
- Cluster별 TF-IDF keyword, 대표 의견(중심에 가장 가까운 문장) 추출
- PCA / UMAP 2D Topic Map 시각화
- Cluster 선택 시 해당 의견만 필터링
- 자연어 Semantic Search (유사도 threshold 조절 가능)
- Topic별 감성(positive/neutral/negative) 비율 시각화
- 분석 결과 CSV 다운로드

## 파일 구성

```
.
├── app.py              # 전체 파이프라인 + Gradio 웹 앱
├── requirements.txt    # 의존 패키지 (Colab 검증 환경 버전 고정)
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

### PyTorch 관련 주의사항

이 프로젝트는 Google Colab **CPU 런타임**에서 개발/검증되었고, `requirements.txt`의
`torch==2.11.0`은 CPU 전용 빌드(`+cpu`) 기준입니다.

- **CPU 서버에 배포하는 경우** — 아래처럼 PyTorch 전용 인덱스를 추가해서 설치해야
  정확히 같은 빌드가 설치됩니다.
  ```bash
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  ```
- **GPU 서버에 배포하는 경우** — `+cpu` 빌드로는 GPU를 활용할 수 없습니다. `torch` 버전
  숫자(`2.11.0`)만 유지하고, 서버의 CUDA 버전에 맞는 GPU 빌드로 별도 설치하세요.
  (PyTorch 공식 설치 페이지에서 CUDA 버전에 맞는 인덱스 URL 확인)

## 실행

```bash
python app.py
```

실행하면 로컬 Gradio 서버가 뜨고(`share=True`로 임시 공개 링크도 함께 생성됩니다),
브라우저에서 다음을 진행합니다.

1. CSV 업로드 (`text` 컬럼 필수)
2. Number of Topics(k) 입력 후 **Analyze** 클릭
3. 분석 결과(Topic Summary, Topic Map, 감성 비율, 추천 k) 확인
4. Cluster 드롭다운으로 특정 주제만 필터링
5. Query 입력 후 **Semantic Search**로 자연어 검색
6. 필요 시 **결과 CSV 다운로드**

## 상시 배포 시 참고

- 코드의 `demo.launch(share=True, debug=True)`는 로컬/임시 테스트용 옵션입니다. 서버에
  상시 호스팅할 경우 `share=False`로 바꾸고, 필요에 따라
  `server_name="0.0.0.0"`, `server_port=<포트>` 등을 지정한 뒤 리버스 프록시(nginx 등)
  뒤에 두는 방식을 권장합니다.
- 감성 분석 모델(`nlptown/bert-base-multilingual-uncased-sentiment`)은 별점 1~5 예측을
  3단계로 매핑한 것으로, 한국어 전용 감성 모델보다 정확도가 낮을 수 있습니다. 필요 시
  교체하세요.
- 데이터 양이 많을수록 embedding·silhouette score·UMAP·감성 분석 계산 시간이
  늘어납니다. GPU 환경에서는 훨씬 빨라집니다.

## 환경 (검증 완료)

| 패키지 | 버전 |
|---|---|
| pandas | 2.2.3 |
| numpy | 2.1.3 |
| sentence-transformers | 6.0.0 |
| scikit-learn | 1.9.0 |
| umap-learn | 0.5.12 |
| transformers | 5.15.0 |
| plotly | 6.9.0 |
| gradio | 6.25.0 |
| torch | 2.11.0 (+cpu) |
