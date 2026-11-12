# RobustBreast

세포핵 측정값 변동에 대한 유방암 분류 모델의 예측 안정성 검사 프로그램
(연구·교육용 프로토타입이며 진단 목적이 아닙니다. 데이터: WDBC, scikit-learn 내장, CC BY 4.0)

## 앱 실행 (Windows 기준)
1. Microsoft Store에서 "Python Install Manager" 설치
2. 이 폴더를 열고 주소창에 cmd 입력 → Enter
3. python -m pip install -r requirements.txt
4. python -m streamlit run app.py   → 브라우저에 앱이 열림 (검은 창을 닫으면 종료)

예시 입력: 사례 ID 291, 특징 mean radius, 방향 +, 변화율 10%
기대 결과: 악성 확률 0.463 → 0.518, 판정 반전 경고 / 자동 안정성 검사(±5%) 취약 특징 4개

## 파일 구성
- app.py           Streamlit 웹 앱 (입력 패널, 결과·경고, 자동 안정성 검사, 전체 분석, CSV 저장)
- model.py         데이터 분할(층화 80/20, random_state=42), StandardScaler + LogisticRegression
- robustness.py    특징별 교란, 전후 비교, 자동 안정성 검사
- 보고서 수치 재현 스크립트:
    python baseline.py      # 5절 기준 성능
    python experiment.py    # 6.1절 H1 → results_all.csv
    python h2h3.py          # 6.2절 H2, 민감도 지수, Spearman
    python h3_more.py       # 6.3절 H3, 2절 시나리오, 무교란 단위 테스트
    python joint.py         # 6.4절 크기 특징 동시 변동
    python seeds.py         # 7.2절 분할 20회 반복

보고서 수치 산출 환경: Python 3.13, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.5
