"""
RobustBreast — 세포핵 측정값 변동에 대한 유방암 분류 모델의 예측 안정성 검사 프로그램
실행:  python -m streamlit run app.py
연구·교육용 프로토타입이며 진단 목적이 아닙니다.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from model import load_data, split, train, malignant_proba, predict_label, MALIGNANT
from robustness import compare, stability_check

DELTAS = [-0.10, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.10]
THRESHOLD = 0.5

st.set_page_config(page_title="RobustBreast", layout="wide")


# ---------------------------------------------------------------- 데이터·모델
@st.cache_resource
def setup():
    X, y, feats = load_data()
    X_train, X_test, y_train, y_test = split(X, y, seed=42)
    model = train(X_train, y_train)
    p = malignant_proba(model, X_test)
    cases = pd.DataFrame({
        "true_malignant": (y_test == MALIGNANT).astype(int).values,
        "p_malignant": p,
    }, index=X_test.index)
    cases["pred_malignant"] = predict_label(p, THRESHOLD)
    cases["dist"] = (cases.p_malignant - 0.5).abs()
    return model, X_test, cases, feats


@st.cache_data
def full_experiment(_model, _X_test, feats):
    """평가셋 114건 × 30개 특징 × 8개 변동률 전체 교란 실험"""
    frames = [compare(_model, _X_test, f, d, THRESHOLD) for f in feats for d in DELTAS]
    res = pd.concat(frames, ignore_index=True)
    res["abs_dp"] = res.dp.abs()
    return res


model, X_test, cases, feats = setup()

lab = lambda v: "악성" if v == 1 else "양성"

# ---------------------------------------------------------------- 상단 고지
st.title("RobustBreast")
st.caption("세포핵 측정값 변동에 대한 유방암 분류 모델의 예측 안정성 검사 프로그램")
st.warning("연구·교육용이며 진단 목적이 아닙니다. 변동률은 가상 변동이며 실제 측정 오차나 진단 오류율이 아닙니다.")

# ---------------------------------------------------------------- 왼쪽 입력 패널
with st.sidebar:
    st.header("입력")
    near_only = st.checkbox("경계 근처 사례만 보기 (악성 확률 0.3~0.7)", value=False)
    pool = cases[cases.dist < 0.2] if near_only else cases
    ids = sorted(pool.index.tolist())
    default_id = 291 if 291 in ids else ids[0]
    case_id = st.selectbox(
        "사례 ID (원 데이터 행 번호)", ids, index=ids.index(default_id),
        format_func=lambda i: f"{i}  ·  실제 {lab(cases.loc[i, 'true_malignant'])}, 악성 확률 {cases.loc[i, 'p_malignant']:.3f}",
    )
    feature = st.selectbox("변경할 특징", feats, index=feats.index("mean radius"))
    direction = st.radio("변화 방향", ["+ (증가)", "- (감소)"], horizontal=True)
    rate = st.slider("변화율 (%)", min_value=1, max_value=10, value=5, step=1)
    delta = (rate / 100) * (1 if direction.startswith("+") else -1)

    st.divider()
    st.subheader("자동 안정성 검사")
    max_rate = st.slider("허용 변동률 ± (%)", min_value=1, max_value=10, value=5, step=1)
    run_check = st.button("이 사례 자동 검사 실행", width="stretch")

# ---------------------------------------------------------------- 오른쪽 결과 패널
x = X_test.loc[[case_id]]
r = compare(model, x, feature, delta, THRESHOLD).iloc[0]
info = cases.loc[case_id]

st.subheader(f"사례 {case_id} 검사 결과")
c1, c2, c3, c4 = st.columns(4)
c1.metric("실제 라벨", lab(info.true_malignant))
c2.metric("원본 악성 확률", f"{r.p_orig:.3f}", help=f"원본 판정: {lab(r.label_orig)}")
c3.metric("변경 후 악성 확률", f"{r.p_pert:.3f}", delta=f"{r.dp * 100:+.2f}%p", delta_color="off")
c4.metric("판정", f"{lab(r.label_orig)} → {lab(r.label_pert)}")

orig_val = float(x[feature].iloc[0])
st.write(f"**{feature}**: {orig_val:.4g} → {orig_val * (1 + delta):.4g} ({delta * 100:+.0f}%)")

if r.flipped:
    st.error(f"판정 불안정: {feature}를 {delta * 100:+.0f}% 바꾸자 판정이 {lab(r.label_orig)}에서 {lab(r.label_pert)}(으)로 바뀌었습니다. "
             "이 사례의 판정은 측정값 변동에 민감합니다(환자의 위험도가 높다는 뜻은 아닙니다).")
elif info.dist < 0.2:
    st.info("판정은 유지됐지만, 원래 확률이 판정 경계(0.5)에서 0.2 이내라 다른 특징이나 더 큰 변동에서는 바뀔 수 있습니다.")
else:
    st.success("판정이 유지됐습니다. 단, 안정적인 판정이 정답이라는 뜻은 아닙니다.")

chart_df = pd.DataFrame({"구분": ["원본", "변경 후"], "악성 확률": [r.p_orig, r.p_pert],
                         "판정": [lab(r.label_orig), lab(r.label_pert)]})
bars = alt.Chart(chart_df).mark_bar(size=34).encode(
    x=alt.X("악성 확률:Q", scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("구분:N", sort=["원본", "변경 후"], title=None),
    color=alt.Color("판정:N", scale=alt.Scale(domain=["양성", "악성"], range=["#4C78A8", "#E45756"])),
    tooltip=["구분", alt.Tooltip("악성 확률:Q", format=".3f"), "판정"])
rule = alt.Chart(pd.DataFrame({"악성 확률": [THRESHOLD]})).mark_rule(strokeDash=[6, 4], color="gray", size=2).encode(x="악성 확률:Q")
st.altair_chart((bars + rule).properties(height=150), width="stretch")
st.caption("회색 점선 = 판정 기준(악성 확률 0.5). 막대가 점선을 넘으면 악성으로 판정합니다.")

# 자동 안정성 검사 결과
if run_check:
    sc = stability_check(model, x, feats, max_rate / 100, THRESHOLD)
    st.session_state["check"] = (case_id, max_rate, sc)

if "check" in st.session_state and st.session_state["check"][0] == case_id:
    _, mr, sc = st.session_state["check"]
    weak = sc[sc.flipped]
    st.markdown(f"#### 자동 안정성 검사 (±{mr}%, 30개 특징 × 2방향 = 60회)")
    if len(weak):
        st.error(f"취약 특징 {weak.feature.nunique()}개: " + ", ".join(sorted(weak.feature.unique())))
        st.dataframe(weak[["feature", "delta", "p_orig", "p_pert"]].rename(
            columns={"feature": "특징", "delta": "변동률", "p_orig": "원본 확률", "p_pert": "변경 확률"}),
            hide_index=True, width="stretch")
    else:
        st.success(f"±{mr}% 안에서는 어떤 특징으로도 판정이 바뀌지 않았습니다.")

# CSV 저장 (현재 검사 + 자동 검사)
rows = [r.to_dict() | {"check": "single", "threshold": THRESHOLD}]
if "check" in st.session_state and st.session_state["check"][0] == case_id:
    for _, s in st.session_state["check"][2].iterrows():
        rows.append(s.to_dict() | {"check": f"auto_±{st.session_state['check'][1]}%", "threshold": THRESHOLD})
csv_df = pd.DataFrame(rows)[["case_id", "check", "feature", "delta", "threshold", "p_orig", "p_pert", "dp",
                             "label_orig", "label_pert", "flipped"]]
st.download_button("이 사례 결과 CSV 저장", csv_df.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"robustbreast_case{case_id}.csv", mime="text/csv")

# ---------------------------------------------------------------- 하단 분석 패널
st.divider()
st.subheader("평가셋 전체 분석 (114건 × 30개 특징 × 8개 변동률)")
with st.spinner("전체 교란 실험 계산 중..."):
    res = full_experiment(model, X_test, tuple(feats))

tab1, tab2, tab3 = st.tabs(["변동률별 결과", "특징별 민감도", "경계 거리별 반전"])

with tab1:
    h1 = res.groupby("delta").agg(평균_확률변화_pp=("abs_dp", lambda s: s.mean() * 100),
                                  반전_수=("flipped", "sum"), 시행_수=("flipped", "size")).reset_index()
    h1["반전율_%"] = h1.반전_수 / h1.시행_수 * 100
    h1["변동률"] = (h1.delta * 100).map(lambda v: f"{v:+.0f}%")
    st.bar_chart(h1, x="변동률", y="반전_수", height=260)
    st.dataframe(h1[["변동률", "평균_확률변화_pp", "반전_수", "시행_수", "반전율_%"]].round(3),
                 hide_index=True, width="stretch")

with tab2:
    mag = st.radio("변동 크기", ["±1%", "±3%", "±5%", "±10%"], index=3, horizontal=True)
    m = float(mag.strip("±%")) / 100
    sub = res[np.isclose(res.delta.abs(), m)]
    h2 = sub.groupby("feature").agg(평균_확률변화_pp=("abs_dp", lambda s: s.mean() * 100),
                                    반전_수=("flipped", "sum")).sort_values("평균_확률변화_pp", ascending=False).reset_index()
    st.bar_chart(h2, x="feature", y="평균_확률변화_pp", horizontal=True, height=620, sort="-평균_확률변화_pp")
    st.dataframe(h2.round(3), hide_index=True, width="stretch")

with tab3:
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5001]
    labels = ["0.0~0.1", "0.1~0.2", "0.2~0.3", "0.3~0.4", "0.4~0.5"]
    res["경계_거리"] = pd.cut((res.p_orig - 0.5).abs(), bins=bins, labels=labels, right=False)
    h3 = res.groupby("경계_거리", observed=False).agg(사례_수=("case_id", "nunique"), 시행_수=("flipped", "size"),
                                                   반전_수=("flipped", "sum")).reset_index()
    h3["반전율_%"] = (h3.반전_수 / h3.시행_수 * 100).round(2)
    st.dataframe(h3, hide_index=True, width="stretch")
    st.caption("경계 거리 = |원래 악성 확률 − 0.5|. 반전은 경계에 가까운 사례에 몰립니다.")

st.download_button("전체 실험 결과 CSV 저장 (27,360행)",
                   res.drop(columns=["경계_거리"]).to_csv(index=False).encode("utf-8-sig"),
                   file_name="robustbreast_all_results.csv", mime="text/csv")
