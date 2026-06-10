# -*- coding: utf-8 -*-
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
from utils.common import (set_page, sidebar_nav, load_model, load_reference,
                          diagnose, KC_NAMES, N_ITEM, N_TRAIN_USER,
                          BRAND_COLOR, TEMPLATE_CSV)

set_page()
sidebar_nav()

st.title("📤 데이터 업로드 & 진단 실행")

# ── CSV 양식 다운로드 ─────────────────────────────────────────────────────────
with st.expander("📋 CSV 입력 양식 안내 (클릭하여 확인 & 다운로드)", expanded=False):
    st.markdown(f"""
**필수 컬럼**: `S_ID`, `Q1` ~ `Q{N_ITEM}`

| 컬럼 | 설명 |
|------|------|
| `S_ID` | 학생 번호 (중복 없이, 임의 숫자 가능) |
| `Q1`~`Q{N_ITEM}` | 각 문항 정오답 (`1` = 정답, `0` = 오답) |
""")
    # 빈 양식 생성
    template_cols = ["S_ID"] + [f"Q{i}" for i in range(1, N_ITEM+1)]
    template_df   = pd.DataFrame(
        [[""] * len(template_cols)],
        columns=template_cols
    )
    st.dataframe(template_df, hide_index=True, use_container_width=True)

    # 실제 예시 파일이 있으면 제공
    if os.path.exists(TEMPLATE_CSV):
        ex_df = pd.read_csv(TEMPLATE_CSV)
        csv_bytes = ex_df.to_csv(index=False).encode("utf-8-sig")
    else:
        csv_bytes = template_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "⬇️ CSV 양식 다운로드",
        data=csv_bytes,
        file_name="QPCS_input_template.csv",
        mime="text/csv",
        use_container_width=True
    )

st.divider()

# ── 파일 업로드 ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "학급 응답 데이터 업로드",
    type="csv",
    help=f"S_ID, Q1~Q{N_ITEM} 컬럼이 포함된 CSV 파일"
)

if uploaded is None:
    st.info("👆 CSV 파일을 업로드하세요.")
    st.stop()

df_wide = pd.read_csv(uploaded)

# 검증
required = ["S_ID"] + [f"Q{i}" for i in range(1, N_ITEM+1)]
missing  = [c for c in required if c not in df_wide.columns]
if missing:
    st.error(f"❌ 필수 컬럼 누락: {missing}")
    st.stop()

score_vals = df_wide[[f"Q{i}" for i in range(1, N_ITEM+1)]].values.flatten()
if not set(np.unique(score_vals[~np.isnan(score_vals.astype(float))])).issubset({0, 1, 0.0, 1.0}):
    st.warning("⚠️ 응답 값에 0/1 이외의 값이 포함되어 있습니다. 확인하세요.")

# 미리보기
st.success(f"✅ 파일 로드 완료: **{len(df_wide)}명** · **{N_ITEM}문항**")
with st.expander("📄 업로드된 데이터 미리보기"):
    st.dataframe(df_wide, use_container_width=True)

# 정답률 간단 표시
sr_raw = df_wide[[f"Q{i}" for i in range(1, N_ITEM+1)]].mean(axis=1).values
c1, c2, c3 = st.columns(3)
c1.metric("업로드 학생 수", f"{len(df_wide)}명")
c2.metric("평균 정답률", f"{sr_raw.mean():.1%}")
c3.metric("정답률 범위", f"{sr_raw.min():.1%} ~ {sr_raw.max():.1%}")

st.divider()

# ── 진단 실행 ─────────────────────────────────────────────────────────────────
if st.button("🔍 인지진단 실행", type="primary", use_container_width=True):

    # 단계별 진행 애니메이션
    steps = [
        ("⚙️", "모델 로딩 중..."),
        ("📊", "기준 데이터 준비 중..."),
        ("🧮", "응답 로그 행렬 구성 중..."),
        ("🧠", "KC 숙달도(θ) 즉각 진단 중..."),
        ("🗺️", "UMAP 좌표 변환 중..."),
        ("✅", "진단 완료!"),
    ]
    prog  = st.progress(0)
    ph    = st.empty()

    for i, (icon, msg) in enumerate(steps[:-1]):
        ph.markdown(f"""
        <div style="background:#EBF5FB; border-radius:10px; padding:0.8rem 1.2rem;
                    border-left:4px solid {BRAND_COLOR};">
          <span style="font-size:1.2rem;">{icon}</span>
          &nbsp; <b>{msg}</b>
        </div>""", unsafe_allow_html=True)
        prog.progress((i + 1) / len(steps))

        if i == 0:
            net = load_model()
        elif i == 1:
            theta_ref, sr_ref, c2d_ref, c3d_ref, r2d, r3d = load_reference()
        elif i == 3:
            theta_new, sr_new, c2d_new, c3d_new = diagnose(
                df_wide, net, r2d, r3d
            )
        time.sleep(0.4)

    # 완료 표시
    ph.markdown(f"""
    <div style="background:#EAFAF1; border-radius:10px; padding:0.8rem 1.2rem;
                border-left:4px solid #27AE60;">
      <span style="font-size:1.2rem;">✅</span>
      &nbsp; <b>진단 완료! 결과를 확인하세요.</b>
    </div>""", unsafe_allow_html=True)
    prog.progress(1.0)

    # 백분위 계산
    percentiles = [int(np.mean(sr_ref <= s) * 100) for s in sr_new]
    kc_mean_ref = theta_ref.mean(axis=0)

    # session_state 저장
    st.session_state["diag"] = dict(
        df_wide=df_wide,
        theta_new=theta_new, sr_new=sr_new,
        c2d_new=c2d_new, c3d_new=c3d_new,
        original_sids=df_wide["S_ID"].tolist(),
        percentiles=percentiles,
        theta_ref=theta_ref, sr_ref=sr_ref,
        c2d_ref=c2d_ref, c3d_ref=c3d_ref,
        kc_mean_ref=kc_mean_ref,
        n_new=len(df_wide),
    )

    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#1A5276,{BRAND_COLOR});
                border-radius:10px; padding:1.2rem 1.8rem; color:white; margin-top:1rem;">
      <div style="font-size:1.2rem; font-weight:700;">✅ 인지진단 완료</div>
      <div style="margin-top:0.4rem; opacity:0.92;">
        {len(df_wide)}명의 KC 숙달도(θ) 계산이 완료되었습니다.<br>
        사이드바에서 <b>[학급 결과]</b> 또는 <b>[개별 학생]</b> 페이지로 이동하세요.
      </div>
    </div>
    """, unsafe_allow_html=True)

elif "diag" in st.session_state:
    st.success("✅ 이미 진단된 결과가 있습니다. **[학급 결과]** 탭으로 이동하세요.")
