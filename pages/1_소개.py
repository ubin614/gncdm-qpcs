# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import streamlit as st
from utils.common import (set_page, sidebar_nav, KC_DEF_CSV,
                          N_ITEM, N_TRAIN_USER, BRAND_COLOR, EXAM_PDF)

set_page()
sidebar_nav()

# ── 히어로 섹션 ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background: linear-gradient(135deg, {BRAND_COLOR} 0%, #1A5276 100%);
            padding: 2.5rem 2rem; border-radius: 16px; color: white;
            margin-bottom: 1.5rem; display: flex; align-items: center; gap: 2rem;">

  <!-- 좌측: 서비스 제목 -->
  <div style="flex: 1.2; min-width: 0;">
    <h1 style="margin:0; font-size:2.2rem;">🧠 G-NCDM 인지진단 서비스</h1>
    <p style="margin:0.5rem 0 0; font-size:1.05rem; opacity:0.9;">
      Generative Neural Cognitive Diagnostic Model<br>
      학생 개개인의 <b>지식요소(KC) 숙달도</b>를 즉각 진단합니다
    </p>
  </div>

  <!-- 구분선 -->
  <div style="width:1px; background:rgba(255,255,255,0.3); align-self:stretch;"></div>

  <!-- 우측: 논문 출처 -->
  <div style="flex: 1; min-width: 0;">
    <div style="font-size:0.72rem; opacity:0.7; letter-spacing:0.08em; text-transform:uppercase;
                margin-bottom:0.4rem;">📌 원본 모델 출처</div>
    <div style="font-size:0.92rem; font-weight:600; line-height:1.4;">
      Generative Cognitive Diagnosis
    </div>
    <div style="font-size:0.82rem; opacity:0.85; margin-top:0.2rem;">
      Jiatong Li, Qi Liu, Mengxiao Zhu (2025)
    </div>
    <div style="margin-top:0.6rem; display:flex; gap:0.6rem; flex-wrap:wrap;">
      <a href="https://arxiv.org/abs/2507.09831" target="_blank"
         style="background:rgba(255,255,255,0.18); color:white; text-decoration:none;
                border:1px solid rgba(255,255,255,0.35); border-radius:6px;
                padding:0.25rem 0.7rem; font-size:0.78rem;">
        📄 arXiv:2507.09831
      </a>
      <a href="https://github.com/CSLiJT/Generative-CD" target="_blank"
         style="background:rgba(255,255,255,0.18); color:white; text-decoration:none;
                border:1px solid rgba(255,255,255,0.35); border-radius:6px;
                padding:0.25rem 0.7rem; font-size:0.78rem;">
        💻 GitHub
      </a>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

# ── BibTeX 인용 ───────────────────────────────────────────────────────────────
with st.expander("📎 논문 인용 (BibTeX)", expanded=False):
    st.code("""@misc{li2025generativecognitivediagnosis,
      title={Generative Cognitive Diagnosis},
      author={Jiatong Li and Qi Liu and Mengxiao Zhu},
      year={2025},
      eprint={2507.09831},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.09831},
}""", language="bibtex")

# ── 핵심 지표 ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, icon, val, label in [
    (c1, "👥", f"{N_TRAIN_USER}명", "훈련 학습자"),
    (c2, "📝", f"{N_ITEM}개", "진단 문항"),
    (c3, "🔑", "6개", "지식요소(KC)"),
    (c4, "⚡", "즉각", "신규 학생 진단"),
]:
    col.markdown(f"""
    <div style="background:#F0F4F8; border-radius:12px; padding:1rem;
                text-align:center; border-left: 4px solid {BRAND_COLOR};">
      <div style="font-size:1.8rem;">{icon}</div>
      <div style="font-size:1.5rem; font-weight:700; color:{BRAND_COLOR};">{val}</div>
      <div style="font-size:0.85rem; color:#555;">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 모델 소개 ─────────────────────────────────────────────────────────────────
st.subheader("📌 G-NCDM이란?")

col_a, col_b = st.columns([3, 2])
with col_a:
    st.markdown("""
<b>G-NCDM(Generative Neural Cognitive Diagnostic Model)</b>은 딥러닝 기반의
차세대 인지진단모델입니다.

기존 CDM(인지진단모델)과의 핵심 차이는 <b>생성형 진단 함수(GDF)</b>입니다.
학습자의 응답 로그를 입력받아 KC 숙달도(θ)를 <b>즉석에서 생성</b>하므로,
새로운 학생의 데이터가 들어오면 <b>재학습 없이</b> 바로 진단할 수 있습니다.
""", unsafe_allow_html=True)

    st.markdown("**전통 CDM vs G-NCDM**")
    comp_df = pd.DataFrame({
        "항목": ["새 학생 진단 방식", "파라미터 저장", "진단 속도", "확장성"],
        "전통 CDM": ["재학습 필요", "학생별 파라미터 저장", "느림", "제한적"],
        "G-NCDM": ["즉각 진단 (GDF)", "함수로 생성", "빠름", "높음"],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

with col_b:
    st.markdown("""
    <div style="background:#EBF5FB; border-radius:12px; padding:1.2rem;">
      <h4 style="color:{c}; margin-top:0;">🔄 진단 흐름</h4>
      <div style="font-size:0.9rem; line-height:2;">
        📥 학생 응답 로그<br>
        &nbsp;&nbsp;↓<br>
        🧩 <b>생성형 진단 함수(f_nn)</b><br>
        &nbsp;&nbsp;↓<br>
        📊 KC 숙달도 벡터 θ<br>
        &nbsp;&nbsp;↓<br>
        🗺️ UMAP 시각화 + 리포트
      </div>
    </div>
    """.format(c=BRAND_COLOR), unsafe_allow_html=True)

st.divider()

# ── KC 정의 카드 ──────────────────────────────────────────────────────────────
st.subheader("🔑 지식요소(KC) 정의")
st.caption("본 문항지(QPCS)가 측정하는 6개의 지식요소입니다. 각 카드를 클릭해 자세한 정의를 확인하세요.")

df_kc = pd.read_csv(KC_DEF_CSV)

KC_COLORS = ["#2980B9","#27AE60","#8E44AD","#E67E22","#C0392B","#16A085"]

for i, row in df_kc.iterrows():
    color = KC_COLORS[i % len(KC_COLORS)]
    with st.expander(
        f"**{row['KC']}  ·  {row['인지요소명']}**",
        expanded=(i == 0)
    ):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 0.8rem;">
              <b>개념적 정의</b><br>
              <span style="color:#333;">{row['개념적 정의']}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 0.8rem;">
              <b>조작적 정의 (숙달 기준)</b><br>
              <span style="color:#333;">{row['조작적 정의']}</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background:#F8F9FA; border-radius:8px; padding:0.8rem;">
              <b>포함 개념 및 기능</b><br>
              <span style="font-size:0.88rem; color:#555;">{row['포함 개념 및 기능']}</span>
              <br><br>
              <b>관련 성취기준</b>&nbsp;
              <code>{row['관련 과학과 성취기준']}</code>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ── 문항지 다운로드 ───────────────────────────────────────────────────────────
st.subheader("📄 진단 문항지")
col_pdf, col_desc = st.columns([1, 3])
with col_pdf:
    import os
    if os.path.exists(EXAM_PDF):
        with open(EXAM_PDF, "rb") as f:
            st.download_button(
                label="⬇️ QPCS 문항지 다운로드 (PDF)",
                data=f.read(),
                file_name="QPCS_문항지.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
    else:
        st.info("문항지 파일을 찾을 수 없습니다.")
with col_desc:
    st.markdown(f"""
본 진단 서비스에 사용된 <b>QPCS(Quantitative Physics Concept Survey)</b> 문항지입니다.

- <b>문항 수</b>: {N_ITEM}개
- <b>측정 개념</b>: 6개 지식요소(KC)
- <b>대상</b>: 고등학교 물리 개념 이해도 측정

학생들에게 직접 배포하여 응답 데이터를 수집한 후,<br>
<b>[데이터 업로드]</b> 페이지에서 진단을 실행하세요.
""", unsafe_allow_html=True)

st.divider()

# ── 사용 흐름 안내 ────────────────────────────────────────────────────────────
st.subheader("📋 서비스 사용 순서")
steps = [
    ("🔬", "모델 훈련", "사이드바 → [모델 훈련] 페이지에서 훈련 과정 확인"),
    ("📤", "데이터 업로드", "학생 응답 CSV 파일을 업로드 (양식 다운로드 제공)"),
    ("📊", "학급 결과 확인", "학급 전체 KC 분포, UMAP, 히트맵 등 인터랙티브 시각화"),
    ("👤", "개별 학생 프로파일", "학생을 선택하면 레이더 차트, 3D UMAP, KC 숙달도 확인"),
]
cols = st.columns(4)
for col, (icon, title, desc) in zip(cols, steps):
    col.markdown(f"""
    <div style="background:white; border:1px solid #DDE; border-radius:12px;
                padding:1rem; text-align:center; height:130px;">
      <div style="font-size:2rem;">{icon}</div>
      <div style="font-weight:700; color:{BRAND_COLOR};">{title}</div>
      <div style="font-size:0.8rem; color:#666; margin-top:0.3rem;">{desc}</div>
    </div>""", unsafe_allow_html=True)
