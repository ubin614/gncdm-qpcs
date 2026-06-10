# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils.common import (set_page, sidebar_nav, KC_NAMES,
                          N_TRAIN_USER, BRAND_COLOR, make_umap_2d, make_umap_3d)

set_page()
sidebar_nav()

st.title("👤 개별 학생 인지진단 프로파일")

if "diag" not in st.session_state:
    st.warning("⚠️ 먼저 **[진단 실행]** 페이지에서 데이터를 업로드하고 진단을 실행하세요.")
    st.stop()

D = st.session_state["diag"]
theta_new   = D["theta_new"]
sr_new      = D["sr_new"]
c2d_new     = D["c2d_new"]
c3d_new     = D["c3d_new"]
sids        = D["original_sids"]
theta_ref   = D["theta_ref"]
sr_ref      = D["sr_ref"]
c2d_ref     = D["c2d_ref"]
c3d_ref     = D["c3d_ref"]
kc_mean_ref = D["kc_mean_ref"]
percentiles = D["percentiles"]
n_new       = D["n_new"]

# ── 학생 선택 ─────────────────────────────────────────────────────────────────
options = [f"S{sids[i]}  ({sr_new[i]:.1%})" for i in range(n_new)]
sel     = st.selectbox("학생 선택", options, index=0)
idx     = options.index(sel)

sid     = sids[idx]
theta   = theta_new[idx]
sr      = sr_new[idx]
pct     = percentiles[idx]
color   = "#2471A3" if sr >= 0.6 else "#E67E22" if sr >= 0.3 else "#C0392B"

def hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
label   = "상위" if sr >= 0.6 else "중위" if sr >= 0.3 else "하위"

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(90deg,{color}22,white);
            border-left:5px solid {color}; border-radius:8px;
            padding:1rem 1.5rem; margin-bottom:1rem;">
  <h2 style="margin:0; color:{color};">Student S{sid}</h2>
  <span style="font-size:1rem; color:#444;">
    정답률 <b>{sr:.1%}</b> &nbsp;|&nbsp;
    백분위 <b>{pct}%</b> <span style="color:{color};">[{label}]</span>
    &nbsp;(훈련 학생 {N_TRAIN_USER}명 기준)
  </span>
</div>
""", unsafe_allow_html=True)

# 강점/취약 KC
weak   = [KC_NAMES[k] for k, v in enumerate(theta) if v < 0.4]
strong = [KC_NAMES[k] for k, v in enumerate(theta) if v >= 0.7]
c1, c2 = st.columns(2)
c1.success(f"**💪 강점 KC**: {', '.join(strong) if strong else '없음'}")
c2.error(  f"**⚠️ 취약 KC**: {', '.join(weak)   if weak   else '없음'}")

st.divider()

# ── 상단 3패널 ────────────────────────────────────────────────────────────────
p1, p2, p3 = st.columns([1, 1.1, 1.1])

# ── [1] 레이더 차트 ───────────────────────────────────────────────────────────
with p1:
    st.markdown("**🕸️ KC 숙달도 레이더**")
    n   = len(KC_NAMES)
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    fig_r = go.Figure()

    # 훈련 평균 (회색)
    fig_r.add_trace(go.Scatterpolar(
        r=np.append(kc_mean_ref, kc_mean_ref[0]),
        theta=KC_NAMES + [KC_NAMES[0]],
        mode="lines", fill="toself",
        fillcolor="rgba(150,150,150,0.12)",
        line=dict(color="gray", width=1.5, dash="dot"),
        name="훈련 평균"
    ))
    # 해당 학생
    fig_r.add_trace(go.Scatterpolar(
        r=np.append(theta, theta[0]),
        theta=KC_NAMES + [KC_NAMES[0]],
        mode="lines+markers", fill="toself",
        fillcolor=hex_to_rgba(color, 0.18),
        line=dict(color=color, width=2.5),
        marker=dict(size=7, color=color),
        name=f"S{sid}"
    ))
    fig_r.update_layout(
        polar=dict(radialaxis=dict(range=[0,1], tickvals=[.2,.4,.6,.8],
                                   tickfont=dict(size=8))),
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
        height=340, margin=dict(t=20, b=60, l=30, r=30)
    )
    st.plotly_chart(fig_r, use_container_width=True)

# ── [2] KC 바 차트 ────────────────────────────────────────────────────────────
with p2:
    st.markdown("**📊 KC별 숙달도**")
    bar_colors = ["#E74C3C" if v < 0.4 else "#F39C12" if v < 0.6 else "#2ECC71"
                  for v in theta]
    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(
        y=KC_NAMES, x=theta,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in theta],
        textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"
    ))
    # 훈련 평균 점선
    for ki, mv in enumerate(kc_mean_ref):
        fig_b.add_shape(type="line",
                        x0=mv, x1=mv, y0=ki-0.4, y1=ki+0.4,
                        line=dict(color="#333", width=2.5, dash="solid"))
    fig_b.add_vline(x=0.5, line_dash="dash", line_color="gray",
                    annotation_text="기준(0.5)")
    fig_b.update_layout(
        xaxis=dict(range=[0, 1.1], title="Proficiency (θ)"),
        height=340, plot_bgcolor="#FAFAFA",
        showlegend=False, margin=dict(t=20, b=30, r=60)
    )
    st.plotly_chart(fig_b, use_container_width=True)

# ── [3] 2D UMAP 위치 ──────────────────────────────────────────────────────────
with p3:
    st.markdown("**🗺️ 전체 분포 내 위치 (2D)**")
    fig_u2 = make_umap_2d(
        c2d_ref, sr_ref, c2d_new, sr_new, sids,
        highlight_idx=idx,
        title=f"S{sid} 위치"
    )
    fig_u2.update_layout(height=340, margin=dict(t=40, b=50, l=5, r=70))
    st.plotly_chart(fig_u2, use_container_width=True)

st.divider()

# ── 3D UMAP (전체 너비) ───────────────────────────────────────────────────────
st.markdown("**🌐 3D UMAP — 전체 분포 내 위치**")
st.caption("드래그로 회전 · 스크롤로 확대/축소 · 마우스 오버로 정보 표시")

fig_3d = make_umap_3d(
    c3d_ref, sr_ref, c3d_new, sr_new, sids,
    highlight_idx=idx,
    title=f"S{sid} 위치 — 3D UMAP"
)
st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# ── KC 상세 해석 테이블 ───────────────────────────────────────────────────────
st.markdown("**📋 KC별 숙달도 상세**")
df_detail = pd.DataFrame({
    "KC": KC_NAMES,
    "숙달도 (θ)": [f"{v:.3f}" for v in theta],
    "훈련 평균": [f"{v:.3f}" for v in kc_mean_ref],
    "차이": [f"{v-r:+.3f}" for v, r in zip(theta, kc_mean_ref)],
    "수준": ["🟢 양호" if v >= 0.7 else "🟡 보통" if v >= 0.4 else "🔴 취약"
             for v in theta]
})
st.dataframe(df_detail, hide_index=True, use_container_width=True)

# ── 개별 리포트 CSV 다운로드 ──────────────────────────────────────────────────
row_dl = {"S_ID": sid, "score_rate": sr, "percentile": pct}
row_dl.update({kc: round(float(theta[i]), 4) for i, kc in enumerate(KC_NAMES)})
df_single = pd.DataFrame([row_dl])
st.download_button(
    f"⬇️ S{sid} 진단 결과 CSV",
    data=df_single.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"diagnosis_S{sid}.csv",
    mime="text/csv"
)
