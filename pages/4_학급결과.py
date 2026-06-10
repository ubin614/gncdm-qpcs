# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.common import (set_page, sidebar_nav, KC_NAMES, N_TRAIN_USER,
                          BRAND_COLOR, make_umap_2d, make_umap_3d)

set_page()
sidebar_nav()

st.title("📊 학급 인지진단 결과")

if "diag" not in st.session_state:
    st.warning("⚠️ 먼저 **[진단 실행]** 페이지에서 데이터를 업로드하고 진단을 실행하세요.")
    st.stop()

D = st.session_state["diag"]
theta_new    = D["theta_new"]
sr_new       = D["sr_new"]
c2d_new      = D["c2d_new"]
c3d_new      = D["c3d_new"]
sids         = D["original_sids"]
theta_ref    = D["theta_ref"]
sr_ref       = D["sr_ref"]
c2d_ref      = D["c2d_ref"]
c3d_ref      = D["c3d_ref"]
kc_mean_ref  = D["kc_mean_ref"]
n_new        = D["n_new"]
percentiles  = D["percentiles"]

# ── 요약 지표 ─────────────────────────────────────────────────────────────────
n_high = int((sr_new >= 0.6).sum())
n_mid  = int(((sr_new >= 0.3) & (sr_new < 0.6)).sum())
n_low  = int((sr_new < 0.3).sum())

c1, c2, c3, c4, c5 = st.columns(5)
for col, icon, val, lbl, delta in [
    (c1, "👥", f"{n_new}명",         "전체 학생",      None),
    (c2, "📈", f"{sr_new.mean():.1%}","평균 정답률",   f"{sr_new.mean()-sr_ref.mean():+.1%} vs 훈련"),
    (c3, "🟢", f"{n_high}명",         "상위 (≥60%)",   None),
    (c4, "🟡", f"{n_mid}명",          "중위 (30~60%)", None),
    (c5, "🔴", f"{n_low}명",          "하위 (<30%)",   None),
]:
    col.metric(f"{icon} {lbl}", val, delta)

st.divider()

# ── 탭 구성 ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 정답률 분포", "📊 KC 숙달도", "🌡️ 히트맵", "🗺️ 2D UMAP", "🌐 3D UMAP"
])

# ── 탭1: 정답률 분포 ──────────────────────────────────────────────────────────
with tab1:
    col_l, col_r = st.columns(2)

    with col_l:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=sr_new, xbins=dict(start=0, end=1, size=0.05),
            marker=dict(
                color=["#E74C3C" if v < 0.3 else "#F39C12" if v < 0.6 else "#27AE60"
                       for v in np.arange(0.025, 1, 0.05)],
                line=dict(color="white", width=1)
            ),
            name="학급", opacity=0.85
        ))
        fig_hist.add_vline(x=sr_new.mean(), line_dash="dash",
                           line_color=BRAND_COLOR, line_width=2,
                           annotation_text=f"학급 평균 {sr_new.mean():.1%}",
                           annotation_position="top right")
        fig_hist.add_vline(x=sr_ref.mean(), line_dash="dot",
                           line_color="gray", line_width=1.5,
                           annotation_text=f"훈련 평균 {sr_ref.mean():.1%}",
                           annotation_position="top left")
        fig_hist.update_layout(
            title="학급 정답률 분포",
            xaxis_title="정답률", yaxis_title="학생 수",
            bargap=0.05, height=360, plot_bgcolor="#FAFAFA", showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r:
        sorted_idx = np.argsort(sr_new)[::-1]
        colors_bar = ["#27AE60" if sr_new[i] >= 0.6
                      else "#F39C12" if sr_new[i] >= 0.3
                      else "#E74C3C" for i in sorted_idx]
        fig_bar = go.Figure(go.Bar(
            x=[f"S{sids[i]}" for i in sorted_idx],
            y=[sr_new[i] for i in sorted_idx],
            marker_color=colors_bar,
            text=[f"{sr_new[i]:.0%}" for i in sorted_idx],
            textposition="outside",
            hovertemplate="학생 S%{x}<br>정답률: %{y:.1%}<extra></extra>"
        ))
        fig_bar.add_hline(y=0.6, line_dash="dash", line_color="#27AE60",
                          annotation_text="상위(60%)")
        fig_bar.add_hline(y=0.3, line_dash="dash", line_color="#E74C3C",
                          annotation_text="하위(30%)")
        fig_bar.update_layout(
            title="학생별 정답률 (내림차순)",
            xaxis_title="학생", yaxis_title="정답률",
            yaxis_range=[0, min(1.15, sr_new.max() + 0.15)],
            height=360, plot_bgcolor="#FAFAFA", showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ── 탭2: KC 숙달도 ────────────────────────────────────────────────────────────
with tab2:
    col_l2, col_r2 = st.columns(2)
    kc_class_mean = theta_new.mean(axis=0)

    with col_l2:
        colors_kc = ["#27AE60" if v >= 0.6 else "#F39C12" if v >= 0.4
                     else "#E74C3C" for v in kc_class_mean]
        fig_kc = go.Figure()
        fig_kc.add_trace(go.Bar(
            name="훈련 평균", x=KC_NAMES, y=kc_mean_ref,
            marker_color="#BDC3C7", opacity=0.75
        ))
        fig_kc.add_trace(go.Bar(
            name="학급 평균", x=KC_NAMES, y=kc_class_mean,
            marker_color=colors_kc, opacity=0.9,
            text=[f"{v:.2f}" for v in kc_class_mean],
            textposition="outside"
        ))
        fig_kc.add_hline(y=0.5, line_dash="dash", line_color="gray",
                         annotation_text="숙달 기준(0.5)")
        fig_kc.update_layout(
            title="KC별 평균 숙달도 (학급 vs 훈련)",
            barmode="group", yaxis_range=[0, 1.15],
            yaxis_title="숙달도 (θ)", height=380, plot_bgcolor="#FAFAFA",
            legend=dict(orientation="h", y=-0.22)
        )
        st.plotly_chart(fig_kc, use_container_width=True)

    with col_r2:
        fig_vio = go.Figure()
        for ki, kc in enumerate(KC_NAMES):
            fig_vio.add_trace(go.Violin(
                y=theta_new[:, ki], name=kc,
                box_visible=True, meanline_visible=True,
                points="all", pointpos=0,
                jitter=0.3, marker_size=5, opacity=0.7
            ))
        fig_vio.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig_vio.update_layout(
            title="KC별 숙달도 분포 (바이올린)",
            yaxis_title="숙달도 (θ)", yaxis_range=[0, 1],
            height=380, plot_bgcolor="#FAFAFA", showlegend=False
        )
        st.plotly_chart(fig_vio, use_container_width=True)

# ── 탭3: 히트맵 ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown("**학생 × KC 숙달도 히트맵 (정답률 내림차순)**")
    sort_idx      = np.argsort(sr_new)[::-1]
    theta_sorted  = theta_new[sort_idx]
    sids_sorted   = [f"S{sids[i]}<br>({sr_new[i]:.0%})" for i in sort_idx]

    fig_heat = go.Figure(go.Heatmap(
        z=theta_sorted.T,
        x=sids_sorted,
        y=KC_NAMES,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        colorbar=dict(title="θ"),
        hovertemplate="학생: %{x}<br>KC: %{y}<br>θ: %{z:.3f}<extra></extra>",
        text=[[f"{theta_sorted[si, ki]:.2f}"
               for si in range(n_new)]
              for ki in range(len(KC_NAMES))],
        texttemplate="%{text}" if n_new <= 40 else "",
        textfont=dict(size=10)
    ))
    fig_heat.update_layout(
        xaxis_title="학생 (정답률 내림차순)",
        yaxis_title="지식요소",
        height=320,
        margin=dict(t=20, b=60)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── 탭4: 2D UMAP ──────────────────────────────────────────────────────────────
with tab4:
    st.caption(
        "회색 등고선 = 기존 730명 학생 밀도 분포 · "
        "● 원형 = 기존 학생 (정답률 색상) · "
        "★ 별형 = 학급 학생 · 색상: 🔵 고득점 → 🔴 저득점"
    )
    fig_2d = make_umap_2d(
        c2d_ref, sr_ref, c2d_new, sr_new, sids,
        title="2D UMAP — 기존 학생 분포 내 학급 위치"
    )
    st.plotly_chart(fig_2d, use_container_width=True)

# ── 탭5: 3D UMAP ──────────────────────────────────────────────────────────────
with tab5:
    st.caption("드래그로 회전 · 스크롤로 확대/축소 · 마우스 오버로 정보 확인")
    fig_3d = make_umap_3d(
        c3d_ref, sr_ref, c3d_new, sr_new, sids,
        title="3D UMAP — 기존 학생 분포 내 학급 위치"
    )
    st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# ── 결과 다운로드 ─────────────────────────────────────────────────────────────
df_dl = pd.DataFrame(theta_new, columns=KC_NAMES)
df_dl.insert(0, "S_ID", sids)
df_dl["score_rate"]  = sr_new
df_dl["percentile"]  = percentiles
st.download_button(
    "⬇️ 학급 진단 결과 CSV 다운로드",
    data=df_dl.to_csv(index=False).encode("utf-8-sig"),
    file_name="class_diagnosis_result.csv",
    mime="text/csv"
)
