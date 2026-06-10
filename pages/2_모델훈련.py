# -*- coding: utf-8 -*-
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils.common import (set_page, sidebar_nav, load_train_log,
                          N_TRAIN_USER, N_ITEM, BRAND_COLOR)

set_page()
sidebar_nav()

st.title("🔬 모델 훈련")
st.caption("G-NCDM이 730명의 학생 데이터로 학습하는 과정을 확인합니다.")

# ── 훈련 정보 카드 ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, icon, val, lbl in [
    (c1, "👥", f"{N_TRAIN_USER}명", "훈련 학습자"),
    (c2, "📝", f"{N_ITEM}개", "문항 수"),
    (c3, "🔄", "20 epochs", "학습 횟수"),
    (c4, "⚙️", "Adam / lr=0.001", "옵티마이저"),
]:
    col.markdown(f"""
    <div style="background:#EBF5FB; border-radius:10px; padding:0.9rem;
                text-align:center; border-top:3px solid {BRAND_COLOR};">
      <div style="font-size:1.4rem;">{icon}</div>
      <div style="font-weight:700; color:{BRAND_COLOR}; font-size:1.1rem;">{val}</div>
      <div style="font-size:0.8rem; color:#555;">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 훈련 실행 버튼 ────────────────────────────────────────────────────────────
col_btn, col_info = st.columns([1, 3])
with col_btn:
    run = st.button("▶ 훈련 시뮬레이션 실행", type="primary", use_container_width=True)
with col_info:
    st.info("실제 학습이 완료된 모델을 기반으로 epoch별 학습 과정을 재현합니다.")

if not run and "train_done" not in st.session_state:
    st.markdown("""
    <div style="background:#F8F9FA; border-radius:12px; padding:2rem;
                text-align:center; color:#888;">
      <div style="font-size:3rem;">▶</div>
      <div>위 버튼을 클릭하면 훈련 과정이 시각화됩니다</div>
    </div>""", unsafe_allow_html=True)

if run or "train_done" in st.session_state:
    epochs, train_acc, valid_acc, valid_auc = load_train_log()
    n_ep = len(epochs)

    # ── 플레이스홀더 ──────────────────────────────────────────────────────────
    prog_bar  = st.progress(0, text="훈련 준비 중...")
    status_ph = st.empty()
    chart_ph  = st.empty()
    metric_ph = st.columns(3)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy (Train vs Valid)", "Validation AUC"),
        horizontal_spacing=0.12
    )
    fig.update_layout(height=380, margin=dict(t=50, b=30),
                      paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                      legend=dict(orientation="h", y=-0.15))

    # 이미 완료된 경우 전체 표시
    if "train_done" in st.session_state:
        replay_epochs = epochs
        delay = 0
    else:
        replay_epochs = range(1, n_ep + 1)
        delay = 0.25

    for ep in replay_epochs:
        idx = ep - 1 if isinstance(ep, int) else list(epochs).index(ep)
        if isinstance(ep, int):
            idx = ep - 1
        else:
            idx = ep

        ep_num = epochs[idx] if not isinstance(ep, int) else ep

        # 진행 표시
        prog_bar.progress(ep_num / n_ep,
                          text=f"Epoch {ep_num}/{n_ep} 학습 중...")
        status_ph.markdown(
            f"**Epoch {ep_num}** &nbsp;|&nbsp; "
            f"Train ACC: `{train_acc[idx]:.4f}` &nbsp;|&nbsp; "
            f"Valid ACC: `{valid_acc[idx]:.4f}` &nbsp;|&nbsp; "
            f"Valid AUC: `{valid_auc[idx]:.4f}`"
        )

        # 차트 업데이트
        fig.data = []
        ep_slice = epochs[:idx+1]
        fig.add_trace(go.Scatter(x=ep_slice, y=train_acc[:idx+1],
                                 mode="lines+markers", name="Train ACC",
                                 line=dict(color=BRAND_COLOR, width=2),
                                 marker=dict(size=6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ep_slice, y=valid_acc[:idx+1],
                                 mode="lines+markers", name="Valid ACC",
                                 line=dict(color="#E74C3C", width=2,
                                           dash="dash"),
                                 marker=dict(size=6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ep_slice, y=valid_auc[:idx+1],
                                 mode="lines+markers", name="Valid AUC",
                                 line=dict(color="#27AE60", width=2),
                                 marker=dict(size=6),
                                 fill="tozeroy",
                                 fillcolor="rgba(39,174,96,0.08)"),
                      row=1, col=2)
        fig.update_xaxes(title_text="Epoch", range=[0.5, n_ep + 0.5])
        fig.update_yaxes(range=[0.4, 1.0], row=1, col=1)
        fig.update_yaxes(range=[0.4, 1.0], row=1, col=2)

        chart_ph.plotly_chart(fig, use_container_width=True,
                              key=f"train_chart_{ep_num}")
        if delay > 0:
            time.sleep(delay)

    prog_bar.progress(1.0, text="✅ 훈련 완료!")
    st.session_state["train_done"] = True

    # ── 최종 결과 지표 ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 최종 테스트 결과")

    result_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "result/QPCS/test_result.json"
    )
    if os.path.exists(result_path):
        import json
        with open(result_path) as f:
            res = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        for col, key, label, fmt in [
            (m1, "acc",  "Accuracy",  ".4f"),
            (m2, "f1",   "F1 Score",  ".4f"),
            (m3, "rmse", "RMSE",      ".4f"),
            (m4, "auc",  "AUC",       ".4f"),
        ]:
            val   = res.get(key, 0)
            val_r = res.get("without_buf", {}).get(key, 0)
            col.metric(
                label,
                f"{val:{fmt}}",
                delta=f"Reconstruction: {val_r:{fmt}}",
                delta_color="normal"
            )

        st.info("""
**Score Prediction**: 버퍼에 저장된 θ/ψ로 예측  
**Score Reconstruction** (delta): 응답 로그에서 즉석 생성 → 새 학생 진단의 핵심 지표
""")

    # ── θ 분포 시각화 ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🗺️ 학습된 학생 θ 분포 (UMAP)")

    import pandas as pd
    from utils.common import load_reference, KC_NAMES
    theta_ref, sr_ref, c2d, c3d, _, _ = load_reference()

    fig_u = go.Figure()
    fig_u.add_trace(go.Scatter(
        x=c2d[:, 0], y=c2d[:, 1],
        mode="markers",
        marker=dict(
            color=sr_ref, colorscale="RdYlBu", cmin=0, cmax=1,
            size=7, opacity=0.7,
            colorbar=dict(title="정답률", thickness=14),
            line=dict(color="white", width=0.4)
        ),
        hovertemplate="정답률: %{marker.color:.1%}<extra></extra>"
    ))
    fig_u.update_layout(
        title="훈련 학생 730명 — θ 공간 UMAP (2D)",
        xaxis_title="UMAP Dim 1", yaxis_title="UMAP Dim 2",
        height=450, plot_bgcolor="#F2F4F6",
        margin=dict(t=50, b=30)
    )
    st.plotly_chart(fig_u, use_container_width=True)
