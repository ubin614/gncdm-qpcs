# -*- coding: utf-8 -*-
"""공통 유틸리티: 경로, 모델 로드, 진단 함수"""

import os, pickle
import numpy as np
import pandas as pd
import torch
import streamlit as st

BASE          = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH    = os.path.join(BASE, "result/QPCS_full/params_32_32.pt")
THETA_CSV     = os.path.join(BASE, "result/QPCS/diagnosis/theta_all.csv")
RESULT_NPY    = os.path.join(BASE, "result/QPCS/result_all.npy")
REDUCER_2D    = os.path.join(BASE, "result/QPCS/umap_reducer_2d.pkl")
REDUCER_3D    = os.path.join(BASE, "result/QPCS/umap_reducer_3d.pkl")
COORDS_2D     = os.path.join(BASE, "result/QPCS/umap_train_coords_2d.npy")
COORDS_3D     = os.path.join(BASE, "result/QPCS/umap_train_coords_3d.npy")
KC_DEF_CSV    = os.path.join(BASE, "QPCS_KC6_definitions.csv")
TEMPLATE_CSV  = os.path.join(BASE, "QPCS_new.csv")

KC_NAMES      = ["KC1","KC2","KC3","KC4","KC5","KC6"]
N_TRAIN_USER  = 730
N_ITEM        = 32

BRAND_COLOR   = "#2E86C1"
PAGE_CONFIG   = dict(
    page_title="G-NCDM 인지진단",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def set_page(cfg=PAGE_CONFIG):
    st.set_page_config(**cfg)

def sidebar_nav():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
        st.markdown("## G-NCDM 인지진단")
        st.caption("Generative Neural CDM")
        st.divider()
        st.markdown("""
**페이지 안내**
1. 🏠 소개
2. 🔬 모델 훈련
3. 📤 데이터 업로드
4. 📊 학급 결과
5. 👤 개별 학생
""")
        st.divider()
        st.caption(f"훈련 데이터: {N_TRAIN_USER}명 · {N_ITEM}문항 · KC×6")

@st.cache_resource(show_spinner="모델 로딩 중...")
def load_model():
    return torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

@st.cache_data(show_spinner="기준 데이터 로딩 중...")
def load_reference():
    df       = pd.read_csv(THETA_CSV)
    theta    = df[KC_NAMES].values
    sr       = df["score_rate"].values
    c2d      = np.load(COORDS_2D)
    c3d      = np.load(COORDS_3D)
    with open(REDUCER_2D,"rb") as f: r2d = pickle.load(f)
    with open(REDUCER_3D,"rb") as f: r3d = pickle.load(f)
    return theta, sr, c2d, c3d, r2d, r3d

@st.cache_data(show_spinner="훈련 로그 로딩 중...")
def load_train_log():
    raw = np.load(RESULT_NPY, allow_pickle=True)
    epochs, train_acc, valid_acc, valid_auc = [], [], [], []
    for i, ep in enumerate(raw):
        epochs.append(i + 1)
        train_acc.append(float(ep.get("train_eval", {}).get("acc", 0)))
        ve = ep.get("valid_eval", {})
        valid_acc.append(float(ve.get("acc", 0)))
        valid_auc.append(float(ve.get("auc", 0)))
    return epochs, train_acc, valid_acc, valid_auc

def make_umap_2d(c2d_ref, sr_ref, c2d_new, sr_new, sids,
                 highlight_idx=None, title="2D UMAP"):
    """KDE 등고선 + 산점도 + 학급 마커가 포함된 Plotly 2D UMAP"""
    import plotly.graph_objects as go

    fig = go.Figure()

    # ① KDE 밀도 등고선 (기존 학생 배경)
    fig.add_trace(go.Histogram2dContour(
        x=c2d_ref[:, 0], y=c2d_ref[:, 1],
        colorscale="Greys",
        reversescale=False,
        showscale=False,
        ncontours=12,
        opacity=0.55,
        contours=dict(coloring="fill", showlabels=False),
        hoverinfo="skip",
        name=""
    ))

    # ② 기존 학생 산점도
    fig.add_trace(go.Scatter(
        x=c2d_ref[:, 0], y=c2d_ref[:, 1],
        mode="markers",
        marker=dict(
            color=sr_ref, colorscale="RdYlBu", cmin=0, cmax=1,
            size=5, opacity=0.35,
            colorbar=dict(title="정답률", thickness=13,
                          tickformat=".0%", x=1.02),
            line=dict(color="white", width=0.3)
        ),
        name=f"기존 학생 ({N_TRAIN_USER}명)",
        hovertemplate="정답률: %{marker.color:.1%}<extra>기존</extra>"
    ))

    # ③ 학급 학생 (강조 or 전체)
    if highlight_idx is None:
        # 학급 전체 표시
        fig.add_trace(go.Scatter(
            x=c2d_new[:, 0], y=c2d_new[:, 1],
            mode="markers+text",
            marker=dict(
                color=sr_new, colorscale="RdYlBu", cmin=0, cmax=1,
                size=14, symbol="star",
                line=dict(color="black", width=1.2)
            ),
            text=[f"S{s}" for s in sids],
            textposition="top center",
            textfont=dict(size=8, color="black"),
            name="학급 학생",
            hovertemplate="S%{text}<br>정답률: %{marker.color:.1%}<extra>학급</extra>"
        ))
    else:
        # 다른 학급 학생 (흐리게)
        other = [i for i in range(len(sids)) if i != highlight_idx]
        if other:
            fig.add_trace(go.Scatter(
                x=c2d_new[other, 0], y=c2d_new[other, 1],
                mode="markers",
                marker=dict(
                    color=[sr_new[i] for i in other],
                    colorscale="RdYlBu", cmin=0, cmax=1,
                    size=8, symbol="circle", opacity=0.40,
                    line=dict(color="gray", width=0.6)
                ),
                name="같은 학급", hoverinfo="skip"
            ))
        # 선택 학생 강조
        sr_h  = sr_new[highlight_idx]
        color = ("#2471A3" if sr_h >= 0.6
                 else "#E67E22" if sr_h >= 0.3 else "#C0392B")
        fig.add_trace(go.Scatter(
            x=[c2d_new[highlight_idx, 0]],
            y=[c2d_new[highlight_idx, 1]],
            mode="markers+text",
            marker=dict(color=color, size=22, symbol="star",
                        line=dict(color="black", width=2)),
            text=[f"S{sids[highlight_idx]} ({sr_h:.0%})"],
            textposition="top center",
            textfont=dict(size=11, color=color, family="Arial Black"),
            name=f"S{sids[highlight_idx]} (선택)",
            hovertemplate=f"S{sids[highlight_idx]}<br>정답률: {sr_h:.1%}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        plot_bgcolor="#F0F4F8",
        paper_bgcolor="white",
        height=480,
        margin=dict(t=50, b=20, l=10, r=80),
        legend=dict(
            orientation="v",
            x=0.01, y=0.01,
            xanchor="left", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#CCCCCC",
            borderwidth=1,
            font=dict(size=10),
        ),
    )
    return fig


def make_umap_3d(c3d_ref, sr_ref, c3d_new, sr_new, sids,
                 highlight_idx=None, title="3D UMAP"):
    """라이트 테마 고가시성 3D UMAP — Plotly Scatter3d"""
    import plotly.graph_objects as go

    BG       = "#F8F9FA"
    GRID_CLR = "#DEE2E6"
    AXIS_CLR = "#495057"

    fig = go.Figure()

    # ① 기존 학생 — 작은 반투명 점
    fig.add_trace(go.Scatter3d(
        x=c3d_ref[:, 0], y=c3d_ref[:, 1], z=c3d_ref[:, 2],
        mode="markers",
        marker=dict(
            color=sr_ref, colorscale="RdYlBu",
            cmin=0, cmax=1,
            size=3, opacity=0.40,
            colorbar=dict(
                title=dict(text="정답률", font=dict(color=AXIS_CLR, size=11)),
                tickformat=".0%",
                tickfont=dict(color=AXIS_CLR, size=9),
                thickness=13, x=1.01,
            ),
            line=dict(width=0)
        ),
        name=f"기존 학생 ({N_TRAIN_USER}명)",
        hovertemplate="정답률: %{marker.color:.1%}<extra>기존</extra>"
    ))

    # ② 학급 학생
    if highlight_idx is None:
        # ── 학급 전체: 테두리 있는 다이아몬드 ────────────────────
        fig.add_trace(go.Scatter3d(
            x=c3d_new[:, 0], y=c3d_new[:, 1], z=c3d_new[:, 2],
            mode="markers+text",
            marker=dict(
                color=sr_new, colorscale="RdYlBu",
                cmin=0, cmax=1, size=10,
                symbol="diamond",
                opacity=1.0,
                line=dict(color="black", width=1.5)
            ),
            text=[f"S{s}" for s in sids],
            textfont=dict(size=8, color="#212529"),
            textposition="top center",
            name="학급 학생",
            hovertemplate="S%{text}<br>정답률: %{marker.color:.1%}<extra>학급</extra>"
        ))
    else:
        # ── 다른 학급 학생 (반투명) ───────────────────────────────
        other = [i for i in range(len(sids)) if i != highlight_idx]
        if other:
            fig.add_trace(go.Scatter3d(
                x=c3d_new[other, 0], y=c3d_new[other, 1], z=c3d_new[other, 2],
                mode="markers",
                marker=dict(
                    color=[sr_new[i] for i in other],
                    colorscale="RdYlBu", cmin=0, cmax=1,
                    size=6, opacity=0.45, symbol="diamond",
                    line=dict(color="#888888", width=1)
                ),
                name="같은 학급",
                hovertemplate="정답률: %{marker.color:.1%}<extra>학급</extra>"
            ))

        # ── 선택 학생: 두꺼운 테두리 + 외곽 링으로 강조 ─────────
        sr_h  = float(sr_new[highlight_idx])
        if sr_h >= 0.6:
            pt_color   = "#1565C0"   # 진파랑
            ring_color = "#42A5F5"   # 밝은 파랑
        elif sr_h >= 0.3:
            pt_color   = "#E65100"   # 진주황
            ring_color = "#FFA726"   # 밝은 주황
        else:
            pt_color   = "#B71C1C"   # 진빨강
            ring_color = "#EF5350"   # 밝은 빨강

        # 외곽 링 (강조 halo)
        fig.add_trace(go.Scatter3d(
            x=[c3d_new[highlight_idx, 0]],
            y=[c3d_new[highlight_idx, 1]],
            z=[c3d_new[highlight_idx, 2]],
            mode="markers",
            marker=dict(
                color=ring_color, size=22, opacity=0.30,
                symbol="diamond", line=dict(width=0)
            ),
            hoverinfo="skip", showlegend=False, name=""
        ))
        # 실제 포인트
        fig.add_trace(go.Scatter3d(
            x=[c3d_new[highlight_idx, 0]],
            y=[c3d_new[highlight_idx, 1]],
            z=[c3d_new[highlight_idx, 2]],
            mode="markers+text",
            marker=dict(
                color=pt_color, size=14, opacity=1.0,
                symbol="diamond",
                line=dict(color="white", width=2.5)
            ),
            text=[f"S{sids[highlight_idx]} ({sr_h:.0%})"],
            textfont=dict(size=12, color=pt_color, family="Arial Black"),
            textposition="top center",
            name=f"S{sids[highlight_idx]} (선택)",
            hovertemplate=f"S{sids[highlight_idx]}<br>정답률: {sr_h:.1%}<extra></extra>"
        ))

    # ── 레이아웃 ──────────────────────────────────────────────────
    axis_style = dict(
        backgroundcolor=BG,
        gridcolor=GRID_CLR,
        showbackground=True,
        zerolinecolor=GRID_CLR,
        tickfont=dict(color=AXIS_CLR, size=9),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color="#212529", size=14),
                   x=0.5, xanchor="center"),
        paper_bgcolor="white",
        scene=dict(
            xaxis=dict(**axis_style,
                       title=dict(text="Dim 1",
                                  font=dict(color=AXIS_CLR, size=10))),
            yaxis=dict(**axis_style,
                       title=dict(text="Dim 2",
                                  font=dict(color=AXIS_CLR, size=10))),
            zaxis=dict(**axis_style,
                       title=dict(text="Dim 3",
                                  font=dict(color=AXIS_CLR, size=10))),
            bgcolor=BG,
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            aspectmode="cube",
        ),
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            font=dict(color="#212529", size=10),
            bgcolor="rgba(248,249,250,0.85)",
            bordercolor="#DEE2E6",
            borderwidth=1,
            x=0.01, y=0.99,
            xanchor="left", yanchor="top"
        ),
    )
    return fig


def diagnose(df_wide, net, r2d, r3d):
    q_cols   = [c for c in df_wide.columns if c.startswith("Q")]
    n_new    = len(df_wide)
    log_mat  = np.zeros((n_new, N_ITEM))
    for li, (_, row) in enumerate(df_wide.iterrows()):
        for j, qc in enumerate(q_cols):
            log_mat[li, j] = (int(row[qc]) - 0.5) * 2

    net.eval()
    device = next(net.parameters()).device
    thetas, srs = [], []
    with torch.no_grad():
        for uid in range(n_new):
            ul  = torch.Tensor([log_mat[uid]]).to(device)
            th  = net.diagnose_theta(ul).cpu().numpy().flatten()
            thetas.append(th)
            sv  = log_mat[uid]
            nc  = (sv > 0).sum(); na = (sv != 0).sum()
            srs.append(nc / na if na > 0 else 0.0)

    theta_new = np.stack(thetas)
    sr_new    = np.array(srs)
    c2d = r2d.transform(theta_new)
    c3d = r3d.transform(theta_new)
    return theta_new, sr_new, c2d, c3d
