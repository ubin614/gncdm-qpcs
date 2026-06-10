# -*- coding: utf-8 -*-
"""
G-NCDM 인지진단 서비스 — Streamlit 앱
"""

import io
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import streamlit as st
import torch
from scipy.stats import gaussian_kde

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
def _set_korean_font():
    candidates = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic",
                  "Malgun Gothic", "NanumBarunGothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_PATH      = os.path.join(BASE, "result/QPCS/params_32_32.pt")
THETA_CSV       = os.path.join(BASE, "result/QPCS/diagnosis/theta_all.csv")
REDUCER_2D_PATH = os.path.join(BASE, "result/QPCS/umap_reducer_2d.pkl")
REDUCER_3D_PATH = os.path.join(BASE, "result/QPCS/umap_reducer_3d.pkl")
COORDS_2D_PATH  = os.path.join(BASE, "result/QPCS/umap_train_coords_2d.npy")
COORDS_3D_PATH  = os.path.join(BASE, "result/QPCS/umap_train_coords_3d.npy")

KC_NAMES     = ["KC1", "KC2", "KC3", "KC4", "KC5", "KC6"]
N_TRAIN_USER = 730
N_ITEM       = 32

# ── 캐시: 모델 / 학습 데이터 로드 ─────────────────────────────────────────────
@st.cache_resource(show_spinner="모델 로딩 중...")
def load_model():
    return torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

@st.cache_data(show_spinner="기준 데이터 로딩 중...")
def load_reference():
    df_ref     = pd.read_csv(THETA_CSV)
    theta_ref  = df_ref[KC_NAMES].values
    sr_ref     = df_ref["score_rate"].values
    coords_2d  = np.load(COORDS_2D_PATH)
    coords_3d  = np.load(COORDS_3D_PATH)
    with open(REDUCER_2D_PATH, "rb") as f:
        reducer_2d = pickle.load(f)
    with open(REDUCER_3D_PATH, "rb") as f:
        reducer_3d = pickle.load(f)
    return theta_ref, sr_ref, coords_2d, coords_3d, reducer_2d, reducer_3d

# ── 진단 함수 ─────────────────────────────────────────────────────────────────
def diagnose(df_wide, net, reducer_2d, reducer_3d):
    q_cols = [c for c in df_wide.columns if c.startswith("Q")]
    n_new  = len(df_wide)

    log_mat = np.zeros((n_new, N_ITEM))
    for local_idx, (_, row) in enumerate(df_wide.iterrows()):
        for j, qcol in enumerate(q_cols):
            sc = int(row[qcol])
            log_mat[local_idx, j] = (sc - 0.5) * 2

    net.eval()
    device = next(net.parameters()).device
    theta_list, sr_list = [], []
    with torch.no_grad():
        for uid in range(n_new):
            user_log = torch.Tensor([log_mat[uid]]).to(device)
            theta = net.diagnose_theta(user_log).cpu().numpy().flatten()
            theta_list.append(theta)
            sv = log_mat[uid]
            nc = (sv > 0).sum(); na = (sv != 0).sum()
            sr_list.append(nc / na if na > 0 else 0.0)

    theta_new = np.stack(theta_list)
    sr_new    = np.array(sr_list)
    coords_new_2d = reducer_2d.transform(theta_new)
    coords_new_3d = reducer_3d.transform(theta_new)
    return theta_new, sr_new, coords_new_2d, coords_new_3d

# ── 헬퍼: matplotlib 차트들 ───────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf

def radar_chart(theta_vals, kc_names, color, label, ref_vals=None):
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    n = len(kc_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    if ref_vals is not None:
        rv = ref_vals.tolist() + ref_vals[:1].tolist()
        ax.plot(angles, rv, "-", color="gray", linewidth=1.2, alpha=0.5, label="훈련 평균")
        ax.fill(angles, rv, color="gray", alpha=0.1)
    vals = theta_vals.tolist() + theta_vals[:1].tolist()
    ax.plot(angles, vals, "o-", color=color, linewidth=2, label=label)
    ax.fill(angles, vals, color=color, alpha=0.22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(kc_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8"], fontsize=6, color="gray")
    ax.grid(color="gray", alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=7)
    plt.tight_layout()
    return fig

def kc_bar_chart(theta_vals, kc_names, ref_vals=None):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    fig.patch.set_facecolor("white")
    colors = ["#E74C3C" if v < 0.4 else "#F39C12" if v < 0.6 else "#2ECC71"
              for v in theta_vals]
    bars = ax.barh(kc_names, theta_vals, color=colors, alpha=0.85, height=0.6)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    if ref_vals is not None:
        for i, mv in enumerate(ref_vals):
            ax.plot(mv, i, marker="|", color="#333", markersize=12,
                    markeredgewidth=2, label="훈련 평균" if i == 0 else "")
        ax.legend(fontsize=7, loc="lower right")
    for bar, val in zip(bars, theta_vals):
        ax.text(min(val + 0.02, 0.93), bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Proficiency (θ)", fontsize=9)
    plt.tight_layout()
    return fig

def umap_2d_fig(coords_ref, sr_ref, coords_new, sr_new, highlight_idx=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F2F4F6")
    sc = ax.scatter(coords_ref[:, 0], coords_ref[:, 1],
                    c=sr_ref, cmap="RdYlBu", s=15, alpha=0.35,
                    vmin=0, vmax=1, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Score Rate", shrink=0.75)
    if highlight_idx is None:
        ax.scatter(coords_new[:, 0], coords_new[:, 1],
                   c=sr_new, cmap="RdYlBu", s=90, alpha=1.0,
                   vmin=0, vmax=1, edgecolors="black",
                   linewidths=1.3, marker="*")
    else:
        ax.scatter(coords_new[:, 0], coords_new[:, 1],
                   c=sr_new, cmap="RdYlBu", s=30, alpha=0.45,
                   vmin=0, vmax=1, edgecolors="none")
        x_h, y_h = coords_new[highlight_idx]
        sr_h     = sr_new[highlight_idx]
        color    = "#2471A3" if sr_h >= 0.6 else "#E67E22" if sr_h >= 0.3 else "#C0392B"
        ax.scatter(x_h, y_h, s=500, c="white",
                   edgecolors=color, linewidths=2.5, zorder=6)
        ax.scatter(x_h, y_h, s=180, marker="*", c=color,
                   edgecolors="black", linewidths=0.8, zorder=7)
        ax.text(0.97, 0.97, f"이 학생 ({sr_h:.0%})",
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                color=color, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=color, lw=1.2, alpha=0.92))
    ax.set_xlabel("UMAP Dim 1", fontsize=9)
    ax.set_ylabel("UMAP Dim 2", fontsize=9)
    ax.set_title("전체 분포 내 위치", fontsize=10)
    plt.tight_layout()
    return fig

def plotly_umap_3d(coords_ref, sr_ref, coords_new, sr_new, sids,
                   highlight_idx=None):
    colorscale = "RdYlBu"
    fig = go.Figure()

    # 기존 학생 (반투명)
    fig.add_trace(go.Scatter3d(
        x=coords_ref[:, 0], y=coords_ref[:, 1], z=coords_ref[:, 2],
        mode="markers",
        marker=dict(size=3, color=sr_ref, colorscale=colorscale,
                    cmin=0, cmax=1, opacity=0.25,
                    colorbar=dict(title="Score Rate", x=1.0, thickness=12)),
        name=f"기존 학생 ({N_TRAIN_USER}명)",
        hovertemplate="정답률: %{marker.color:.1%}<extra>기존 학생</extra>"
    ))

    # 새 학생 전체 (반투명)
    not_highlight = [i for i in range(len(sids))
                     if i != highlight_idx]
    if not_highlight:
        fig.add_trace(go.Scatter3d(
            x=coords_new[not_highlight, 0],
            y=coords_new[not_highlight, 1],
            z=coords_new[not_highlight, 2],
            mode="markers+text",
            marker=dict(size=6, color=sr_new[not_highlight],
                        colorscale=colorscale, cmin=0, cmax=1,
                        opacity=0.55, symbol="diamond",
                        line=dict(color="black", width=1)),
            text=[f"S{sids[i]}" for i in not_highlight],
            textposition="top center",
            textfont=dict(size=9),
            name="학급 학생",
            hovertemplate="S%{text}<br>정답률: %{marker.color:.1%}<extra>학급 학생</extra>"
        ))

    # 강조 학생
    if highlight_idx is not None:
        hi = highlight_idx
        sr_h = sr_new[hi]
        color_h = ("#2471A3" if sr_h >= 0.6 else
                   "#E67E22" if sr_h >= 0.3 else "#C0392B")
        fig.add_trace(go.Scatter3d(
            x=[coords_new[hi, 0]],
            y=[coords_new[hi, 1]],
            z=[coords_new[hi, 2]],
            mode="markers+text",
            marker=dict(size=14, color=color_h, symbol="diamond",
                        opacity=1.0, line=dict(color="black", width=2)),
            text=[f"S{sids[hi]} ({sr_h:.0%})"],
            textposition="top center",
            textfont=dict(size=11, color=color_h),
            name=f"S{sids[hi]} (선택)",
            hovertemplate=f"S{sids[hi]}<br>정답률: {sr_h:.1%}<extra>선택 학생</extra>"
        ))

    fig.update_layout(
        title=dict(text="3D UMAP — 전체 분포 내 학급 위치",
                   font=dict(size=14)),
        scene=dict(
            xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3",
            bgcolor="#F2F4F6"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
        legend=dict(x=0, y=1, font=dict(size=10))
    )
    return fig

# ── Streamlit 레이아웃 ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="G-NCDM 인지진단 서비스",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 G-NCDM 인지진단 서비스")
st.caption("Generative Neural Cognitive Diagnostic Model | QPCS 문항 기반")

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 사용 방법")
    st.markdown("""
1. **CSV 파일 업로드**  
   `S_ID, Q1~Q32` 형식의 학급 데이터
2. **[진단 실행]** 버튼 클릭
3. 탭에서 결과 확인
   - 학급 요약
   - 3D UMAP (인터랙티브)
   - 개별 학생 프로파일
""")
    st.divider()
    st.header("⚙️ 모델 정보")
    st.markdown(f"""
- **모델**: G-NCDM  
- **훈련 학생 수**: {N_TRAIN_USER}명  
- **문항 수**: {N_ITEM}개  
- **지식요소(KC)**: {len(KC_NAMES)}개  
""")
    st.divider()
    st.warning("⚠️ 실제 학생 데이터 업로드 시 개인정보 처리 방침을 확인하세요.")

# ── 파일 업로드 ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "학급 응답 데이터 업로드 (CSV)",
    type="csv",
    help="S_ID, Q1, Q2, ..., Q32 컬럼이 포함된 CSV 파일"
)

if uploaded is None:
    st.info("👆 CSV 파일을 업로드하면 인지진단이 시작됩니다.")
    st.stop()

df_wide = pd.read_csv(uploaded)

# 형식 검증
required_cols = ["S_ID"] + [f"Q{i}" for i in range(1, N_ITEM + 1)]
missing = [c for c in required_cols if c not in df_wide.columns]
if missing:
    st.error(f"❌ 필수 컬럼 누락: {missing}")
    st.stop()

st.success(f"✅ 데이터 로드 완료: **{len(df_wide)}명**, **{N_ITEM}문항**")

with st.expander("📄 데이터 미리보기", expanded=False):
    st.dataframe(df_wide, use_container_width=True)

# ── 진단 실행 ─────────────────────────────────────────────────────────────────
if st.button("🔍 인지진단 실행", type="primary", use_container_width=True):
    net = load_model()
    theta_ref, sr_ref, coords_2d_ref, coords_3d_ref, \
        reducer_2d, reducer_3d = load_reference()
    kc_mean_ref = theta_ref.mean(axis=0)

    with st.spinner("θ 계산 중 (즉각 진단)..."):
        theta_new, sr_new, coords_new_2d, coords_new_3d = diagnose(
            df_wide, net, reducer_2d, reducer_3d
        )

    original_sids = df_wide["S_ID"].tolist()
    n_new = len(df_wide)
    percentiles = [int(np.mean(sr_ref <= s) * 100) for s in sr_new]

    # 결과 저장
    st.session_state["result"] = dict(
        theta_new=theta_new, sr_new=sr_new,
        coords_new_2d=coords_new_2d, coords_new_3d=coords_new_3d,
        original_sids=original_sids, percentiles=percentiles,
        theta_ref=theta_ref, sr_ref=sr_ref,
        coords_2d_ref=coords_2d_ref, coords_3d_ref=coords_3d_ref,
        kc_mean_ref=kc_mean_ref, n_new=n_new
    )
    st.success("✅ 진단 완료!")

# ── 결과 표시 ─────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.stop()

R = st.session_state["result"]
theta_new    = R["theta_new"]
sr_new       = R["sr_new"]
coords_new_2d = R["coords_new_2d"]
coords_new_3d = R["coords_new_3d"]
original_sids = R["original_sids"]
percentiles   = R["percentiles"]
theta_ref     = R["theta_ref"]
sr_ref        = R["sr_ref"]
coords_2d_ref = R["coords_2d_ref"]
coords_3d_ref = R["coords_3d_ref"]
kc_mean_ref   = R["kc_mean_ref"]
n_new         = R["n_new"]

st.divider()
tab_class, tab_umap, tab_individual = st.tabs(
    ["📊 학급 요약", "🌐 3D UMAP", "👤 개별 학생 프로파일"]
)

# ── 탭1: 학급 요약 ────────────────────────────────────────────────────────────
with tab_class:
    st.subheader("학급 요약 지표")
    n_high = int((sr_new >= 0.6).sum())
    n_mid  = int(((sr_new >= 0.3) & (sr_new < 0.6)).sum())
    n_low  = int((sr_new < 0.3).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("학생 수", f"{n_new}명")
    c2.metric("평균 정답률", f"{sr_new.mean():.1%}",
              delta=f"{sr_new.mean() - sr_ref.mean():+.1%} vs 훈련 학생")
    c3.metric("상위 (≥60%)", f"{n_high}명 ({n_high/n_new:.0%})")
    c4.metric("하위 (<30%)", f"{n_low}명 ({n_low/n_new:.0%})")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**KC별 평균 숙달도 (학급 vs 훈련 평균)**")
        fig_kc, ax_kc = plt.subplots(figsize=(6, 3.5))
        kc_class_mean = theta_new.mean(axis=0)
        x_pos = np.arange(len(KC_NAMES))
        w = 0.38
        ax_kc.bar(x_pos - w/2, kc_class_mean, w,
                  color="#3498DB", alpha=0.85, label="학급 평균")
        ax_kc.bar(x_pos + w/2, kc_mean_ref, w,
                  color="#BDC3C7", alpha=0.85, label="훈련 평균")
        for xi, val in zip(x_pos - w/2, kc_class_mean):
            ax_kc.text(xi, val + 0.01, f"{val:.2f}", ha="center",
                       va="bottom", fontsize=7, color="#2471A3")
        ax_kc.set_xticks(x_pos)
        ax_kc.set_xticklabels(KC_NAMES)
        ax_kc.set_ylim(0, 1)
        ax_kc.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax_kc.legend(fontsize=9)
        ax_kc.set_ylabel("숙달도 (θ)")
        plt.tight_layout()
        st.pyplot(fig_kc)
        plt.close(fig_kc)

    with col_b:
        st.markdown("**KC별 숙달도 분포 (학급 내)**")
        fig_box, ax_box = plt.subplots(figsize=(6, 3.5))
        bp = ax_box.boxplot(
            [theta_new[:, k] for k in range(len(KC_NAMES))],
            labels=KC_NAMES, patch_artist=True,
            medianprops=dict(color="black", linewidth=2)
        )
        colors_box = ["#AED6F1", "#A9DFBF", "#FAD7A0",
                      "#F1948A", "#C39BD3", "#85C1E9"]
        for patch, col in zip(bp["boxes"], colors_box):
            patch.set_facecolor(col); patch.set_alpha(0.8)
        ax_box.plot(range(1, len(KC_NAMES)+1), kc_mean_ref,
                    "D", color="#2C3E50", markersize=5, label="훈련 평균")
        ax_box.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax_box.set_ylim(0, 1); ax_box.legend(fontsize=8)
        ax_box.set_ylabel("숙달도 (θ)")
        plt.tight_layout()
        st.pyplot(fig_box)
        plt.close(fig_box)

    st.divider()
    st.markdown("**학생 × KC 숙달도 히트맵 (정답률 내림차순)**")
    sort_order   = np.argsort(sr_new)[::-1]
    theta_sorted = theta_new[sort_order]
    sid_sorted   = [original_sids[i] for i in sort_order]
    sr_sorted    = sr_new[sort_order]

    fig_heat, ax_heat = plt.subplots(figsize=(max(10, n_new * 0.55), 3))
    im = ax_heat.imshow(theta_sorted.T, aspect="auto",
                        cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_heat, label="θ", shrink=0.85)
    ax_heat.set_yticks(range(len(KC_NAMES)))
    ax_heat.set_yticklabels(KC_NAMES, fontsize=9)
    ax_heat.set_xticks(range(n_new))
    ax_heat.set_xticklabels(
        [f"S{s}\n{r:.0%}" for s, r in zip(sid_sorted, sr_sorted)],
        fontsize=8
    )
    if n_new <= 40:
        for ki in range(len(KC_NAMES)):
            for si in range(n_new):
                val = theta_sorted[si, ki]
                tc = "white" if val < 0.25 or val > 0.80 else "black"
                ax_heat.text(si, ki, f"{val:.2f}", ha="center",
                             va="center", fontsize=7, color=tc)
    ax_heat.set_title("학생별 KC 숙달도 (정답률 내림차순)")
    plt.tight_layout()
    st.pyplot(fig_heat)
    plt.close(fig_heat)

    # CSV 다운로드
    df_result = pd.DataFrame(theta_new, columns=KC_NAMES)
    df_result.insert(0, "S_ID", original_sids)
    df_result["score_rate"] = sr_new
    df_result["percentile"] = percentiles
    st.download_button(
        "⬇️ 진단 결과 CSV 다운로드",
        data=df_result.to_csv(index=False).encode("utf-8-sig"),
        file_name="theta_result.csv",
        mime="text/csv"
    )

# ── 탭2: 3D UMAP ──────────────────────────────────────────────────────────────
with tab_umap:
    st.subheader("3D UMAP — 전체 분포 내 학급 위치")
    st.caption("점을 드래그해 회전 · 스크롤로 확대/축소 · 점에 마우스를 올리면 정보 표시")

    fig_3d = plotly_umap_3d(
        coords_3d_ref, sr_ref,
        coords_new_3d, sr_new, original_sids,
        highlight_idx=None
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ── 탭3: 개별 학생 ────────────────────────────────────────────────────────────
with tab_individual:
    sid_options = [f"S{s}  ({sr_new[i]:.1%})" for i, s in enumerate(original_sids)]
    selected    = st.selectbox("학생 선택", sid_options)
    sel_idx     = sid_options.index(selected)
    sel_sid     = original_sids[sel_idx]
    sel_theta   = theta_new[sel_idx]
    sel_sr      = sr_new[sel_idx]
    sel_pct     = percentiles[sel_idx]
    sel_color   = ("#2471A3" if sel_sr >= 0.6
                   else "#E67E22" if sel_sr >= 0.3 else "#C0392B")

    st.markdown(
        f"### Student S{sel_sid} &nbsp; | &nbsp; "
        f"정답률 **{sel_sr:.1%}** &nbsp; | &nbsp; "
        f"백분위 **{sel_pct}%** (훈련 학생 {N_TRAIN_USER}명 기준)"
    )

    # 취약/강점 KC
    weak   = [KC_NAMES[k] for k, v in enumerate(sel_theta) if v < 0.4]
    strong = [KC_NAMES[k] for k, v in enumerate(sel_theta) if v >= 0.7]
    col_info1, col_info2 = st.columns(2)
    col_info1.success(f"**강점 KC**: {', '.join(strong) if strong else '없음'}")
    col_info2.error(f"**취약 KC**: {', '.join(weak) if weak else '없음'}")

    st.divider()
    c1, c2, c3 = st.columns([1, 1.2, 1.2])

    with c1:
        st.markdown("**레이더 차트**")
        fig_r = radar_chart(sel_theta, KC_NAMES, sel_color,
                            f"S{sel_sid}", ref_vals=kc_mean_ref)
        st.pyplot(fig_r); plt.close(fig_r)

    with c2:
        st.markdown("**UMAP 위치 (2D)**")
        fig_u = umap_2d_fig(coords_2d_ref, sr_ref,
                            coords_new_2d, sr_new,
                            highlight_idx=sel_idx)
        st.pyplot(fig_u); plt.close(fig_u)

    with c3:
        st.markdown("**KC 숙달도**")
        fig_b = kc_bar_chart(sel_theta, KC_NAMES, ref_vals=kc_mean_ref)
        st.pyplot(fig_b); plt.close(fig_b)

    # 3D UMAP (해당 학생 강조)
    st.markdown("**3D UMAP (이 학생 강조)**")
    fig_3d_ind = plotly_umap_3d(
        coords_3d_ref, sr_ref,
        coords_new_3d, sr_new, original_sids,
        highlight_idx=sel_idx
    )
    st.plotly_chart(fig_3d_ind, use_container_width=True)

    # 개별 리포트 PNG 다운로드
    fig_dl = plt.figure(figsize=(15, 5))
    fig_dl.patch.set_facecolor("white")
    fig_dl.suptitle(
        f"Cognitive Diagnostic Report — S{sel_sid}  |  "
        f"Correct Rate: {sel_sr:.1%}  |  Percentile: {sel_pct}%",
        fontsize=12, fontweight="bold"
    )
    gs_dl = gridspec.GridSpec(1, 3, figure=fig_dl, wspace=0.38)

    ax_r = fig_dl.add_subplot(gs_dl[0], polar=True)
    n = len(KC_NAMES)
    angs = np.linspace(0, 2*np.pi, n, endpoint=False).tolist() + [0]
    rv = kc_mean_ref.tolist() + [kc_mean_ref[0]]
    ax_r.plot(angs, rv, "-", color="gray", linewidth=1, alpha=0.5, label="훈련 평균")
    ax_r.fill(angs, rv, color="gray", alpha=0.1)
    sv = sel_theta.tolist() + [sel_theta[0]]
    ax_r.plot(angs, sv, "o-", color=sel_color, linewidth=2, label=f"S{sel_sid}")
    ax_r.fill(angs, sv, color=sel_color, alpha=0.22)
    ax_r.set_xticks(angs[:-1]); ax_r.set_xticklabels(KC_NAMES, fontsize=8)
    ax_r.set_ylim(0, 1); ax_r.legend(fontsize=7, bbox_to_anchor=(1.3, 1.15))
    ax_r.set_title("Knowledge Proficiencies", pad=12, fontsize=10)

    ax_u = fig_dl.add_subplot(gs_dl[1])
    ax_u.set_facecolor("#F2F4F6")
    ax_u.scatter(coords_2d_ref[:, 0], coords_2d_ref[:, 1],
                 c=sr_ref, cmap="RdYlBu", s=10, alpha=0.3, vmin=0, vmax=1)
    x_h, y_h = coords_new_2d[sel_idx]
    ax_u.scatter(x_h, y_h, s=400, c="white",
                 edgecolors=sel_color, linewidths=2.5, zorder=6)
    ax_u.scatter(x_h, y_h, s=150, marker="*", c=sel_color,
                 edgecolors="black", zorder=7)
    ax_u.text(0.97, 0.97, f"S{sel_sid} ({sel_sr:.0%})",
              transform=ax_u.transAxes, fontsize=8, fontweight="bold",
              color=sel_color, ha="right", va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="white",
                        ec=sel_color, lw=1.2, alpha=0.9))
    ax_u.set_title("Relative Position (UMAP)", fontsize=10)
    ax_u.set_xlabel("Dim 1", fontsize=8); ax_u.set_ylabel("Dim 2", fontsize=8)

    ax_b = fig_dl.add_subplot(gs_dl[2])
    cols_b = ["#E74C3C" if v < 0.4 else "#F39C12" if v < 0.6 else "#2ECC71"
              for v in sel_theta]
    bars_dl = ax_b.barh(KC_NAMES, sel_theta, color=cols_b, alpha=0.85, height=0.6)
    ax_b.set_xlim(0, 1)
    ax_b.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    for i, mv in enumerate(kc_mean_ref):
        ax_b.plot(mv, i, "|", color="#333", markersize=12,
                  markeredgewidth=2, label="훈련 평균" if i == 0 else "")
    for bar, val in zip(bars_dl, sel_theta):
        ax_b.text(min(val+0.02, 0.93), bar.get_y()+bar.get_height()/2,
                  f"{val:.3f}", va="center", fontsize=8)
    ax_b.legend(fontsize=7, loc="lower right")
    ax_b.set_xlabel("Proficiency (θ)", fontsize=9)
    ax_b.set_title("KC Proficiency", fontsize=10)

    plt.tight_layout()
    dl_buf = fig_to_bytes(fig_dl)
    plt.close(fig_dl)
    st.download_button(
        f"⬇️ S{sel_sid} 리포트 PNG 다운로드",
        data=dl_buf,
        file_name=f"report_S{sel_sid}.png",
        mime="image/png"
    )
