# -*- coding: utf-8 -*-
"""
G-NCDM 개별 인지진단 리포트 생성 스크립트.
논문 Figure 11과 유사한 형식으로 출력합니다.

터미널에서 다음과 같이 실행:
    python diagnostic_report.py

출력:
  result/QPCS/diagnosis/
    ├── theta_all.csv          — 전체 학생 KC 숙달도 + 정답률
    ├── umap_all.png           — 전체 학생 UMAP 분포도
    ├── individual/
    │   ├── student_0000.png   — 개별 학생 리포트 (레이더 + UMAP)
    │   ├── student_0001.png
    │   └── ...
    └── sample_report.png      — 상위/하위 대표 학생 비교 리포트
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.font_manager as fm

def _set_korean_font():
    candidates = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic",
                  "Malgun Gothic", "NanumBarunGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()

import torch
from tqdm import tqdm
from umap import UMAP
from train import IDCDataset

# ── 설정 ─────────────────────────────────────────────────────────────────────

RESULT_PATH   = "result/QPCS"
MODEL_PATH    = "result/QPCS/params_32_32.pt"
TRAIN_FILE    = "data/QPCS_train.csv"
VALID_FILE    = "data/QPCS_valid.csv"
TEST_FILE     = "data/QPCS_test.csv"
Q_MATRIX_FILE = "data/Q_matrix.npy"

N_USER = 730
N_ITEM = 32
N_KNOW = 6
KC_NAMES = ["KC1", "KC2", "KC3", "KC4", "KC5", "KC6"]

# 개별 리포트를 생성할 test 학생 수 (너무 많으면 오래 걸림)
N_INDIVIDUAL_REPORTS = 10   # 상위 10명 + 하위 10명

OUT_DIR = os.path.join(RESULT_PATH, "diagnosis")
IND_DIR = os.path.join(OUT_DIR, "individual")
os.makedirs(IND_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ── 모델 및 데이터 로드 ───────────────────────────────────────────────────────

print("모델 로딩 중...")
net = torch.load(MODEL_PATH, weights_only=False)
net.eval()
device = net.device

Q_mat = np.load(Q_MATRIX_FILE)
df_all = pd.concat([
    pd.read_csv(TRAIN_FILE),
    pd.read_csv(VALID_FILE),
    pd.read_csv(TEST_FILE)
], ignore_index=True)
df_test = pd.read_csv(TEST_FILE)
test_user_ids = sorted(df_test["user_id"].unique())

dataset_all = IDCDataset(df_all, n_user=N_USER, n_item=N_ITEM)

# ── 전체 학생 θ 계산 ──────────────────────────────────────────────────────────

print("전체 학생 θ 계산 중...")
theta_list, score_rates = [], []

with torch.no_grad():
    for uid in tqdm(range(N_USER)):
        user_log = torch.Tensor([dataset_all.log_mat[uid]]).to(device)
        theta = net.diagnose_theta(user_log).cpu().numpy().flatten()
        theta_list.append(theta)

        sv = dataset_all.log_mat[uid]
        n_c = len(sv[sv > 0])
        n_a = len(sv[sv != 0])
        score_rates.append(n_c / n_a if n_a > 0 else 0.0)

theta_mat   = np.stack(theta_list)        # (730, 6)
score_rates = np.array(score_rates)       # (730,)

# ── theta_all.csv 저장 ───────────────────────────────────────────────────────

df_theta = pd.DataFrame(theta_mat, columns=KC_NAMES)
df_theta.insert(0, "user_id", range(N_USER))
df_theta["score_rate"] = score_rates
df_theta["is_test"] = df_theta["user_id"].isin(test_user_ids)
df_theta.to_csv(os.path.join(OUT_DIR, "theta_all.csv"), index=False)
print("theta_all.csv 저장 완료")

# ── UMAP 계산 ────────────────────────────────────────────────────────────────

print("UMAP 계산 중 (시간이 걸릴 수 있습니다)...")
reducer = UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=15,      # 로컬 구조 강조 → 내부 패턴 차이 부각
    min_dist=0.15,       # 클러스터 내부 밀집 / 클러스터 간 분리
    spread=2.0,          # 전체 분포 확장
    init="random",       # spectral init 실패 방지
    metric="cosine"      # 절대값 아닌 방향(패턴) 차이 기반 → 저성취군 내부 분리
)
umap_coords = reducer.fit_transform(theta_mat)   # (730, 2)

# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

def draw_radar(ax, theta_vals, kc_names, color, label, fill_alpha=0.25):
    """레이더 차트 그리기"""
    n = len(kc_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    vals = theta_vals.tolist() + theta_vals[:1].tolist()

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, vals, "o-", color=color, linewidth=2, label=label)
    ax.fill(angles, vals, color=color, alpha=fill_alpha)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(kc_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=7, color="gray")
    ax.grid(color="gray", alpha=0.3)


def draw_umap(ax, umap_coords, score_rates, highlight_ids, highlight_labels,
              highlight_colors, highlight_markers):
    """UMAP 분포도: 배경 밀도 + 개별 학생 위치 강조"""
    from scipy.stats import gaussian_kde

    ax.set_facecolor("#F2F4F6")

    # ── 1. KDE 밀도 등고선 (배경 구조 표시) ──────────────────────────────
    try:
        xy = umap_coords.T
        kde = gaussian_kde(xy, bw_method=0.35)
        xmin, xmax = xy[0].min() - 1, xy[0].max() + 1
        ymin, ymax = xy[1].min() - 1, xy[1].max() + 1
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 120),
                             np.linspace(ymin, ymax, 120))
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=6, cmap="Greys", alpha=0.18, zorder=1)
        ax.contour(xx, yy, zz, levels=4, colors="gray", linewidths=0.4,
                   alpha=0.35, zorder=1)
    except Exception:
        pass

    # ── 2. 전체 학생 산점도 (배경, 정답률 색상) ───────────────────────────
    sc = ax.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=score_rates, cmap="RdYlBu",
        s=18, alpha=0.45, vmin=0, vmax=1,
        edgecolors="none", zorder=2
    )
    plt.colorbar(sc, ax=ax, label="Score Rate", shrink=0.7, pad=0.02)

    # ── 3. 강조 학생 마커 + 주석 화살표 ──────────────────────────────────
    for uid, label, color, marker in zip(
        highlight_ids, highlight_labels, highlight_colors, highlight_markers
    ):
        x_h, y_h = umap_coords[uid, 0], umap_coords[uid, 1]

        # 큰 외곽 원 (후광 효과)
        ax.scatter(x_h, y_h, s=500, c="white",
                   edgecolors=color, linewidths=3, zorder=6, alpha=0.85)
        # 실제 마커
        ax.scatter(x_h, y_h, s=200, marker=marker, c=color,
                   edgecolors="black", linewidths=1.2, zorder=7, label=label)

        # 화살표 + 텍스트 박스
        x_range = umap_coords[:, 0].max() - umap_coords[:, 0].min()
        y_range = umap_coords[:, 1].max() - umap_coords[:, 1].min()
        dx = x_range * 0.22
        dy = y_range * 0.22
        ax.annotate(
            label,
            xy=(x_h, y_h), xytext=(x_h + dx, y_h + dy),
            fontsize=8, fontweight="bold", color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=color, lw=1.2, alpha=0.9),
            zorder=8
        )

    ax.set_xlabel("UMAP Dim 1", fontsize=9)
    ax.set_ylabel("UMAP Dim 2", fontsize=9)
    ax.set_title("Relative Position (UMAP)", fontsize=11)
    ax.tick_params(labelsize=8)


def draw_kc_bar(ax, theta_vals, kc_names):
    """KC별 숙달도 바 차트"""
    colors = ["#E74C3C" if v < 0.4 else "#F39C12" if v < 0.6 else "#2ECC71"
              for v in theta_vals]
    bars = ax.barh(kc_names, theta_vals, color=colors, alpha=0.85, height=0.6)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Proficiency (θ)")
    ax.set_title("KC Proficiency Detail", fontsize=12)
    for bar, val in zip(bars, theta_vals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)


# ── 전체 UMAP 분포도 저장 ────────────────────────────────────────────────────

from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor("#F2F4F6")
fig.patch.set_facecolor("white")

# ── KDE 밀도 등고선 ──────────────────────────────────────────────────────
try:
    xy = umap_coords.T
    kde = gaussian_kde(xy, bw_method=0.3)
    xmin, xmax = xy[0].min() - 2, xy[0].max() + 2
    ymin, ymax = xy[1].min() - 2, xy[1].max() + 2
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 150),
                         np.linspace(ymin, ymax, 150))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=8, cmap="Greys", alpha=0.20, zorder=1)
    ax.contour(xx, yy, zz, levels=5, colors="gray", linewidths=0.5,
               alpha=0.40, zorder=1)
except Exception:
    pass

# ── 전체 학생 산점도 (정답률 색상) ──────────────────────────────────────
sc = ax.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=score_rates, cmap="RdYlBu",
    s=50, alpha=0.70, vmin=0, vmax=1,
    edgecolors="white", linewidths=0.4, zorder=2
)
cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label("Correct Rate  (blue = high  /  red = low)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

# ── test 학생: 테두리 강조 ───────────────────────────────────────────────
test_coords = umap_coords[test_user_ids]
test_sr     = score_rates[test_user_ids]
ax.scatter(
    test_coords[:, 0], test_coords[:, 1],
    c=test_sr, cmap="RdYlBu",
    s=90, alpha=1.0, vmin=0, vmax=1,
    edgecolors="black", linewidths=1.3, zorder=4, label="Test students"
)

# ── 정답률 구간별 무게중심 레이블 ────────────────────────────────────────
for label, mask, color in [
    ("High\n(≥60%)",  score_rates >= 0.6,  "#1565C0"),
    ("Mid\n(30-60%)", (score_rates >= 0.3) & (score_rates < 0.6), "#E65100"),
    ("Low\n(<30%)",   score_rates < 0.3,   "#B71C1C"),
]:
    if mask.sum() == 0:
        continue
    cx = umap_coords[mask, 0].mean()
    cy = umap_coords[mask, 1].mean()
    ax.text(cx, cy, label, ha="center", va="center", fontsize=9,
            fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=color, alpha=0.75, lw=1.2), zorder=9)

ax.set_title("UMAP — All Students' Cognitive State (θ)",
             fontweight="bold", fontsize=14, pad=12)
ax.set_xlabel("UMAP Dimension 1", fontsize=11)
ax.set_ylabel("UMAP Dimension 2", fontsize=11)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

n_high = int((score_rates >= 0.6).sum())
n_mid  = int(((score_rates >= 0.3) & (score_rates < 0.6)).sum())
n_low  = int((score_rates < 0.3).sum())
ax.text(0.01, 0.01,
        f"High (≥60%): {n_high}명  |  Mid (30~60%): {n_mid}명  |  Low (<30%): {n_low}명",
        transform=ax.transAxes, fontsize=9, color="gray",
        verticalalignment="bottom")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "umap_all.png"), dpi=180, bbox_inches="tight")
plt.close()
print("umap_all.png 저장 완료")


# ── 개별 학생 리포트 ──────────────────────────────────────────────────────────

# test 학생을 정답률 기준으로 정렬, 상위 N/2명 + 하위 N/2명 선택
test_scores = score_rates[test_user_ids]
sorted_by_score = np.argsort(test_scores)[::-1]
top_half  = [test_user_ids[i] for i in sorted_by_score[:N_INDIVIDUAL_REPORTS // 2]]
bot_half  = [test_user_ids[i] for i in sorted_by_score[-N_INDIVIDUAL_REPORTS // 2:]]
report_ids = top_half + bot_half

print(f"\n개별 리포트 생성 중 ({len(report_ids)}명)...")

for uid in tqdm(report_ids):
    theta_vals = theta_mat[uid]
    sr = score_rates[uid]
    percentile = int(np.mean(score_rates <= sr) * 100)

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(
        f"Diagnostic Report — Student {uid:04d}"
        f"  |  Score Rate: {sr:.1%}  |  Percentile: {percentile}%",
        fontsize=14, fontweight="bold"
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # 왼쪽: 레이더 차트
    ax_radar = fig.add_subplot(gs[0], polar=True)
    draw_radar(ax_radar, theta_vals, KC_NAMES,
               color="#3498DB", label=f"Student {uid:04d}")
    ax_radar.set_title("Knowledge Proficiencies", pad=15, fontsize=12)

    # 가운데: UMAP
    ax_umap = fig.add_subplot(gs[1])
    draw_umap(
        ax_umap, umap_coords, score_rates,
        highlight_ids=[uid],
        highlight_labels=[f"Student {uid:04d} ({sr:.0%})"],
        highlight_colors=["#E74C3C" if sr < 0.4 else "#2ECC71"],
        highlight_markers=["*" if sr >= 0.4 else "^"]
    )

    # 오른쪽: KC별 바 차트
    ax_bar = fig.add_subplot(gs[2])
    draw_kc_bar(ax_bar, theta_vals, KC_NAMES)

    plt.savefig(os.path.join(IND_DIR, f"student_{uid:04d}.png"),
                bbox_inches="tight")
    plt.close()


# ── 대표 비교 리포트 (상위 vs 하위) ──────────────────────────────────────────

print("\n대표 비교 리포트 생성 중...")

top_uid = top_half[0]
bot_uid = bot_half[-1]
top_theta, bot_theta = theta_mat[top_uid], theta_mat[bot_uid]
top_sr,    bot_sr    = score_rates[top_uid], score_rates[bot_uid]

fig = plt.figure(figsize=(16, 7))
fig.suptitle("Sample Diagnostic Report — G-NCDM",
             fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# 레이더 (두 학생 겹쳐서)
ax_radar = fig.add_subplot(gs[0], polar=True)
draw_radar(ax_radar, top_theta, KC_NAMES,
           color="#3498DB", label=f"Student {top_uid} (top, {top_sr:.0%})", fill_alpha=0.2)
draw_radar(ax_radar, bot_theta, KC_NAMES,
           color="#E74C3C", label=f"Student {bot_uid} (low, {bot_sr:.0%})", fill_alpha=0.2)
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
ax_radar.set_title("Knowledge Proficiencies", pad=15, fontsize=12)

# UMAP (두 학생 표시)
ax_umap = fig.add_subplot(gs[1])
draw_umap(
    ax_umap, umap_coords, score_rates,
    highlight_ids=[top_uid, bot_uid],
    highlight_labels=[f"Student {top_uid} (top {int(np.mean(score_rates<=top_sr)*100)}%)",
                      f"Student {bot_uid} (bot {int(np.mean(score_rates<=bot_sr)*100)}%)"],
    highlight_colors=["#3498DB", "#E74C3C"],
    highlight_markers=["*", "^"]
)


# KC 비교 바 (나란히)
ax_bar = fig.add_subplot(gs[2])
x = np.arange(N_KNOW)
w = 0.35
ax_bar.barh(x + w/2, top_theta, w, color="#3498DB", alpha=0.85,
            label=f"Student {top_uid} ({top_sr:.0%})")
ax_bar.barh(x - w/2, bot_theta, w, color="#E74C3C", alpha=0.85,
            label=f"Student {bot_uid} ({bot_sr:.0%})")
ax_bar.set_yticks(x)
ax_bar.set_yticklabels(KC_NAMES)
ax_bar.set_xlim(0, 1)
ax_bar.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax_bar.set_xlabel("Proficiency (θ)")
ax_bar.set_title("KC Proficiency Comparison", fontsize=12)
ax_bar.legend(fontsize=9)

plt.savefig(os.path.join(OUT_DIR, "sample_report.png"), bbox_inches="tight")
plt.close()

# ── 요약 출력 ─────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("진단 리포트 생성 완료!")
print(f"저장 위치: {OUT_DIR}/")
print(f"\n  theta_all.csv          — 전체 {N_USER}명 KC 숙달도 + 정답률")
print(f"  umap_all.png           — 전체 학생 UMAP 분포도")
print(f"  sample_report.png      — 상위/하위 대표 학생 비교")
print(f"  individual/ ({len(report_ids)}개) — 개별 학생 리포트")
print("="*60)

print("\n[KC 평균 숙달도 요약]")
for i, kc in enumerate(KC_NAMES):
    mean_all  = theta_mat[:, i].mean()
    mean_test = theta_mat[test_user_ids, i].mean()
    print(f"  {kc}: 전체 평균={mean_all:.3f}  /  테스트 학생 평균={mean_test:.3f}")
