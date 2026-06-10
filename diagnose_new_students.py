# -*- coding: utf-8 -*-
"""
새로운 학생 데이터(QPCS_new.csv)에 대한 즉각 인지진단 + 시각화 리포트 생성.

[입력]
  QPCS_new.csv         — 새 학생 응답 (wide format: S_ID, Q1~Q32)
  result/QPCS/params_32_32.pt  — 훈련된 GNCDM 모델

[출력]
  result/QPCS/new_students/
    ├── theta_new.csv             — 새 학생 KC 숙달도 + 정답률
    ├── umap_new_in_context.png   — 기존 730명 분포 위에 새 학생 위치 표시
    └── individual/
        ├── new_student_01.png    — 개별 학생 리포트 (레이더 + UMAP + KC바)
        └── ...

[실행]
  python diagnose_new_students.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

# ── 한글 폰트 설정 (macOS) ────────────────────────────────────────────────────
def _set_korean_font():
    candidates = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic",
                  "Malgun Gothic", "NanumBarunGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False   # 마이너스 부호 깨짐 방지

_set_korean_font()

import torch
from tqdm import tqdm
from scipy.stats import gaussian_kde
from umap import UMAP
from train import IDCDataset

# ── 설정 ─────────────────────────────────────────────────────────────────────

NEW_DATA_FILE   = "QPCS_new.csv"           # 새 학생 응답 (wide format)
MODEL_PATH      = "result/QPCS/params_32_32.pt"   # 검증용 모델
# MODEL_PATH    = "result/QPCS_full/params_32_32.pt"  # 배포용 모델 (전체 학습 후)
Q_MATRIX_FILE   = "data/Q_matrix.npy"
TRAIN_THETA_CSV = "result/QPCS/diagnosis/theta_all.csv"  # 기존 730명 θ (참조용)
TRAIN_ALL_CSV   = "data/QPCS_full_train.csv"             # 기존 전체 응답 (없으면 train+valid+test 사용)

N_TRAIN_USER = 730
N_ITEM       = 32
N_KNOW       = 6
KC_NAMES     = ["KC1", "KC2", "KC3", "KC4", "KC5", "KC6"]

OUT_DIR = "result/QPCS/new_students"
IND_DIR = os.path.join(OUT_DIR, "individual")
os.makedirs(IND_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ── 1. 새 학생 데이터 전처리 (wide → long + log_mat) ─────────────────────────

print("새 학생 데이터 로드 중...")
df_wide = pd.read_csv(NEW_DATA_FILE)
q_cols  = [c for c in df_wide.columns if c.startswith("Q")]

records = []
for local_idx, (_, row) in enumerate(df_wide.iterrows()):
    original_sid = int(row["S_ID"])
    for j, qcol in enumerate(q_cols):
        records.append({
            "user_id":      local_idx,          # 0-indexed (새 학생 내부 번호)
            "original_sid": original_sid,       # 원래 S_ID
            "item_id":      j,
            "score":        int(row[qcol]),
        })

df_new_long = pd.DataFrame(records)
n_new = df_wide.shape[0]
original_sids = [int(r["S_ID"]) for _, r in df_wide.iterrows()]

print(f"새 학생 수: {n_new}명  |  응답 수: {len(df_new_long):,}행")

# log_mat 직접 구성 (IDCDataset은 N_USER 크기 할당 필요)
log_mat_new = np.zeros((n_new, N_ITEM))
for _, row in df_new_long.iterrows():
    uid  = int(row["user_id"])
    iid  = int(row["item_id"])
    sc   = int(row["score"])
    log_mat_new[uid, iid] = (sc - 0.5) * 2   # +1 정답, -1 오답

# ── 2. 모델 로드 ──────────────────────────────────────────────────────────────

print("모델 로딩 중...")
net = torch.load(MODEL_PATH, weights_only=False)
net.eval()
device = net.device
print(f"device: {device}")

# ── 3. 새 학생 θ 즉각 진단 (재학습 없음) ──────────────────────────────────────

print("새 학생 θ 계산 중 (즉각 진단)...")
theta_new_list  = []
score_rate_new  = []

with torch.no_grad():
    for uid in tqdm(range(n_new)):
        user_log = torch.Tensor([log_mat_new[uid]]).to(device)
        theta = net.diagnose_theta(user_log).cpu().numpy().flatten()
        theta_new_list.append(theta)

        sv = log_mat_new[uid]
        n_correct = (sv > 0).sum()
        n_answered = (sv != 0).sum()
        score_rate_new.append(n_correct / n_answered if n_answered > 0 else 0.0)

theta_new  = np.stack(theta_new_list)   # (n_new, 6)
sr_new     = np.array(score_rate_new)   # (n_new,)

# ── 4. 결과 CSV 저장 ──────────────────────────────────────────────────────────

df_theta_new = pd.DataFrame(theta_new, columns=KC_NAMES)
df_theta_new.insert(0, "original_sid", original_sids)
df_theta_new["score_rate"] = sr_new

# 백분위 (기존 학생 θ가 있으면 기준으로 사용, 없으면 새 학생끼리 비교)
if os.path.exists(TRAIN_THETA_CSV):
    df_train_theta = pd.read_csv(TRAIN_THETA_CSV)
    ref_sr = df_train_theta["score_rate"].values
    percentiles = [int(np.mean(ref_sr <= s) * 100) for s in sr_new]
else:
    percentiles = [int(np.mean(sr_new <= s) * 100) for s in sr_new]

df_theta_new["percentile_vs_train"] = percentiles
df_theta_new.to_csv(os.path.join(OUT_DIR, "theta_new.csv"), index=False)
print("theta_new.csv 저장 완료")

print("\n=== 새 학생 진단 결과 미리보기 ===")
print(df_theta_new.to_string(index=False))

# ── 5. UMAP 계산 (기존 학생 + 새 학생 통합) ──────────────────────────────────

print("\nUMAP 계산 중...")

# 기존 학생 θ 로드
if os.path.exists(TRAIN_THETA_CSV):
    df_ref = pd.read_csv(TRAIN_THETA_CSV)
    theta_ref = df_ref[KC_NAMES].values
    sr_ref    = df_ref["score_rate"].values
    print(f"기존 학생 θ 로드: {len(theta_ref)}명 (theta_all.csv)")
else:
    print("기존 학생 θ 파일 없음 → 새 학생끼리만 UMAP")
    theta_ref = None
    sr_ref    = None

if theta_ref is not None:
    theta_all_combined = np.vstack([theta_ref, theta_new])   # (730+n_new, 6)
    sr_all_combined    = np.concatenate([sr_ref, sr_new])
    new_indices        = np.arange(len(theta_ref), len(theta_ref) + n_new)
else:
    theta_all_combined = theta_new
    sr_all_combined    = sr_new
    new_indices        = np.arange(n_new)

reducer = UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=15,
    min_dist=0.15,
    spread=2.0,
    init="random",
    metric="cosine"
)
umap_coords = reducer.fit_transform(theta_all_combined)
print("2D UMAP 완료")

# 3D UMAP
print("3D UMAP 계산 중...")
reducer_3d = UMAP(
    n_components=3,
    random_state=42,
    n_neighbors=15,
    min_dist=0.15,
    spread=2.0,
    init="random",
    metric="cosine"
)
umap_coords_3d = reducer_3d.fit_transform(theta_all_combined)
print("3D UMAP 완료")

# ── 6. 헬퍼 함수 ─────────────────────────────────────────────────────────────

def draw_radar(ax, theta_vals, kc_names, color, label, fill_alpha=0.25):
    n = len(kc_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    vals = theta_vals.tolist() + theta_vals[:1].tolist()
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, vals, "o-", color=color, linewidth=2.5, label=label)
    ax.fill(angles, vals, color=color, alpha=fill_alpha)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(kc_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=7, color="gray")
    ax.grid(color="gray", alpha=0.3)


def draw_umap_new(ax, umap_coords, sr_all, new_idx, label, color):
    """기존 학생 배경 + 새 학생 강조 (주석이 플롯 안에 머물도록 처리)"""
    ax.set_facecolor("#F2F4F6")

    # KDE 밀도 등고선
    try:
        xy  = umap_coords.T
        kde = gaussian_kde(xy, bw_method=0.35)
        xmin, xmax = xy[0].min() - 1, xy[0].max() + 1
        ymin, ymax = xy[1].min() - 1, xy[1].max() + 1
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100),
                             np.linspace(ymin, ymax, 100))
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=6, cmap="Greys", alpha=0.18, zorder=1)
        ax.contour(xx, yy, zz, levels=4, colors="gray",
                   linewidths=0.4, alpha=0.35, zorder=1)
    except Exception:
        pass

    # 전체 배경 산점도 (기존 학생)
    mask_bg = np.ones(len(umap_coords), dtype=bool)
    mask_bg[new_idx] = False
    sc = ax.scatter(
        umap_coords[mask_bg, 0], umap_coords[mask_bg, 1],
        c=sr_all[mask_bg], cmap="RdYlBu",
        s=15, alpha=0.35, vmin=0, vmax=1,
        edgecolors="none", zorder=2
    )

    # 강조: 해당 새 학생
    x_h, y_h = umap_coords[new_idx, 0], umap_coords[new_idx, 1]
    ax.scatter(x_h, y_h, s=500, c="white",
               edgecolors=color, linewidths=3, zorder=6, alpha=0.9)
    ax.scatter(x_h, y_h, s=200, marker="*", c=color,
               edgecolors="black", linewidths=0.8, zorder=7)

    ax.set_xlabel("UMAP Dim 1", fontsize=9)
    ax.set_ylabel("UMAP Dim 2", fontsize=9)
    ax.set_title("Relative Position", fontsize=11)
    ax.tick_params(labelsize=8)
    plt.colorbar(sc, ax=ax, label="Score Rate", shrink=0.65, pad=0.02)

    # 우측 상단 고정 레이블 (연결선 없음)
    ax.text(0.97, 0.97, label,
            transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=color,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=color, lw=1.2, alpha=0.92),
            zorder=9)


def draw_kc_bar(ax, theta_vals, kc_names, mean_ref=None):
    colors = ["#E74C3C" if v < 0.4 else "#F39C12" if v < 0.6 else "#2ECC71"
              for v in theta_vals]
    bars = ax.barh(kc_names, theta_vals, color=colors, alpha=0.85, height=0.6)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    # 기존 학생 평균 기준선
    if mean_ref is not None:
        for i, (mv, kc) in enumerate(zip(mean_ref, kc_names)):
            ax.plot(mv, i, marker="|", color="#333333",
                    markersize=14, markeredgewidth=2.5,
                    label="Train avg" if i == 0 else "")

    ax.set_xlabel("Proficiency (θ)")
    ax.set_title("KC Proficiency", fontsize=11)
    for bar, val in zip(bars, theta_vals):
        ax.text(min(val + 0.02, 0.95),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    if mean_ref is not None:
        ax.legend(fontsize=8, loc="lower right")


# ── 7. 전체 분포 위 새 학생 위치 요약 UMAP ───────────────────────────────────

print("\n전체 UMAP 요약 저장 중...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor("#F2F4F6")
fig.patch.set_facecolor("white")

# KDE 배경
try:
    xy  = umap_coords.T
    kde = gaussian_kde(xy, bw_method=0.3)
    xmin2, xmax2 = xy[0].min() - 2, xy[0].max() + 2
    ymin2, ymax2 = xy[1].min() - 2, xy[1].max() + 2
    xx2, yy2 = np.meshgrid(np.linspace(xmin2, xmax2, 130),
                           np.linspace(ymin2, ymax2, 130))
    zz2 = kde(np.vstack([xx2.ravel(), yy2.ravel()])).reshape(xx2.shape)
    ax.contourf(xx2, yy2, zz2, levels=8, cmap="Greys", alpha=0.18, zorder=1)
    ax.contour(xx2, yy2, zz2, levels=5, colors="gray",
               linewidths=0.5, alpha=0.40, zorder=1)
except Exception:
    pass

# 기존 학생 (배경)
mask_train = np.ones(len(umap_coords), dtype=bool)
mask_train[new_indices] = False
sc = ax.scatter(
    umap_coords[mask_train, 0], umap_coords[mask_train, 1],
    c=sr_all_combined[mask_train], cmap="RdYlBu",
    s=40, alpha=0.55, vmin=0, vmax=1,
    edgecolors="white", linewidths=0.3, zorder=2, label="기존 학생"
)
cbar = plt.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
cbar.set_label("Correct Rate  (blue=high / red=low)", fontsize=10)

# 새 학생 (강조)
new_colors = plt.cm.RdYlBu(sr_new)
ax.scatter(
    umap_coords[new_indices, 0], umap_coords[new_indices, 1],
    c=sr_new, cmap="RdYlBu",
    s=180, alpha=1.0, vmin=0, vmax=1,
    edgecolors="black", linewidths=1.8,
    marker="*", zorder=5, label="새 학생"
)

# 새 학생 번호 레이블
for i, ni in enumerate(new_indices):
    ax.annotate(
        f"S{original_sids[i]}",
        xy=(umap_coords[ni, 0], umap_coords[ni, 1]),
        xytext=(3, 4), textcoords="offset points",
        fontsize=7, fontweight="bold", color="black",
        zorder=9
    )

ax.set_title("UMAP — 새 학생들의 전체 분포 내 위치",
             fontweight="bold", fontsize=13, pad=10)
ax.set_xlabel("UMAP Dimension 1", fontsize=11)
ax.set_ylabel("UMAP Dimension 2", fontsize=11)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

n_high = int((sr_all_combined >= 0.6).sum())
n_low  = int((sr_all_combined < 0.3).sum())
ax.text(0.01, 0.01,
        f"기존 {N_TRAIN_USER}명 + 새 학생 {n_new}명  |  "
        f"High(≥60%): {n_high}명  Low(<30%): {n_low}명",
        transform=ax.transAxes, fontsize=8, color="gray",
        verticalalignment="bottom")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "umap_new_in_context.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("umap_new_in_context.png 저장 완료")

# ── 7-b. 3D UMAP 저장 (4개 시점) ─────────────────────────────────────────────

print("3D UMAP 저장 중...")
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# 기존 / 새 학생 마스크
mask_bg_3d    = np.ones(len(umap_coords_3d), dtype=bool)
mask_bg_3d[new_indices] = False
xyz_bg  = umap_coords_3d[mask_bg_3d]
sr_bg   = sr_all_combined[mask_bg_3d]
xyz_new = umap_coords_3d[new_indices]

# 4개 시점 (elev, azim)
viewpoints = [(20, 30), (20, 120), (20, 210), (45, 60)]

fig_3d = plt.figure(figsize=(18, 14))
fig_3d.patch.set_facecolor("white")
fig_3d.suptitle("3D UMAP — 전체 분포 내 학급 위치",
                fontsize=14, fontweight="bold", y=0.99)

for k, (elev, azim) in enumerate(viewpoints):
    ax3 = fig_3d.add_subplot(2, 2, k + 1, projection="3d")
    ax3.set_facecolor("#F2F4F6")

    # 기존 학생 배경
    sc3 = ax3.scatter(
        xyz_bg[:, 0], xyz_bg[:, 1], xyz_bg[:, 2],
        c=sr_bg, cmap="RdYlBu",
        s=18, alpha=0.30, vmin=0, vmax=1,
        edgecolors="none", depthshade=True
    )

    # 새 학생 (★ 마커 → 3D는 별 마커 미지원, 큰 원으로 대체)
    ax3.scatter(
        xyz_new[:, 0], xyz_new[:, 1], xyz_new[:, 2],
        c=sr_new, cmap="RdYlBu",
        s=120, alpha=1.0, vmin=0, vmax=1,
        edgecolors="black", linewidths=1.5,
        marker="^", depthshade=False, zorder=5
    )

    # 새 학생 번호 레이블
    for i, ni in enumerate(new_indices):
        x3, y3, z3 = umap_coords_3d[ni]
        ax3.text(x3, y3, z3, f" S{original_sids[i]}",
                 fontsize=7, fontweight="bold", color="black", zorder=9)

    ax3.view_init(elev=elev, azim=azim)
    ax3.set_xlabel("Dim 1", fontsize=8, labelpad=2)
    ax3.set_ylabel("Dim 2", fontsize=8, labelpad=2)
    ax3.set_zlabel("Dim 3", fontsize=8, labelpad=2)
    ax3.tick_params(labelsize=7)
    ax3.set_title(f"elev={elev}°  azim={azim}°", fontsize=10)

# 공통 컬러바
cbar_ax = fig_3d.add_axes([0.92, 0.15, 0.015, 0.65])
sm = plt.cm.ScalarMappable(cmap="RdYlBu",
                           norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
fig_3d.colorbar(sm, cax=cbar_ax, label="Correct Rate (blue=high / red=low)")

# 범례 (기존/학급 구분)
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markersize=8, alpha=0.6, label=f"기존 학생 ({N_TRAIN_USER}명)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#2C3E50",
           markersize=9, markeredgecolor="black", label=f"학급 학생 ({n_new}명)"),
]
fig_3d.legend(handles=legend_elems, loc="lower center",
              ncol=2, fontsize=10, framealpha=0.9,
              bbox_to_anchor=(0.47, 0.01))

plt.subplots_adjust(left=0.04, right=0.90, top=0.94, bottom=0.07,
                    hspace=0.30, wspace=0.20)
plt.savefig(os.path.join(OUT_DIR, "umap_3d.png"), dpi=150, bbox_inches="tight")
plt.close()
print("umap_3d.png 저장 완료")

# ── 8. 기존 학생 KC 평균 (비교 기준선용) ──────────────────────────────────────

if os.path.exists(TRAIN_THETA_CSV):
    kc_mean_ref = df_ref[KC_NAMES].mean().values
else:
    kc_mean_ref = None

# ── 9. 학급 종합 리포트 생성 ──────────────────────────────────────────────────

print("\n학급 종합 리포트 생성 중...")

fig_cls = plt.figure(figsize=(18, 14))
fig_cls.patch.set_facecolor("white")
fig_cls.suptitle(
    f"학급 인지진단 종합 리포트  |  학생 수: {n_new}명  |  "
    f"평균 정답률: {sr_new.mean():.1%}  |  기준: 훈련 학생 {N_TRAIN_USER}명",
    fontsize=14, fontweight="bold", y=0.98
)

gs_cls = gridspec.GridSpec(2, 3, figure=fig_cls,
                           hspace=0.42, wspace=0.38)

# ── [1,1] KC별 숙달도 비교 (학급 평균 vs 기존 평균) ─────────────────────────
ax_kc = fig_cls.add_subplot(gs_cls[0, 0])
kc_class_mean = theta_new.mean(axis=0)
x_pos = np.arange(N_KNOW)
w = 0.38
bars1 = ax_kc.bar(x_pos - w/2, kc_class_mean, w,
                  color="#3498DB", alpha=0.85, label="학급 평균")
if kc_mean_ref is not None:
    bars2 = ax_kc.bar(x_pos + w/2, kc_mean_ref, w,
                      color="#BDC3C7", alpha=0.85, label="훈련 학생 평균")
for bar, val in zip(bars1, kc_class_mean):
    ax_kc.text(bar.get_x() + bar.get_width()/2, val + 0.01,
               f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#2471A3")
ax_kc.set_xticks(x_pos)
ax_kc.set_xticklabels(KC_NAMES, fontsize=9)
ax_kc.set_ylim(0, 1)
ax_kc.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax_kc.set_ylabel("평균 숙달도 (θ)", fontsize=10)
ax_kc.set_title("KC별 숙달도 비교", fontsize=11, fontweight="bold")
ax_kc.legend(fontsize=9)

# ── [1,2] 정답률 분포 히스토그램 ─────────────────────────────────────────────
ax_hist = fig_cls.add_subplot(gs_cls[0, 1])
bins = np.linspace(0, 1, 11)
n_hist, _, patches = ax_hist.hist(sr_new, bins=bins, edgecolor="white",
                                   linewidth=0.8, rwidth=0.85)
for patch, left in zip(patches, bins[:-1]):
    mid = left + 0.05
    if mid < 0.3:
        patch.set_facecolor("#E74C3C")
    elif mid < 0.6:
        patch.set_facecolor("#F39C12")
    else:
        patch.set_facecolor("#2ECC71")

ax_hist.axvline(sr_new.mean(), color="#2C3E50", linewidth=2,
                linestyle="--", label=f"학급 평균 {sr_new.mean():.1%}")
if kc_mean_ref is not None:
    ax_hist.axvline(ref_sr.mean(), color="gray", linewidth=1.5,
                    linestyle=":", label=f"훈련 평균 {ref_sr.mean():.1%}")
ax_hist.set_xlabel("정답률", fontsize=10)
ax_hist.set_ylabel("학생 수", fontsize=10)
ax_hist.set_title("학급 정답률 분포", fontsize=11, fontweight="bold")
ax_hist.legend(fontsize=9)

n_high_cls = int((sr_new >= 0.6).sum())
n_mid_cls  = int(((sr_new >= 0.3) & (sr_new < 0.6)).sum())
n_low_cls  = int((sr_new < 0.3).sum())
ax_hist.text(0.98, 0.95,
             f"상위(≥60%): {n_high_cls}명\n중위(30~60%): {n_mid_cls}명\n하위(<30%): {n_low_cls}명",
             transform=ax_hist.transAxes, fontsize=9, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

# ── [1,3] KC별 분포 박스플롯 ─────────────────────────────────────────────────
ax_box = fig_cls.add_subplot(gs_cls[0, 2])
bp = ax_box.boxplot(
    [theta_new[:, k] for k in range(N_KNOW)],
    labels=KC_NAMES, patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)
box_colors = ["#AED6F1", "#A9DFBF", "#FAD7A0",
              "#F1948A", "#C39BD3", "#85C1E9"]
for patch, col in zip(bp["boxes"], box_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.8)
if kc_mean_ref is not None:
    ax_box.plot(range(1, N_KNOW + 1), kc_mean_ref,
                "D", color="#2C3E50", markersize=6,
                label="훈련 평균", zorder=5)
    ax_box.legend(fontsize=9)
ax_box.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax_box.set_ylabel("숙달도 (θ)", fontsize=10)
ax_box.set_title("KC별 숙달도 분포", fontsize=11, fontweight="bold")
ax_box.set_ylim(0, 1)

# ── [2,1:2] 학생 × KC 히트맵 (정답률 내림차순) ──────────────────────────────
ax_heat = fig_cls.add_subplot(gs_cls[1, :2])
sort_order = np.argsort(sr_new)[::-1]
theta_sorted = theta_new[sort_order]
sid_sorted   = [original_sids[i] for i in sort_order]
sr_sorted    = sr_new[sort_order]

im = ax_heat.imshow(theta_sorted.T, aspect="auto",
                    cmap="RdYlGn", vmin=0, vmax=1,
                    interpolation="nearest")
plt.colorbar(im, ax=ax_heat, label="숙달도 (θ)", shrink=0.85)
ax_heat.set_yticks(range(N_KNOW))
ax_heat.set_yticklabels(KC_NAMES, fontsize=9)
ax_heat.set_xticks(range(n_new))
ax_heat.set_xticklabels(
    [f"S{s}\n({r:.0%})" for s, r in zip(sid_sorted, sr_sorted)],
    fontsize=7, rotation=0, ha="center"
)
ax_heat.set_title("학생별 KC 숙달도 히트맵  (정답률 내림차순 정렬)",
                  fontsize=11, fontweight="bold")
ax_heat.set_xlabel("학생 (S_ID / 정답률)", fontsize=10)

# 셀 값 표시 (학생 수가 적을 때만)
if n_new <= 40:
    for ki in range(N_KNOW):
        for si in range(n_new):
            val = theta_sorted[si, ki]
            txt_color = "white" if val < 0.25 or val > 0.80 else "black"
            ax_heat.text(si, ki, f"{val:.2f}", ha="center", va="center",
                         fontsize=6, color=txt_color)

# ── [2,3] 학급 UMAP ──────────────────────────────────────────────────────────
ax_umap_cls = fig_cls.add_subplot(gs_cls[1, 2])
ax_umap_cls.set_facecolor("#F2F4F6")

# 기존 학생 배경
mask_bg_cls = np.ones(len(umap_coords), dtype=bool)
mask_bg_cls[new_indices] = False
ax_umap_cls.scatter(
    umap_coords[mask_bg_cls, 0], umap_coords[mask_bg_cls, 1],
    c=sr_all_combined[mask_bg_cls], cmap="RdYlBu",
    s=12, alpha=0.30, vmin=0, vmax=1, edgecolors="none", zorder=2
)
# 학급 학생 (별 마커)
sc_cls = ax_umap_cls.scatter(
    umap_coords[new_indices, 0], umap_coords[new_indices, 1],
    c=sr_new, cmap="RdYlBu",
    s=120, alpha=1.0, vmin=0, vmax=1,
    edgecolors="black", linewidths=1.5,
    marker="*", zorder=5
)
plt.colorbar(sc_cls, ax=ax_umap_cls, label="정답률", shrink=0.75)
ax_umap_cls.set_title("전체 분포 내 학급 위치", fontsize=11, fontweight="bold")
ax_umap_cls.set_xlabel("UMAP Dim 1", fontsize=9)
ax_umap_cls.set_ylabel("UMAP Dim 2", fontsize=9)
ax_umap_cls.tick_params(labelsize=8)

plt.savefig(os.path.join(OUT_DIR, "class_report.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("class_report.png 저장 완료")

# ── 10. 개별 학생 리포트 생성 ──────────────────────────────────────────────────

print(f"\n개별 학생 리포트 생성 중 ({n_new}명)...")

for i in tqdm(range(n_new)):
    sid        = original_sids[i]
    theta_vals = theta_new[i]
    sr         = sr_new[i]
    pct        = percentiles[i]
    new_idx_global = new_indices[i]

    # 색상: 정답률에 따라
    if sr >= 0.6:
        color = "#2471A3"    # 파랑 (상위)
    elif sr >= 0.3:
        color = "#E67E22"    # 주황 (중위)
    else:
        color = "#C0392B"    # 빨강 (하위)

    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Cognitive Diagnostic Report  —  Student S{sid}"
        f"   |   Correct Rate: {sr:.1%}   |   Percentile: {pct}%  (vs. {N_TRAIN_USER} train students)",
        fontsize=13, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── 왼쪽: 레이더 차트 ──────────────────────────────────────────────────
    ax_radar = fig.add_subplot(gs[0], polar=True)

    # 기존 학생 평균 (회색 배경)
    if kc_mean_ref is not None:
        n = len(KC_NAMES)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]
        mean_vals = kc_mean_ref.tolist() + kc_mean_ref[:1].tolist()
        ax_radar.plot(angles, mean_vals, "-", color="gray",
                      linewidth=1.2, alpha=0.5, label="Train avg")
        ax_radar.fill(angles, mean_vals, color="gray", alpha=0.1)

    draw_radar(ax_radar, theta_vals, KC_NAMES, color=color,
               label=f"S{sid}", fill_alpha=0.22)
    ax_radar.set_title("Knowledge Proficiencies", pad=15, fontsize=11)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    # ── 가운데: UMAP 위치 ──────────────────────────────────────────────────
    ax_umap = fig.add_subplot(gs[1])
    draw_umap_new(
        ax_umap, umap_coords, sr_all_combined,
        new_idx=new_idx_global,
        label=f"S{sid} ({sr:.0%})",
        color=color
    )

    # ── 오른쪽: KC 바 차트 ────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[2])
    draw_kc_bar(ax_bar, theta_vals, KC_NAMES, mean_ref=kc_mean_ref)

    # 해석 텍스트 (취약 KC 자동 표시)
    weak_kcs  = [KC_NAMES[k] for k, v in enumerate(theta_vals) if v < 0.4]
    strong_kcs = [KC_NAMES[k] for k, v in enumerate(theta_vals) if v >= 0.7]
    note_parts = []
    if strong_kcs:
        note_parts.append(f"강점: {', '.join(strong_kcs)}")
    if weak_kcs:
        note_parts.append(f"취약: {', '.join(weak_kcs)}")
    if note_parts:
        fig.text(0.5, -0.03, "  |  ".join(note_parts),
                 ha="center", fontsize=10, color="dimgray",
                 style="italic")

    plt.savefig(os.path.join(IND_DIR, f"new_student_S{sid:03d}.png"),
                bbox_inches="tight", dpi=150)
    plt.close()

# ── 10. 완료 요약 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("새 학생 인지진단 완료!")
print(f"저장 위치: {OUT_DIR}/")
print(f"\n  theta_new.csv              — 새 학생 {n_new}명 KC 숙달도 + 정답률")
print(f"  class_report.png           — 학급 종합 리포트 (교사용)")
print(f"  umap_new_in_context.png    — 전체 분포 내 학급 위치 (2D)")
print(f"  umap_3d.png                — 전체 분포 내 학급 위치 (3D, 4개 시점)")
print(f"  individual/ ({n_new}개)       — 개별 학생 리포트")
print("=" * 60)

print("\n[새 학생 KC 평균 숙달도]")
for i, kc in enumerate(KC_NAMES):
    new_mean = theta_new[:, i].mean()
    ref_mean = kc_mean_ref[i] if kc_mean_ref is not None else float("nan")
    diff = new_mean - ref_mean if kc_mean_ref is not None else 0
    sign = "▲" if diff > 0 else "▼"
    print(f"  {kc}: 새 학생 {new_mean:.3f}  /  기존 평균 {ref_mean:.3f}  "
          f"({sign}{abs(diff):.3f})")
