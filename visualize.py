# -*- coding: utf-8 -*-
"""
GNCDM 학습 결과 종합 시각화 스크립트.
터미널에서 다음과 같이 실행:
    python visualize.py

저장 위치: RESULT_PATH/figures/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
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
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
from train import IDCDataset

# ── 설정 ────────────────────────────────────────────────────────────────────

RESULT_PATH   = "result/QPCS"
MODEL_PATH    = "result/QPCS/params_32_32.pt"
TRAIN_FILE    = "data/my_train.csv"
VALID_FILE    = "data/my_valid.csv"
TEST_FILE     = "data/my_test.csv"
Q_MATRIX_FILE = "data/my_Q_matrix.npy"

N_USER = 730
N_ITEM = 32
N_KNOW = 6
KC_NAMES = [f"KC{i+1}" for i in range(N_KNOW)]

FIG_DIR = os.path.join(RESULT_PATH, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

# ── 데이터 로드 ──────────────────────────────────────────────────────────────

print("데이터 및 모델 로딩 중...")
result_all = np.load(os.path.join(RESULT_PATH, "result_all.npy"), allow_pickle=True)
with open(os.path.join(RESULT_PATH, "test_result.json")) as f:
    test_result = json.load(f)

Q_mat = np.load(Q_MATRIX_FILE)
df_train = pd.read_csv(TRAIN_FILE)
df_test  = pd.read_csv(TEST_FILE)
df_all   = pd.concat([pd.read_csv(TRAIN_FILE),
                      pd.read_csv(VALID_FILE),
                      pd.read_csv(TEST_FILE)], ignore_index=True)

net = torch.load(MODEL_PATH, weights_only=False)
net.eval()
device = net.device

# ── 에폭별 지표 추출 ──────────────────────────────────────────────────────────

epochs       = list(range(len(result_all)))
theta_norms  = [e["Theta_norm"]          for e in result_all]
train_acc    = [e["train_eval"]["acc"]   for e in result_all]
valid_acc    = [e["valid_eval"]["acc"]   for e in result_all]
valid_f1     = [e["valid_eval"]["f1"]    for e in result_all]
valid_rmse   = [e["valid_eval"]["rmse"]  for e in result_all]
valid_auc    = [e["valid_eval"]["auc"]   for e in result_all]

# ── 테스트셋 예측값 수집 ──────────────────────────────────────────────────────

print("테스트셋 예측값 수집 중...")
dataset_test = IDCDataset(df_test, n_user=N_USER, n_item=N_ITEM)
dataset_all  = IDCDataset(df_all,  n_user=N_USER, n_item=N_ITEM)

y_true, y_pred_buf, y_pred_recon = [], [], []

from torch.utils.data import DataLoader
loader = DataLoader(dataset_test, batch_size=256, shuffle=False)

with torch.no_grad():
    for user_log, item_log, user_id, item_id, score in loader:
        user_log = user_log.to(device)
        item_log = item_log.to(device)
        user_id  = user_id.to(device)
        item_id  = item_id.to(device)
        p_buf   = net.forward_using_buf(user_id, item_id).cpu().numpy().flatten()
        p_recon = net(user_log, item_log, user_id, item_id).cpu().numpy().flatten()
        y_true    += score.numpy().flatten().tolist()
        y_pred_buf   += p_buf.tolist()
        y_pred_recon += p_recon.tolist()

y_true       = np.array(y_true)
y_pred_buf   = np.array(y_pred_buf)
y_pred_recon = np.array(y_pred_recon)

# ── θ 행렬 계산 ───────────────────────────────────────────────────────────────

print("θ 행렬 계산 중...")
theta_list = []
with torch.no_grad():
    for uid in range(N_USER):
        user_log = torch.Tensor([dataset_all.log_mat[uid]]).to(device)
        theta = net.diagnose_theta(user_log).cpu().numpy()
        theta_list.append(theta)
theta_mat = np.concatenate(theta_list, axis=0)   # (730, 6)

# ── ψ 행렬 계산 ───────────────────────────────────────────────────────────────

psi_mat = net.get_Psi_buf().detach().numpy()      # (32, 6)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — 학습 수렴 곡선 (Theta norm + Train/Valid Acc)
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 1 — Training Convergence", fontweight="bold")

ax = axes[0]
ax.plot(epochs, theta_norms, "o-", color="#E74C3C", linewidth=2)
ax.set_title("Θ Norm per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Θ Norm (L2)")
ax.grid(alpha=0.3)
ax.annotate(f"Final: {theta_norms[-1]:.3f}",
            xy=(epochs[-1], theta_norms[-1]),
            xytext=(-40, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->"))

ax = axes[1]
ax.plot(epochs, train_acc, "s-", color="#3498DB", label="Train Acc", linewidth=2)
ax.plot(epochs, valid_acc, "o--", color="#E67E22", label="Valid Acc", linewidth=2)
ax.set_title("Accuracy per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_convergence.png"))
plt.close()
print("fig1_convergence.png 저장")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Validation 지표 추이 (F1, RMSE, AUC)
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Figure 2 — Validation Metrics per Epoch", fontweight="bold")

for ax, vals, label, color, ylim in zip(
    axes,
    [valid_f1, valid_rmse, valid_auc],
    ["F1 Score", "RMSE", "AUC"],
    ["#2ECC71", "#E74C3C", "#9B59B6"],
    [(0, 1), (0, 1), (0.5, 1)]
):
    ax.plot(epochs, vals, "o-", color=color, linewidth=2)
    ax.set_title(label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    ax.set_ylim(*ylim)
    ax.axhline(vals[-1], color=color, linestyle=":", alpha=0.5)
    ax.annotate(f"{vals[-1]:.4f}", xy=(epochs[-1], vals[-1]),
                xytext=(-30, 8), textcoords="offset points")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_valid_metrics.png"))
plt.close()
print("fig2_valid_metrics.png 저장")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — ROC 곡선 (Prediction vs Reconstruction)
# ════════════════════════════════════════════════════════════════════════════

fpr_b, tpr_b, _ = roc_curve(y_true, y_pred_buf)
fpr_r, tpr_r, _ = roc_curve(y_true, y_pred_recon)
auc_b = auc(fpr_b, tpr_b)
auc_r = auc(fpr_r, tpr_r)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_b, tpr_b, color="#3498DB", linewidth=2,
        label=f"Score Prediction  (AUC = {auc_b:.4f})")
ax.plot(fpr_r, tpr_r, color="#E67E22", linewidth=2, linestyle="--",
        label=f"Score Reconstruction  (AUC = {auc_r:.4f})")
ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Random (AUC = 0.50)")
ax.set_title("Figure 3 — ROC Curve", fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_roc_curve.png"))
plt.close()
print("fig3_roc_curve.png 저장")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Calibration Plot (확률 보정)
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 4 — Calibration Plot", fontweight="bold")

for ax, y_pred, title in zip(
    axes,
    [y_pred_buf, y_pred_recon],
    ["Score Prediction", "Score Reconstruction"]
):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    ax.plot(prob_pred, prob_true, "s-", color="#E74C3C", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_title(title)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_calibration.png"))
plt.close()
print("fig4_calibration.png 저장")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Confusion Matrix
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 5 — Confusion Matrix", fontweight="bold")

for ax, y_pred, title in zip(
    axes,
    [y_pred_buf, y_pred_recon],
    ["Score Prediction", "Score Reconstruction"]
):
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Incorrect(0)", "Correct(1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_confusion_matrix.png"))
plt.close()
print("fig5_confusion_matrix.png 저장")


# ════════════════════════════════════════════════════════════════════════════
# Figure 6 — Prediction vs Reconstruction 지표 비교 바 차트
# ════════════════════════════════════════════════════════════════════════════

metrics = ["ACC", "F1", "RMSE", "AUC"]
pred_vals  = [test_result["acc"], test_result["f1"],
              test_result["rmse"], test_result["auc"]]
recon_vals = [test_result["without_buf"]["acc"], test_result["without_buf"]["f1"],
              test_result["without_buf"]["rmse"], test_result["without_buf"]["auc"]]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
bars1 = ax.bar(x - width/2, pred_vals,  width, label="Score Prediction",     color="#3498DB", alpha=0.85)
bars2 = ax.bar(x + width/2, recon_vals, width, label="Score Reconstruction",  color="#E67E22", alpha=0.85)

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

ax.set_title("Figure 6 — Score Prediction vs Reconstruction", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_metric_comparison.png"))
plt.close()
print("fig6_metric_comparison.png 저장")


# ════════════════════════════════════════════════════════════════════════════
# Figure 7 — θ 분포 (KC별 학생 숙달도 분포)
# ════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))
parts = ax.violinplot(
    [theta_mat[:, k] for k in range(N_KNOW)],
    positions=range(N_KNOW), showmedians=True, showmeans=False
)
for pc in parts["bodies"]:
    pc.set_facecolor("#3498DB")
    pc.set_alpha(0.6)
parts["cmedians"].set_color("#E74C3C")
parts["cmedians"].set_linewidth(2)

ax.set_title("Figure 7 — θ Distribution per KC (Learner Proficiency)", fontweight="bold")
ax.set_xticks(range(N_KNOW))
ax.set_xticklabels(KC_NAMES)
ax.set_ylabel("Proficiency (θ)")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold = 0.5")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# KC별 평균 표시
for k in range(N_KNOW):
    mean_val = theta_mat[:, k].mean()
    ax.text(k, mean_val + 0.03, f"μ={mean_val:.2f}",
            ha="center", fontsize=9, color="#2C3E50")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig7_theta_distribution.png"))
plt.close()
print("fig7_theta_distribution.png 저장")


# ════════════════════════════════════════════════════════════════════════════
# Figure 8 — θ 히트맵 (학생 × KC)
# ════════════════════════════════════════════════════════════════════════════

# 전체 정답률로 학생 정렬 (내림차순 → 위쪽이 높은 정답률)
score_rates = []
for uid in range(N_USER):
    sv = dataset_all.log_mat[uid]
    n_c = len(sv[sv > 0])
    n_a = len(sv[sv != 0])
    score_rates.append(n_c / n_a if n_a > 0 else 0)
sort_idx = np.argsort(score_rates)[::-1]   # 내림차순: 위쪽 = 높은 정답률
theta_sorted = theta_mat[sort_idx]

cmap = LinearSegmentedColormap.from_list("prof", ["#D32F2F", "#FFC107", "#388E3C"])
fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(theta_sorted, aspect="auto", cmap=cmap, vmin=0, vmax=1)
ax.set_title("Figure 8 — θ Heatmap (Students × KC)\n(top = high score rate  /  bottom = low score rate)",
             fontweight="bold")
ax.set_xlabel("Knowledge Component")
ax.set_ylabel("Student  ▲ high score rate  /  low score rate ▼")
ax.set_xticks(range(N_KNOW))
ax.set_xticklabels(KC_NAMES)
ax.set_yticks([])
plt.colorbar(im, ax=ax, label="Proficiency (θ)", shrink=0.6)

# 상단·하단 경계 표시
n_top = int(N_USER * 0.2)
n_bot = int(N_USER * 0.2)
ax.axhline(n_top - 0.5, color="white", linewidth=1.2, linestyle="--", alpha=0.7)
ax.axhline(N_USER - n_bot - 0.5, color="white", linewidth=1.2, linestyle="--", alpha=0.7)
ax.text(N_KNOW - 0.5, n_top / 2, "Top 20%", color="white",
        fontsize=9, ha="right", va="center", fontweight="bold")
ax.text(N_KNOW - 0.5, N_USER - n_bot / 2, "Bottom 20%", color="white",
        fontsize=9, ha="right", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig8_theta_heatmap.png"))
plt.close()
print("fig8_theta_heatmap.png 저장")


# ════════════════════════════════════════════════════════════════════════════
# Figure 9 — ψ 히트맵 (문항 × KC 난이도)
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Figure 9 — Item Feature (ψ) Heatmap", fontweight="bold")

ax = axes[0]
im = ax.imshow(psi_mat, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
ax.set_title("ψ (Item Features from g_nn)")
ax.set_xlabel("Knowledge Component")
ax.set_ylabel("Item ID (0~31)")
ax.set_xticks(range(N_KNOW))
ax.set_xticklabels(KC_NAMES)
ax.set_yticks(range(N_ITEM))
ax.set_yticklabels([f"Q{i+1}" for i in range(N_ITEM)], fontsize=8)
plt.colorbar(im, ax=ax, label="ψ value", shrink=0.8)

ax = axes[1]
im2 = ax.imshow(Q_mat, aspect="auto", cmap="Greys", vmin=0, vmax=1)
ax.set_title("Q-matrix (Reference)")
ax.set_xlabel("Knowledge Component")
ax.set_ylabel("Item ID (0~31)")
ax.set_xticks(range(N_KNOW))
ax.set_xticklabels(KC_NAMES)
ax.set_yticks(range(N_ITEM))
ax.set_yticklabels([f"Q{i+1}" for i in range(N_ITEM)], fontsize=8)
plt.colorbar(im2, ax=ax, label="Q value", shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig9_psi_heatmap.png"))
plt.close()
print("fig9_psi_heatmap.png 저장")


# ════════════════════════════════════════════════════════════════════════════
# Figure 10 — 예측 확률 분포 히스토그램
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 10 — Predicted Probability Distribution", fontweight="bold")

for ax, y_pred, title in zip(
    axes,
    [y_pred_buf, y_pred_recon],
    ["Score Prediction", "Score Reconstruction"]
):
    ax.hist(y_pred[y_true == 0], bins=40, alpha=0.6, color="#E74C3C",
            label="Actual Incorrect (0)", density=True)
    ax.hist(y_pred[y_true == 1], bins=40, alpha=0.6, color="#2ECC71",
            label="Actual Correct (1)", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold = 0.5")
    ax.set_title(title)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig10_pred_distribution.png"))
plt.close()
print("fig10_pred_distribution.png 저장")


# ════════════════════════════════════════════════════════════════════════════
# 최종 요약 출력
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("시각화 완료! 저장 위치:", FIG_DIR)
print("="*60)
print("\n[테스트 결과 요약]")
print(f"  Score Prediction    | ACC={test_result['acc']:.4f}  F1={test_result['f1']:.4f}"
      f"  RMSE={test_result['rmse']:.4f}  AUC={test_result['auc']:.4f}")
wb = test_result["without_buf"]
print(f"  Score Reconstruction| ACC={wb['acc']:.4f}  F1={wb['f1']:.4f}"
      f"  RMSE={wb['rmse']:.4f}  AUC={wb['auc']:.4f}")
print("\n[생성된 파일]")
figs = [
    "fig1_convergence.png        — 학습 수렴 곡선 (Theta norm + Train/Valid Acc)",
    "fig2_valid_metrics.png      — Validation F1/RMSE/AUC 추이",
    "fig3_roc_curve.png          — ROC 곡선",
    "fig4_calibration.png        — 확률 보정(Calibration) 플롯",
    "fig5_confusion_matrix.png   — 혼동 행렬",
    "fig6_metric_comparison.png  — Prediction vs Reconstruction 지표 비교",
    "fig7_theta_distribution.png — KC별 학습자 숙달도 분포 (바이올린)",
    "fig8_theta_heatmap.png      — 학습자×KC 숙달도 히트맵",
    "fig9_psi_heatmap.png        — 문항 특성(ψ) + Q-행렬 비교",
    "fig10_pred_distribution.png — 예측 확률 분포",
]
for f in figs:
    print(f"  {f}")
