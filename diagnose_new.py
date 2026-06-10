# -*- coding: utf-8 -*-
"""
훈련된 GNCDM 모델로 새로운 학생들의 인지 상태(θ)를 즉각 진단합니다.

[입력 준비]
  새 학생 응답 데이터를 long format CSV로 준비:
    user_id (0-indexed), item_id (0-indexed), score (0/1)

[실행]
  python diagnose_new.py

[출력]
  OUTPUT_PATH/theta.npy       : shape (학생수, KC수) — KC별 숙달도 (0~1)
  OUTPUT_PATH/score_rate.npy  : shape (학생수,)     — 학생별 전체 정답률
  OUTPUT_PATH/theta_df.csv    : theta를 보기 좋은 CSV로 저장
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from train import IDCDataset

# ── 설정 ─────────────────────────────────────────────────────────────────────

# 새 학생 응답 데이터 (long format: user_id, item_id, score)
NEW_DATA_FILE = "data/my_test.csv"

# 훈련된 모델 경로
MODEL_PATH = "result/mydata/params_32_32.pt"

# Q-행렬 경로 (없으면 빈 문자열 '')
Q_MATRIX_FILE = "data/my_Q_matrix.npy"

# 결과 저장 경로
OUTPUT_PATH = "result/mydata/diagnosis"

# 데이터 전체 규모 (훈련 시와 동일하게 설정)
N_USER = 730   # 새 학생 수에 맞게 변경
N_ITEM = 32
N_KNOW = 6

KC_NAMES = ["KC1", "KC2", "KC3", "KC4", "KC5", "KC6"]  # Q_matrix.csv 컬럼명

# ── 실행 ─────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_PATH, exist_ok=True)

# 모델 로드
print("모델 로딩 중...")
net = torch.load(MODEL_PATH, weights_only=False)
net.eval()
device = net.device

# Q-행렬 로드
Q_mat = np.load(Q_MATRIX_FILE) if Q_MATRIX_FILE != "" \
    else np.ones((N_ITEM, N_KNOW))

# 새 학생 응답 데이터 로드
df_new = pd.read_csv(NEW_DATA_FILE)
n_new_user = df_new["user_id"].nunique()
print(f"새 학생 수: {n_new_user}명, 응답 수: {len(df_new)}행")

# 응답 로그 행렬 구성 (+1 정답, -1 오답, 0 미응시)
dataset = IDCDataset(df_new, n_user=N_USER, n_item=N_ITEM)

# ── θ 계산 (즉각 진단 — 재학습 없음) ─────────────────────────────────────────
print("θ 계산 중 (재학습 없음)...")
theta_list = []

for user_id in tqdm(range(N_USER)):
    user_log = torch.Tensor([dataset.log_mat[user_id]]).to(device)
    with torch.no_grad():
        theta = net.diagnose_theta(user_log).detach().cpu().numpy()
    theta_list.append(theta)

theta_mat = np.concatenate(theta_list, axis=0)   # (N_USER, N_KNOW)

# ── 정답률 계산 ────────────────────────────────────────────────────────────────
score_rates = []
for user_id in range(N_USER):
    score_vec = dataset.log_mat[user_id]
    n_correct = len(score_vec[score_vec > 0])
    n_all     = len(score_vec[score_vec != 0])
    acc = n_correct / n_all if n_all != 0 else 0.0
    score_rates.append(acc)
score_rates = np.array(score_rates)

# ── 저장 ──────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUTPUT_PATH, "theta.npy"),      theta_mat)
np.save(os.path.join(OUTPUT_PATH, "score_rate.npy"), score_rates)

# 보기 좋은 CSV로도 저장
theta_df = pd.DataFrame(
    theta_mat,
    columns=KC_NAMES[:N_KNOW]
)
theta_df.insert(0, "user_id", range(N_USER))
theta_df["score_rate"] = score_rates
theta_df.to_csv(os.path.join(OUTPUT_PATH, "theta_df.csv"), index=False)

# ── 결과 미리보기 ──────────────────────────────────────────────────────────────
print("\n=== 진단 결과 미리보기 (상위 5명) ===")
print(theta_df.head().to_string(index=False))
print(f"\n평균 숙달도:")
for kc in KC_NAMES[:N_KNOW]:
    print(f"  {kc}: {theta_df[kc].mean():.3f}")

print(f"\n결과 저장 완료: {OUTPUT_PATH}/")
print(f"  theta.npy      → shape {theta_mat.shape}")
print(f"  score_rate.npy → shape {score_rates.shape}")
print(f"  theta_df.csv   → 확인하기 쉬운 CSV")
