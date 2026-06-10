# -*- coding: utf-8 -*-
"""
배포용: 전체 730명 데이터로 GNCDM을 학습하고 모델을 저장합니다.

검증용 run_qpcs.py 와의 차이:
  - 전체 데이터(QPCS_full_train.csv)를 훈련에 사용
  - valid / test 평가 없음
  - 저장 경로: result/QPCS_full/

실행:
  python run_qpcs_full.py
"""

import gc
import json
import os
import numpy as np
import pandas as pd
import torch

import model
import train

# ── 파라미터 설정 ────────────────────────────────────────────────────────────

TRAIN_FILE    = "data/QPCS_full_train.csv"
Q_MATRIX_FILE = "data/Q_matrix.npy"
SAVE_PATH     = "./result/QPCS_full"

N_USER   = 730
N_ITEM   = 32
N_KNOW   = 6
USER_DIM = 32
ITEM_DIM = 32
ALPHA    = 0.99

TRAINING_CONFIG = "config/training_config_QPCS.json"

# ── 데이터 로드 ──────────────────────────────────────────────────────────────

os.makedirs(SAVE_PATH, exist_ok=True)

df_train = pd.read_csv(TRAIN_FILE)
Q_mat    = np.load(Q_MATRIX_FILE)


def add_knowledge_code(data, Q_mat):
    knowledge = []
    for i in range(data.shape[0]):
        knowledge.append(Q_mat[data.loc[i, "item_id"]])
    data["knowledge"] = knowledge
    return data


df_train = add_knowledge_code(df_train, Q_mat)

# ── 설정 로드 ────────────────────────────────────────────────────────────────

with open(TRAINING_CONFIG, "r") as fp:
    config = json.load(fp)

batch_size = int(config["batch_size"])
lr         = float(config["lr"])
n_epoch    = int(config["n_epoch"])
device     = torch.device(config["device"])

print("=" * 55)
print("[배포용 전체 데이터 학습]")
print(f"  훈련 데이터 : {len(df_train):,} rows (전체 {N_USER}명)")
print(f"  device      : {device}")
print(f"  n_user={N_USER}, n_item={N_ITEM}, n_know={N_KNOW}")
print(f"  user_dim={USER_DIM}, item_dim={ITEM_DIM}, alpha={ALPHA}")
print(f"  lr={lr}, batch_size={batch_size}, n_epoch={n_epoch}")
print("=" * 55 + "\n")

# ── 모델 초기화 및 학습 ──────────────────────────────────────────────────────

net = model.GNCDM(
    N_USER, N_ITEM, N_KNOW,
    USER_DIM, ITEM_DIM, ALPHA,
    Q_mat=Q_mat,
    monotonicity_assumption=True,
    device=device
)

# valid_data=None → 검증 없이 전 epoch 학습
result_all = train.train(net, df_train, valid_data=None,
                         batch_size=batch_size, lr=lr, n_epoch=n_epoch)

np.save(os.path.join(SAVE_PATH, "result_all.npy"), result_all)

# ── 모델 저장 ────────────────────────────────────────────────────────────────

model_path = os.path.join(SAVE_PATH, f"params_{USER_DIM}_{ITEM_DIM}.pt")
torch.save(net, model_path)

with open(os.path.join(SAVE_PATH, "cmd.txt"), "w") as fp:
    fp.write("python run_qpcs_full.py")

print("\n" + "=" * 55)
print("배포용 모델 학습 완료!")
print(f"  모델 저장 위치: {model_path}")
print(f"  학습 곡선 저장: {SAVE_PATH}/result_all.npy")
print("=" * 55)

gc.collect()
