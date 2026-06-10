# -*- coding: utf-8 -*-
"""
QPCS 데이터로 G-NCDM을 학습/평가하는 실행 스크립트.
터미널에서 다음과 같이 실행:
    python run_mydata.py
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

TRAIN_FILE    = "data/QPCS_train.csv"
VALID_FILE    = "data/QPCS_valid.csv"
TEST_FILE     = "data/QPCS_test.csv"
Q_MATRIX_FILE = "data/Q_matrix.npy"
SAVE_PATH     = "./result/QPCS"

N_USER   = 730
N_ITEM   = 32
N_KNOW   = 6
USER_DIM = 32
ITEM_DIM = 32
ALPHA    = 0.99

TRAINING_CONFIG = "config/training_config_QPCS.json"

# ── 실행 ─────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_PATH, exist_ok=True)

df_train = pd.read_csv(TRAIN_FILE)
df_valid = pd.read_csv(VALID_FILE)
df_test  = pd.read_csv(TEST_FILE)

Q_mat = np.load(Q_MATRIX_FILE)

def add_knowledge_code(data, Q_mat):
    knowledge = []
    for i in range(data.shape[0]):
        knowledge.append(Q_mat[data.loc[i, "item_id"]])
    data["knowledge"] = knowledge
    return data

df_train = add_knowledge_code(df_train, Q_mat)
df_valid = add_knowledge_code(df_valid, Q_mat)
df_test  = add_knowledge_code(df_test,  Q_mat)

with open(TRAINING_CONFIG, "r") as fp:
    config = json.load(fp)

batch_size = int(config["batch_size"])
lr         = float(config["lr"])
n_epoch    = int(config["n_epoch"])
device     = torch.device(config["device"])
print(f"device : {device}")
print(f"n_user={N_USER}, n_item={N_ITEM}, n_know={N_KNOW}")
print(f"user_dim={USER_DIM}, item_dim={ITEM_DIM}, alpha={ALPHA}")
print(f"lr={lr}, batch_size={batch_size}, n_epoch={n_epoch}\n")

net = model.GNCDM(
    N_USER, N_ITEM, N_KNOW,
    USER_DIM, ITEM_DIM, ALPHA,
    Q_mat=Q_mat,
    monotonicity_assumption=True,
    device=device
)

result_all = train.train(net, df_train, df_valid,
                         batch_size=batch_size, lr=lr, n_epoch=n_epoch)

np.save(os.path.join(SAVE_PATH, "result_all.npy"), result_all)

print("\n=== 최종 테스트 평가 ===")
test_result = train.eval(net, df_test, batch_size=256)

with open(os.path.join(SAVE_PATH, "cmd.txt"), "w") as fp:
    fp.write("python run_qpcs.py")

with open(os.path.join(SAVE_PATH, "test_result.json"), "w") as fp:
    json.dump(test_result, fp, indent=2)

torch.save(net, os.path.join(SAVE_PATH, f"params_{USER_DIM}_{ITEM_DIM}.pt"))

print(f"\n결과 저장 완료: {SAVE_PATH}/")
gc.collect()
