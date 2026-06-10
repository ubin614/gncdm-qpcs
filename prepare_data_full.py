# -*- coding: utf-8 -*-
"""
배포용: QPCS.csv + Q_matrix.csv → 전체 데이터를 훈련셋으로 변환.

검증용 prepare_data.py 와의 차이:
  - train/valid/test 분할 없음
  - 모든 730명 × 32문항 데이터를 단일 훈련 파일로 저장

출력:
  data/QPCS_full_train.csv   — 전체 응답 데이터 (23,360 rows)
  data/Q_matrix.npy          — Q행렬 (32 × 6), 이미 존재하면 덮어쓰지 않음
"""

import os
import numpy as np
import pandas as pd


def main():
    os.makedirs("data", exist_ok=True)

    # ── 1. Q_matrix.csv → npy ───────────────────────────────────────────────
    qmat_df = pd.read_csv("Q_matrix.csv")
    kc_cols = [c for c in qmat_df.columns if c.startswith("KC")]
    Q_matrix = qmat_df[kc_cols].values.astype(np.float32)   # (32, 6)

    print(f"Q_matrix shape : {Q_matrix.shape}")
    np.save("data/Q_matrix.npy", Q_matrix)
    print("Saved → data/Q_matrix.npy\n")

    # ── 2. QPCS.csv → long format (전체, 분할 없음) ─────────────────────────
    qpcs_df = pd.read_csv("QPCS.csv")
    q_cols  = [c for c in qpcs_df.columns if c.startswith("Q")]

    records = []
    for _, row in qpcs_df.iterrows():
        user_id = int(row["S_ID"]) - 1          # 0-indexed
        for j, qcol in enumerate(q_cols):
            records.append({
                "user_id": user_id,
                "item_id": j,                   # 0-indexed
                "score":   int(row[qcol]),
            })

    full_df = pd.DataFrame(records)
    n_users = full_df["user_id"].nunique()
    n_items = full_df["item_id"].nunique()

    print(f"전체 데이터: {len(full_df):,} rows  |  "
          f"{n_users} users × {n_items} items")
    print(f"Score 분포:\n{full_df['score'].value_counts().to_string()}\n")

    full_df.to_csv("data/QPCS_full_train.csv", index=False)
    print("Saved → data/QPCS_full_train.csv")

    print("\n" + "=" * 55)
    print("[배포용 run_qpcs_full.py 파라미터]")
    print(f"  N_USER = {n_users}")
    print(f"  N_ITEM = {n_items}")
    print(f"  N_KNOW = {len(kc_cols)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
