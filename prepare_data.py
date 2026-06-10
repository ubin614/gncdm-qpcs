# -*- coding: utf-8 -*-
"""
QPCS.csv와 Q_matrix.csv를 GNCDM 학습에 필요한 형식으로 전처리합니다.

출력:
  data/my_train.csv, data/my_valid.csv, data/my_test.csv
    - 컬럼: user_id (0-indexed), item_id (0-indexed), score (0/1)
    - 분할 비율: train 70%, valid 10%, test 20%

  data/my_Q_matrix.npy
    - shape: (32, 6) — 문항 수 × 지식요소 수
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def main():
    os.makedirs("data", exist_ok=True)

    # ── 1. Q_matrix.csv → npy ───────────────────────────────────────────────
    # item 컬럼은 무시; 행 순서가 곧 문항 인덱스 (0~31)
    qmat_df = pd.read_csv("Q_matrix.csv")
    kc_cols = [c for c in qmat_df.columns if c.startswith("KC")]
    Q_matrix = qmat_df[kc_cols].values.astype(np.float32)   # (32, 6)

    print(f"Q_matrix shape : {Q_matrix.shape}")
    print(f"KC coverage per item (sum per row):\n{Q_matrix.sum(axis=1)}\n")

    np.save("data/Q_matrix.npy", Q_matrix)
    print("Saved → data/Q_matrix.npy\n")

    # ── 2. QPCS.csv → long format ───────────────────────────────────────────
    qpcs_df = pd.read_csv("QPCS.csv")
    q_cols = [c for c in qpcs_df.columns if c.startswith("Q")]

    records = []
    for _, row in qpcs_df.iterrows():
        user_id = int(row["S_ID"]) - 1          # 0-indexed
        for j, qcol in enumerate(q_cols):
            records.append({
                "user_id": user_id,
                "item_id": j,                   # 0-indexed
                "score":   int(row[qcol]),
            })

    long_df = pd.DataFrame(records)
    n_users = long_df["user_id"].nunique()
    n_items = long_df["item_id"].nunique()
    print(f"Long format: {len(long_df):,} rows  |  "
          f"{n_users} users  ×  {n_items} items")
    print(f"Score distribution:\n{long_df['score'].value_counts().to_string()}\n")

    # ── 3. Train / Valid / Test split (70 / 10 / 20) ────────────────────────
    train_df, temp_df = train_test_split(
        long_df, test_size=0.30, random_state=RANDOM_SEED
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.667, random_state=RANDOM_SEED   # 10 / 20
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    train_df.to_csv("data/QPCS_train.csv", index=False)
    valid_df.to_csv("data/QPCS_valid.csv", index=False)
    test_df.to_csv( "data/QPCS_test.csv",  index=False)

    print(f"Train : {len(train_df):>6,} rows  ({len(train_df)/len(long_df)*100:.1f}%)")
    print(f"Valid : {len(valid_df):>6,} rows  ({len(valid_df)/len(long_df)*100:.1f}%)")
    print(f"Test  : {len(test_df):>6,} rows  ({len(test_df)/len(long_df)*100:.1f}%)")
    print("\nSaved → data/QPCS_train.csv, data/QPCS_valid.csv, data/QPCS_test.csv")

    # ── 4. 실행 정보 출력 ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("GNCDM 실행 파라미터 (config/training_config_QPCS.json 참고)")
    print(f"  --n_user  {n_users}")
    print(f"  --n_item  {n_items}")
    print(f"  --n_know  {len(kc_cols)}")
    print("="*60)


if __name__ == "__main__":
    main()
