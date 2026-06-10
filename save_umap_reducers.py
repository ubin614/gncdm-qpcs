# -*- coding: utf-8 -*-
"""
훈련 학생의 θ를 기반으로 UMAP reducer(2D/3D)를 학습하여 저장합니다.
Streamlit 앱에서 새 학생을 transform()으로 빠르게 위치시키기 위해 1회 실행합니다.

실행:
    python save_umap_reducers.py

출력:
    result/QPCS/umap_reducer_2d.pkl
    result/QPCS/umap_reducer_3d.pkl
    result/QPCS/umap_train_coords_2d.npy   ← 730명 2D 좌표
    result/QPCS/umap_train_coords_3d.npy   ← 730명 3D 좌표
"""

import os
import pickle
import numpy as np
import pandas as pd
from umap import UMAP

THETA_CSV  = "result/QPCS/diagnosis/theta_all.csv"
SAVE_DIR   = "result/QPCS"
KC_NAMES   = ["KC1", "KC2", "KC3", "KC4", "KC5", "KC6"]

df = pd.read_csv(THETA_CSV)
theta_mat = df[KC_NAMES].values
print(f"훈련 학생 θ 로드: {theta_mat.shape}")

UMAP_PARAMS = dict(
    random_state=42,
    n_neighbors=15,
    min_dist=0.15,
    spread=2.0,
    init="random",
    metric="cosine"
)

print("2D UMAP 학습 중...")
reducer_2d = UMAP(n_components=2, **UMAP_PARAMS)
coords_2d  = reducer_2d.fit_transform(theta_mat)

print("3D UMAP 학습 중...")
reducer_3d = UMAP(n_components=3, **UMAP_PARAMS)
coords_3d  = reducer_3d.fit_transform(theta_mat)

with open(os.path.join(SAVE_DIR, "umap_reducer_2d.pkl"), "wb") as f:
    pickle.dump(reducer_2d, f)
with open(os.path.join(SAVE_DIR, "umap_reducer_3d.pkl"), "wb") as f:
    pickle.dump(reducer_3d, f)
np.save(os.path.join(SAVE_DIR, "umap_train_coords_2d.npy"), coords_2d)
np.save(os.path.join(SAVE_DIR, "umap_train_coords_3d.npy"), coords_3d)

print("저장 완료:")
print(f"  umap_reducer_2d.pkl  / umap_train_coords_2d.npy  ({coords_2d.shape})")
print(f"  umap_reducer_3d.pkl  / umap_train_coords_3d.npy  ({coords_3d.shape})")
