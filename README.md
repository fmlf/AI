

# 🌟 AI를 활용한 침임탐지 예측모델
 
> 데이터셋 : https://www.kaggle.com/datasets/dhoogla/cicidscollection/data

> 모델: RandomForest

> 정확도 : 98%

---

## 🔧 기능

학습한 데이터셋을 기반으로 비정상적인 행위를 학습해서 침입탐지/공격 행위를 예측함

---

## 💻 코드 예시
┌─────────────────────────┐
│ AI_code
├─────────────────────────┤
│# -*- coding: utf-8 -*-
|import warnings
|warnings.filterwarnings("ignore")
|
|import numpy as np
|import pandas as pd
|import matplotlib.pyplot as plt
|import seaborn as sns 
|
|from sklearn.model_selection import train_test_split
|from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
|from sklearn.ensemble import RandomForestClassifier
|from sklearn.preprocessing import StandardScaler
|from sklearn.pipeline import Pipeline
|from imblearn.over_sampling import SMOTE
|
|# ----------------------
|# 0) 데이터 로드
|# ----------------------
|PARQUET_PATH = "AI/cic-collection.parquet"
|df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
|print(f"[OK] Loaded: {df.shape}")
|
|# ----------------------
|# 1) 전처리
|# ----------------------
|TARGET = "ClassLabel"
|X_all = df.drop(columns=["Label", "ClassLabel"], errors="ignore")
|X = X_all.select_dtypes(include=[np.number]).copy()
|y = df[TARGET].astype(str)
|
|# 결측/inf
|X = X.replace([np.inf, -np.inf], np.nan)
|if X.isna().any().any():
|    X = X.fillna(X.median(numeric_only=True))
|
|# 빠른 샘플(속도/메모리 맞춰 조절)
|N = min(len(df), 300_000)  # 처음엔 300k 권장 (여유되면 500k~1M)
|X_s, _, y_s, _ = train_test_split(X, y, train_size=N, stratify=y, random_state=42)
|
|# ----------------------
|# 2) 1단계: 정상 vs 공격 (이진)
|# ----------------------
|is_attack = (y_s != "Benign").astype(int)  # 1=공격, 0=정상
|X_train, X_test, yb_train, yb_test = train_test_split(
|    X_s, is_attack, test_size=0.2, stratify=is_attack, random_state=42
|)
|
|rf_bin = RandomForestClassifier(
|    n_estimators=200,
|    class_weight="balanced",
|    max_depth=None,
|    min_samples_leaf=2,
|    n_jobs=-1,
|    random_state=42,
|    verbose=0
|)
|rf_bin.fit(X_train, yb_train)
|
|# 임계값: 자신 있을 때만 공격으로 (정확도↑/공격 재현율↓)
|tau_attack = 0.85
|proba_attack = rf_bin.predict_proba(X_test)[:, 1]
|pred_attack_mask = (proba_attack >= tau_attack).astype(int)
|
|print("\n[Stage1] Binary (Benign vs Attack)")
|print("accuracy:", accuracy_score(yb_test, pred_attack_mask))
|print(classification_report(yb_test, pred_attack_mask, digits=4))
|
|# ----------------------
|# 3) 2단계: 공격만 세부 분류 (SMOTE)
|# ----------------------
|# 2단계 학습셋: X_train 중 공격만
|mask_train_attack = (yb_train == 1)
|X_train_attack = X_train[mask_train_attack]
|y_train_attack = y_s.loc[X_train_attack.index]  # 원래 다중 라벨
|
|# SMOTE로 희귀 클래스 보강 (필요 수치 조정)
|minor_targets = {
|    "Infiltration": 8000,
|    "Webattack": 8000,
|    "Portscan": 8000
|}
|smote = SMOTE(random_state=42, sampling_strategy=minor_targets)
|X_tr_bal, y_tr_bal = smote.fit_resample(X_train_attack, y_train_attack)
|
|print("\n[SMOTE] Before:")
|print(y_train_attack.value_counts())
|print("\n[SMOTE] After:")
|print(y_tr_bal.value_counts())
|
|rf_multi = RandomForestClassifier(
|    n_estimators=300,
|    class_weight="balanced_subsample",
|    max_depth=None,
|    min_samples_leaf=1,
|    n_jobs=-1,
|    random_state=42,
|    verbose=0
|)
|rf_multi.fit(X_tr_bal, y_tr_bal)
|
|# ----------------------
|# 4) 최종 예측 결합
|# ----------------------
|y_test_full = y_s.loc[X_test.index]  # 원래 다중 타깃
|
|# 1단계에서 Benign으로 본 건 그대로 Benign
|final_pred = np.array(["Benign"] * len(X_test), dtype=object)
|
|# 1단계에서 공격으로 본 샘플만 2단계 분류기 적용
|attack_idx = np.where(pred_attack_mask == 1)[0]
|if len(attack_idx) > 0:
|    sub_preds = rf_multi.predict(X_test.iloc[attack_idx])
|    final_pred[attack_idx] = sub_preds
|
|# ----------------------
|# 5) 평가
|# ----------------------
|print("\n=== FINAL REPORT (Two-Stage RF + threshold) ===")
|print(classification_report(y_test_full, final_pred, digits=4))
|acc = accuracy_score(y_test_full, final_pred)
|print("Final accuracy:", round(acc, 4))
|
|# 혼동행렬
|
|labels_sorted = sorted(y_s.unique().tolist())
|cm = confusion_matrix(y_test_full, final_pred, labels=labels_sorted)
|cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in labels_sorted],
|                        columns=[f"pred_{c}" for c in labels_sorted])
|print("\nConfusion matrix (head):")
|print(cm_df.head(min(8, len(cm_df))))
|
|
|
|# confusion matrix 생성
|cm = confusion_matrix(y_test_full, final_pred, labels=labels_sorted)
|#cm = confusion_matrix(y_test_full, final_pred, labels=labels_sorted)
|cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in labels_sorted],
|                        columns=[f"pred_{c}" for c in labels_sorted])
|print("\nConfusion matrix (head):")
|print(cm_df.head(min(8, len(cm_df))))
|
|plt.figure(figsize=(8,6))
|sns.heatmap(cm,
|            annot=True, fmt="d", cmap="Blues",
|            xticklabels=labels_sorted,
|            yticklabels=labels_sorted)
|
|#plt.imshow(cm, aspect="auto")
|plt.title("Confusion Matrix (Two-Stage RF)")
|plt.xlabel("Predicted")
|plt.ylabel("True")
|#plt.colorbar()
|plt.tight_layout()
|plt.show()
|
|
|"""
|plt.figure()
|plt.imshow(cm, aspect="auto")
|plt.title("Confusion Matrix (Two-Stage RF)")
|plt.xlabel("Predicted")
|plt.ylabel("True")
|plt.colorbar()
|plt.tight_layout()
|plt.show()
|"""
|"""
|def plot_top_features(importances, feature_names, topn=20, title="Feature Importances"):
|    imp = np.asarray(importances)
|    feats = np.asarray(feature_names)
|
|    order = np.argsort(imp)[::-1][:topn]
|    top_feats = feats[order]
|    top_vals = imp[order]
|
|    plt.figure(figsize=(8, 6))
|    plt.barh(top_feats[::-1], top_vals[::-1], color="skyblue", edgecolor="black")
|    plt.title(title)
|    plt.xlabel("Importance")
|    plt.ylabel("Feature")
|    plt.tight_layout()
|    plt.show()
|"""
|
|# ----------------------
|# 6) 상위 특징 (RandomForest 중요도 기준)
|# ----------------------
|importances = rf_multi.feature_importances_
|feat_importance = (
|
|    pd.DataFrame({"feature": X.columns, "importance": importances})
|    .sort_values("importance", ascending=False)
|    .head(20)
|)
|print("\nTop 20 features (RandomForest)")
|print(feat_importance)
|
|plt.figure(figsize=(8, 6))
|plt.barh(feat_importance["feature"][::-1], feat_importance["importance"][::-1])
|plt.title("Top 20 Features (RandomForest Importance)")
|plt.xlabel("Importance")
|plt.ylabel("Feature")
|
|plt.tight_layout()
|plt.show()
|
|
│ } │
│ } │
└─────────────────────────┘






