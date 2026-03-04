"""
train_supervised.py  - 教師あり学習モデルのトレーニング
==========================================================
教師あり学習パイプライン ステップ4:
  label_tool.py でラベリングしたデータを使い、
  高精度な教師あり分類モデルを訓練して保存する。

使い方:
    python train_supervised.py

必要なもの:
    labeled_data.csv  (label_tool.py で作成)

出力:
    smurf_model.pkl         ← 保存されたモデル (smurf_ai.py が自動読み込み)
    smurf_output/04_feature_importance.png
    smurf_output/05_confusion_matrix.png
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
LABELED_CSV    = Path("labeled_data.csv")
FULL_RESULTS   = Path("smurf_output/smurf_results_full.csv")
MODEL_PATH     = Path("smurf_model.pkl")
OUTPUT_DIR     = Path("smurf_output")
MIN_SAMPLES    = 30  # 学習に最低必要なラベル数

# ─────────────────────────────────────────
# 使用する特徴量リスト (教師あり学習用)
# *連続値のみ使用。score系はリークになるため注意
# ─────────────────────────────────────────
SUPERVISED_FEATURES = [
    # 戦闘性能
    "avg_kd", "avg_kda", "avg_hs_pct",
    "avg_dpr", "avg_spr", "avg_kpr",
    "median_kd", "median_kpr",
    # 安定性
    "kd_std", "kd_cv", "kills_std", "kills_cv",
    "hs_std", "hs_cv", "dpr_std", "dpr_cv", "score_cv",
    # 勝利
    "win_rate", "max_win_streak", "avg_margin",
    # エージェント
    "agent_diversity", "top_agent_ratio",
    # ランク乖離 (最重要グループ)
    "rank_gap", "tier_range", "tier_trend",
    "perf_rank_ratio_kd", "perf_rank_ratio_hs",
    "perf_rank_ratio_dpr", "perf_level_ratio",
    # ★ランク期待値正規化 (高精度化)
    "kd_rank_deviation", "hs_rank_deviation", "hs_kd_compound",
    # アカウント
    "account_level", "current_tier", "match_frequency",
    "activity_span_days",
    # ★デランク検出
    "intentional_loss_rate", "tank_streak_max",
    "kd_cliff_drop_rate", "kd_bimodal_score",
    # MMR系
    "peak_tier_v3", "peak_current_gap_v3",
    "seasons_played", "seasonal_rank_growth", "seasonal_avg_winrate",
    "avg_rr_change", "positive_rr_rate", "elo_velocity", "max_rr_streak",
    # 行動
    "avg_afk_rounds", "economy_efficiency", "solo_queue_rate",
    # ルールスコア (ドメイン知識エンコード済み)
    "rule_score",
]

# ─────────────────────────────────────────
# 日本語フォント設定
# ─────────────────────────────────────────
def _setup_font():
    for fp in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
        try:
            fn = fm.FontProperties(fname=fp).get_name()
            if "Gothic" in fn or "Meiryo" in fn or "Yu" in fn or "Noto" in fn:
                plt.rcParams["font.family"] = fn
                return
        except Exception:
            pass
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────
# XGBoost / LightGBM を試みる、失敗したら sklearn
# ─────────────────────────────────────────
def _build_model(n_smurf: int, n_normal: int):
    """利用可能な最良のモデルを返す"""
    ratio = n_normal / max(n_smurf, 1)

    # XGBoost を優先
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,  # 不均衡対策: 負例/正例の比率
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        print("[MODEL] XGBoost を使用します")
        return model, "xgboost"
    except ImportError:
        pass

    # LightGBM を試みる
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight={0: 1.0, 1: ratio},
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        print("[MODEL] LightGBM を使用します")
        return model, "lightgbm"
    except ImportError:
        pass

    # sklearn GradientBoosting (フォールバック)
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    print("[MODEL] sklearn GradientBoostingClassifier を使用します")
    print("  ※ pip install xgboost でより高精度なモデルを使えます")
    return model, "sklearn_gb"


def load_training_data():
    """ラベルデータ + フル特徴量をマージで読み込む"""
    if not LABELED_CSV.exists():
        print(f"[ERROR] {LABELED_CSV} が見つかりません。先に label_tool.py を実行してください。")
        sys.exit(1)

    labels_df = pd.read_csv(LABELED_CSV)
    n_total = len(labels_df)
    n_smurf = (labels_df["label"] == 1).sum()
    n_normal = (labels_df["label"] == 0).sum()

    print(f"[DATA] ラベルデータ読み込み: {n_total}件  (スマーフ: {n_smurf}, 通常: {n_normal})")

    if n_total < MIN_SAMPLES:
        print(f"[ERROR] 学習には最低 {MIN_SAMPLES} 件のラベルが必要です (現在: {n_total}件)")
        sys.exit(1)
    if n_smurf < 5:
        print(f"[ERROR] スマーフラベル(1)が少なすぎます。最低5件必要です (現在: {n_smurf}件)")
        sys.exit(1)
    if n_normal < 5:
        print(f"[ERROR] 通常ラベル(0)が少なすぎます。最低5件必要です (現在: {n_normal}件)")
        sys.exit(1)

    # フル特徴量をマージ (labeled_dataにない特徴量を補完)
    if FULL_RESULTS.exists():
        full_df = pd.read_csv(FULL_RESULTS)
        # labeled_dataにある特徴量を優先、追加特徴量はfullから補完
        merge_cols = ["puuid"] + [c for c in SUPERVISED_FEATURES
                                  if c not in labels_df.columns and c in full_df.columns]
        if merge_cols:
            labels_df = labels_df.merge(full_df[merge_cols], on="puuid", how="left")
            print(f"[DATA] フル特徴量から {len(merge_cols)-1} 列を補完しました")

    return labels_df, n_smurf, n_normal


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """特徴量行列とラベルを準備"""
    available = [c for c in SUPERVISED_FEATURES if c in df.columns]
    missing   = [c for c in SUPERVISED_FEATURES if c not in df.columns]
    if missing:
        print(f"[WARN] {len(missing)}個の特徴量が欠損 (スキップ): {missing[:5]}...")

    X = df[available].values.astype(float)
    y = df["label"].values.astype(int)

    # NaN/Inf を中央値で補完
    for i in range(X.shape[1]):
        mask = ~np.isfinite(X[:, i])
        if mask.any():
            X[mask, i] = np.nanmedian(X[:, i])

    print(f"[DATA] 使用特徴量数: {len(available)}")
    return X, y, available


def plot_feature_importance(model, feature_names: list[str], model_type: str):
    """特徴量重要度グラフを保存"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        if model_type in ("xgboost", "lightgbm"):
            importances = model.feature_importances_
        else:
            importances = model.feature_importances_

        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=True).tail(30)

        fig, ax = plt.subplots(figsize=(10, 10))
        bars = ax.barh(fi_df["feature"], fi_df["importance"],
                       color=plt.cm.RdYlGn(fi_df["importance"] / fi_df["importance"].max()))
        ax.set_xlabel("重要度 (Feature Importance)", fontsize=12)
        ax.set_title("スマーフ検出: 特徴量重要度 TOP30", fontsize=14, fontweight="bold")
        ax.axvline(fi_df["importance"].mean(), color="gray", linestyle="--", alpha=0.5, label="平均")
        ax.legend()
        plt.tight_layout()
        out_path = OUTPUT_DIR / "04_feature_importance.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[VIZ] 特徴量重要度: {out_path}")
    except Exception as e:
        print(f"[WARN] 特徴量重要度グラフ生成失敗: {e}")


def plot_confusion_matrix(y_true, y_pred):
    """混同行列を保存"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["通常", "スマーフ"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("混同行列 (交差検証最終fold)", fontsize=13)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "05_confusion_matrix.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] 混同行列: {out_path}")


def main():
    _setup_font()
    print()
    print("=" * 58)
    print(" VALORANT スマーフ - 教師あり学習モデル訓練")
    print("=" * 58)
    print()

    # ─── データ読み込み ───
    df, n_smurf, n_normal = load_training_data()
    X, y, feature_names = prepare_features(df)

    # ─── モデル選択 ───
    model, model_type = _build_model(n_smurf, n_normal)

    # ─── 交差検証 (5-fold Stratified) ───
    print()
    print("[CV] 5-fold 交差検証を実行中...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores  = []
    all_y_true  = []
    all_y_pred  = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # 不均衡対策: サンプル重みを計算
        sw = compute_sample_weight("balanced", y_tr)

        if model_type in ("xgboost", "lightgbm"):
            model.fit(X_tr, y_tr, sample_weight=sw,
                      eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_tr, y_tr, sample_weight=sw)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_prob)
        auc_scores.append(auc)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        print(f"  Fold {fold}: AUC = {auc:.4f}")

    mean_auc = np.mean(auc_scores)
    std_auc  = np.std(auc_scores)
    print()
    print(f"  平均 AUC : {mean_auc:.4f} ± {std_auc:.4f}")
    if mean_auc >= 0.90:
        print("  ✅ 優秀 (AUC >= 0.90)")
    elif mean_auc >= 0.80:
        print("  📈 良好 (AUC >= 0.80)  → もっとラベルを増やすと改善します")
    else:
        print("  ⚠  まだ改善が必要 (AUC < 0.80)  → ラベル数を増やしてください")

    # ─── 最終評価 ───
    print()
    print("[EVAL] 分類レポート (5-fold 全val合計):")
    print(classification_report(all_y_true, all_y_pred,
                                 target_names=["通常", "スマーフ"]))

    # ─── 全データで最終学習 ───
    print("[TRAIN] 全データで最終モデルを訓練中...")
    sw_all = compute_sample_weight("balanced", y)
    if model_type in ("xgboost", "lightgbm"):
        model.fit(X, y, sample_weight=sw_all, verbose=False)
    else:
        model.fit(X, y, sample_weight=sw_all)

    # ─── モデル保存 ───
    save_obj = {
        "model": model,
        "model_type": model_type,
        "feature_names": feature_names,
        "mean_auc": mean_auc,
        "trained_at": pd.Timestamp.now().isoformat(),
        "n_labeled": len(df),
        "n_smurf": int(n_smurf),
        "n_normal": int(n_normal),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_obj, f)
    print(f"[SAVE] モデル保存: {MODEL_PATH}")
    print(f"       → smurf_ai.py が次回起動時に自動的に利用します")

    # ─── 可視化 ───
    plot_feature_importance(model, feature_names, model_type)
    plot_confusion_matrix(all_y_true, all_y_pred)

    # ─── TOP特徴量 表示 ───
    try:
        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(10)
        print()
        print("[INFO] 最も重要な特徴量 TOP10:")
        for rank, (_, row) in enumerate(fi.iterrows(), 1):
            bar = "█" * int(row["importance"] / fi["importance"].max() * 20)
            print(f"  {rank:2}. {row['feature']:<35} {bar} {row['importance']:.4f}")
    except Exception:
        pass

    print()
    print("=" * 58)
    print(f" 訓練完了！ AUC: {mean_auc:.4f}  ラベル数: {len(df)}")
    print(" 次のステップ: python smurf_ai.py")
    print(" (教師ありモデルが自動で使われ、結果が改善されます)")
    print("=" * 58)


if __name__ == "__main__":
    main()
