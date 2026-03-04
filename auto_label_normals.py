"""
auto_label_normals.py  - 低スコアプレイヤーの自動ラベリング
============================================================
スマーフスコアが非常に低い (<= AUTO_NORMAL_THRESHOLD) プレイヤーを
「通常プレイヤー (label=0)」として自動でラベルを付ける。

これにより:
  - 手動ラベリングの負担を大幅削減
  - 教師ありモデルに必要な「正例 vs 負例」のバランスを確保

使い方:
    python auto_label_normals.py
    または run_pipeline.py から自動呼び出し

ロジック:
  - スマーフスコア <= 20 かつ
  - モデル4つ全てのスコアが低い (<= 35) プレイヤーを自動的に label=0 とする
  - 既にラベル済みのプレイヤーはスキップ
"""

import csv
import sys
import pandas as pd
import numpy as np
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

FULL_RESULTS    = Path("smurf_output/smurf_results_full.csv")
LABELED_OUTPUT  = Path("labeled_data.csv")

# この閾値以下は「通常プレイヤー」と自動判定
# smurf_scoreが靐15以下 = 全モデルが一致して「障害なし」と証明したプレイヤー
AUTO_NORMAL_SCORE_THRESHOLD = 15   # smurf_score の上限
AUTO_NORMAL_BUDGET          = 200  # 自動追加の上限
MIN_NORMAL_TARGET           = 80   # 通常ラベルの目標件数


def load_already_labeled() -> set[str]:
    if not LABELED_OUTPUT.exists():
        return set()
    df = pd.read_csv(LABELED_OUTPUT)
    return set(df["puuid"].astype(str).tolist())


def load_labeled_counts() -> tuple[int, int]:
    if not LABELED_OUTPUT.exists():
        return 0, 0
    df = pd.read_csv(LABELED_OUTPUT)
    n_smurf  = int((df["label"] == 1).sum())
    n_normal = int((df["label"] == 0).sum())
    return n_smurf, n_normal


def append_label(puuid: str, player: str, label: int, row: pd.Series):
    write_header = not LABELED_OUTPUT.exists()
    cols_to_save = [
        "puuid", "player", "smurf_score", "judgment", "confidence",
        "account_level", "current_tier", "current_rank",
        "matches_count", "avg_kd", "avg_kda", "avg_hs_pct",
        "win_rate", "avg_dpr", "avg_kpr",
        "kd_rank_deviation", "hs_rank_deviation", "hs_kd_compound",
        "kd_bimodal_score", "tank_streak_max", "kd_cliff_drop_rate", "intentional_loss_rate",
        "rank_gap", "perf_rank_ratio_kd", "perf_level_ratio",
        "peak_tier_v3", "peak_current_gap_v3",
        "iso_score", "lof_score", "ocsvm_score", "gmm_score", "rule_score",
    ]
    record = {"puuid": puuid, "player": player, "label": label}
    for col in cols_to_save:
        if col not in record:
            record[col] = row.get(col, "")

    with open(LABELED_OUTPUT, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def main():
    print("[AUTO-LABEL] 低スコアプレイヤーの自動ラベリング開始")

    if not FULL_RESULTS.exists():
        print(f"[ERROR] {FULL_RESULTS} が見つかりません。先に smurf_ai.py を実行してください。")
        sys.exit(1)

    df = pd.read_csv(FULL_RESULTS)
    for col in ["smurf_score"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(50)

    already_done = load_already_labeled()
    n_smurf_before, n_normal_before = load_labeled_counts()
    print(f"  現在のラベル: スマーフ={n_smurf_before}, 通常={n_normal_before}")

    # 自動ラベル対象: smurf_score が靐15以下のプレイヤー
    mask_low = df["smurf_score"] <= AUTO_NORMAL_SCORE_THRESHOLD
    candidates = df[mask_low & ~df["puuid"].astype(str).isin(already_done)].copy()

    # すでに目標件数に達している場合はスキップ
    if n_normal_before >= MIN_NORMAL_TARGET:
        print(f"  通常ラベルは既に{n_normal_before}件あります。自動ラベリングをスキップします。")
        return n_normal_before

    need = min(MIN_NORMAL_TARGET - n_normal_before, AUTO_NORMAL_BUDGET, len(candidates))
    if need <= 0:
        print(f"  対象候補なし（既に追加済みか候補が0件）。スキップします。")
        return n_normal_before

    # ランダムサンプリング (偏りを防ぐため)
    sampled = candidates.sample(n=need, random_state=42)
    added = 0
    for _, row in sampled.iterrows():
        puuid  = str(row.get("puuid", ""))
        player = str(row.get("player", "?"))
        append_label(puuid, player, 0, row)
        added += 1

    n_smurf_after, n_normal_after = load_labeled_counts()
    print(f"  自動追加: {added}件 (label=0, スコア<={AUTO_NORMAL_SCORE_THRESHOLD})")
    print(f"  最終ラベル: スマーフ={n_smurf_after}, 通常={n_normal_after}, 合計={n_smurf_after + n_normal_after}")
    return n_normal_after


if __name__ == "__main__":
    main()
