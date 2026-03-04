"""
check_puuid.py  - PUUIDを入力してスマーフ判定 (教師ありモデルのみ)
=====================================================================
使い方:
    python check_puuid.py <PUUID>
    python check_puuid.py               ← 対話入力
"""

import sys
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# ── パス設定 ────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "collected_data"
MODEL_PATH = BASE_DIR / "smurf_model.pkl"
REGIONS    = ["ap", "kr", "na", "eu", "latam", "br"]

# ── collector / smurf_ai をインポート ────────────────────
sys.path.insert(0, str(BASE_DIR))
from collector import (
    fetch_account_by_puuid,
    fetch_stored_matches_by_puuid,
    fetch_mmr_by_puuid,
    fetch_mmr_v3_by_puuid,
    fetch_stored_mmr_history_by_puuid,
    fetch_match_v4_by_puuid,
)
import importlib.util
_spec = importlib.util.spec_from_file_location("smurf_ai", BASE_DIR / "smurf_ai.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
extract_features = _mod.extract_features

# ── 教師ありモデルで使う特徴量 ──────────────────────────
SUPERVISED_FEATURES = [
    "avg_kd", "avg_kda", "avg_hs_pct",
    "avg_dpr", "avg_spr", "avg_kpr",
    "median_kd", "median_kpr",
    "kd_std", "kd_cv", "kills_std", "kills_cv",
    "hs_std", "hs_cv", "dpr_std", "dpr_cv", "score_cv",
    "win_rate", "max_win_streak", "avg_margin",
    "agent_diversity", "top_agent_ratio",
    "rank_gap", "tier_range", "tier_trend",
    "perf_rank_ratio_kd", "perf_rank_ratio_hs",
    "perf_rank_ratio_dpr", "perf_level_ratio",
    "kd_rank_deviation", "hs_rank_deviation", "hs_kd_compound",
    "account_level", "current_tier", "match_frequency",
    "activity_span_days",
    "intentional_loss_rate", "tank_streak_max",
    "kd_cliff_drop_rate", "kd_bimodal_score",
    "peak_tier_v3", "peak_current_gap_v3",
    "seasons_played", "seasonal_rank_growth", "seasonal_avg_winrate",
    "avg_rr_change", "positive_rr_rate", "elo_velocity", "max_rr_streak",
    "avg_afk_rounds", "economy_efficiency", "solo_queue_rate",
    "rule_score",
]

TIER_NAMES = {
    0:"Unranked", 1:"Iron 1", 2:"Iron 2", 3:"Iron 3",
    4:"Bronze 1", 5:"Bronze 2", 6:"Bronze 3",
    7:"Silver 1", 8:"Silver 2", 9:"Silver 3",
    10:"Gold 1", 11:"Gold 2", 12:"Gold 3",
    13:"Platinum 1", 14:"Platinum 2", 15:"Platinum 3",
    16:"Diamond 1", 17:"Diamond 2", 18:"Diamond 3",
    19:"Ascendant 1", 20:"Ascendant 2", 21:"Ascendant 3",
    22:"Immortal 1", 23:"Immortal 2", 24:"Immortal 3",
    25:"Radiant",
}

def tier_name(t):
    try:
        return TIER_NAMES.get(int(t), f"Tier{int(t)}")
    except Exception:
        return str(t)


# ══════════════════════════════════════════
# 1. データ取得
# ══════════════════════════════════════════
def fetch_and_save(puuid: str) -> dict | None:
    """APIからデータを取得してJSONに保存。既存ファイルがあればそれを返す。"""
    json_path = DATA_DIR / f"{puuid}.json"

    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        # 旧フォーマット(topレベルにpuuidなし)の場合は補完
        if not data.get("puuid"):
            acc = data.get("account") or {}
            data["puuid"]  = puuid
            data["name"]   = acc.get("name", "")
            data["tag"]    = acc.get("tag", "")
            data["player"] = f"{acc.get('name','')}#{acc.get('tag','')}"
            data["region"] = acc.get("region", "ap")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        print(f"  キャッシュ使用: {json_path.name}")
        return data

    # ── APIから新規取得 ──
    print("  APIからデータを取得中...")
    account = fetch_account_by_puuid(puuid)
    if not account:
        print("  [ERROR] アカウント情報の取得に失敗しました (PUUIDが無効の可能性)")
        return None

    region = (account.get("region") or "ap").lower()
    if region not in REGIONS:
        region = "ap"

    # stored-matches
    matches = None
    used_region = None
    for r in [region] + [x for x in REGIONS if x != region]:
        m = fetch_stored_matches_by_puuid(r, puuid)
        if m:
            matches = m
            used_region = r
            break

    if not matches:
        print("  [ERROR] 試合データが取得できませんでした")
        return None

    mmr         = fetch_mmr_by_puuid(used_region, puuid)
    mmr_v3      = fetch_mmr_v3_by_puuid(used_region, puuid)
    mmr_history = fetch_stored_mmr_history_by_puuid(used_region, puuid)
    match_v4    = fetch_match_v4_by_puuid(used_region, puuid, size=5)

    data = {
        "puuid":         puuid,
        "name":          account.get("name", ""),
        "tag":           account.get("tag", ""),
        "player":        f"{account.get('name','')}#{account.get('tag','')}",
        "region":        used_region,
        "collected_at":  datetime.now().isoformat(),
        "account":       account,
        "stored_matches": matches,
        "mmr":           mmr,
        "mmr_v3":        mmr_v3,
        "mmr_history":   mmr_history,
        "match_v4":      match_v4,
    }
    DATA_DIR.mkdir(exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  データ保存: {json_path.name}")
    return data


# ══════════════════════════════════════════
# 2. 教師ありモデルで予測
# ══════════════════════════════════════════
def predict_supervised(feat: dict) -> dict:
    """教師ありモデルのみでスマーフ確率を返す"""
    if not MODEL_PATH.exists():
        return {"error": "smurf_model.pkl が見つかりません。先に train_supervised.py を実行してください。"}

    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)

    model          = obj["model"]
    feature_names  = obj["feature_names"]

    # 特徴量ベクトルを組み立て
    row = {k: feat.get(k, 0.0) for k in feature_names}
    X   = pd.DataFrame([row])[feature_names].values.astype(float)
    X   = np.nan_to_num(X, nan=0.0)

    prob = float(model.predict_proba(X)[0][1])  # スマーフ確率 0~1

    return {
        "prob":         prob,
        "score":        round(prob * 100, 1),
        "auc":          obj.get("mean_auc"),
        "n_labels":     obj.get("n_labeled"),
        "n_smurf":      obj.get("n_smurf"),
        "trained_date": obj.get("trained_at"),
        "feature_names": feature_names,
        "feature_values": row,
    }


# ══════════════════════════════════════════
# 3. 結果表示
# ══════════════════════════════════════════
def print_result(feat: dict, pred: dict):
    if "error" in pred:
        print(f"\n[ERROR] {pred['error']}")
        return

    prob  = pred["prob"]
    score = pred["score"]

    # 判定
    if prob >= 0.80:
        judgment = "🔴 スマーフ確定"
        color_bar = "████████████████████"
    elif prob >= 0.55:
        judgment = "🟠 スマーフ可能性高"
        color_bar = "████████████░░░░░░░░"
    elif prob >= 0.35:
        judgment = "🟡 グレーゾーン"
        color_bar = "████████░░░░░░░░░░░░"
    else:
        judgment = "🟢 通常プレイヤー"
        color_bar = "████░░░░░░░░░░░░░░░░"

    print()
    print("=" * 58)
    print(f"  VALORANT スマーフ判定結果 (教師ありモデル)")
    print("=" * 58)
    print(f"  プレイヤー     : {feat.get('player', '?')}")
    print(f"  PUUID          : {feat.get('puuid', '?')}")
    print()
    print(f"  ━━━ 判定 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {judgment}")
    print(f"  スマーフ確率   : {score:.1f}%  [{color_bar}]")
    print()
    print(f"  ━━━ プロフィール ━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  現在ランク     : {feat.get('current_rank', '?')}")
    print(f"  最高ランク     : {feat.get('highest_rank', '?')}")
    print(f"  アカウントLv   : {int(feat.get('account_level', 0))}")
    print(f"  試合数         : {int(feat.get('matches_count', 0))}")
    print()
    print(f"  ━━━ 戦闘指標 ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  平均KD         : {feat.get('avg_kd', 0):.2f}")
    print(f"  平均HS%        : {feat.get('avg_hs_pct', 0)*100:.1f}%")
    print(f"  平均ダメージ   : {feat.get('avg_dpr', 0):.0f}")
    print(f"  勝率           : {feat.get('win_rate', 0)*100:.1f}%")
    print(f"  KDランク偏差   : {feat.get('kd_rank_deviation', 0):.2f}  (高い=ランク以上の実力)")
    print()
    print(f"  ━━━ スマーフシグナル ━━━━━━━━━━━━━━━━━━")
    rg = feat.get("rank_gap", 0)
    tt = feat.get("tier_trend", 0)
    il = feat.get("intentional_loss_rate", 0)
    ts = feat.get("tank_streak_max", 0)
    kcd = feat.get("kd_cliff_drop_rate", 0)
    kbm = feat.get("kd_bimodal_score", 0)
    rs = feat.get("rule_score", 0)
    print(f"  ランク乖離     : {rg:.1f}  (現在vs最高ランクの差)")
    print(f"  ランク降下傾向 : {tt:.3f}  (負=下がり続けている)")
    print(f"  意図的負け率   : {il*100:.1f}%")
    print(f"  連続タンク      : {ts:.0f}試合")
    print(f"  KD急落率       : {kcd:.3f}")
    print(f"  KD二峰性スコア : {kbm:.2f}  (高=高低KDが混在=デランク疑惑)")
    print(f"  ルールスコア   : {rs:.1f} / 100")
    print()
    print(f"  ━━━ モデル情報 ━━━━━━━━━━━━━━━━━━━━━━━━")
    auc_str     = f"{pred['auc']:.4f}"    if pred.get('auc')         is not None else "N/A"
    nlabels_str = str(pred['n_labels'])   if pred.get('n_labels')    is not None else "N/A"
    nsmurf_str  = str(pred['n_smurf'])    if pred.get('n_smurf')     is not None else "N/A"
    date_str    = pred.get('trained_date') or "N/A"
    print(f"  訓練AUC        : {auc_str}")
    print(f"  訓練ラベル数   : {nlabels_str}件 (スマーフ: {nsmurf_str}件)")
    print(f"  訓練日         : {date_str}")
    print("=" * 58)

    # ── 特徴量の注目ポイントを自動コメント ──
    comments = []
    if feat.get("kd_rank_deviation", 0) > 2.0:
        comments.append("  → KDがランク平均より大幅に高い (スマーフ特徴)")
    if rg >= 6:
        comments.append(f"  → 現在ランクと最高ランクが{rg:.0f}段階も乖離")
    if il > 0.3:
        comments.append(f"  → 意図的な負け行動が{il*100:.0f}%のマッチで疑われる")
    if ts >= 5:
        comments.append(f"  → {ts:.0f}連続でタンク(急落)パターンを検出")
    if feat.get("account_level", 99) < 30 and prob > 0.5:
        comments.append("  → アカウントLv30未満の低レベル高パフォーマンス")
    if rs >= 70:
        comments.append(f"  → ルールベーススコアが{rs:.0f}点 (高スマーフ疑惑)")

    if comments:
        print()
        print("  ⚠️  注目ポイント:")
        for c in comments:
            print(c)
        print()


# ══════════════════════════════════════════
# メイン
# ══════════════════════════════════════════
def main():
    # PUUID の取得
    if len(sys.argv) >= 2:
        puuid = sys.argv[1].strip()
    else:
        print("VALORANT スマーフ判定ツール (教師ありモデル)")
        print("-" * 42)
        puuid = input("PUUID を入力してください: ").strip()

    if not puuid:
        print("[ERROR] PUUIDが入力されていません")
        sys.exit(1)

    print(f"\n[CHECK] {puuid}")
    print("-" * 58)

    # 1. データ取得
    print("  [1/3] データ取得...")
    data = fetch_and_save(puuid)
    if data is None:
        sys.exit(1)

    # 2. 特徴量抽出
    print("  [2/3] 特徴量抽出...")
    feat = extract_features(data)
    if feat is None:
        print("  [ERROR] 特徴量の抽出に失敗しました (コンペマッチが3試合未満の可能性)")
        sys.exit(1)

    # puuid / player を補完
    feat["puuid"]  = puuid
    feat["player"] = data.get("player", "?")

    # ランク名を付与
    mmr = data.get("mmr", {}) or {}
    cur = mmr.get("current_data", {}) or {}
    hi  = mmr.get("highest_rank", {}) or {}
    feat["current_rank"] = cur.get("currenttierpatched") or tier_name(feat.get("current_tier", 0))
    feat["highest_rank"] = hi.get("patched_tier") or tier_name(feat.get("peak_tier_v3", 0))

    # 3. 教師ありモデルで判定
    print("  [3/3] 教師ありモデルで判定中...")
    pred = predict_supervised(feat)

    # 4. 結果表示
    print_result(feat, pred)


if __name__ == "__main__":
    main()
