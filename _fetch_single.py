"""指定PUUIDのデータを収集して教師ありモデルで判定するスクリプト"""
import sys, json, os, pickle, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

from collector import (fetch_account_by_puuid, fetch_stored_matches_by_puuid,
                       fetch_mmr_by_puuid, fetch_mmr_v3_by_puuid,
                       fetch_stored_mmr_history_by_puuid, fetch_match_v4_by_puuid)

PUUID = "26d91571-3e25-5727-b3c5-99563f4cddf8"
REGIONS = ["ap", "kr", "na", "eu", "latam", "br"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "collected_data")

# ─── 1. アカウント情報 ────────────────────────────────────
print("=== STEP1: アカウント情報取得 ===")
account = fetch_account_by_puuid(PUUID)
if account:
    name = account.get("name")
    tag  = account.get("tag")
    print(f"  name  : {name}#{tag}")
    print(f"  level : {account.get('account_level')}")
    region = (account.get("region") or "ap").lower()
    if region not in REGIONS:
        region = "ap"
    print(f"  region: {region}")
else:
    print("  アカウント情報取得失敗 → 全リージョン試行")
    region = None

# ─── 2. stored-matches ───────────────────────────────────
print("\n=== STEP2: stored-matches 取得 ===")
matches = None
used_region = None
regions_to_try = ([region] + [r for r in REGIONS if r != region]) if region else REGIONS
for r in regions_to_try:
    m = fetch_stored_matches_by_puuid(r, PUUID)
    if m:
        matches = m
        used_region = r
        print(f"  成功 region={r}, {len(m)}試合")
        break
    else:
        print(f"  {r}: データなし")

if not matches:
    print("  全リージョン失敗 → 終了")
    sys.exit(1)

# ─── 3. 追加データ ────────────────────────────────────────
print(f"\n=== STEP3: 追加API (region={used_region}) ===")
mmr         = fetch_mmr_by_puuid(used_region, PUUID)
print(f"  mmr       : {'OK' if mmr else 'NG'}")
mmr_v3      = fetch_mmr_v3_by_puuid(used_region, PUUID)
print(f"  mmr_v3    : {'OK' if mmr_v3 else 'NG'}")
mmr_history = fetch_stored_mmr_history_by_puuid(used_region, PUUID)
print(f"  mmr_history: {len(mmr_history) if mmr_history else 0}件")
match_v4    = fetch_match_v4_by_puuid(used_region, PUUID, size=5)
print(f"  match_v4  : {len(match_v4) if match_v4 else 0}件")

# ─── 4. JSON保存 ──────────────────────────────────────────
data = {
    "account":       account,
    "stored_matches": matches,
    "mmr":           mmr,
    "mmr_v3":        mmr_v3,
    "mmr_history":   mmr_history,
    "match_v4":      match_v4,
}
out_path = os.path.join(DATA_DIR, f"{PUUID}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
print(f"\n  → {out_path} 保存完了")

# ─── 5. smurf_ai で特徴抽出 & 判定 ──────────────────────
print("\n=== STEP4: スマーフ判定 ===")
import importlib.util, pathlib

ai_path = pathlib.Path(__file__).parent / "smurf_ai.py"
spec = importlib.util.spec_from_file_location("smurf_ai", ai_path)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

detector = mod.EnsembleSmurfDetector()
result   = detector.fit_predict([PUUID])

if result is None or len(result) == 0:
    print("  判定失敗（データ不足の可能性）")
    sys.exit(1)

row = result.iloc[0]
print()
print("=" * 50)
print(f"  プレイヤー    : {row.get('player', '?')}")
print(f"  最終スコア    : {row.get('smurf_score', '?'):.1f} / 100")
print(f"  判定          : {row.get('judgment', '?')}")
print(f"  信頼度        : {row.get('confidence', '?')}")
print(f"  現在ランク    : {row.get('current_rank', '?')}")
print(f"  最高ランク    : {row.get('highest_rank', '?')}")
print(f"  アカウントLv  : {row.get('account_level', '?')}")
print(f"  試合数        : {row.get('matches_count', '?')}")
print(f"  平均KD        : {row.get('avg_kd', '?'):.2f}")
print(f"  平均HS%       : {row.get('avg_hs_pct', '?'):.1f}%")
print(f"  勝率          : {row.get('win_rate', '?'):.1f}%")
print(f"  KD rank偏差   : {row.get('kd_rank_deviation', '?'):.2f}")
print()
# 教師ありモデルスコアがある場合
if "supervised_prob" in row.index or "sup_prob" in row.index:
    sp = row.get("supervised_prob") or row.get("sup_prob")
    print(f"  教師あり確率  : {sp:.4f}")
print(f"  IsolForest    : {row.get('iso_score', '?'):.1f}")
print(f"  LOF           : {row.get('lof_score', '?'):.1f}")
print(f"  OCSVM         : {row.get('ocsvm_score', '?'):.1f}")
print(f"  GMM           : {row.get('gmm_score', '?'):.1f}")
print(f"  Rule          : {row.get('rule_score', '?'):.1f}")
print("=" * 50)
