import pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8")

PUUID = "26d91571-3e25-5727-b3c5-99563f4cddf8"
df = pd.read_csv("smurf_output/smurf_results_full.csv")
r = df[df["puuid"] == PUUID]

if len(r) == 0:
    print("NOT FOUND")
    sys.exit(1)

row = r.iloc[0]
str_cols = {"player", "judgment", "confidence", "current_rank", "highest_rank"}
print("=" * 52)
print("  スマーフ判定結果: RGX#欲しい")
print("=" * 52)
for col in ["player", "judgment", "smurf_score", "confidence",
            "current_rank", "highest_rank", "account_level",
            "matches_count", "avg_kd", "avg_hs_pct", "win_rate",
            "kd_rank_deviation", "iso_score", "lof_score",
            "ocsvm_score", "gmm_score", "rule_score"]:
    if col not in row.index:
        continue
    v = row[col]
    if col in str_cols:
        print(f"  {col:<22}: {v}")
    else:
        try:
            print(f"  {col:<22}: {float(v):.2f}")
        except Exception:
            print(f"  {col:<22}: {v}")

rank = int(df["smurf_score"].rank(ascending=False)[r.index[0]])
print(f"  {'rank_in_dataset':<22}: {rank} / {len(df)} 位 (疑惑スコア高い順)")
print("=" * 52)
