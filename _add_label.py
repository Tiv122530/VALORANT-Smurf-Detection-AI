import pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8")

PUUID = "26d91571-3e25-5727-b3c5-99563f4cddf8"
df = pd.read_csv("smurf_output/smurf_results_full.csv")
row = df[df["puuid"] == PUUID].iloc[0]

ld = pd.read_csv("labeled_data.csv")
if PUUID in ld["puuid"].values:
    ld.loc[ld["puuid"] == PUUID, "label"] = 1
    print("既存ラベルを 1 (スマーフ) に更新")
else:
    new_row = {"puuid": PUUID, "player": row["player"], "smurf_score": row["smurf_score"], "label": 1}
    ld = pd.concat([ld, pd.DataFrame([new_row])], ignore_index=True)
    print("新規ラベル追加: label=1 (スマーフ)")

ld.to_csv("labeled_data.csv", index=False)
n_s = int((ld["label"] == 1).sum())
n_n = int((ld["label"] == 0).sum())
print(f"現在のラベル: スマーフ={n_s}, 通常={n_n}, 合計={len(ld)}")
