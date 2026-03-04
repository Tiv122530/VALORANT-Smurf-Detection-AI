"""なぜこのPUUIDがextract_featuresでスキップされるか調べる"""
import sys, json, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

PUUID = "26d91571-3e25-5727-b3c5-99563f4cddf8"

# JSONロード
with open(f"collected_data/{PUUID}.json", encoding="utf-8") as f:
    data = json.load(f)

# smurf_aiのextract_featuresをインポート
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("smurf_ai", pathlib.Path("smurf_ai.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# load_all_playersで読み込まれるフォーマットを確認
players = mod.load_all_players()
matching = [p for p in players if p.get("puuid") == PUUID]
if not matching:
    print(f"load_all_players()にpuuid={PUUID}が含まれない")
    print("load_all_playersの仕組みを確認します...")
    # 直接JSONを渡してextract_featuresを試す
    feat = mod.extract_features(data)
    if feat:
        print("extract_features(data)は成功しました!")
        print(f"  smurf_score計算用キー: puuid={feat.get('puuid')}, player={feat.get('player')}")
    else:
        print("extract_features(data)がNoneを返しました")
        # デバッグ: dataの構造
        print(f"  stored_matches: {len(data.get('stored_matches') or [])}")
        acc = data.get("account") or {}
        print(f"  account name  : {acc.get('name')}#{acc.get('tag')}")
        # puuidがdataのトップレベルにあるか
        print(f"  top-level keys: {list(data.keys())}")
else:
    p = matching[0]
    print(f"load_all_playersで見つかりました: {list(p.keys())[:5]}")
    feat = mod.extract_features(p)
    if feat:
        print("extract_featuresは成功!")
    else:
        print("extract_featuresがNoneを返した → スキップ条件に引っかかっている")
