"""
既存 collected_data/ の1002人JSONに新しいAPIデータを追加する。
追加データ:
  1. mmr_v3    : v3/by-puuid/mmr (seasonal, peak)
  2. mmr_history: v1/by-puuid/stored-mmr-history (elo変動履歴)
  3. match_v4  : v4/by-puuid/matches (ability, economy, behavior)

中断再開対応・並列実行。
"""

import json
import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# collector.py の共有部品をインポート
from collector import (
    api_get, rotator, save_player_data,
    fetch_mmr_v3_by_puuid,
    fetch_stored_mmr_history_by_puuid,
    fetch_match_v4_by_puuid,
    DELAY_BETWEEN_CALLS, DATA_DIR,
)

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
NUM_WORKERS = 6
PROGRESS_FILE = "update_progress.json"
REGION = "ap"


def load_update_progress() -> set:
    """更新済みpuuidセットをロード"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("updated_puuids", []))
        except:
            pass
    return set()


def save_update_progress(updated: set):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"updated_puuids": list(updated), "last_save": datetime.now().isoformat()},
                  f, ensure_ascii=False)


def update_one_player(filepath: str, region: str) -> dict | None:
    """
    1人のJSONを読み込み、新データを追加して上書き保存。
    戻り値: {"puuid": ..., "name": ..., "status": "ok"/"skip"/"error"}
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"puuid": "?", "name": "?", "status": "error", "error": str(e)}

    puuid = data.get("puuid", "")
    player = data.get("player", "?")

    # 既にデータがあるかチェック
    has_v3 = data.get("mmr_v3") is not None
    has_hist = data.get("mmr_history") is not None
    has_v4 = data.get("match_v4") is not None

    if has_v3 and has_hist and has_v4:
        return {"puuid": puuid, "name": player, "status": "skip"}

    # --- 1. mmr_v3 (seasonal / peak) ---
    if not has_v3:
        mmr_v3 = fetch_mmr_v3_by_puuid(region, puuid)
        data["mmr_v3"] = mmr_v3
        time.sleep(DELAY_BETWEEN_CALLS)
    
    # --- 2. stored-mmr-history ---
    if not has_hist:
        mmr_history = fetch_stored_mmr_history_by_puuid(region, puuid)
        data["mmr_history"] = mmr_history
        time.sleep(DELAY_BETWEEN_CALLS)

    # --- 3. v4 match detail ---
    if not has_v4:
        match_v4 = fetch_match_v4_by_puuid(region, puuid, size=3)
        data["match_v4"] = match_v4
        time.sleep(DELAY_BETWEEN_CALLS)

    data["updated_at"] = datetime.now().isoformat()

    # 上書き保存
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"puuid": puuid, "name": player, "status": "ok"}


def main():
    print("=" * 65)
    print("  既存データ更新: mmr_v3 + mmr_history + match_v4 追加")
    print(f"  ワーカー数: {NUM_WORKERS}")
    print("=" * 65)

    # ファイル一覧
    data_dir = Path(DATA_DIR)
    json_files = sorted(data_dir.glob("*.json"))
    total = len(json_files)
    print(f"\n[DATA] {total} ファイル検出")

    # 進捗ロード
    updated_set = load_update_progress()
    print(f"[RESUME] 更新済み: {len(updated_set)} 人")

    # 未更新ファイルを抽出
    remaining = []
    for fp in json_files:
        puuid = fp.stem  # ファイル名 = puuid
        if puuid not in updated_set:
            remaining.append(fp)

    print(f"[TODO] 残り: {len(remaining)} 人\n")

    if not remaining:
        print("[DONE] 全プレイヤーが既に更新済みです")
        return

    lock = threading.Lock()
    start_time = time.time()
    ok_count = 0
    skip_count = 0
    error_count = 0
    batch_size = NUM_WORKERS

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]

            elapsed = time.time() - start_time
            done = ok_count + skip_count + error_count
            rate = done / elapsed * 60 if elapsed > 0 else 0
            remaining_count = len(remaining) - done
            eta = remaining_count / rate if rate > 0 else 0

            print(f"--- バッチ {batch_start//batch_size + 1} | "
                  f"完了: {done}/{len(remaining)} | "
                  f"{rate:.1f}人/分 | ETA: {eta:.0f}分 | "
                  f"{rotator.report()} ---")

            futures = {}
            for fp in batch:
                future = executor.submit(update_one_player, str(fp), REGION)
                futures[future] = fp

            for future in as_completed(futures):
                fp = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  [ERROR] {fp.name}: {e}")
                    error_count += 1
                    continue

                if result is None:
                    error_count += 1
                    continue

                status = result["status"]
                puuid = result["puuid"]
                name = result["name"]

                if status == "ok":
                    ok_count += 1
                    with lock:
                        updated_set.add(puuid)
                    print(f"  [OK] {name} (+mmr_v3, +mmr_history, +match_v4) "
                          f"({ok_count + skip_count + error_count}/{len(remaining)})")
                elif status == "skip":
                    skip_count += 1
                    with lock:
                        updated_set.add(puuid)
                else:
                    error_count += 1
                    print(f"  [ERR] {name}: {result.get('error', '?')}")

            # 進捗保存 (バッチごと)
            save_update_progress(updated_set)

    elapsed_total = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"  更新完了!")
    print(f"  成功: {ok_count} | スキップ: {skip_count} | エラー: {error_count}")
    print(f"  所要時間: {elapsed_total/60:.1f} 分")
    print(f"  {rotator.report()}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
