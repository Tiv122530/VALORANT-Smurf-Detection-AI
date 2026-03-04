"""
VALORANT 大規模データ収集器 (puuid版)
- /valorant/v1/by-puuid/stored-matches/{region}/{puuid} を使用
- 起点プレイヤーから BFS でプレイヤーを自動発見
- 重複なし (puuidベース) で1000人分の stored-matches を取得・保存
- トークンを順番にローテーション
- レート制限対策 + 中断再開対応
"""

import requests
import time
import json
import os
import sys
import threading
from collections import deque
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from henrik import henriktoken

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
BASE_URL = "https://api.henrikdev.xyz"
DATA_DIR = "collected_data"
PROGRESS_FILE = "collect_progress.json"
TARGET_PLAYERS = 1000
NUM_WORKERS = 6  # 並列ワーカー数 (12トークン ÷ 1人3API = 4だが余裕見て6)

# レート制限設定 (12トークン × 各30req/min)
# 安全マージンを取って同一トークン2秒間隔
DELAY_PER_TOKEN = 2.0
DELAY_BETWEEN_CALLS = 0.05
RATE_LIMIT_WAIT = 10
MAX_RETRIES = 3


# ─────────────────────────────────────────
# トークンローテーター
# ─────────────────────────────────────────
class TokenRotator:
    """トークンを順番にローテーションしてレート制限を回避"""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.index = 0
        self.last_used = {t: 0.0 for t in tokens}
        self.lock = threading.Lock()
        self.total_calls = 0
        self.rate_limited_count = 0

    def get_next(self) -> tuple[str, dict]:
        """次に使えるトークンを取得 (必要なら待機)"""
        with self.lock:
            best_token = None
            best_wait = float("inf")
            best_idx = 0

            for i in range(len(self.tokens)):
                idx = (self.index + i) % len(self.tokens)
                token = self.tokens[idx]
                elapsed = time.time() - self.last_used[token]
                wait_needed = max(0, DELAY_PER_TOKEN - elapsed)

                if wait_needed < best_wait:
                    best_wait = wait_needed
                    best_token = token
                    best_idx = idx

            if best_wait > 0:
                time.sleep(best_wait)

            self.last_used[best_token] = time.time()
            self.index = (best_idx + 1) % len(self.tokens)
            self.total_calls += 1

            return best_token, {"Authorization": best_token}

    def report(self):
        return (f"API呼出: {self.total_calls}回, "
                f"429エラー: {self.rate_limited_count}回")


rotator = TokenRotator(henriktoken)


# ─────────────────────────────────────────
# API呼び出し (リトライ付き)
# ─────────────────────────────────────────
def api_get(url: str, params: dict = None) -> dict | None:
    """レート制限対策付きGETリクエスト"""
    for attempt in range(MAX_RETRIES):
        token, headers = rotator.get_next()
        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)

            if r.status_code == 429:
                rotator.rate_limited_count += 1
                wait = RATE_LIMIT_WAIT * (attempt + 1)
                print(f"  [429] レート制限 → {wait}秒待機 (token ...{token[-8:]})")
                time.sleep(wait)
                continue

            if r.status_code == 404:
                return None

            if r.status_code >= 500:
                print(f"  [5xx] サーバーエラー {r.status_code} → リトライ")
                time.sleep(3)
                continue

            r.raise_for_status()
            return r.json()

        except requests.exceptions.Timeout:
            print(f"  [TIMEOUT] → リトライ ({attempt+1}/{MAX_RETRIES})")
            time.sleep(2)
        except requests.exceptions.ConnectionError:
            print(f"  [CONN] 接続エラー → 5秒待機")
            time.sleep(5)
        except Exception as e:
            print(f"  [ERROR] {e}")
            time.sleep(2)

    return None


# ─────────────────────────────────────────
# API関数 (puuidベース)
# ─────────────────────────────────────────

def fetch_account(name: str, tag: str) -> dict | None:
    """アカウント情報を取得 → puuidを得る"""
    url = f"{BASE_URL}/valorant/v2/account/{name}/{tag}"
    result = api_get(url)
    if result:
        return result.get("data")
    return None


def fetch_account_by_puuid(puuid: str) -> dict | None:
    """puuidからアカウント情報を取得"""
    url = f"{BASE_URL}/valorant/v1/by-puuid/account/{puuid}"
    result = api_get(url)
    if result:
        return result.get("data")
    return None


def fetch_stored_matches_by_puuid(region: str, puuid: str,
                                   mode: str = "competitive") -> list | None:
    """
    puuidベースで stored-matches を全件取得
    GET /valorant/v1/by-puuid/stored-matches/{region}/{puuid}
    """
    url = f"{BASE_URL}/valorant/v1/by-puuid/stored-matches/{region}/{puuid}"
    params = {"mode": mode}
    result = api_get(url, params)
    if result:
        return result.get("data", [])
    return None


def fetch_mmr_by_puuid(region: str, puuid: str) -> dict | None:
    """puuidベースでMMR情報を取得 (v2)"""
    url = f"{BASE_URL}/valorant/v2/by-puuid/mmr/{region}/{puuid}"
    result = api_get(url)
    if result:
        return result.get("data")
    return None


def fetch_mmr_v3_by_puuid(region: str, puuid: str) -> dict | None:
    """
    v3/by-puuid/mmr でシーズン別データ・peak・seasonal を取得
    GET /valorant/v3/by-puuid/mmr/{region}/pc/{puuid}
    """
    url = f"{BASE_URL}/valorant/v3/by-puuid/mmr/{region}/pc/{puuid}"
    result = api_get(url)
    if result:
        return result.get("data")
    return None


def fetch_stored_mmr_history_by_puuid(region: str, puuid: str) -> list | None:
    """
    stored-mmr-history: 試合ごとの elo / tier / last_mmr_change を取得
    GET /valorant/v1/by-puuid/stored-mmr-history/{region}/{puuid}
    """
    url = f"{BASE_URL}/valorant/v1/by-puuid/stored-mmr-history/{region}/{puuid}"
    result = api_get(url)
    if result:
        return result.get("data", [])
    return None


def fetch_match_v4_by_puuid(region: str, puuid: str,
                            mode: str = "competitive", size: int = 3) -> list | None:
    """
    v4/by-puuid/matches で試合詳細を取得
    ability_casts, behavior, economy, party_id, rounds, kills 等
    GET /valorant/v4/by-puuid/matches/{region}/pc/{puuid}
    """
    url = f"{BASE_URL}/valorant/v4/by-puuid/matches/{region}/pc/{puuid}"
    params = {"mode": mode, "size": size}
    result = api_get(url, params)
    if result:
        return result.get("data", [])
    return None


def fetch_match_v3(region: str, name: str, tag: str,
                   mode: str = "competitive", size: int = 5) -> list:
    """v3/matches で試合詳細を取得 (プレイヤー発見用)"""
    url = f"{BASE_URL}/valorant/v3/matches/{region}/{name}/{tag}"
    params = {"mode": mode, "size": size}
    result = api_get(url, params)
    if result:
        return result.get("data", [])
    return []


# ─────────────────────────────────────────
# プレイヤー発見
# ─────────────────────────────────────────

def extract_players_from_v3(matches: list) -> list[dict]:
    """
    v3/matchesから全プレイヤーの puuid, name, tag を抽出
    """
    players = []
    seen = set()
    for match in matches:
        players_data = match.get("players") or {}
        all_p = players_data.get("all_players", [])
        if not all_p:
            reds = players_data.get("red", []) or []
            blues = players_data.get("blue", []) or []
            all_p = reds + blues

        for p in all_p:
            puuid = p.get("puuid", "")
            name = p.get("name", "")
            tag = p.get("tag", "")
            if puuid and puuid not in seen:
                seen.add(puuid)
                players.append({"puuid": puuid, "name": name, "tag": tag})
    return players


# ─────────────────────────────────────────
# 進捗管理 & 保存
# ─────────────────────────────────────────

def load_progress() -> dict:
    """中断再開用に進捗をロード"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, ValueError):
            print("[WARN] 進捗ファイルが壊れています。リセットします")
            os.remove(PROGRESS_FILE)
    return {
        "collected_puuids": [],
        "queue": [],
        "failed_puuids": [],
        "total_api_calls": 0,
    }


def save_progress(progress: dict):
    """進捗を保存"""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def save_player_data(puuid: str, data: dict):
    """1プレイヤーのデータをJSONファイルに保存 (ファイル名=puuid)"""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{puuid}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────
# 1プレイヤー処理 (ワーカー用)
# ─────────────────────────────────────────

def process_one_player(puuid: str, item: dict, region: str) -> dict | None:
    """
    1人分のデータを取得して返す (スレッドセーフ)
    成功時: {"puuid": ..., "data": ..., "name": ..., "tag": ...}
    失敗時: None
    """
    # --- stored-matches ---
    stored = fetch_stored_matches_by_puuid(region, puuid)
    if stored is None or len(stored) == 0:
        return None

    time.sleep(DELAY_BETWEEN_CALLS)

    # --- account ---
    account = fetch_account_by_puuid(puuid)
    time.sleep(DELAY_BETWEEN_CALLS)

    if account:
        real_name = account.get("name", item.get("name", ""))
        real_tag = account.get("tag", item.get("tag", ""))
    else:
        real_name = item.get("name", "")
        real_tag = item.get("tag", "")

    # --- MMR (v2) ---
    mmr = fetch_mmr_by_puuid(region, puuid)
    time.sleep(DELAY_BETWEEN_CALLS)

    # --- MMR v3 (seasonal / peak) ---
    mmr_v3 = fetch_mmr_v3_by_puuid(region, puuid)
    time.sleep(DELAY_BETWEEN_CALLS)

    # --- Stored MMR History (elo変動履歴) ---
    mmr_history = fetch_stored_mmr_history_by_puuid(region, puuid)
    time.sleep(DELAY_BETWEEN_CALLS)

    # --- v4 Match Detail (ability, economy, behavior) ---
    match_v4 = fetch_match_v4_by_puuid(region, puuid, size=3)

    player_data = {
        "puuid": puuid,
        "name": real_name,
        "tag": real_tag,
        "player": f"{real_name}#{real_tag}",
        "region": region,
        "collected_at": datetime.now().isoformat(),
        "stored_matches": stored,
        "account": account,
        "mmr": mmr,
        "mmr_v3": mmr_v3,
        "mmr_history": mmr_history,
        "match_v4": match_v4,
    }

    level = account.get("account_level", "?") if account else "?"
    rank = mmr.get("current_data", {}).get("currenttierpatched", "?") if mmr else "?"

    return {
        "puuid": puuid,
        "data": player_data,
        "name": real_name,
        "tag": real_tag,
        "stored_count": len(stored),
        "level": level,
        "rank": rank,
    }


# ─────────────────────────────────────────
# メイン収集ループ (並列版)
# ─────────────────────────────────────────

def collect(seed_name: str, seed_tag: str, seed_puuid: str,
            region: str = "ap", target: int = TARGET_PLAYERS):
    """
    BFSで stored-matches を puuidベースで大量収集 (並列版)

    流れ:
    1. シードプレイヤーの v3/matches から対戦相手の puuid を発見
    2. NUM_WORKERS 並列で by-puuid/stored-matches を取得
    3. 定期的に v3/matches でさらに新しい puuid を発見
    4. target人に達するまで繰り返す
    """
    print("=" * 60)
    print("  VALORANT データ収集器 (puuid版・並列)")
    print(f"  トークン数: {len(henriktoken)}")
    print(f"  ワーカー数: {NUM_WORKERS}")
    print(f"  リージョン: {region}")
    print(f"  目標: {target} 人")
    print(f"  エンドポイント: /v1/by-puuid/stored-matches/")
    print("=" * 60)

    # 進捗ロード
    progress = load_progress()
    collected_set = set(progress["collected_puuids"])
    failed_set = set(progress["failed_puuids"])
    lock = threading.Lock()

    # キュー復元 or 初期化
    queue = deque()
    if progress["queue"]:
        for item in progress["queue"]:
            puuid = item["puuid"] if isinstance(item, dict) else item
            if puuid not in collected_set and puuid not in failed_set:
                queue.append(item if isinstance(item, dict) else {"puuid": puuid, "name": "", "tag": ""})
        print(f"\n[RESUME] 前回の続き: 収集済み {len(collected_set)}人, キュー {len(queue)}人")
    else:
        queue.append({"puuid": seed_puuid, "name": seed_name, "tag": seed_tag})
        print(f"\n[START] シード: {seed_name}#{seed_tag} ({seed_puuid[:8]}...)")

    start_time = time.time()
    discover_counter = 0
    last_discovered_name = seed_name
    last_discovered_tag = seed_tag

    def take_batch(batch_size: int) -> list[dict]:
        """キューからバッチを取り出す (メインスレッドで呼ぶ)"""
        batch = []
        while queue and len(batch) < batch_size:
            item = queue.popleft()
            puuid = item["puuid"]
            if puuid not in collected_set and puuid not in failed_set:
                batch.append(item)
        return batch

    def do_discover(d_name: str, d_tag: str):
        """プレイヤー発見 (メインスレッドで呼ぶ)"""
        nonlocal last_discovered_name, last_discovered_tag
        if not d_name or not d_tag:
            return
        print(f"  [DISCOVER] {d_name}#{d_tag} の試合からプレイヤー探索...")
        v3_matches = fetch_match_v3(region, d_name, d_tag, size=5)
        time.sleep(DELAY_BETWEEN_CALLS)

        new_found = 0
        queue_puuids = {q["puuid"] for q in queue}

        for p in extract_players_from_v3(v3_matches):
            p_puuid = p["puuid"]
            if (p_puuid not in collected_set
                    and p_puuid not in failed_set
                    and p_puuid not in queue_puuids):
                queue.append(p)
                queue_puuids.add(p_puuid)
                new_found += 1

        if new_found > 0:
            print(f"  [DISCOVER] +{new_found}人発見 (キュー:{len(queue)})")
        last_discovered_name = d_name
        last_discovered_tag = d_tag

    # --- 初回発見 (キューが少ない場合) ---
    if len(queue) < NUM_WORKERS * 2:
        do_discover(seed_name or last_discovered_name,
                    seed_tag or last_discovered_tag)

    # --- メインループ ---
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        while len(collected_set) < target:
            # バッチ取り出し
            batch = take_batch(NUM_WORKERS)

            if not batch:
                # キューが空 → 発見を試みる
                do_discover(last_discovered_name, last_discovered_tag)
                batch = take_batch(NUM_WORKERS)
                if not batch:
                    print("\n[WARN] キューが枯渇しました。終了します。")
                    break

            elapsed = time.time() - start_time
            rate = len(collected_set) / elapsed * 60 if elapsed > 0 else 0
            print(f"\n--- バッチ {len(batch)}人投入 | "
                  f"収集済:{len(collected_set)}/{target} | "
                  f"{rate:.1f}人/分 | キュー:{len(queue)} | "
                  f"{rotator.report()} ---")

            # 並列実行
            futures = {}
            for item in batch:
                puuid = item["puuid"]
                future = executor.submit(process_one_player, puuid, item, region)
                futures[future] = item

            for future in as_completed(futures):
                item = futures[future]
                puuid = item["puuid"]
                display = f"{item.get('name', '???')}#{item.get('tag', '???')}"

                try:
                    result = future.result()
                except Exception as e:
                    print(f"  [ERROR] {display}: {e}")
                    with lock:
                        failed_set.add(puuid)
                    continue

                if result is None:
                    print(f"  [SKIP] {display}")
                    with lock:
                        failed_set.add(puuid)
                    continue

                # 保存
                save_player_data(puuid, result["data"])
                with lock:
                    collected_set.add(puuid)
                    discover_counter += 1

                print(f"  [OK] {result['name']}#{result['tag']} "
                      f"| stored:{result['stored_count']}件 "
                      f"| Lv.{result['level']} | {result['rank']} "
                      f"({len(collected_set)}/{target})")

                # 発見用の名前を更新
                if result["name"] and result["tag"]:
                    last_discovered_name = result["name"]
                    last_discovered_tag = result["tag"]

            # --- 定期的にプレイヤー発見 ---
            if discover_counter >= 5 or len(queue) < NUM_WORKERS * 3:
                do_discover(last_discovered_name, last_discovered_tag)
                discover_counter = 0

            # --- 進捗保存 (バッチごと) ---
            progress["collected_puuids"] = list(collected_set)
            progress["queue"] = [{"puuid": q["puuid"], "name": q.get("name", ""),
                                  "tag": q.get("tag", "")} for q in queue]
            progress["failed_puuids"] = list(failed_set)
            progress["total_api_calls"] = rotator.total_calls
            save_progress(progress)

    elapsed_total = time.time() - start_time
    minutes = elapsed_total / 60

    print("\n" + "=" * 60)
    print("  収集完了!")
    print(f"  収集済み: {len(collected_set)} 人")
    print(f"  失敗: {len(failed_set)} 人")
    print(f"  所要時間: {minutes:.1f} 分")
    print(f"  {rotator.report()}")
    print(f"  データ保存先: {DATA_DIR}/")
    print("=" * 60)

    return collected_set


# ─────────────────────────────────────────
# 収集データ → CSV変換
# ─────────────────────────────────────────

def export_to_csv(output_file: str = "collected_players.csv"):
    """collected_data/ のJSONを読み込んでCSVにまとめる"""
    import numpy as np
    import pandas as pd

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] {DATA_DIR}/ が見つかりません")
        return

    RANK_MAP = {
        "Iron 1": 3, "Iron 2": 4, "Iron 3": 5,
        "Bronze 1": 6, "Bronze 2": 7, "Bronze 3": 8,
        "Silver 1": 9, "Silver 2": 10, "Silver 3": 11,
        "Gold 1": 12, "Gold 2": 13, "Gold 3": 14,
        "Platinum 1": 15, "Platinum 2": 16, "Platinum 3": 17,
        "Diamond 1": 18, "Diamond 2": 19, "Diamond 3": 20,
        "Ascendant 1": 21, "Ascendant 2": 22, "Ascendant 3": 23,
        "Immortal 1": 24, "Immortal 2": 25, "Immortal 3": 26,
        "Radiant": 27,
    }

    rows = []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    print(f"[EXPORT] {len(files)} ファイルを処理中...")

    for fname in files:
        filepath = os.path.join(DATA_DIR, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        matches = data.get("stored_matches", [])
        account = data.get("account")
        mmr = data.get("mmr")
        puuid = data.get("puuid", fname.replace(".json", ""))
        name = data.get("name", "")
        tag = data.get("tag", "")

        if not matches:
            continue

        kills, deaths, assists = [], [], []
        hs_list, score_list, dmg_list, rounds_list = [], [], [], []
        wins, losses = 0, 0
        agents = set()

        for m in matches:
            stats = m.get("stats", {})
            teams = m.get("teams", {})

            k = stats.get("kills", 0)
            d = stats.get("deaths", 0)
            a = stats.get("assists", 0)
            shots = stats.get("shots", {})
            hs = shots.get("head", 0)
            bs = shots.get("body", 0)
            ls = shots.get("leg", 0)
            score = stats.get("score", 0)
            dmg = stats.get("damage", {}).get("made", 0) if isinstance(stats.get("damage"), dict) else 0
            agent = stats.get("character", {}).get("name", "") if isinstance(stats.get("character"), dict) else ""
            team_id = stats.get("team", "")

            kills.append(k)
            deaths.append(d)
            assists.append(a)
            total_shots = hs + bs + ls
            hs_list.append(hs / total_shots if total_shots else 0)
            score_list.append(score)
            dmg_list.append(dmg)
            if agent:
                agents.add(agent)

            red = teams.get("red", 0) if isinstance(teams.get("red"), (int, float)) else 0
            blue = teams.get("blue", 0) if isinstance(teams.get("blue"), (int, float)) else 0
            total_r = red + blue or 24
            rounds_list.append(total_r)

            if isinstance(teams.get("red"), int) and isinstance(teams.get("blue"), int):
                if team_id.lower() == "red":
                    if red > blue: wins += 1
                    else: losses += 1
                elif team_id.lower() == "blue":
                    if blue > red: wins += 1
                    else: losses += 1

        n = len(matches)
        avg_k = np.mean(kills)
        avg_d = np.mean(deaths)
        avg_a = np.mean(assists)
        avg_hs = np.mean(hs_list)
        avg_sc = np.mean(score_list)
        avg_dm = np.mean(dmg_list)
        avg_rnd = np.mean(rounds_list)

        kd = avg_k / avg_d if avg_d else avg_k
        kda = (avg_k + avg_a) / avg_d if avg_d else avg_k + avg_a
        wr = wins / (wins + losses) if (wins + losses) else 0

        rank_str = ""
        rank_num = 0
        if mmr:
            rank_str = mmr.get("current_data", {}).get("currenttierpatched", "")
            rank_num = RANK_MAP.get(rank_str, 0)

        acct_level = account.get("account_level", 0) if account else 0

        rows.append({
            "puuid": puuid,
            "player": f"{name}#{tag}",
            "matches_count": n,
            "avg_kills": round(avg_k, 2),
            "avg_deaths": round(avg_d, 2),
            "avg_assists": round(avg_a, 2),
            "kd_ratio": round(kd, 2),
            "kda_ratio": round(kda, 2),
            "avg_hs_pct": round(avg_hs, 4),
            "kill_per_round": round(avg_k / avg_rnd, 4),
            "dmg_per_round": round(avg_dm / avg_rnd, 2),
            "score_per_round": round(avg_sc / avg_rnd, 2),
            "win_rate": round(wr, 4),
            "kill_consistency": round(np.std(kills), 2),
            "agent_diversity": len(agents),
            "rank": rank_str,
            "rank_num": rank_num,
            "account_level": acct_level,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[EXPORT] {len(df)}人のデータを {output_file} に保存")
    return df


# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VALORANT データ収集器 (puuid版)")
    print(f"  エンドポイント: /v1/by-puuid/stored-matches/")
    print(f"  トークン: {len(henriktoken)}個")
    print("=" * 60)

    # 既存の進捗確認
    if os.path.exists(PROGRESS_FILE):
        prog = load_progress()
        n_done = len(prog.get("collected_puuids", []))
        if n_done > 0:
            print(f"\n[INFO] 前回の進捗あり ({n_done}人収集済み)")
            choice = input("続きから再開? [Y/n]: ").strip().lower()
            if choice != "n":
                region = input("リージョン [ap]: ").strip() or "ap"
                target = int(input(f"目標人数 [{TARGET_PLAYERS}]: ").strip() or str(TARGET_PLAYERS))
                collect("", "", "", region=region, target=target)
                print("\n[INFO] CSVエクスポート中...")
                export_to_csv()
                sys.exit(0)
            else:
                os.remove(PROGRESS_FILE)
                print("[INFO] 進捗をリセット")

    # 新規収集
    print("\n起点プレイヤーを入力:")
    seed_input = input("プレイヤー名#タグ (例: Player#JP1): ").strip()

    if "#" not in seed_input:
        print("[ERROR] 'name#tag' の形式で入力してください")
        sys.exit(1)

    seed_name, seed_tag = seed_input.rsplit("#", 1)

    # アカウント確認 → puuid取得
    print(f"\n[CHECK] {seed_name}#{seed_tag} のアカウント確認中...")
    account = fetch_account(seed_name, seed_tag)
    if not account:
        print("[ERROR] アカウントが見つかりません")
        sys.exit(1)

    seed_puuid = account.get("puuid", "")
    if not seed_puuid:
        print("[ERROR] puuidが取得できませんでした")
        sys.exit(1)

    print(f"  -> OK | Lv.{account.get('account_level', '?')} | puuid: {seed_puuid[:12]}...")

    region = input("\nリージョン [ap]: ").strip() or "ap"
    target = int(input(f"目標人数 [{TARGET_PLAYERS}]: ").strip() or str(TARGET_PLAYERS))

    print(f"\n{'='*60}")
    print(f"  シード: {seed_name}#{seed_tag}")
    print(f"  puuid:  {seed_puuid}")
    print(f"  目標:   {target}人")
    print(f"  API:    /v1/by-puuid/stored-matches/{region}/{{puuid}}")
    print(f"{'='*60}")

    confirm = input("\n開始? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("キャンセル")
        sys.exit(0)

    # 既存データの確認
    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        n_existing = len([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])
        if n_existing > 0:
            print(f"\n[INFO] {DATA_DIR}/ に既存データが {n_existing}件 あります")
            reset = input("リセットして最初から? [n/Y=追加]: ").strip().lower()
            if reset == "y":
                import shutil
                shutil.rmtree(DATA_DIR)
                print("[INFO] データをリセットしました")

    collect(seed_name, seed_tag, seed_puuid, region=region, target=target)

    print("\n[INFO] CSVエクスポート中...")
    export_to_csv()
