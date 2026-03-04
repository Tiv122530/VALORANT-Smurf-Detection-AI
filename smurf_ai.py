"""
VALORANT 高精度スマーフ検出AI (教師なし学習 - アンサンブル) v2
============================================================
collected_data/ の1000人分のローカルJSONデータを使用。
API呼び出し不要 - 完全ローカル実行。

データソース:
  - stored_matches: 保存された試合データ
  - account: アカウントレベル
  - mmr: 現在MMR, 最高ランク
  - mmr_v3: シーズン履歴, ピークランク, 勝率
  - mmr_history: 試合ごとのMMR推移, RR変動
  - match_v4: アビリティ使用, エコノミー, 行動, パーティ

モデル構成:
  1. Isolation Forest    - 異常値検出
  2. Local Outlier Factor - 密度ベース異常検出
  3. One-Class SVM       - カーネル空間での異常検出
  4. DBSCAN              - ノイズ点検出
  5. Gaussian Mixture     - 確率的クラスタリング
  6. ルールベースブースト - ドメイン知識による補正

特徴量 (75+):
  - 戦闘性能 (KD, KDA, HS%, ダメージ, スコア)
  - ラウンド正規化性能 (キル/R, ダメージ/R, スコア/R)
  - 安定性指標 (各種std, 変動係数)
  - ランク vs 実力の乖離
  - アカウント年齢・レベル関連
  - 勝率・連勝パターン
  - エージェント多様性
  - ティア変動パターン
  - [NEW] シーズン履歴 (プレイ期間, ランク成長)
  - [NEW] MMR推移 (RR変動, Elo速度, 連勝パターン)
  - [NEW] 行動指標 (AFK, スポーン, エコノミー効率)
  - [NEW] ソーシャル指標 (ソロキュー率, パーティ)
  - [NEW] デランク検出 (KD急落, 意図的低KDストリーク, KD二峰性)
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
DATA_DIR = Path("collected_data")
OUTPUT_DIR = Path("smurf_output")

# ティアマッピング (tier数値 → ランク名)
TIER_TO_RANK = {
    0: "Unrated", 1: "Unused", 2: "Unused",
    3: "Iron 1", 4: "Iron 2", 5: "Iron 3",
    6: "Bronze 1", 7: "Bronze 2", 8: "Bronze 3",
    9: "Silver 1", 10: "Silver 2", 11: "Silver 3",
    12: "Gold 1", 13: "Gold 2", 14: "Gold 3",
    15: "Platinum 1", 16: "Platinum 2", 17: "Platinum 3",
    18: "Diamond 1", 19: "Diamond 2", 20: "Diamond 3",
    21: "Ascendant 1", 22: "Ascendant 2", 23: "Ascendant 3",
    24: "Immortal 1", 25: "Immortal 2", 26: "Immortal 3",
    27: "Radiant",
}

RANK_PATCHED_TO_TIER = {v: k for k, v in TIER_TO_RANK.items()}


def _safe_div(a, b, default=0.0):
    """ゼロ除算安全"""
    return a / b if b and b != 0 else default


def _coeff_variation(arr):
    """変動係数"""
    if len(arr) < 2:
        return 0.0
    m = np.mean(arr)
    if m == 0:
        return 0.0
    return np.std(arr) / abs(m)


# ─────────────────────────────────────────
# 1. データ読み込み
# ─────────────────────────────────────────

def load_all_players(data_dir: Path = DATA_DIR) -> list[dict]:
    """collected_data/ の全JSONを読み込む"""
    players = []
    json_files = sorted(data_dir.glob("*.json"))
    print(f"[DATA] {len(json_files)} ファイル検出")

    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            players.append(data)
        except Exception as e:
            print(f"  [WARN] {fp.name}: {e}")

    print(f"[DATA] {len(players)} プレイヤー読み込み完了")
    return players


# ─────────────────────────────────────────
# 2. 高度な特徴量抽出 (70+ 特徴量)
# ─────────────────────────────────────────

def extract_features(player_data: dict) -> dict | None:
    """
    1人のプレイヤーデータから70+の特徴量を抽出する。
    stored_matches, mmr, mmr_v3, mmr_history, match_v4 を統合分析。
    """
    matches = player_data.get("stored_matches", [])
    account = player_data.get("account", {}) or {}
    mmr = player_data.get("mmr", {}) or {}
    current_mmr = mmr.get("current_data", {}) or {}
    highest_rank = mmr.get("highest_rank", {}) or {}

    # ──── 新データソース ────
    mmr_v3 = player_data.get("mmr_v3", {}) or {}
    mmr_history = player_data.get("mmr_history", []) or []
    match_v4_data = player_data.get("match_v4", []) or []

    # Competitiveのみ抽出
    comp_matches = [m for m in matches if m.get("meta", {}).get("mode") == "Competitive"]

    if len(comp_matches) < 3:
        return None

    # 基本情報
    puuid = player_data.get("puuid", "")
    player_name = player_data.get("player", "")
    account_level = account.get("account_level", 0) or 0
    current_tier = current_mmr.get("currenttier", 0) or 0
    current_elo = current_mmr.get("elo", 0) or 0
    highest_tier = highest_rank.get("tier", 0) or 0

    # ──────── 試合ごとの統計を蓄積 ────────
    kills_list = []
    deaths_list = []
    assists_list = []
    hs_pct_list = []
    score_list = []
    dmg_made_list = []
    dmg_received_list = []
    rounds_list = []
    kd_list = []
    kda_list = []
    kpr_list = []      # kill per round
    dpr_list = []      # damage per round
    spr_list = []      # score per round
    wins = 0
    losses = 0
    draws = 0
    agent_counter = Counter()
    tier_list = []
    level_list = []
    timestamps = []
    win_streak = 0
    max_win_streak = 0
    current_streak = 0
    match_margins = []  # 勝利マージン

    for match in comp_matches:
        stats = match.get("stats", {})
        meta = match.get("meta", {})
        teams = match.get("teams", {})

        k = stats.get("kills", 0) or 0
        d = stats.get("deaths", 0) or 0
        a = stats.get("assists", 0) or 0
        shots = stats.get("shots", {}) or {}
        head = shots.get("head", 0) or 0
        body = shots.get("body", 0) or 0
        leg = shots.get("leg", 0) or 0
        total_shots = head + body + leg
        score = stats.get("score", 0) or 0
        damage = stats.get("damage", {}) or {}
        dmg_made = damage.get("made", 0) or 0
        dmg_received = damage.get("received", 0) or 0
        agent_name = ""
        char = stats.get("character")
        if isinstance(char, dict):
            agent_name = char.get("name", "")
        tier = stats.get("tier", 0) or 0
        level = stats.get("level", 0) or 0
        team_id = (stats.get("team", "") or "").lower()

        # ラウンド数
        red_r = teams.get("red", 0) if isinstance(teams.get("red"), (int, float)) else 0
        blue_r = teams.get("blue", 0) if isinstance(teams.get("blue"), (int, float)) else 0
        total_rounds = red_r + blue_r
        if total_rounds == 0:
            total_rounds = 24

        # 勝敗判定
        my_rounds = 0
        opp_rounds = 0
        if team_id == "red":
            my_rounds = red_r
            opp_rounds = blue_r
        elif team_id == "blue":
            my_rounds = blue_r
            opp_rounds = red_r

        won = my_rounds > opp_rounds
        lost = my_rounds < opp_rounds
        drew = my_rounds == opp_rounds
        if won:
            wins += 1
            current_streak += 1
            max_win_streak = max(max_win_streak, current_streak)
        elif lost:
            losses += 1
            current_streak = 0
        else:
            draws += 1
            current_streak = 0

        match_margins.append(my_rounds - opp_rounds)

        # 蓄積
        kills_list.append(k)
        deaths_list.append(d)
        assists_list.append(a)
        hs_pct_list.append(_safe_div(head, total_shots))
        score_list.append(score)
        dmg_made_list.append(dmg_made)
        dmg_received_list.append(dmg_received)
        rounds_list.append(total_rounds)
        kd_list.append(_safe_div(k, d, default=float(k)))
        kda_list.append(_safe_div(k + a, max(d, 1)))
        kpr_list.append(_safe_div(k, total_rounds))
        dpr_list.append(_safe_div(dmg_made, total_rounds))
        spr_list.append(_safe_div(score, total_rounds))

        if agent_name:
            agent_counter[agent_name] += 1
        if tier > 0:
            tier_list.append(tier)
        if level > 0:
            level_list.append(level)

        started = meta.get("started_at")
        if started:
            try:
                ts = datetime.fromisoformat(started.replace("Z", "+00:00"))
                timestamps.append(ts)
            except:
                pass

    n = len(comp_matches)

    # ──────── 戦闘性能特徴量 ────────
    avg_kills = np.mean(kills_list)
    avg_deaths = np.mean(deaths_list)
    avg_assists = np.mean(assists_list)
    avg_hs_pct = np.mean(hs_pct_list)
    avg_score = np.mean(score_list)
    avg_dmg_made = np.mean(dmg_made_list)
    avg_dmg_received = np.mean(dmg_received_list)
    avg_rounds = np.mean(rounds_list)

    avg_kd = np.mean(kd_list)
    avg_kda = np.mean(kda_list)
    avg_kpr = np.mean(kpr_list)
    avg_dpr = np.mean(dpr_list)
    avg_spr = np.mean(spr_list)

    # 中央値KD (外れ値に強い)
    median_kd = np.median(kd_list)
    median_kpr = np.median(kpr_list)

    # ──────── 安定性・一貫性特徴量 ────────
    # スマーフは安定して高い → 低い変動係数
    kd_std = np.std(kd_list) if len(kd_list) > 1 else 0
    kd_cv = _coeff_variation(kd_list)
    kills_std = np.std(kills_list) if len(kills_list) > 1 else 0
    kills_cv = _coeff_variation(kills_list)
    hs_std = np.std(hs_pct_list) if len(hs_pct_list) > 1 else 0
    hs_cv = _coeff_variation(hs_pct_list)
    dpr_std = np.std(dpr_list) if len(dpr_list) > 1 else 0
    dpr_cv = _coeff_variation(dpr_list)
    score_cv = _coeff_variation(score_list)

    # ──────── 勝率関連 ────────
    total_games = wins + losses + draws
    win_rate = _safe_div(wins, wins + losses)
    avg_margin = np.mean(match_margins) if match_margins else 0

    # ──────── エージェント関連 ────────
    agent_diversity = len(agent_counter)
    # 最多エージェント使用率 (スマーフは1-2エージェントに偏る)
    top_agent_ratio = 0.0
    if agent_counter and n > 0:
        top_agent_ratio = agent_counter.most_common(1)[0][1] / n

    # ──────── ランク乖離特徴量 (最重要) ────────
    # 現在ランク vs 最高ランク の差
    rank_gap = highest_tier - current_tier if highest_tier > 0 and current_tier > 0 else 0

    # ティアの変動ぶり (試合中の tier フィールド)
    tier_range = 0
    tier_trend = 0.0  # 正=上昇トレンド
    avg_match_tier = 0
    if tier_list:
        tier_range = max(tier_list) - min(tier_list)
        avg_match_tier = np.mean(tier_list)
        if len(tier_list) >= 2:
            # 線形回帰の傾きでトレンドを算出
            x = np.arange(len(tier_list))
            tier_trend = np.polyfit(x, tier_list, 1)[0]

    # ──────── 性能/ランク乖離スコア (スマーフの最大シグナル) ────────
    # 低ランクなのに高パフォーマンス = スマーフ
    # tier基準: performance_rank_ratio
    effective_tier = current_tier if current_tier > 0 else (avg_match_tier if avg_match_tier > 0 else 10)
    perf_rank_ratio_kd = _safe_div(avg_kd, effective_tier / 15.0)  # 15 = Plat1 を基準
    perf_rank_ratio_hs = _safe_div(avg_hs_pct, effective_tier / 15.0)
    perf_rank_ratio_dpr = _safe_div(avg_dpr, effective_tier / 15.0)

    # ──────── アカウント年齢関連 ────────
    # レベルが低いのに高パフォ = スマーフ
    level_to_use = account_level if account_level > 0 else (max(level_list) if level_list else 100)
    perf_level_ratio = _safe_div(avg_kd, max(level_to_use / 200.0, 0.1))

    # 最近の試合の活動頻度
    match_frequency = 0.0  # 1日あたりの試合数
    activity_span_days = 0.0
    if len(timestamps) >= 2:
        timestamps_sorted = sorted(timestamps)
        span = (timestamps_sorted[-1] - timestamps_sorted[0]).total_seconds()
        activity_span_days = span / 86400.0
        if activity_span_days > 0:
            match_frequency = n / activity_span_days

    # ──────── レベル変動 ────────
    level_range = 0
    if level_list:
        level_range = max(level_list) - min(level_list)

    # ──────── ダメージ効率 ────────
    dmg_efficiency = _safe_div(avg_dmg_made, avg_dmg_made + avg_dmg_received)

    # ──────── 上位パフォーマンス率 ────────
    # KD > 1.5 の試合の割合
    high_perf_rate = sum(1 for kd in kd_list if kd >= 1.5) / n if n > 0 else 0
    # KD > 2.0 の試合の割合
    dominant_rate = sum(1 for kd in kd_list if kd >= 2.0) / n if n > 0 else 0

    # ──────── FIRST BLOOD的指標 (キル-デス差の安定性) ────────
    kd_diff_list = [k - d for k, d in zip(kills_list, deaths_list)]
    avg_kd_diff = np.mean(kd_diff_list)
    kd_diff_positive_rate = sum(1 for x in kd_diff_list if x > 0) / n if n > 0 else 0

    # ──────── デランク・意図的負け検出 ────────
    # KD < 0.3 の割合 (意図的負け / 捨て試合)
    intentional_loss_rate = sum(1 for kd in kd_list if kd < 0.3) / n if n > 0 else 0

    # 連続低KDストリーク最大値 (kd < 0.4 が続く = タンク期間)
    _tank_streak = 0
    tank_streak_max = 0
    for _kd in kd_list:
        if _kd < 0.4:
            _tank_streak += 1
            tank_streak_max = max(tank_streak_max, _tank_streak)
        else:
            _tank_streak = 0

    # 高KD → 急落 の回数 (1.5以上 → 次の試合0.5以下)
    kd_cliff_drops = sum(
        1 for i in range(1, len(kd_list))
        if kd_list[i - 1] >= 1.5 and kd_list[i] <= 0.5
    )
    kd_cliff_drop_rate = kd_cliff_drops / (n - 1) if n > 1 else 0

    # KD二峰性スコア: 高KD率 × 低KD率
    # 高いと「圧倒的な試合」と「捨て試合」が混在 = デランクスマーフの典型
    _high_kd_rate = sum(1 for kd in kd_list if kd >= 1.5) / n if n > 0 else 0
    _very_low_kd_rate = sum(1 for kd in kd_list if kd < 0.4) / n if n > 0 else 0
    kd_bimodal_score = _high_kd_rate * _very_low_kd_rate

    # ═══════════════════════════════════════════════════════════
    # 新データソースからの特徴量 (mmr_v3, mmr_history, match_v4)
    # ═══════════════════════════════════════════════════════════

    # ──────── mmr_v3: シーズン履歴・ピークランク ────────
    v3_peak = mmr_v3.get("peak", {}) or {}
    v3_current = mmr_v3.get("current", {}) or {}
    v3_seasonal = mmr_v3.get("seasonal", []) or []

    # ピークティア (v3はより正確)
    peak_tier_v3 = 0
    peak_tier_obj = v3_peak.get("tier", {}) or {}
    if isinstance(peak_tier_obj, dict):
        peak_tier_v3 = peak_tier_obj.get("id", 0) or 0
    peak_rr = v3_peak.get("rr", 0) or 0

    # 現在のv3データ
    v3_cur_tier_obj = v3_current.get("tier", {}) or {}
    v3_cur_tier = 0
    if isinstance(v3_cur_tier_obj, dict):
        v3_cur_tier = v3_cur_tier_obj.get("id", 0) or 0
    v3_cur_elo = v3_current.get("elo", 0) or 0
    v3_last_change = v3_current.get("last_change", 0) or 0

    # シーズンデータ解析
    seasons_played = len(v3_seasonal)
    seasonal_tiers = []
    seasonal_winrates = []
    seasonal_games_list = []
    total_act_wins = 0

    for season in v3_seasonal:
        end_tier_obj = season.get("end_tier", {}) or {}
        end_tier_id = 0
        if isinstance(end_tier_obj, dict):
            end_tier_id = end_tier_obj.get("id", 0) or 0
        if end_tier_id > 0:
            seasonal_tiers.append(end_tier_id)

        s_wins = season.get("wins", 0) or 0
        s_games = season.get("games", 0) or 0
        seasonal_games_list.append(s_games)
        if s_games > 0:
            seasonal_winrates.append(s_wins / s_games)

        act_wins = season.get("act_wins", []) or []
        total_act_wins += len(act_wins)

    # シーズン平均勝率
    seasonal_avg_winrate = np.mean(seasonal_winrates) if seasonal_winrates else 0.0

    # ランク成長速度 (最初のシーズン → 最後のシーズン)
    seasonal_rank_growth = 0.0
    first_season_tier = 0
    latest_season_tier = 0
    if len(seasonal_tiers) >= 2:
        first_season_tier = seasonal_tiers[-1]  # リスト末尾が最も古い
        latest_season_tier = seasonal_tiers[0]  # リスト先頭が最新
        seasonal_rank_growth = (latest_season_tier - first_season_tier) / len(seasonal_tiers)
    elif len(seasonal_tiers) == 1:
        first_season_tier = seasonal_tiers[0]
        latest_season_tier = seasonal_tiers[0]

    # ピークと現在の乖離 (v3版、より正確)
    peak_current_gap_v3 = peak_tier_v3 - v3_cur_tier if peak_tier_v3 > 0 and v3_cur_tier > 0 else 0

    # シーズンあたり平均試合数
    avg_games_per_season = np.mean(seasonal_games_list) if seasonal_games_list else 0.0

    # ──────── mmr_history: 試合ごとのMMR推移 ────────
    mmr_hist_count = len(mmr_history)
    rr_changes = []
    elo_values = []

    for entry in mmr_history:
        rr_change = entry.get("last_mmr_change", 0) or 0
        elo_val = entry.get("elo", 0) or 0
        rr_changes.append(rr_change)
        if elo_val > 0:
            elo_values.append(elo_val)

    avg_rr_change = np.mean(rr_changes) if rr_changes else 0.0
    rr_change_std = np.std(rr_changes) if len(rr_changes) > 1 else 0.0
    positive_rr_rate = sum(1 for r in rr_changes if r > 0) / len(rr_changes) if rr_changes else 0.0
    max_rr_gain = max(rr_changes) if rr_changes else 0
    max_rr_loss = min(rr_changes) if rr_changes else 0

    # Elo推移分析
    avg_elo = np.mean(elo_values) if elo_values else 0.0
    elo_range = (max(elo_values) - min(elo_values)) if len(elo_values) >= 2 else 0
    elo_velocity = 0.0
    if len(elo_values) >= 3:
        x = np.arange(len(elo_values))
        elo_velocity = np.polyfit(x, elo_values, 1)[0]  # 正=上昇中

    # 連続RR獲得パターン (スマーフは連勝しやすい)
    rr_streak = 0
    max_rr_streak = 0
    for r in rr_changes:
        if r > 0:
            rr_streak += 1
            max_rr_streak = max(max_rr_streak, rr_streak)
        else:
            rr_streak = 0

    # ──────── match_v4: 詳細試合データ ────────
    ability_totals = []
    ultimate_casts_list = []
    afk_rounds_list = []
    rounds_in_spawn_list = []
    economy_spent_list = []
    loadout_value_list = []
    party_ids = set()
    session_playtimes = []
    ff_incoming_list = []
    ff_outgoing_list = []
    v4_comp_count = 0

    for match in match_v4_data:
        if not isinstance(match, dict):
            continue

        meta = match.get("metadata", {}) or {}
        if meta.get("queue", {}) and isinstance(meta.get("queue"), dict):
            queue_id = meta["queue"].get("id", "")
            if queue_id and queue_id != "competitive":
                continue

        players_list = match.get("players", []) or []
        puuid_target = puuid

        for p in players_list:
            if not isinstance(p, dict):
                continue
            if p.get("puuid") != puuid_target:
                continue

            v4_comp_count += 1

            # アビリティ使用
            ability = p.get("ability_casts", {}) or {}
            grenade = ability.get("grenade", 0) or 0
            ab1 = ability.get("ability1", 0) or 0
            ab2 = ability.get("ability2", 0) or 0
            ult = ability.get("ultimate", 0) or 0
            ability_totals.append(grenade + ab1 + ab2 + ult)
            ultimate_casts_list.append(ult)

            # 行動指標
            behavior = p.get("behavior", {}) or {}
            afk_r = behavior.get("afk_rounds", 0) or 0
            afk_rounds_list.append(afk_r)
            spawn_r = behavior.get("rounds_in_spawn", 0) or 0
            rounds_in_spawn_list.append(spawn_r)
            ff = behavior.get("friendly_fire", {}) or {}
            ff_incoming_list.append(ff.get("incoming", 0) or 0)
            ff_outgoing_list.append(ff.get("outgoing", 0) or 0)

            # エコノミー
            economy = p.get("economy", {}) or {}
            spent = economy.get("spent", {}) or {}
            loadout = economy.get("loadout_value", {}) or {}
            economy_spent_list.append(spent.get("average", 0) or 0)
            loadout_value_list.append(loadout.get("average", 0) or 0)

            # パーティ
            party = p.get("party_id", "")
            if party:
                party_ids.add(party)

            # セッション
            session_ms = p.get("session_playtime_in_ms", 0) or 0
            if session_ms > 0:
                session_playtimes.append(session_ms / 1000.0 / 60.0)  # 分に変換

            break  # 自分のデータを見つけた

    # match_v4 集計
    avg_ability_total = np.mean(ability_totals) if ability_totals else 0.0
    avg_ultimate_casts = np.mean(ultimate_casts_list) if ultimate_casts_list else 0.0
    avg_afk_rounds = np.mean(afk_rounds_list) if afk_rounds_list else 0.0
    avg_rounds_in_spawn = np.mean(rounds_in_spawn_list) if rounds_in_spawn_list else 0.0
    avg_economy_spent = np.mean(economy_spent_list) if economy_spent_list else 0.0
    avg_loadout_value = np.mean(loadout_value_list) if loadout_value_list else 0.0
    party_count = len(party_ids)
    solo_queue_rate = 0.0
    if v4_comp_count > 0 and party_count > 0:
        # 各パーティIDに所属 → ユニークパーティ数/試合数 ≈ ソロ率の近似
        solo_queue_rate = min(party_count / v4_comp_count, 1.0)
    avg_session_playtime = np.mean(session_playtimes) if session_playtimes else 0.0
    avg_ff_outgoing = np.mean(ff_outgoing_list) if ff_outgoing_list else 0.0

    # エコノミー効率 = パフォーマンス / 投資 (KD per credit)
    economy_efficiency = 0.0
    if avg_economy_spent > 0 and avg_kd > 0:
        economy_efficiency = avg_kd / (avg_economy_spent / 4000.0)  # 4000クレジットを基準

    # ──────── 特徴量辞書 ────────
    features = {
        # ID
        "puuid": puuid,
        "player": player_name,

        # 基本情報
        "account_level": level_to_use,
        "current_tier": current_tier,
        "current_elo": current_elo,
        "highest_tier": highest_tier,
        "matches_count": n,

        # === 戦闘性能 (平均) ===
        "avg_kills": round(avg_kills, 3),
        "avg_deaths": round(avg_deaths, 3),
        "avg_assists": round(avg_assists, 3),
        "avg_kd": round(avg_kd, 4),
        "avg_kda": round(avg_kda, 4),
        "avg_hs_pct": round(avg_hs_pct, 5),
        "avg_dmg_made": round(avg_dmg_made, 2),
        "avg_dpr": round(avg_dpr, 3),
        "avg_spr": round(avg_spr, 3),
        "avg_kpr": round(avg_kpr, 5),

        # === 中央値 (外れ値耐性) ===
        "median_kd": round(median_kd, 4),
        "median_kpr": round(median_kpr, 5),

        # === 安定性・一貫性 ===
        "kd_std": round(kd_std, 4),
        "kd_cv": round(kd_cv, 4),
        "kills_std": round(kills_std, 3),
        "kills_cv": round(kills_cv, 4),
        "hs_std": round(hs_std, 5),
        "hs_cv": round(hs_cv, 4),
        "dpr_std": round(dpr_std, 3),
        "dpr_cv": round(dpr_cv, 4),
        "score_cv": round(score_cv, 4),

        # === 勝利関連 ===
        "win_rate": round(win_rate, 4),
        "max_win_streak": max_win_streak,
        "avg_margin": round(avg_margin, 3),

        # === エージェント ===
        "agent_diversity": agent_diversity,
        "top_agent_ratio": round(top_agent_ratio, 4),

        # === ランク乖離 (最重要) ===
        "rank_gap": rank_gap,
        "tier_range": tier_range,
        "tier_trend": round(tier_trend, 5),

        # === 性能/ランク乖離 ===
        "perf_rank_ratio_kd": round(perf_rank_ratio_kd, 4),
        "perf_rank_ratio_hs": round(perf_rank_ratio_hs, 5),
        "perf_rank_ratio_dpr": round(perf_rank_ratio_dpr, 4),
        "perf_level_ratio": round(perf_level_ratio, 4),

        # === アカウント活動 ===
        "match_frequency": round(match_frequency, 4),
        "activity_span_days": round(activity_span_days, 2),
        "level_range": level_range,

        # === ダメージ効率 ===
        "dmg_efficiency": round(dmg_efficiency, 5),

        # === 支配力 ===
        "high_perf_rate": round(high_perf_rate, 4),
        "dominant_rate": round(dominant_rate, 4),
        "avg_kd_diff": round(avg_kd_diff, 3),
        "kd_diff_positive_rate": round(kd_diff_positive_rate, 4),

        # === デランク・意図的負け検出 ===
        "intentional_loss_rate": round(intentional_loss_rate, 4),
        "tank_streak_max": tank_streak_max,
        "kd_cliff_drop_rate": round(kd_cliff_drop_rate, 4),
        "kd_bimodal_score": round(kd_bimodal_score, 4),

        # === mmr_v3: シーズン・ピーク ===
        "peak_tier_v3": peak_tier_v3,
        "peak_rr": peak_rr,
        "peak_current_gap_v3": peak_current_gap_v3,
        "seasons_played": seasons_played,
        "first_season_tier": first_season_tier,
        "latest_season_tier": latest_season_tier,
        "seasonal_rank_growth": round(seasonal_rank_growth, 4),
        "seasonal_avg_winrate": round(seasonal_avg_winrate, 4),
        "total_act_wins": total_act_wins,
        "avg_games_per_season": round(avg_games_per_season, 2),

        # === mmr_history: MMR推移 ===
        "mmr_hist_count": mmr_hist_count,
        "avg_rr_change": round(avg_rr_change, 3),
        "rr_change_std": round(rr_change_std, 3),
        "positive_rr_rate": round(positive_rr_rate, 4),
        "max_rr_gain": max_rr_gain,
        "elo_range": elo_range,
        "elo_velocity": round(elo_velocity, 4),
        "max_rr_streak": max_rr_streak,

        # === match_v4: 詳細マッチ ===
        "avg_ability_total": round(avg_ability_total, 2),
        "avg_ultimate_casts": round(avg_ultimate_casts, 3),
        "avg_afk_rounds": round(avg_afk_rounds, 3),
        "avg_rounds_in_spawn": round(avg_rounds_in_spawn, 3),
        "avg_economy_spent": round(avg_economy_spent, 2),
        "avg_loadout_value": round(avg_loadout_value, 2),
        "economy_efficiency": round(economy_efficiency, 4),
        "party_count": party_count,
        "solo_queue_rate": round(solo_queue_rate, 4),
        "avg_session_playtime": round(avg_session_playtime, 2),
        "avg_ff_outgoing": round(avg_ff_outgoing, 3),
    }

    return features


# ─────────────────────────────────────────
# 3. アンサンブル教師なし学習モデル
# ─────────────────────────────────────────

# モデルに使用する特徴量カラム (ID・メタデータ以外)
FEATURE_COLS = [
    # 戦闘性能
    "avg_kills", "avg_deaths", "avg_assists",
    "avg_kd", "avg_kda", "avg_hs_pct",
    "avg_dmg_made", "avg_dpr", "avg_spr", "avg_kpr",
    "median_kd", "median_kpr",
    # 安定性
    "kd_std", "kd_cv", "kills_std", "kills_cv",
    "hs_std", "hs_cv", "dpr_std", "dpr_cv", "score_cv",
    # 勝利
    "win_rate", "max_win_streak", "avg_margin",
    # エージェント
    "agent_diversity", "top_agent_ratio",
    # ランク乖離
    "rank_gap", "tier_range", "tier_trend",
    "perf_rank_ratio_kd", "perf_rank_ratio_hs",
    "perf_rank_ratio_dpr", "perf_level_ratio",
    # アカウント
    "account_level", "current_tier", "match_frequency",
    "activity_span_days", "level_range",
    # ダメージ効率
    "dmg_efficiency",
    # 支配力
    "high_perf_rate", "dominant_rate",
    "avg_kd_diff", "kd_diff_positive_rate",
    # === 新特徴量: デランク・意図的負け検出 ===
    "intentional_loss_rate", "tank_streak_max",
    "kd_cliff_drop_rate", "kd_bimodal_score",
    # === 新特徴量: mmr_v3 シーズン・ピーク ===
    "peak_tier_v3", "peak_current_gap_v3",
    "seasons_played", "first_season_tier", "latest_season_tier",
    "seasonal_rank_growth", "seasonal_avg_winrate",
    "total_act_wins", "avg_games_per_season",
    # === 新特徴量: mmr_history MMR推移 ===
    "mmr_hist_count", "avg_rr_change", "rr_change_std",
    "positive_rr_rate", "max_rr_gain", "elo_range",
    "elo_velocity", "max_rr_streak",
    # === 新特徴量: match_v4 詳細マッチ ===
    "avg_ability_total", "avg_ultimate_casts",
    "avg_afk_rounds", "avg_rounds_in_spawn",
    "avg_economy_spent", "avg_loadout_value", "economy_efficiency",
    "party_count", "solo_queue_rate",
    "avg_session_playtime", "avg_ff_outgoing",
]


class EnsembleSmurfDetector:
    """
    複数の教師なし学習モデルをアンサンブルしてスマーフを高精度検出。

    モデル:
      1. Isolation Forest (異常値検出)
      2. Local Outlier Factor (密度ベース)
      3. One-Class SVM (カーネル空間)
      4. DBSCAN (密度クラスタリング)
      5. Gaussian Mixture (確率的)
      6. K-Means (スキルクラスタ分類)

    各モデルのスコアを重み付き平均で統合。
    """

    def __init__(self, contamination: float = 0.12):
        self.contamination = contamination
        self.scaler = RobustScaler()  # 外れ値に強い
        self.standard_scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.is_fitted = False

        # モデル群
        self.iso_forest = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            max_samples=0.8,
            max_features=0.8,
            random_state=42,
            bootstrap=True,
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=False,
            metric="euclidean",
        )
        self.ocsvm = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=contamination,
        )
        self.gmm = GaussianMixture(
            n_components=5,
            covariance_type="full",
            random_state=42,
            max_iter=500,
        )
        self.kmeans = KMeans(
            n_clusters=6,
            random_state=42,
            n_init=20,
            max_iter=500,
        )

        # モデル重み (検出力の高い順、ルールベースは新データで強化)
        self.weights = {
            "isolation_forest": 0.28,
            "lof": 0.22,
            "ocsvm": 0.18,
            "gmm": 0.14,
            "rule_based": 0.18,
        }

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """スコアを0-100に正規化 (高い = スマーフ疑い)"""
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-10:
            return np.full_like(scores, 50.0)
        return (scores - mn) / (mx - mn) * 100.0

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """全モデルで学習→予測→アンサンブル"""
        result = df.copy()
        X_raw = df[FEATURE_COLS].values.astype(float)

        # 欠損値を中央値で補完
        col_medians = np.nanmedian(X_raw, axis=0)
        for i in range(X_raw.shape[1]):
            mask = np.isnan(X_raw[:, i]) | np.isinf(X_raw[:, i])
            X_raw[mask, i] = col_medians[i]

        X_scaled = self.scaler.fit_transform(X_raw)
        X_standard = self.standard_scaler.fit_transform(X_raw)
        self.pca.fit(X_standard)

        n_samples = len(df)
        print(f"[MODEL] {n_samples} プレイヤー × {len(FEATURE_COLS)} 特徴量")
        print(f"[MODEL] PCA 寄与率: PC1={self.pca.explained_variance_ratio_[0]:.3f}, PC2={self.pca.explained_variance_ratio_[1]:.3f}")

        # ───── 1. Isolation Forest ─────
        print("[MODEL] 1/5 Isolation Forest 学習中...")
        self.iso_forest.fit(X_scaled)
        iso_raw = -self.iso_forest.decision_function(X_scaled)  # 高い=異常
        iso_scores = self._normalize_scores(iso_raw)
        result["iso_score"] = np.round(iso_scores, 2)

        # ───── 2. Local Outlier Factor ─────
        print("[MODEL] 2/5 Local Outlier Factor 学習中...")
        lof_labels = self.lof.fit_predict(X_scaled)
        lof_raw = -self.lof.negative_outlier_factor_  # 高い=異常
        lof_scores = self._normalize_scores(lof_raw)
        result["lof_score"] = np.round(lof_scores, 2)

        # ───── 3. One-Class SVM ─────
        print("[MODEL] 3/5 One-Class SVM 学習中...")
        self.ocsvm.fit(X_scaled)
        ocsvm_raw = -self.ocsvm.decision_function(X_scaled)
        ocsvm_scores = self._normalize_scores(ocsvm_raw)
        result["ocsvm_score"] = np.round(ocsvm_scores, 2)

        # ───── 4. Gaussian Mixture Model ─────
        print("[MODEL] 4/5 Gaussian Mixture Model 学習中...")
        self.gmm.fit(X_scaled)
        gmm_log_probs = self.gmm.score_samples(X_scaled)
        gmm_raw = -gmm_log_probs  # 低い確率 = 異常
        gmm_scores = self._normalize_scores(gmm_raw)
        result["gmm_score"] = np.round(gmm_scores, 2)

        # ───── 5. K-Means スキルクラスタ ─────
        print("[MODEL] 5/5 K-Means クラスタリング学習中...")
        self.kmeans.fit(X_scaled)
        clusters = self.kmeans.predict(X_scaled)
        result["cluster"] = clusters

        # クラスタ中心からの距離 (遠い = 異常)
        distances = self.kmeans.transform(X_scaled)
        min_distances = distances.min(axis=1)

        # Silhouette score
        if n_samples > 5:
            sil = silhouette_score(X_scaled, clusters, sample_size=min(1000, n_samples))
            print(f"[MODEL] K-Means Silhouette Score: {sil:.4f}")

        # クラスタ別平均KDでスキルレベルラベル付与
        cluster_kd = result.groupby("cluster")["avg_kd"].mean().sort_values()
        n_clusters = len(cluster_kd)
        skill_labels_map = {}
        label_names = ["最低スキル帯", "低スキル帯", "中低スキル帯", "中スキル帯", "高スキル帯", "最高スキル帯"]
        for i, c in enumerate(cluster_kd.index):
            idx = min(i, len(label_names) - 1)
            skill_labels_map[c] = label_names[int(round(i / max(n_clusters - 1, 1) * (len(label_names) - 1)))]
        result["skill_cluster"] = result["cluster"].map(skill_labels_map)

        # ───── 6. ルールベーススコア (ドメイン知識) ─────
        print("[MODEL] ルールベーススコア計算中...")
        rule_scores = self._calc_rule_scores(df)
        result["rule_score"] = np.round(rule_scores, 2)

        # ───── アンサンブル統合 ─────
        print("[MODEL] アンサンブル統合中...")
        ensemble_scores = (
            self.weights["isolation_forest"] * iso_scores +
            self.weights["lof"] * lof_scores +
            self.weights["ocsvm"] * ocsvm_scores +
            self.weights["gmm"] * gmm_scores +
            self.weights["rule_based"] * rule_scores
        )

        # アンサンブルスコアを0-100に再正規化
        final_scores = self._normalize_scores(ensemble_scores)
        result["smurf_score"] = np.round(final_scores, 1)

        # ───── DBSCAN ノイズ検出でブースト ─────
        dbscan = DBSCAN(eps=1.5, min_samples=5)
        db_labels = dbscan.fit_predict(X_scaled)
        noise_mask = db_labels == -1
        # DBSCAN でノイズ判定 → スコア+5
        result.loc[noise_mask, "smurf_score"] = np.minimum(
            result.loc[noise_mask, "smurf_score"] + 5.0, 100.0
        )
        result["is_dbscan_noise"] = noise_mask.astype(int)

        # ───── 最終判定 ─────
        result["judgment"] = result["smurf_score"].apply(self._judge)
        result["confidence"] = result.apply(self._calc_confidence, axis=1)

        # ───── ランク名追加 ─────
        result["current_rank"] = result["current_tier"].map(
            lambda t: TIER_TO_RANK.get(int(t), "Unknown") if pd.notna(t) else "Unknown"
        )
        result["highest_rank"] = result["highest_tier"].map(
            lambda t: TIER_TO_RANK.get(int(t), "Unknown") if pd.notna(t) else "Unknown"
        )

        self.is_fitted = True
        return result

    def _calc_rule_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        ドメイン知識に基づくルールベーススマーフスコア (0-100)。
        スマーフの典型的パターンを数値化。
        新データソース(mmr_v3, mmr_history, match_v4)を活用した拡張版。
        """
        scores = np.zeros(len(df))

        for i, (_, row) in enumerate(df.iterrows()):
            s = 0.0
            lvl = row.get("account_level", 200)
            kd = row.get("avg_kd", 1.0)
            hs = row.get("avg_hs_pct", 0.15)
            tier = row.get("current_tier", 10)
            highest = row.get("highest_tier", 0)
            win_rate = row.get("win_rate", 0.5)
            kpr = row.get("avg_kpr", 0.7)
            dpr = row.get("avg_dpr", 130)
            dominant = row.get("dominant_rate", 0)
            gap = row.get("rank_gap", 0)
            perf_rank_kd = row.get("perf_rank_ratio_kd", 1.0)
            perf_lvl = row.get("perf_level_ratio", 1.0)
            agent_div = row.get("agent_diversity", 5)
            top_agent = row.get("top_agent_ratio", 0.3)
            kd_cv = row.get("kd_cv", 0.5)
            tier_trend = row.get("tier_trend", 0)

            # ── 新データ変数 ──
            seasons = row.get("seasons_played", 0)
            peak_v3 = row.get("peak_tier_v3", 0)
            peak_gap_v3 = row.get("peak_current_gap_v3", 0)
            rank_growth = row.get("seasonal_rank_growth", 0)
            seasonal_wr = row.get("seasonal_avg_winrate", 0.5)
            act_wins = row.get("total_act_wins", 0)
            avg_rr = row.get("avg_rr_change", 0)
            pos_rr_rate = row.get("positive_rr_rate", 0.5)
            elo_vel = row.get("elo_velocity", 0)
            rr_streak = row.get("max_rr_streak", 0)
            afk_rounds = row.get("avg_afk_rounds", 0)
            spawn_rounds = row.get("avg_rounds_in_spawn", 0)
            econ_eff = row.get("economy_efficiency", 1.0)
            solo_rate = row.get("solo_queue_rate", 0.5)
            avg_games_season = row.get("avg_games_per_season", 50)
            first_tier = row.get("first_season_tier", 0)

            # --- 低レベル + 高パフォーマンス ---
            if lvl < 30:
                if kd > 1.8: s += 25
                elif kd > 1.3: s += 15
                if hs > 0.25: s += 15
                if win_rate > 0.65: s += 10
            elif lvl < 60:
                if kd > 1.8: s += 15
                elif kd > 1.3: s += 8
                if hs > 0.25: s += 8
                if win_rate > 0.65: s += 5
            elif lvl < 100:
                if kd > 2.0: s += 10
                if hs > 0.28: s += 5

            # --- 現ランク vs 最高ランク ---
            if gap >= 8: s += 20
            elif gap >= 5: s += 12
            elif gap >= 3: s += 5

            # --- 性能/ランク乖離 ---
            if perf_rank_kd > 2.0: s += 15
            elif perf_rank_kd > 1.5: s += 8

            # --- 性能/レベル乖離 ---
            if perf_lvl > 3.0: s += 15
            elif perf_lvl > 2.0: s += 8

            # --- 支配率 ---
            if dominant > 0.4: s += 10
            elif dominant > 0.25: s += 5

            # --- エージェント偏り (スマーフは少数エージェント) ---
            if agent_div <= 2 and lvl < 50: s += 8
            if top_agent > 0.7: s += 5

            # --- KD安定性 (スマーフは安定して高い) ---
            if kd > 1.5 and kd_cv < 0.3: s += 8
            elif kd > 1.3 and kd_cv < 0.25: s += 5

            # --- 急上昇トレンド ---
            if tier_trend > 0.05: s += 8
            elif tier_trend > 0.02: s += 3

            # ═══════════════════════════════════════
            # 新ルール: mmr_v3 シーズンデータ
            # ═══════════════════════════════════════

            # --- 少シーズンなのに高パフォ (新アカウント) ---
            if seasons <= 1 and kd > 1.5: s += 15
            elif seasons <= 2 and kd > 1.5: s += 10
            elif seasons <= 3 and kd > 1.3: s += 5

            # --- ピーク vs 現在の乖離 (v3版、より正確) ---
            if peak_gap_v3 >= 8: s += 15
            elif peak_gap_v3 >= 5: s += 8
            elif peak_gap_v3 >= 3: s += 3

            # --- 急激なシーズンランク成長 ---
            if rank_growth > 2.0: s += 12  # 1シーズンで2ティア以上
            elif rank_growth > 1.0: s += 6
            elif rank_growth > 0.5: s += 3

            # --- 初シーズンからいきなり高ランク ---
            if first_tier >= 15 and seasons <= 2: s += 15  # Plat+で開始
            elif first_tier >= 12 and seasons <= 2: s += 8  # Gold+

            # --- アクトウィン数少 = プレイ歴短い (スマーフ兆候) ---
            if act_wins < 10 and kd > 1.5: s += 5

            # ═══════════════════════════════════════
            # 新ルール: mmr_history MMR推移
            # ═══════════════════════════════════════

            # --- 異常に高い平均RR獲得 (上位帯に急上昇中) ---
            if avg_rr > 18: s += 12  # ほぼ毎試合大勝利
            elif avg_rr > 12: s += 6
            elif avg_rr > 8: s += 3

            # --- RR獲得率が高い (勝ちまくっている) ---
            if pos_rr_rate > 0.75: s += 10
            elif pos_rr_rate > 0.65: s += 5

            # --- Elo急上昇 ---
            if elo_vel > 5.0: s += 10  # 試合ごとにEloが5以上増加
            elif elo_vel > 2.0: s += 5

            # --- 長い連勝RRストリーク ---
            if rr_streak >= 8: s += 10
            elif rr_streak >= 5: s += 5
            elif rr_streak >= 3: s += 2

            # ═══════════════════════════════════════
            # 新ルール: match_v4 行動・エコノミー
            # ═══════════════════════════════════════

            # --- 高エコノミー効率 (少ない経済で高KD = スキルフル) ---
            if econ_eff > 2.0 and kd > 1.5: s += 8
            elif econ_eff > 1.5 and kd > 1.3: s += 4

            # --- AFK・スポーンが少ない = 真剣にプレイ ---
            # (スマーフはAFKしない or 逆に舐めプでAFKする → 参考程度)
            if afk_rounds > 2: s -= 3  # AFK多い = 故意の妨害/通常プレイヤー

            # --- ソロキュー率が高い (スマーフはソロで暴れがち) ---
            if solo_rate > 0.8 and kd > 1.5: s += 5
            elif solo_rate > 0.8 and kd > 1.3: s += 3

            # ═══════════════════════════════════════
            # デランク・意図的負けルール
            # ═══════════════════════════════════════

            bimodal = row.get("kd_bimodal_score", 0)
            tank_streak = row.get("tank_streak_max", 0)
            cliff_rate  = row.get("kd_cliff_drop_rate", 0)
            inten_loss  = row.get("intentional_loss_rate", 0)

            # --- KD二峰性: 圧倒的試合と捨て試合が混在 ---
            # 例: 40%がKD1.5+、30%がKD0.4未満 → bimodal = 0.12
            if bimodal > 0.12: s += 20
            elif bimodal > 0.08: s += 12
            elif bimodal > 0.05: s += 6

            # --- 連続低KDストリーク (タンク期間の証拠) ---
            if tank_streak >= 5: s += 18
            elif tank_streak >= 3: s += 10
            elif tank_streak >= 2: s += 4

            # --- 高KD直後の急落頻度 ---
            if cliff_rate > 0.15: s += 12
            elif cliff_rate > 0.08: s += 6
            elif cliff_rate > 0.04: s += 3

            # --- 意図的負け率 (KD<0.3) ---
            if inten_loss > 0.25: s += 15
            elif inten_loss > 0.15: s += 8
            elif inten_loss > 0.08: s += 4

            # ═══════════════════════════════════════
            # 減点ルール (通常プレイヤー判定)
            # ═══════════════════════════════════════

            # --- 高レベルなのに低ランクは通常プレイヤー → 減点 ---
            if lvl > 200 and tier > 0 and tier <= 14 and kd < 1.3:
                s -= 10

            # --- 多シーズンプレイ = 長期プレイヤー (スマーフの可能性低) ---
            if seasons >= 6 and kd < 1.5:
                s -= 8
            elif seasons >= 4 and kd < 1.3:
                s -= 5

            # --- シーズンあたり試合数多い = ガチプレイヤー ---
            if avg_games_season > 80 and seasons >= 3:
                s -= 5

            scores[i] = max(0, min(s, 100))

        return scores

    def _judge(self, score: float) -> str:
        """スコアに基づく最終判定"""
        if score >= 75:
            return "🔴 スマーフ濃厚"
        elif score >= 55:
            return "🟠 スマーフ可能性高"
        elif score >= 40:
            return "🟡 スマーフ疑い"
        elif score >= 25:
            return "🔵 やや疑わしい"
        else:
            return "🟢 通常プレイヤー"

    def _calc_confidence(self, row) -> str:
        """モデル間の一致度で信頼度を計算"""
        scores = [
            row.get("iso_score", 50),
            row.get("lof_score", 50),
            row.get("ocsvm_score", 50),
            row.get("gmm_score", 50),
        ]
        # 全モデルが高スコア or 低スコアで一致 → 高信頼
        above_60 = sum(1 for s in scores if s >= 60)
        below_40 = sum(1 for s in scores if s <= 40)

        if above_60 >= 3 or below_40 >= 3:
            return "高"
        elif above_60 >= 2 or below_40 >= 2:
            return "中"
        else:
            return "低"

    def visualize(self, df: pd.DataFrame, save_dir: Path = OUTPUT_DIR):
        """6種の分析グラフを生成"""
        if not self.is_fitted:
            raise RuntimeError("先にfit_predict()を実行してください")

        save_dir.mkdir(exist_ok=True)

        X_raw = df[FEATURE_COLS].values.astype(float)
        col_medians = np.nanmedian(X_raw, axis=0)
        for i in range(X_raw.shape[1]):
            mask = np.isnan(X_raw[:, i]) | np.isinf(X_raw[:, i])
            X_raw[mask, i] = col_medians[i]
        X_standard = self.standard_scaler.transform(X_raw)
        coords = self.pca.transform(X_standard)

        # ─── 1. 全体概要 (4パネル) ───
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle("VALORANT Smurf Detection - Ensemble Analysis", fontsize=16, fontweight="bold")

        # (1) PCA + スマーフスコア
        ax = axes[0, 0]
        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                            c=df["smurf_score"], cmap="RdYlGn_r",
                            alpha=0.7, edgecolors="k", linewidths=0.3, s=30)
        plt.colorbar(scatter, ax=ax, label="Smurf Score")
        ax.set_title("PCA - Smurf Score Heatmap")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # スマーフ濃厚のプレイヤーにラベル
        smurf_mask = df["smurf_score"] >= 70
        if smurf_mask.any():
            for idx in df[smurf_mask].index:
                ax.annotate(df.loc[idx, "player"][:15],
                           (coords[idx, 0], coords[idx, 1]),
                           fontsize=5, alpha=0.8, color="red")

        # (2) K-Means クラスタ
        ax = axes[0, 1]
        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                            c=df["cluster"], cmap="viridis",
                            alpha=0.7, edgecolors="k", linewidths=0.3, s=30)
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_title("K-Means Skill Clusters")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # (3) スコア分布ヒストグラム
        ax = axes[1, 0]
        ax.hist(df["smurf_score"], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(x=75, color="red", linestyle="--", linewidth=2, label="Smurf Line (75)")
        ax.axvline(x=55, color="orange", linestyle="--", linewidth=1.5, label="Suspect Line (55)")
        ax.axvline(x=40, color="gold", linestyle="--", linewidth=1.5, label="Watch Line (40)")
        ax.set_title("Smurf Score Distribution")
        ax.set_xlabel("Smurf Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        # (4) 各モデルスコアの相関
        ax = axes[1, 1]
        model_cols = ["iso_score", "lof_score", "ocsvm_score", "gmm_score", "rule_score"]
        corr = df[model_cols].corr()
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(model_cols)))
        ax.set_yticks(range(len(model_cols)))
        ax.set_xticklabels(["IsoForest", "LOF", "OCSVM", "GMM", "Rule"], fontsize=8, rotation=45)
        ax.set_yticklabels(["IsoForest", "LOF", "OCSVM", "GMM", "Rule"], fontsize=8)
        for ii in range(len(model_cols)):
            for jj in range(len(model_cols)):
                ax.text(jj, ii, f"{corr.iloc[ii, jj]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax)
        ax.set_title("Model Score Correlation")

        plt.tight_layout()
        fig.savefig(save_dir / "01_overview.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIS] 01_overview.png 保存")

        # ─── 2. スマーフ特徴量詳細 ───
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Smurf Feature Analysis", fontsize=16, fontweight="bold")

        # (1) KD vs Account Level
        ax = axes[0, 0]
        scatter = ax.scatter(df["account_level"], df["avg_kd"],
                            c=df["smurf_score"], cmap="RdYlGn_r",
                            alpha=0.7, edgecolors="k", linewidths=0.3, s=25)
        ax.set_xlabel("Account Level")
        ax.set_ylabel("Average K/D")
        ax.set_title("K/D vs Account Level")
        plt.colorbar(scatter, ax=ax, label="Smurf Score")

        # (2) HS% vs Tier
        ax = axes[0, 1]
        scatter = ax.scatter(df["current_tier"], df["avg_hs_pct"],
                            c=df["smurf_score"], cmap="RdYlGn_r",
                            alpha=0.7, edgecolors="k", linewidths=0.3, s=25)
        ax.set_xlabel("Current Tier")
        ax.set_ylabel("Avg Headshot %")
        ax.set_title("HS% vs Rank Tier")
        plt.colorbar(scatter, ax=ax, label="Smurf Score")

        # (3) Win Rate vs KD
        ax = axes[0, 2]
        scatter = ax.scatter(df["avg_kd"], df["win_rate"],
                            c=df["smurf_score"], cmap="RdYlGn_r",
                            alpha=0.7, edgecolors="k", linewidths=0.3, s=25)
        ax.set_xlabel("Average K/D")
        ax.set_ylabel("Win Rate")
        ax.set_title("Win Rate vs K/D")
        plt.colorbar(scatter, ax=ax, label="Smurf Score")

        # (4) Performance-Rank Ratio
        ax = axes[1, 0]
        scatter = ax.scatter(df["perf_rank_ratio_kd"], df["perf_level_ratio"],
                            c=df["smurf_score"], cmap="RdYlGn_r",
                            alpha=0.7, edgecolors="k", linewidths=0.3, s=25)
        ax.set_xlabel("Performance/Rank Ratio (KD)")
        ax.set_ylabel("Performance/Level Ratio")
        ax.set_title("Rank vs Level Discrepancy")
        plt.colorbar(scatter, ax=ax, label="Smurf Score")

        # (5) Dominant Rate vs Account Level
        ax = axes[1, 1]
        scatter = ax.scatter(df["account_level"], df["dominant_rate"],
                            c=df["smurf_score"], cmap="RdYlGn_r",
                            alpha=0.7, edgecolors="k", linewidths=0.3, s=25)
        ax.set_xlabel("Account Level")
        ax.set_ylabel("Dominant Rate (KD>2.0)")
        ax.set_title("Dominance vs Account Level")
        plt.colorbar(scatter, ax=ax, label="Smurf Score")

        # (6) Rank Gap (highest - current)
        ax = axes[1, 2]
        df_with_gap = df[df["rank_gap"] > 0]
        if len(df_with_gap) > 0:
            scatter = ax.scatter(df_with_gap["rank_gap"], df_with_gap["avg_kd"],
                                c=df_with_gap["smurf_score"], cmap="RdYlGn_r",
                                alpha=0.7, edgecolors="k", linewidths=0.3, s=25)
            plt.colorbar(scatter, ax=ax, label="Smurf Score")
        ax.set_xlabel("Rank Gap (Highest - Current)")
        ax.set_ylabel("Average K/D")
        ax.set_title("Rank Gap vs Performance")

        plt.tight_layout()
        fig.savefig(save_dir / "02_features.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIS] 02_features.png 保存")

        # ─── 3. モデル比較 ───
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")

        model_names = ["Isolation Forest", "LOF", "One-Class SVM", "GMM"]
        score_cols = ["iso_score", "lof_score", "ocsvm_score", "gmm_score"]

        for idx, (ax, mname, scol) in enumerate(zip(axes.flat, model_names, score_cols)):
            scatter = ax.scatter(coords[:, 0], coords[:, 1],
                                c=df[scol], cmap="RdYlGn_r",
                                alpha=0.7, edgecolors="k", linewidths=0.2, s=20)
            ax.set_title(f"{mname} Score")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.colorbar(scatter, ax=ax)

        plt.tight_layout()
        fig.savefig(save_dir / "03_model_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIS] 03_model_comparison.png 保存")


# ─────────────────────────────────────────
# 4. メイン実行
# ─────────────────────────────────────────

def run():
    """スマーフ検出パイプライン実行"""
    print("=" * 70)
    print("  VALORANT 高精度スマーフ検出AI v2")
    print("  教師なし学習アンサンブル + 6データソース統合 (71特徴量)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── STEP 1: データ読み込み ──
    print("\n[STEP 1/5] データ読み込み")
    players = load_all_players()
    if len(players) < 10:
        print("[ERROR] 最低10人のプレイヤーデータが必要です")
        sys.exit(1)

    # ── STEP 2: 特徴量抽出 ──
    print("\n[STEP 2/5] 特徴量抽出 (70+ features, 6データソース統合)")
    features_list = []
    skipped = 0
    for p in players:
        feat = extract_features(p)
        if feat:
            features_list.append(feat)
        else:
            skipped += 1
    print(f"  -> {len(features_list)} プレイヤーの特徴量抽出完了 (スキップ: {skipped})")

    df = pd.DataFrame(features_list)

    # 基本統計
    print(f"\n  【データ概要】")
    print(f"    プレイヤー数 : {len(df)}")
    print(f"    特徴量数     : {len(FEATURE_COLS)}")
    print(f"    平均試合数   : {df['matches_count'].mean():.1f}")
    print(f"    平均KD       : {df['avg_kd'].mean():.3f}")
    print(f"    平均HS%      : {df['avg_hs_pct'].mean():.4f}")
    print(f"    平均レベル   : {df['account_level'].mean():.1f}")
    # 新データの取得状況
    has_v3 = (df['seasons_played'] > 0).sum()
    has_hist = (df['mmr_hist_count'] > 0).sum()
    has_v4 = (df['avg_ability_total'] > 0).sum()
    print(f"    mmr_v3あり   : {has_v3}/{len(df)}")
    print(f"    mmr_histあり : {has_hist}/{len(df)}")
    print(f"    match_v4あり : {has_v4}/{len(df)}")

    # ── STEP 3: モデル学習 & 予測 ──
    print(f"\n[STEP 3/5] アンサンブルモデル学習 & 予測")
    detector = EnsembleSmurfDetector(contamination=0.12)
    results = detector.fit_predict(df)

    # ── STEP 4: 結果表示 ──
    print(f"\n[STEP 4/5] 結果分析")
    results_sorted = results.sort_values("smurf_score", ascending=False)

    # 判定カウント
    judgment_counts = results["judgment"].value_counts()
    print("\n  【判定結果サマリー】")
    for j, c in judgment_counts.items():
        print(f"    {j}: {c} 人")

    # Top スマーフ候補
    print("\n  【スマーフ濃厚 TOP 20】")
    display_cols = [
        "player", "account_level", "current_rank", "highest_rank",
        "avg_kd", "avg_hs_pct", "win_rate", "dominant_rate",
        "seasons_played", "elo_velocity",
        "smurf_score", "confidence", "judgment",
    ]
    top_smurfs = results_sorted.head(20)[display_cols]
    pd.set_option("display.max_colwidth", 25)
    pd.set_option("display.width", 200)
    print(top_smurfs.to_string(index=False))

    # 通常プレイヤーTOP (検証用)
    print("\n  【通常プレイヤー例 (低スコア TOP 10)】")
    normal = results_sorted.tail(10)[display_cols]
    print(normal.to_string(index=False))

    # ── STEP 5: 保存 ──
    print(f"\n[STEP 5/5] 結果保存")

    # CSV (全結果)
    csv_path = OUTPUT_DIR / "smurf_results_full.csv"
    results_sorted.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  -> {csv_path}")

    # CSV (スマーフ候補のみ)
    smurfs_only = results_sorted[results_sorted["smurf_score"] >= 40]
    smurf_csv = OUTPUT_DIR / "smurf_suspects.csv"
    smurfs_only.to_csv(smurf_csv, index=False, encoding="utf-8-sig")
    print(f"  -> {smurf_csv} ({len(smurfs_only)} 人)")

    # グラフ
    print("\n[VIS] グラフ生成中...")
    detector.visualize(results)

    # サマリーレポート
    report_path = OUTPUT_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  VALORANT スマーフ検出AIレポート\n")
        f.write(f"  生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"分析プレイヤー数: {len(df)}\n")
        f.write(f"使用特徴量数: {len(FEATURE_COLS)}\n")
        f.write(f"contamination: {detector.contamination}\n\n")
        f.write("【判定結果サマリー】\n")
        for j, c in judgment_counts.items():
            f.write(f"  {j}: {c} 人\n")
        f.write("\n【アンサンブルモデル重み】\n")
        for k, v in detector.weights.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n【スマーフ濃厚 TOP 30】\n")
        top30 = results_sorted.head(30)[display_cols]
        f.write(top30.to_string(index=False))
        f.write("\n")
    print(f"  -> {report_path}")

    print("\n" + "=" * 70)
    print("  完了!")
    print(f"  出力先: {OUTPUT_DIR}/")
    print(f"    - smurf_results_full.csv  (全{len(results)}人の結果)")
    print(f"    - smurf_suspects.csv      (スマーフ候補 {len(smurfs_only)}人)")
    print(f"    - 01_overview.png         (全体概要グラフ)")
    print(f"    - 02_features.png         (特徴量分析グラフ)")
    print(f"    - 03_model_comparison.png (モデル比較グラフ)")
    print(f"    - report.txt              (テキストレポート)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run()
