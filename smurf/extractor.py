"""
extractor.py  - VALORANT プレイヤー特徴量抽出 (78特徴量)
smurf_ai.py の extract_features を単体パッケージ用に切り出したモジュール。
"""
import numpy as np
from datetime import datetime
from collections import Counter

# ─── ランク期待値テーブル ─────────────────────────────────────
# (tier_min, tier_max): (期待KD, KD_std, 期待HS%, HS_std)
RANK_EXPECTED_STATS: dict[tuple[int, int], tuple[float, float, float, float]] = {
    (3,  5):  (0.74, 0.18, 0.14, 0.06),   # Iron
    (6,  8):  (0.85, 0.19, 0.15, 0.06),   # Bronze
    (9,  11): (0.95, 0.21, 0.16, 0.06),   # Silver
    (12, 14): (1.05, 0.24, 0.17, 0.06),   # Gold
    (15, 17): (1.15, 0.27, 0.18, 0.06),   # Platinum
    (18, 20): (1.30, 0.30, 0.19, 0.06),   # Diamond
    (21, 23): (1.50, 0.34, 0.21, 0.07),   # Ascendant
    (24, 26): (1.70, 0.38, 0.22, 0.07),   # Immortal
    (27, 27): (2.00, 0.42, 0.24, 0.08),   # Radiant
}

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


def _rank_expected(tier: int) -> tuple:
    for (lo, hi), vals in RANK_EXPECTED_STATS.items():
        if lo <= tier <= hi:
            return vals
    return (1.0, 0.25, 0.16, 0.06)


def _div(a, b, default=0.0):
    return a / b if b and b != 0 else default


def _cv(arr):
    if len(arr) < 2:
        return 0.0
    m = np.mean(arr)
    return np.std(arr) / abs(m) if m != 0 else 0.0


def extract_features(player_data: dict) -> dict | None:
    """
    1人分のプレイヤーJSONから78特徴量を抽出して返す。
    コンペティティブマッチが3試合未満の場合は None を返す。
    """
    matches       = player_data.get("stored_matches", [])
    account       = player_data.get("account", {}) or {}
    mmr           = player_data.get("mmr", {}) or {}
    current_mmr   = mmr.get("current_data", {}) or {}
    highest_rank  = mmr.get("highest_rank", {}) or {}
    mmr_v3        = player_data.get("mmr_v3", {}) or {}
    mmr_history   = player_data.get("mmr_history", []) or []
    match_v4_data = player_data.get("match_v4", []) or []

    comp = [m for m in matches if m.get("meta", {}).get("mode") == "Competitive"]
    if len(comp) < 3:
        return None

    puuid         = player_data.get("puuid", "")
    account_level = account.get("account_level", 0) or 0
    current_tier  = current_mmr.get("currenttier", 0) or 0
    current_elo   = current_mmr.get("elo", 0) or 0
    highest_tier  = highest_rank.get("tier", 0) or 0

    # ─── 試合ごと集計 ──────────────────────────────────────
    kills_l, deaths_l, assists_l = [], [], []
    hs_pct_l, score_l = [], []
    dmg_made_l, dmg_recv_l, rounds_l = [], [], []
    kd_l, kda_l, kpr_l, dpr_l, spr_l = [], [], [], [], []
    wins = losses = draws = 0
    max_win_streak = cur_streak = 0
    agent_ctr = Counter()
    tier_l, level_l, timestamps = [], [], []
    margins = []

    for m in comp:
        s  = m.get("stats", {})
        mt = m.get("meta", {})
        tm = m.get("teams", {})

        k  = s.get("kills", 0) or 0
        d  = s.get("deaths", 0) or 0
        a  = s.get("assists", 0) or 0
        sh = s.get("shots", {}) or {}
        head  = sh.get("head", 0) or 0
        total_sh = head + (sh.get("body", 0) or 0) + (sh.get("leg", 0) or 0)
        score = s.get("score", 0) or 0
        dmg   = s.get("damage", {}) or {}
        dm    = dmg.get("made", 0) or 0
        dr    = dmg.get("received", 0) or 0
        char  = s.get("character", {})
        agent = char.get("name", "") if isinstance(char, dict) else ""
        tier  = s.get("tier", 0) or 0
        level = s.get("level", 0) or 0
        team  = (s.get("team", "") or "").lower()

        red_r  = tm.get("red", 0) if isinstance(tm.get("red"), (int, float)) else 0
        blue_r = tm.get("blue", 0) if isinstance(tm.get("blue"), (int, float)) else 0
        total_r = red_r + blue_r or 24
        my_r   = red_r if team == "red" else (blue_r if team == "blue" else 0)
        op_r   = blue_r if team == "red" else (red_r if team == "blue" else 0)

        won  = my_r > op_r
        lost = my_r < op_r
        if won:
            wins += 1; cur_streak += 1
            max_win_streak = max(max_win_streak, cur_streak)
        elif lost:
            losses += 1; cur_streak = 0
        else:
            draws += 1; cur_streak = 0
        margins.append(my_r - op_r)

        kd = _div(k, d, default=float(k))
        kills_l.append(k); deaths_l.append(d); assists_l.append(a)
        hs_pct_l.append(_div(head, total_sh))
        score_l.append(score); dmg_made_l.append(dm); dmg_recv_l.append(dr)
        rounds_l.append(total_r)
        kd_l.append(kd); kda_l.append(_div(k + a, max(d, 1)))
        kpr_l.append(_div(k, total_r)); dpr_l.append(_div(dm, total_r))
        spr_l.append(_div(score, total_r))
        if agent: agent_ctr[agent] += 1
        if tier > 0:  tier_l.append(tier)
        if level > 0: level_l.append(level)
        started = mt.get("started_at")
        if started:
            try:
                timestamps.append(datetime.fromisoformat(started.replace("Z", "+00:00")))
            except Exception:
                pass

    n = len(comp)

    # ─── 基本性能 ──────────────────────────────────────────
    avg_kd    = np.mean(kd_l);    avg_kda   = np.mean(kda_l)
    avg_hs    = np.mean(hs_pct_l); avg_dpr  = np.mean(dpr_l)
    avg_kpr   = np.mean(kpr_l);   avg_spr   = np.mean(spr_l)
    avg_kills = np.mean(kills_l); avg_deaths= np.mean(deaths_l)
    avg_assists=np.mean(assists_l)
    avg_dmg   = np.mean(dmg_made_l)
    med_kd    = np.median(kd_l);  med_kpr   = np.median(kpr_l)

    # ─── 安定性 ──────────────────────────────────────────
    kd_std  = np.std(kd_l)    if len(kd_l) > 1    else 0
    kd_cv   = _cv(kd_l)
    ks_std  = np.std(kills_l) if len(kills_l) > 1  else 0
    ks_cv   = _cv(kills_l)
    hs_std  = np.std(hs_pct_l)if len(hs_pct_l) > 1 else 0
    hs_cv   = _cv(hs_pct_l)
    dpr_std = np.std(dpr_l)   if len(dpr_l) > 1    else 0
    dpr_cv  = _cv(dpr_l)
    sc_cv   = _cv(score_l)

    # ─── 勝率 ────────────────────────────────────────────
    win_rate    = _div(wins, wins + losses)
    avg_margin  = np.mean(margins) if margins else 0

    # ─── エージェント ─────────────────────────────────────
    agent_div = len(agent_ctr)
    top_agent = agent_ctr.most_common(1)[0][1] / n if agent_ctr and n > 0 else 0.0

    # ─── ランク乖離 ───────────────────────────────────────
    rank_gap   = highest_tier - current_tier if highest_tier > 0 and current_tier > 0 else 0
    tier_range = max(tier_l) - min(tier_l) if tier_l else 0
    tier_trend = 0.0
    avg_match_tier = 0
    if tier_l:
        avg_match_tier = np.mean(tier_l)
        if len(tier_l) >= 2:
            tier_trend = float(np.polyfit(np.arange(len(tier_l)), tier_l, 1)[0])

    eff_tier = current_tier if current_tier > 0 else (avg_match_tier if avg_match_tier > 0 else 10)
    perf_kd  = _div(avg_kd,  eff_tier / 15.0)
    perf_hs  = _div(avg_hs,  eff_tier / 15.0)
    perf_dpr = _div(avg_dpr, eff_tier / 15.0)

    lv_use = account_level if account_level > 0 else (max(level_l) if level_l else 100)
    perf_lv = _div(avg_kd, max(lv_use / 200.0, 0.1))

    # ─── ランク期待値偏差 ─────────────────────────────────
    _ek, _eks, _eh, _ehs = _rank_expected(int(eff_tier))
    kd_dev = (avg_kd - _ek)  / max(_eks, 0.01)
    hs_dev = (avg_hs - _eh)  / max(_ehs, 0.01)
    hs_kd  = (avg_hs / 100.0) * avg_kd if avg_hs <= 1 else (avg_hs / 10000.0) * avg_kd

    # ─── 活動頻度 ─────────────────────────────────────────
    freq = span = 0.0
    if len(timestamps) >= 2:
        ts = sorted(timestamps)
        span = (ts[-1] - ts[0]).total_seconds() / 86400.0
        freq = n / span if span > 0 else 0.0
    lr  = max(level_l) - min(level_l) if level_l else 0
    dmg_eff = _div(avg_dmg, avg_dmg + np.mean(dmg_recv_l))

    # ─── 支配力 ───────────────────────────────────────────
    hi_perf   = sum(1 for x in kd_l if x >= 1.5) / n
    dominant  = sum(1 for x in kd_l if x >= 2.0) / n
    kd_diff_l = [k - d for k, d in zip(kills_l, deaths_l)]
    avg_kdd   = np.mean(kd_diff_l)
    kd_pos    = sum(1 for x in kd_diff_l if x > 0) / n

    # ─── デランク検出 ─────────────────────────────────────
    il_rate = sum(1 for x in kd_l if x < 0.3) / n
    _ts = _tsmax = 0
    for x in kd_l:
        if x < 0.4:
            _ts += 1; _tsmax = max(_tsmax, _ts)
        else:
            _ts = 0
    cliff = sum(1 for i in range(1, len(kd_l))
                if kd_l[i-1] >= 1.5 and kd_l[i] <= 0.5) / max(n - 1, 1)
    hkr   = sum(1 for x in kd_l if x >= 1.5) / n
    vlkr  = sum(1 for x in kd_l if x < 0.4) / n
    bimod = hkr * vlkr

    # ─── mmr_v3 ──────────────────────────────────────────
    v3_peak    = mmr_v3.get("peak", {}) or {}
    v3_current = mmr_v3.get("current", {}) or {}
    v3_seasonal= mmr_v3.get("seasonal", []) or []

    peak_t_obj = v3_peak.get("tier", {}) or {}
    peak_tier  = peak_t_obj.get("id", 0) if isinstance(peak_t_obj, dict) else 0
    peak_rr    = v3_peak.get("rr", 0) or 0
    v3_ct_obj  = v3_current.get("tier", {}) or {}
    v3_ct      = v3_ct_obj.get("id", 0) if isinstance(v3_ct_obj, dict) else 0
    peak_gap   = peak_tier - v3_ct if peak_tier > 0 and v3_ct > 0 else 0

    seas_tiers = []; seas_wr = []; seas_games = []; total_aw = 0
    for s in v3_seasonal:
        et = (s.get("end_tier", {}) or {})
        tid= et.get("id", 0) if isinstance(et, dict) else 0
        if tid > 0: seas_tiers.append(tid)
        g = s.get("games", 0) or 0; w = s.get("wins", 0) or 0
        seas_games.append(g)
        if g > 0: seas_wr.append(w / g)
        total_aw += len(s.get("act_wins", []) or [])

    seas_played  = len(v3_seasonal)
    seas_avgwr   = float(np.mean(seas_wr) if seas_wr else 0.0)
    first_st = latest_st = 0; seas_growth = 0.0
    if len(seas_tiers) >= 2:
        first_st = seas_tiers[-1]; latest_st = seas_tiers[0]
        seas_growth = (latest_st - first_st) / len(seas_tiers)
    elif len(seas_tiers) == 1:
        first_st = latest_st = seas_tiers[0]
    avg_gperseas = float(np.mean(seas_games) if seas_games else 0.0)

    # ─── mmr_history ─────────────────────────────────────
    rr_changes = [e.get("last_mmr_change", 0) or 0 for e in mmr_history]
    elo_vals   = [e.get("elo", 0) for e in mmr_history if (e.get("elo") or 0) > 0]
    avg_rrc    = float(np.mean(rr_changes) if rr_changes else 0.0)
    rrc_std    = float(np.std(rr_changes)  if len(rr_changes) > 1 else 0.0)
    pos_rr     = sum(1 for r in rr_changes if r > 0) / len(rr_changes) if rr_changes else 0.0
    max_rr_g   = max(rr_changes) if rr_changes else 0
    elo_range  = (max(elo_vals) - min(elo_vals)) if len(elo_vals) >= 2 else 0
    elo_vel    = 0.0
    if len(elo_vals) >= 3:
        elo_vel = float(np.polyfit(np.arange(len(elo_vals)), elo_vals, 1)[0])
    rrs = rrsmax = 0
    for r in rr_changes:
        if r > 0: rrs += 1; rrsmax = max(rrsmax, rrs)
        else: rrs = 0

    # ─── match_v4 ─────────────────────────────────────────
    ab_tot_l = []; ult_l = []; afk_l = []; spawn_l = []
    eco_l = []; lv_l = []; party_ids = set()
    sess_l = []; ff_out_l = []; v4c = 0

    for mv in match_v4_data:
        if not isinstance(mv, dict): continue
        meta = mv.get("metadata", {}) or {}
        q = meta.get("queue", {})
        if isinstance(q, dict) and q.get("id") and q.get("id") != "competitive":
            continue
        for p in (mv.get("players", []) or []):
            if not isinstance(p, dict) or p.get("puuid") != puuid: continue
            v4c += 1
            ab = p.get("ability_casts", {}) or {}
            ab_tot_l.append((ab.get("grenade",0) or 0) + (ab.get("ability1",0) or 0) +
                            (ab.get("ability2",0) or 0) + (ab.get("ultimate",0) or 0))
            ult_l.append(ab.get("ultimate", 0) or 0)
            bh = p.get("behavior", {}) or {}
            afk_l.append(bh.get("afk_rounds", 0) or 0)
            spawn_l.append(bh.get("rounds_in_spawn", 0) or 0)
            ff = (bh.get("friendly_fire", {}) or {})
            ff_out_l.append(ff.get("outgoing", 0) or 0)
            ec = p.get("economy", {}) or {}
            eco_l.append((ec.get("spent", {}) or {}).get("average", 0) or 0)
            lv_l.append((ec.get("loadout_value", {}) or {}).get("average", 0) or 0)
            if p.get("party_id"): party_ids.add(p["party_id"])
            sm = p.get("session_playtime_in_ms", 0) or 0
            if sm > 0: sess_l.append(sm / 60000.0)
            break

    avg_ab   = float(np.mean(ab_tot_l) if ab_tot_l else 0.0)
    avg_ult  = float(np.mean(ult_l)    if ult_l    else 0.0)
    avg_afk  = float(np.mean(afk_l)    if afk_l    else 0.0)
    avg_spwn = float(np.mean(spawn_l)  if spawn_l  else 0.0)
    avg_eco  = float(np.mean(eco_l)    if eco_l    else 0.0)
    avg_lv   = float(np.mean(lv_l)     if lv_l     else 0.0)
    p_cnt    = len(party_ids)
    solo_r   = min(p_cnt / v4c, 1.0) if v4c > 0 and p_cnt > 0 else 0.0
    avg_sess = float(np.mean(sess_l)   if sess_l   else 0.0)
    avg_ff   = float(np.mean(ff_out_l) if ff_out_l else 0.0)
    eco_eff  = _div(avg_kd, avg_eco / 4000.0) if avg_eco > 0 and avg_kd > 0 else 0.0

    # ─── 組み立て ─────────────────────────────────────────
    return {
        "puuid": puuid, "player": player_data.get("player", ""),
        "account_level": lv_use, "current_tier": current_tier,
        "current_elo": current_elo, "highest_tier": highest_tier,
        "matches_count": n,
        "avg_kills": round(avg_kills, 3), "avg_deaths": round(avg_deaths, 3),
        "avg_assists": round(avg_assists, 3),
        "avg_kd": round(avg_kd, 4), "avg_kda": round(avg_kda, 4),
        "avg_hs_pct": round(avg_hs, 5), "avg_dmg_made": round(avg_dmg, 2),
        "avg_dpr": round(avg_dpr, 3), "avg_spr": round(avg_spr, 3),
        "avg_kpr": round(avg_kpr, 5),
        "median_kd": round(med_kd, 4), "median_kpr": round(med_kpr, 5),
        "kd_std": round(kd_std, 4), "kd_cv": round(kd_cv, 4),
        "kills_std": round(ks_std, 3), "kills_cv": round(ks_cv, 4),
        "hs_std": round(hs_std, 5), "hs_cv": round(hs_cv, 4),
        "dpr_std": round(dpr_std, 3), "dpr_cv": round(dpr_cv, 4),
        "score_cv": round(sc_cv, 4),
        "win_rate": round(win_rate, 4), "max_win_streak": max_win_streak,
        "avg_margin": round(avg_margin, 3),
        "agent_diversity": agent_div, "top_agent_ratio": round(top_agent, 4),
        "rank_gap": rank_gap, "tier_range": tier_range,
        "tier_trend": round(tier_trend, 5),
        "perf_rank_ratio_kd": round(perf_kd, 4),
        "perf_rank_ratio_hs": round(perf_hs, 5),
        "perf_rank_ratio_dpr": round(perf_dpr, 4),
        "perf_level_ratio": round(perf_lv, 4),
        "kd_rank_deviation": round(kd_dev, 4),
        "hs_rank_deviation": round(hs_dev, 4),
        "hs_kd_compound": round(hs_kd, 5),
        "match_frequency": round(freq, 4),
        "activity_span_days": round(span, 2),
        "level_range": lr, "dmg_efficiency": round(dmg_eff, 5),
        "high_perf_rate": round(hi_perf, 4), "dominant_rate": round(dominant, 4),
        "avg_kd_diff": round(avg_kdd, 3),
        "kd_diff_positive_rate": round(kd_pos, 4),
        "intentional_loss_rate": round(il_rate, 4),
        "tank_streak_max": _tsmax,
        "kd_cliff_drop_rate": round(cliff, 4),
        "kd_bimodal_score": round(bimod, 4),
        "peak_tier_v3": peak_tier, "peak_rr": peak_rr,
        "peak_current_gap_v3": peak_gap,
        "seasons_played": seas_played,
        "first_season_tier": first_st, "latest_season_tier": latest_st,
        "seasonal_rank_growth": round(seas_growth, 4),
        "seasonal_avg_winrate": round(seas_avgwr, 4),
        "total_act_wins": total_aw,
        "avg_games_per_season": round(avg_gperseas, 2),
        "mmr_hist_count": len(mmr_history),
        "avg_rr_change": round(avg_rrc, 3),
        "rr_change_std": round(rrc_std, 3),
        "positive_rr_rate": round(pos_rr, 4),
        "max_rr_gain": max_rr_g,
        "elo_range": elo_range, "elo_velocity": round(elo_vel, 4),
        "max_rr_streak": rrsmax,
        "avg_ability_total": round(avg_ab, 2),
        "avg_ultimate_casts": round(avg_ult, 3),
        "avg_afk_rounds": round(avg_afk, 3),
        "avg_rounds_in_spawn": round(avg_spwn, 3),
        "avg_economy_spent": round(avg_eco, 2),
        "avg_loadout_value": round(avg_lv, 2),
        "economy_efficiency": round(eco_eff, 4),
        "party_count": p_cnt,
        "solo_queue_rate": round(solo_r, 4),
        "avg_session_playtime": round(avg_sess, 2),
        "avg_ff_outgoing": round(avg_ff, 3),
    }
