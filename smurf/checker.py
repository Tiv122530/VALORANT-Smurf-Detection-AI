"""
checker.py  - PUUIDを受け取って教師ありモデルでスマーフ判定するメインロジック
"""
import json
from datetime import datetime
from pathlib import Path

from .collector import fetch_all
from .extractor import extract_features, TIER_TO_RANK
from .model import SmurfModel

_DATA_DIR = Path(__file__).parent / "data"
_DATA_DIR.mkdir(exist_ok=True)

_model: SmurfModel | None = None


def _get_model() -> SmurfModel:
    global _model
    if _model is None:
        _model = SmurfModel()
    return _model


def _load_or_fetch(puuid: str, force_refresh: bool = False) -> dict | None:
    """キャッシュがあれば使用、なければAPIから取得してキャッシュ保存"""
    cache = _DATA_DIR / f"{puuid}.json"

    if cache.exists() and not force_refresh:
        with open(cache, encoding="utf-8") as f:
            data = json.load(f)
        # 旧フォーマット補完
        if not data.get("puuid"):
            acc = data.get("account") or {}
            data["puuid"]  = puuid
            data["name"]   = acc.get("name", "")
            data["tag"]    = acc.get("tag", "")
            data["player"] = f"{acc.get('name','')}#{acc.get('tag','')}"
            data["region"] = acc.get("region", "ap")
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        return data

    data = fetch_all(puuid)
    if data:
        data["collected_at"] = datetime.now().isoformat()
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    return data


def check(puuid: str, force_refresh: bool = False) -> dict:
    """
    PUUIDを受け取り、教師ありモデルでスマーフ判定する。

    Args:
        puuid:         VALORANTプレイヤーのPUUID
        force_refresh: Trueにするとキャッシュを無視してAPIを再取得

    Returns:
        {
            # 判定
            "judgment":    str,    # "🔴 スマーフ確定" など
            "prob":        float,  # スマーフ確率 0.0~1.0
            "score":       float,  # 0~100換算スコア

            # プロフィール
            "player":      str,
            "puuid":       str,
            "current_rank":str,
            "highest_rank":str,
            "account_level":int,
            "matches_count":int,

            # 戦闘指標
            "avg_kd":      float,
            "avg_hs_pct":  float,  # 0.0~1.0 (1.0=100%)
            "avg_dpr":     float,
            "win_rate":    float,

            # スマーフシグナル
            "kd_rank_deviation": float,
            "rank_gap":         int,
            "tier_trend":       float,
            "intentional_loss_rate": float,
            "tank_streak_max":  int,
            "kd_cliff_drop_rate": float,
            "kd_bimodal_score": float,
            "rule_score":       float,

            # モデル情報
            "model_auc":   float,
            "n_labeled":   int,
            "n_smurf":     int,
            "trained_at":  str,

            # 生特徴量 (全78特徴量)
            "features":    dict,
        }

    Raises:
        ValueError: PUUIDが無効またはデータ不足
        FileNotFoundError: smurf_model.pklが存在しない
    """
    # 1. データ取得
    data = _load_or_fetch(puuid, force_refresh=force_refresh)
    if data is None:
        raise ValueError(f"プレイヤーデータを取得できませんでした。PUUID: {puuid}")

    # 2. 特徴量抽出
    feat = extract_features(data)
    if feat is None:
        raise ValueError("コンペティティブマッチが3試合未満のため判定できません。")

    feat["puuid"]  = puuid
    feat["player"] = data.get("player", "?")
    feat["rule_score"] = feat.get("rule_score", 0.0)  # 念のため確保

    # ランク名付与
    mmr = data.get("mmr", {}) or {}
    cur = mmr.get("current_data", {}) or {}
    hi  = mmr.get("highest_rank", {}) or {}
    feat["current_rank"] = (
        cur.get("currenttierpatched")
        or TIER_TO_RANK.get(int(feat.get("current_tier", 0)), "Unranked")
    )
    feat["highest_rank"] = (
        hi.get("patched_tier")
        or TIER_TO_RANK.get(int(feat.get("peak_tier_v3", 0)), "Unranked")
    )

    # 3. 教師ありモデルで判定
    model  = _get_model()
    result = model.predict(feat)

    # 4. レスポンス組み立て
    return {
        # 判定
        "judgment":    result["judgment"],
        "prob":        result["prob"],
        "score":       result["score"],

        # プロフィール
        "player":       feat["player"],
        "puuid":        puuid,
        "current_rank": feat["current_rank"],
        "highest_rank": feat["highest_rank"],
        "account_level":int(feat.get("account_level", 0)),
        "matches_count":int(feat.get("matches_count", 0)),

        # 戦闘指標
        "avg_kd":      round(float(feat.get("avg_kd", 0)), 2),
        "avg_hs_pct":  round(float(feat.get("avg_hs_pct", 0)), 4),
        "avg_dpr":     round(float(feat.get("avg_dpr", 0)), 1),
        "win_rate":    round(float(feat.get("win_rate", 0)), 4),

        # スマーフシグナル
        "kd_rank_deviation":    round(float(feat.get("kd_rank_deviation", 0)), 3),
        "rank_gap":             int(feat.get("rank_gap", 0)),
        "tier_trend":           round(float(feat.get("tier_trend", 0)), 4),
        "intentional_loss_rate":round(float(feat.get("intentional_loss_rate", 0)), 4),
        "tank_streak_max":      int(feat.get("tank_streak_max", 0)),
        "kd_cliff_drop_rate":   round(float(feat.get("kd_cliff_drop_rate", 0)), 4),
        "kd_bimodal_score":     round(float(feat.get("kd_bimodal_score", 0)), 4),
        "rule_score":           round(float(feat.get("rule_score", 0)), 1),

        # モデル情報
        "model_auc":  result["model_auc"],
        "n_labeled":  result["n_labeled"],
        "n_smurf":    result["n_smurf"],
        "trained_at": result["trained_at"],

        # 生特徴量
        "features": feat,
    }
