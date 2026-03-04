"""
model.py  - 教師ありモデルによるスマーフ推論
"""
import pickle
import numpy as np
from pathlib import Path

# 教師ありモデルで使用する特徴量
FEATURES = [
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

# モデルファイルのデフォルトパス
_DEFAULT_MODEL = Path(__file__).parent / "smurf_model.pkl"


class SmurfModel:
    """
    教師ありスマーフ分類モデルのラッパー。
    smurf_model.pkl を読み込んでスマーフ確率を返す。
    """

    def __init__(self, model_path: Path | str | None = None):
        path = Path(model_path) if model_path else _DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"モデルファイルが見つかりません: {path}\n"
                "先に train_supervised.py を実行してください。"
            )
        with open(path, "rb") as f:
            obj = pickle.load(f)

        self._model         = obj["model"]
        self._feat_names    = obj["feature_names"]
        self.auc            = obj.get("mean_auc")
        self.n_labeled      = obj.get("n_labeled")
        self.n_smurf        = obj.get("n_smurf")
        self.trained_at     = obj.get("trained_at", "")[:10]

    def predict(self, feat: dict) -> dict:
        """
        特徴量辞書を受け取り判定結果を返す。

        Returns:
            {
                "prob":       float,   # スマーフ確率 0.0~1.0
                "score":      float,   # 0~100換算
                "judgment":   str,     # 判定ラベル
                "model_auc":  float,
                "n_labeled":  int,
                "n_smurf":    int,
                "trained_at": str,
            }
        """
        row = {k: feat.get(k, 0.0) for k in self._feat_names}
        X   = np.array([[row[k] for k in self._feat_names]], dtype=float)
        X   = np.nan_to_num(X, nan=0.0)
        prob = float(self._model.predict_proba(X)[0][1])

        if prob >= 0.80:
            judgment = "🔴 スマーフ確定"
        elif prob >= 0.55:
            judgment = "🟠 スマーフ可能性高"
        elif prob >= 0.35:
            judgment = "🟡 グレーゾーン"
        else:
            judgment = "🟢 通常プレイヤー"

        return {
            "prob":       prob,
            "score":      round(prob * 100, 1),
            "judgment":   judgment,
            "model_auc":  self.auc,
            "n_labeled":  self.n_labeled,
            "n_smurf":    self.n_smurf,
            "trained_at": self.trained_at,
        }
