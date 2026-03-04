"""
label_tool.py  - インタラクティブ・ラベリングツール
=====================================================
教師あり学習パイプライン ステップ3:
  教師なしAIが「怪しい」と判定したプレイヤーだけを
  人間がサクッと 1(スマーフ) / 0(一般) とラベリングする。

使い方:
    python label_tool.py

操作:
    1  → スマーフ (正例)
    0  → 通常プレイヤー (負例)
    s  → スキップ (後で判断)
    q  → 中断 (進捗は保存済み)
    ?  → 現在の統計を表示

出力:
    labeled_data.csv  ← これを train_supervised.py に渡す
"""

import os
import sys
import csv
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
FULL_RESULTS   = Path("smurf_output/smurf_results_full.csv")
LABELED_OUTPUT = Path("labeled_data.csv")
# このスコア以上のプレイヤーだけをラベリング対象にする
SUSPECT_THRESHOLD = 55.0
# 1セッションで表示する最大件数
MAX_PER_SESSION = 200

# ANSIカラー (Windows Terminal / VSCode Terminal で動作)
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def _c(text: str, color: str) -> str:
    """カラーコードを付与 (TTY以外は無効化)"""
    if sys.stdout.isatty():
        return f"{color}{text}{RESET}"
    return text


def load_already_labeled() -> set[str]:
    """既にラベルが付いたpuuidセットを返す"""
    if not LABELED_OUTPUT.exists():
        return set()
    df = pd.read_csv(LABELED_OUTPUT)
    return set(df["puuid"].astype(str).tolist())


def load_suspects() -> pd.DataFrame:
    """教師なし結果を読み込み、怪しい順に並べる"""
    if not FULL_RESULTS.exists():
        print(_c(f"[ERROR] {FULL_RESULTS} が見つかりません。先に smurf_ai.py を実行してください。", RED))
        sys.exit(1)

    df = pd.read_csv(FULL_RESULTS)
    df["smurf_score"] = pd.to_numeric(df["smurf_score"], errors="coerce").fillna(0)
    suspects = df[df["smurf_score"] >= SUSPECT_THRESHOLD].copy()
    suspects = suspects.sort_values("smurf_score", ascending=False).reset_index(drop=True)
    return suspects


def display_player(row: pd.Series, index: int, total: int):
    """1プレイヤーの情報をわかりやすく表示"""
    score = float(row.get("smurf_score", 0))
    judgment = str(row.get("judgment", ""))

    if score >= 80:
        score_color = RED
    elif score >= 65:
        score_color = YELLOW
    else:
        score_color = GREEN

    kd = float(row.get("avg_kd", 0))
    hs = float(row.get("avg_hs_pct", 0)) * 100
    tier = row.get("current_rank", "Unknown")
    lvl  = row.get("account_level", "?")
    wr   = float(row.get("win_rate", 0)) * 100
    n    = row.get("matches_count", "?")
    peak = row.get("highest_rank", "Unknown")
    kd_dev = row.get("kd_rank_deviation", None)
    bimodal = row.get("kd_bimodal_score", None)
    tank    = row.get("tank_streak_max", None)
    cliff   = row.get("kd_cliff_drop_rate", None)

    print()
    print(_c("=" * 62, BOLD))
    print(_c(f"  [{index}/{total}]  {row.get('player', '?')}", BOLD))
    print(_c("=" * 62, BOLD))
    print(f"  スマーフスコア : {_c(f'{score:.1f}/100', score_color)}")
    print(f"  判定           : {judgment}")
    print(f"  信頼度         : {row.get('confidence', '?')}")
    print()
    print(_c("  ── 基本情報 ──────────────────────────────", DIM))
    print(f"  現在ランク  : {tier}  (ピーク: {peak})")
    print(f"  アカウントLv: {lvl}")
    print(f"  試合数      : {n}")
    print()
    print(_c("  ── 戦闘性能 ──────────────────────────────", DIM))
    _kd_str = f"{kd:.2f}"
    if kd_dev is not None and str(kd_dev) != "nan":
        _kd_str += f"  (ランク期待値偏差: {_c(f'+{float(kd_dev):.1f}σ', RED if float(kd_dev) >= 2.0 else YELLOW)})"
    print(f"  KD          : {_kd_str}")
    print(f"  HS%         : {hs:.1f}%")
    print(f"  勝率        : {wr:.0f}%")
    print(f"  DPR         : {row.get('avg_dpr', '?')}")
    print()

    # デランクシグナルがある場合のみ表示
    signals = []
    if bimodal is not None and str(bimodal) != "nan" and float(bimodal) > 0.03:
        signals.append(f"KD二峰性スコア: {float(bimodal):.3f} (高い=デランク疑い)")
    if tank is not None and str(tank) != "nan" and int(float(tank)) >= 2:
        signals.append(f"連続低KDストリーク: {int(float(tank))}試合")
    if cliff is not None and str(cliff) != "nan" and float(cliff) > 0.04:
        signals.append(f"KD急落率: {float(cliff):.2f}")
    if signals:
        print(_c("  ── デランクシグナル ───────────────────────", DIM))
        for sig in signals:
            print(f"  ⚠  {sig}")
        print()

    print(_c("  ── モデルスコア ─────────────────────────", DIM))
    iso  = float(row.get("iso_score", 0))
    lof  = float(row.get("lof_score", 0))
    svm  = float(row.get("ocsvm_score", 0))
    gmm  = float(row.get("gmm_score", 0))
    rule = float(row.get("rule_score", 0))
    print(f"  IF={iso:.0f}  LOF={lof:.0f}  SVM={svm:.0f}  GMM={gmm:.0f}  Rule={rule:.0f}")
    print()


def append_label(puuid: str, player: str, label: int, row: pd.Series):
    """labeled_data.csv に1行追記"""
    write_header = not LABELED_OUTPUT.exists()
    cols_to_save = [
        "puuid", "player", "smurf_score", "judgment", "confidence",
        "account_level", "current_tier", "current_rank",
        "matches_count", "avg_kd", "avg_kda", "avg_hs_pct",
        "win_rate", "avg_dpr", "avg_kpr",
        "kd_rank_deviation", "hs_rank_deviation", "hs_kd_compound",
        "kd_bimodal_score", "tank_streak_max", "kd_cliff_drop_rate", "intentional_loss_rate",
        "rank_gap", "perf_rank_ratio_kd", "perf_level_ratio",
        "peak_tier_v3", "peak_current_gap_v3",
        "iso_score", "lof_score", "ocsvm_score", "gmm_score", "rule_score",
    ]
    record = {"puuid": puuid, "player": player, "label": label}
    for col in cols_to_save:
        if col not in record:
            record[col] = row.get(col, "")

    with open(LABELED_OUTPUT, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def show_stats(labeled_path: Path):
    """現在のラベリング統計を表示"""
    if not labeled_path.exists():
        print("  まだラベルデータがありません。")
        return
    df = pd.read_csv(labeled_path)
    n_smurf  = (df["label"] == 1).sum()
    n_normal = (df["label"] == 0).sum()
    total    = len(df)
    print()
    print(_c(f"  ラベリング済み: {total}件  (スマーフ: {n_smurf}, 通常: {n_normal})", CYAN))
    if total >= 30:
        ratio = n_smurf / total
        print(f"  スマーフ率: {ratio:.0%}  {'← 教師あり学習に十分です！' if total >= 50 else '← あと少し頑張りましょう'}")
    print()


def main():
    print()
    print(_c("╔══════════════════════════════════════════════════╗", CYAN))
    print(_c("║   VALORANT スマーフ ラベリングツール            ║", CYAN))
    print(_c("╚══════════════════════════════════════════════════╝", CYAN))
    print()
    print(f"  対象: スマーフスコア {SUSPECT_THRESHOLD}+ のプレイヤー")
    print(f"  操作: [1]=スマーフ  [0]=通常  [s]=スキップ  [?]=統計  [q]=終了")
    print()

    suspects      = load_suspects()
    already_done  = load_already_labeled()
    remaining     = suspects[~suspects["puuid"].astype(str).isin(already_done)]

    print(f"  怪しいプレイヤー数  : {len(suspects)}")
    print(f"  ラベリング済み      : {len(already_done)}")
    print(f"  残り作業            : {len(remaining)}")
    print()

    if len(remaining) == 0:
        print(_c("  全プレイヤーのラベリングが完了しています！", GREEN))
        print(f"  次は: python train_supervised.py")
        show_stats(LABELED_OUTPUT)
        return

    show_stats(LABELED_OUTPUT)
    input("  Enterキーで開始...")

    labeled_this_session = 0
    skipped = 0
    total_to_do = min(len(remaining), MAX_PER_SESSION)

    for idx, (_, row) in enumerate(remaining.iterrows(), 1):
        if labeled_this_session + skipped >= MAX_PER_SESSION:
            break

        puuid  = str(row.get("puuid", ""))
        player = str(row.get("player", "?"))

        display_player(row, idx, total_to_do)

        while True:
            try:
                raw = input(_c("  判定 [1=スマーフ / 0=通常 / s=スキップ / ?=統計 / q=終了]: ", BOLD)).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  中断しました。")
                show_stats(LABELED_OUTPUT)
                return

            if raw == "q":
                print()
                show_stats(LABELED_OUTPUT)
                print(f"  次は: python train_supervised.py")
                return
            elif raw == "?":
                show_stats(LABELED_OUTPUT)
                continue
            elif raw == "s":
                skipped += 1
                break
            elif raw in ("0", "1"):
                label = int(raw)
                append_label(puuid, player, label, row)
                labeled_this_session += 1
                verdict = _c("スマーフ", RED) if label == 1 else _c("通常", GREEN)
                print(f"  → {verdict} としてラベリング完了")
                break
            else:
                print("  ※ 1, 0, s, ?, q のいずれかを入力してください")

    print()
    print(_c("  セッション終了！", GREEN))
    show_stats(LABELED_OUTPUT)

    already_done_now = load_already_labeled()
    n_done = len(already_done_now)
    if n_done >= 50:
        print(_c(f"  {n_done}件のラベルがあります。教師あり学習を実行できます！", GREEN))
        print(_c("  次のコマンドを実行: python train_supervised.py", BOLD))
    else:
        need = 50 - n_done
        print(f"  教師あり学習には最低50件必要です。あと {need} 件ラベリングしてください。")


if __name__ == "__main__":
    main()
