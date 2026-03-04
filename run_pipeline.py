"""
run_pipeline.py  - 全自動スマーフ検出パイプライン
===================================================
このスクリプト1本で全ステップを自動実行します。

実行:
    python run_pipeline.py          # 通常実行 (データ収集スキップ)
    python run_pipeline.py --collect # 新規プレイヤーを収集してから実行

流れ:
    [自動] 1. (--collect時) collector.py でデータ収集
    [自動] 2. update_data.py で既存JSONを最新データに更新
    [自動] 3. smurf_ai.py で教師なし検出
    ★手動★ 4. label_tool.py でラベリング (50件以上推奨)
    [自動] 5. labeled_data.csv があれば train_supervised.py で教師ありモデル訓練
    [自動] 6. 教師ありモデルがあれば smurf_ai.py を再実行してブレンド

ステップ4だけ人間の作業が必要です。
    → python label_tool.py を実行し 1/0 でラベリングしてください。
    → 50件以上ラベルしたら、このスクリプトを再実行すると自動で続きをやります。
"""

import subprocess
import sys
import time
from pathlib import Path

# Windowsターミナルの文字コード問題対策
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

PYTHON = sys.executable
LABELED_CSV     = Path("labeled_data.csv")
MODEL_PATH      = Path("smurf_model.pkl")
FULL_RESULTS    = Path("smurf_output/smurf_results_full.csv")
MIN_LABELS      = 30   # 教師あり学習の最低ライン

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def c(text, color): return f"{color}{text}{RESET}"
def header(title):
    print()
    print(c("=" * 60, CYAN))
    print(c(f"  {title}", BOLD))
    print(c("=" * 60, CYAN))

def run(script: str, label: str) -> bool:
    """サブプロセスでスクリプトを実行。成功=True"""
    header(f"STEP: {label}")
    t0 = time.time()
    import os
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([PYTHON, script], check=False, env=env)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(c(f"  [OK] 完了 ({elapsed:.0f}秒)", GREEN))
        return True
    else:
        print(c(f"  [!!] エラー (終了コード {result.returncode})", RED))
        return False

def count_labels() -> tuple[int, int, int]:
    """(総数, スマーフ数, 通常数) を返す"""
    if not LABELED_CSV.exists():
        return 0, 0, 0
    try:
        import pandas as pd
        df = pd.read_csv(LABELED_CSV)
        n_s = int((df["label"] == 1).sum())
        n_n = int((df["label"] == 0).sum())
        return len(df), n_s, n_n
    except Exception:
        return 0, 0, 0

def main():
    do_collect = "--collect" in sys.argv

    print()
    print(c("╔══════════════════════════════════════════════════════════╗", CYAN))
    print(c("║    VALORANT スマーフ検出 - 全自動パイプライン            ║", CYAN))
    print(c("╠══════════════════════════════════════════════════════════╣", CYAN))
    print(c("║  python run_pipeline.py          # 検出のみ              ║", CYAN))
    print(c("║  python run_pipeline.py --collect # データ収集から実行   ║", CYAN))
    print(c("╚══════════════════════════════════════════════════════════╝", CYAN))

    n_labels, n_smurf, n_normal = count_labels()
    model_exists = MODEL_PATH.exists()

    print()
    print(c("  現在の状態:", BOLD))
    print(f"    ラベルデータ  : {n_labels}件 (スマーフ:{n_smurf} / 通常:{n_normal})")
    print(f"    教師ありモデル: {'あり ✅' if model_exists else 'なし (ステップ5で作成)'}")
    print()

    # ─── ステップ1: データ収集 (オプション) ───
    if do_collect:
        ok = run("collector.py", "1/5  新規プレイヤーデータ収集 (collector.py)")
        if not ok:
            print(c("  collector.py で問題が発生しました。続行します。", YELLOW))

    # ─── ステップ2: 既存データ更新 ───
    if Path("update_data.py").exists():
        ok = run("update_data.py", "2/5  既存JSONデータ更新 (update_data.py)")
        if not ok:
            print(c("  update_data.py で問題が発生しました。続行します。", YELLOW))
    else:
        print(c("\n  [SKIP] update_data.py が見つかりません", YELLOW))

    # ─── ステップ3: 教師なし検出 ───
    ok = run("smurf_ai.py", "3/5  教師なし異常検知 (smurf_ai.py)")
    if not ok:
        print(c("  smurf_ai.py が失敗しました。処理を中断します。", RED))
        sys.exit(1)

    # ─── ステップ4: ラベリング状況を確認 ───
    n_labels, n_smurf, n_normal = count_labels()  # 再確認
    header("4/5  ラベリング状況確認")

    if n_labels < MIN_LABELS:
        need = MIN_LABELS - n_labels
        print(c(f"  ⚠  教師ありモデルの学習には最低{MIN_LABELS}件のラベルが必要です", YELLOW))
        print(f"     現在: {n_labels}件  あと {need} 件必要")
        print()
        print(c("  ★ あなたがやること (1回だけ) ★", BOLD))
        print(c("  ─────────────────────────────────────────────────────", YELLOW))
        print(c("    python label_tool.py", BOLD))
        print()
        print("  スマーフっぽいプレイヤーの一覧が表示されます。")
        print("  1 または 0 を押してラベルをつけてください。")
        print("  50件ラベルしたら q で終了し、このスクリプトを再実行：")
        print(c("    python run_pipeline.py", BOLD))
        print(c("  ─────────────────────────────────────────────────────", YELLOW))
        print()
        if FULL_RESULTS.exists():
            import pandas as pd
            df = pd.read_csv(FULL_RESULTS)
            df["smurf_score"] = pd.to_numeric(df["smurf_score"], errors="coerce").fillna(0)
            suspects = df[df["smurf_score"] >= 55]
            print(f"  ラベリング候補: {len(suspects)}件 (スマーフスコア55+)")
        print()
        print(c("  教師なし検出の結果は smurf_output/ に保存済みです。", GREEN))
        sys.exit(0)

    else:
        print(c(f"  [OK] {n_labels}件のラベルがあります (スマーフ:{n_smurf} / 通常:{n_normal})", GREEN))

    # ─── ステップ5: 教師ありモデル訓練 ───
    ok = run("train_supervised.py", "5/5  教師ありモデル訓練 (train_supervised.py)")
    if not ok:
        print(c("  train_supervised.py が失敗しました。", RED))
        sys.exit(1)

    # ─── ステップ6: 教師ありモデルが揃ったので再推論 ───
    if MODEL_PATH.exists():
        if n_labels >= MIN_LABELS:
            header("6/6  教師ありモデルでスコアを再計算 (smurf_ai.py)")
            print("  (教師ありモデルのブレンドスコアで最終結果を生成します)")
            ok = run("smurf_ai.py", "最終推論 - smurf_ai.py")
            if not ok:
                print(c("  再推論に失敗しました。", RED))

    # ─── 完了サマリー ───
    print()
    print(c("=" * 60, GREEN))
    print(c("  *** 全パイプライン完了! ***", BOLD))
    print(c("=" * 60, GREEN))
    print()
    print("  出力ファイル:")
    for f in [
        "smurf_output/smurf_results_full.csv  ← 全プレイヤーの詳細スコア",
        "smurf_output/smurf_suspects.csv      ← 怪しいプレイヤーだけ抽出",
        "smurf_output/01_overview.png         ← 可視化グラフ",
        "smurf_output/report.txt              ← テキストサマリー",
    ]:
        print(f"    {f}")
    if MODEL_PATH.exists():
        print(f"    smurf_model.pkl               ← 教師ありモデル (次回から自動使用)")
    print()
    print("  ラベルを増やすほど精度が上がります:")
    print(f"    現在  {n_labels}件 → python label_tool.py で追加 → python run_pipeline.py で再学習")
    print()


if __name__ == "__main__":
    main()
