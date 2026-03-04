"""
python -m smurf <PUUID>  で実行できるCLIエントリポイント
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from . import check


def _bar(prob: float) -> str:
    filled = round(prob * 20)
    return "█" * filled + "░" * (20 - filled)


def main():
    if len(sys.argv) >= 2:
        puuid = sys.argv[1].strip()
    else:
        print("VALORANT スマーフ判定ツール (教師ありモデル)")
        print("-" * 44)
        puuid = input("PUUID を入力 > ").strip()

    if not puuid:
        print("[ERROR] PUUIDが空です")
        sys.exit(1)

    print(f"\n🔍  {puuid}")
    print("  データ取得中...", end=" ", flush=True)

    try:
        r = check(puuid)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print("完了\n")
    prob  = r["prob"]
    score = r["score"]

    print("=" * 56)
    print("  VALORANT スマーフ判定結果 (教師ありモデル)")
    print("=" * 56)
    print(f"  プレイヤー     : {r['player']}")
    print()
    print(f"  ━━━ 判定 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {r['judgment']}")
    print(f"  スマーフ確率   : {score:.1f}%  [{_bar(prob)}]")
    print()
    print(f"  ━━━ プロフィール ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  現在ランク     : {r['current_rank']}")
    print(f"  最高ランク     : {r['highest_rank']}")
    print(f"  アカウントLv   : {r['account_level']}")
    print(f"  試合数         : {r['matches_count']}")
    print()
    print(f"  ━━━ 戦闘指標 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━─")
    print(f"  平均KD         : {r['avg_kd']:.2f}")
    print(f"  平均HS%        : {r['avg_hs_pct']*100:.1f}%")
    print(f"  平均ダメージ   : {r['avg_dpr']:.0f}")
    print(f"  勝率           : {r['win_rate']*100:.1f}%")
    print(f"  KDランク偏差   : {r['kd_rank_deviation']:.2f}")
    print()
    print(f"  ━━━ スマーフシグナル ━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  ランク乖離     : {r['rank_gap']} 段階")
    print(f"  ランク降下傾向 : {r['tier_trend']:.3f}  (負=下降中)")
    print(f"  意図的負け率   : {r['intentional_loss_rate']*100:.1f}%")
    print(f"  連続タンク     : {r['tank_streak_max']} 試合")
    print(f"  KD急落率       : {r['kd_cliff_drop_rate']:.3f}")
    print(f"  KD二峰性スコア : {r['kd_bimodal_score']:.3f}")
    print(f"  ルールスコア   : {r['rule_score']:.0f} / 100")
    print()
    print(f"  ━━━ モデル情報 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    auc = f"{r['model_auc']:.4f}" if r["model_auc"] else "N/A"
    print(f"  訓練AUC        : {auc}")
    print(f"  ラベル数       : {r['n_labeled']} 件 (スマーフ: {r['n_smurf']} 件)")
    print(f"  訓練日         : {r['trained_at']}")
    print("=" * 56)

    # 注目ポイント
    tips = []
    if r["kd_rank_deviation"] > 2.0:
        tips.append("→ KDがランク平均より大幅に高い")
    if r["rank_gap"] >= 6:
        tips.append(f"→ 現在ランクと最高ランクが {r['rank_gap']} 段階乖離")
    if r["intentional_loss_rate"] > 0.3:
        tips.append(f"→ 意図的な負け行動が {r['intentional_loss_rate']*100:.0f}% のマッチで疑われる")
    if r["tank_streak_max"] >= 5:
        tips.append(f"→ {r['tank_streak_max']} 連続タンクパターンを検出")
    if tips:
        print("\n  ⚠️  注目ポイント:")
        for t in tips:
            print(f"     {t}")
        print()


if __name__ == "__main__":
    main()
