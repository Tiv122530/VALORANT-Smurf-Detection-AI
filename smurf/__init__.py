"""
smurf  - VALORANT スマーフ検出パッケージ (教師ありモデル)
=========================================================
使い方:
    from smurf import check

    result = check("26d91571-3e25-5727-b3c5-99563f4cddf8")
    print(result["judgment"])   # 🔴 スマーフ確定
    print(result["score"])      # 99.5
    print(result["prob"])       # 0.995

CLIとして実行:
    python -m smurf <PUUID>
    python -m smurf                  ← 対話入力
"""

from .checker import check

__all__ = ["check"]
__version__ = "1.0.0"
