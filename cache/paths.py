from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PLAYER_HISTORY_DIR = CACHE_DIR / "player_history"
PLAYER_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

PLAYER_INVESTMENT_DIR = CACHE_DIR / "player_investment"
PLAYER_INVESTMENT_DIR.mkdir(parents=True, exist_ok=True)
