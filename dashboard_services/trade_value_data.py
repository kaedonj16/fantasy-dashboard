# dashboard_services/trade_value_data.py

from typing import Dict, Any

from dashboard_services.utils import load_players_index, load_teams_index


def load_player_metadata() -> Dict[str, dict]:
    """
    Wrapper around whatever your players index is.
    Expected shape (example):
    {
      "1234": {"name": "Justin Jefferson", "position": "WR", "team": "MIN", "age": 25, ...},
      ...
    }
    """
    return load_players_index()


def load_team_metadata() -> Dict[str, dict]:
    """
    Wrapper around your teams index.
    Example expected:
    {
      "MIN": {"teamId": "8", "byeWeek": 6, ...},
      ...
    }
    """
    return load_teams_index()


def load_projections_for_season(season: int) -> Dict[str, float]:
    """
    Return projected fantasy points in your league's scoring
    keyed by Sleeper player_id (or whatever id you use in players_index).

    Example:
    { "1234": 285.3, "5678": 212.1, ... }

    TODO: wire this into your Tank01 or existing projections pipeline.
    """
    projections: Dict[str, float] = {}

    # Pseudocode example if you have a projections table already:
    # for row in tank01_projection_rows:
    #     sleeper_id = row["sleeper_id"]
    #     projections[sleeper_id] = row["projected_fpts"]

    return projections


def load_usage_metrics(season: int) -> Dict[str, dict]:
    """
    Build per-player usage metrics from your stats data.

    Expected output shape:
    {
      "1234": {
        "snap_share": 0.86,
        "target_share": 0.28,
        "carry_share": 0.05,
        "route_participation": 0.94,
        "redzone_share": 0.30,
        "injury_risk_flag": False,
        "suspension_risk_flag": False,
      },
      ...
    }

    TODO: implement using your stats source (Sleeper or CSVs).
    """
    usage: Dict[str, dict] = {}

    # Pseudocode:
    # for each team:
    #   team_snaps = sum(snaps for all players on team in window)
    #   team_pass_attempts = ...
    #   team_rush_attempts = ...
    #   team_dropbacks = ...
    #   team_rz_opps = ...
    #
    #   for each player:
    #       usage[player_id] = {
    #           "snap_share": player_snaps / team_snaps,
    #           "target_share": player_targets / team_pass_attempts,
    #           "carry_share": player_carries / team_rush_attempts,
    #           "route_participation": player_routes / team_dropbacks,
    #           "redzone_share": (rz_targets + rz_carries) / team_rz_opps,
    #       }

    return usage


def load_team_context(season: int) -> Dict[str, dict]:
    """
    Returns team-level context for each NFL team.

    Example:
    {
      "MIN": {
        "offense_tier": 2,
        "pass_rate": 0.62,
        "plays_per_game": 66.5,
      },
      ...
    }

    You can base offense_tier on points per game or EPA and bucket into 4 tiers.
    """
    context: Dict[str, dict] = {}

    # Pseudocode:
    # for each team:
    #   points_per_game = ...
    #   rank teams by points_per_game and assign tier 1–4
    #   pass_rate = team_dropbacks / (team_dropbacks + team_rush_attempts)
    #   plays_per_game = total_plays / games
    #
    #   context[team_abbr] = {
    #       "offense_tier": tier,
    #       "pass_rate": pass_rate,
    #       "plays_per_game": plays_per_game,
    #   }

    return context
