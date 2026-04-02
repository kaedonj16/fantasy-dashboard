"""
LLM-based projection engine for breakout candidates.

Uses AI to project full stat lines based on role changes and previous usage.
"""

import json
from typing import Dict, Any, Optional
import os

# Try to import anthropic, but gracefully fall back if not available
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def project_player_stats(
    player_info: Dict[str, Any],
    previous_usage: Dict[str, Any],
    efficiency_metrics: Optional[Dict[str, Any]],
    role_change: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use Claude to project player stats based on role changes.

    Args:
        player_info: {position, team, age}
        previous_usage: Previous season usage stats
        efficiency_metrics: Efficiency metrics (yards per carry, etc.)
        role_change: Expected changes in usage (carries_delta, targets_delta, etc.)

    Returns:
        {
            "projected_usage": {...},
            "projected_stats": {...},
            "fantasy_points": {...},
            "efficiency_adjustments": {...},
            "confidence": 0-100,
            "notes": "..."
        }
    """

    # Build the prompt with all context
    system_prompt = """You are a fantasy football projection engine.

Your job is to project a player's next-season production based on a change in role.

Inputs:
- player_info:
    - position
    - team
    - age (if available)
- previous_usage:
    - games
    - snap_share
    - carries
    - targets
    - receptions
    - rush_yards
    - rec_yards
    - rush_tds
    - rec_tds
    - pass_attempts (QB)
    - pass_yards (QB)
    - pass_tds (QB)
    - interceptions (QB)
- efficiency_metrics (if available):
    - yards_per_carry
    - yards_per_target
    - catch_rate
    - td_rate (rush/rec/pass)
- role_change:
    - carries_delta
    - targets_delta
    - routes_delta (if applicable)
    - snap_share_delta (optional)
    - pass_attempts_delta (QB only)

Instructions:

1. Establish baseline per-game and per-touch efficiency using previous_usage.
   - If efficiency is missing or sample size is small, regress toward reasonable positional averages.

2. Apply role changes:
   - Add the deltas (e.g., +75 carries, +30 targets) to previous totals.
   - Distribute increases across games (do NOT assume perfect health; cap at 17 games).

3. Apply diminishing returns:
   - As volume increases, slightly reduce efficiency (e.g., 3–10% drop depending on size of increase).
   - Larger increases = larger efficiency regression.

4. Account for role realism:
   - RB: carries and targets should scale differently (targets more volatile)
   - WR/TE: targets drive everything; receptions depend on catch rate
   - QB: pass attempts drive yards and TDs, but TD rate may regress

5. Recalculate full stat line:
   - carries, targets, receptions
   - yards (rush/rec/pass)
   - touchdowns
   - turnovers (QB)

6. Output fantasy scoring:
   - PPR points
   - Half PPR
   - Standard
   - Points per game

7. Include a confidence adjustment:
   - Lower confidence if:
       - previous sample size is small
       - role increase is very large
       - efficiency was previously extreme (unsustainable)

Output format (JSON only, no markdown):
{
  "projected_usage": {
    "games": 16,
    "carries": 180,
    "targets": 50,
    "receptions": 40,
    "snap_share": 0.65
  },
  "projected_stats": {
    "rush_yards": 750,
    "rush_tds": 6,
    "rec_yards": 350,
    "rec_tds": 2,
    "pass_yards": 0,
    "pass_tds": 0,
    "interceptions": 0
  },
  "fantasy_points": {
    "ppr_total": 198.0,
    "ppr_ppg": 12.4,
    "half_ppr_total": 178.0,
    "half_ppr_ppg": 11.1,
    "standard_total": 158.0,
    "standard_ppg": 9.9
  },
  "efficiency_adjustments": {
    "yards_per_touch_change": -0.2,
    "td_rate_change": -0.01
  },
  "confidence": 75,
  "notes": "Moderate confidence. Role increase is significant but player has shown efficiency in limited sample. Slight regression expected due to increased volume."
}

IMPORTANT: Return ONLY valid JSON. Do not include markdown code blocks or any other formatting."""

    user_prompt = f"""Project stats for this player:

Player Info:
{json.dumps(player_info, indent=2)}

Previous Usage (last season):
{json.dumps(previous_usage, indent=2)}

Efficiency Metrics:
{json.dumps(efficiency_metrics or {}, indent=2)}

Role Change (expected deltas):
{json.dumps(role_change, indent=2)}

Return your projection as JSON following the exact format specified in the system prompt."""

    try:
        # Check if anthropic library is available
        if not HAS_ANTHROPIC:
            return _fallback_projection(player_info, previous_usage, role_change)

        # Initialize Claude client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # Fallback to simple projection if no API key
            return _fallback_projection(player_info, previous_usage, role_change)

        client = anthropic.Anthropic(api_key=api_key)

        # Call Claude with the projection prompt
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.3,  # Lower temperature for more consistent projections
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract JSON from response
        response_text = message.content[0].text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            # Find the actual JSON content
            lines = response_text.split("\n")
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (not line.strip().startswith("```") and "{" in response_text):
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        # Parse JSON response
        projection = json.loads(response_text)

        return projection

    except Exception as e:
        print(f"[projections] LLM projection failed: {e}")
        # Fallback to simple projection
        return _fallback_projection(player_info, previous_usage, role_change)


def _fallback_projection(
    player_info: Dict[str, Any],
    previous_usage: Dict[str, Any],
    role_change: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simple fallback projection when LLM is unavailable.

    Uses basic math to estimate stats based on role changes.
    """
    position = player_info.get('position', 'RB')

    # Previous stats
    prev_games = previous_usage.get('games', 16)
    prev_carries = previous_usage.get('carries', 0)
    prev_targets = previous_usage.get('targets', 0)
    prev_receptions = previous_usage.get('receptions', 0)
    prev_rush_yards = previous_usage.get('rush_yards', 0) or previous_usage.get('avg_rush_yards', 0) * prev_games
    prev_rec_yards = previous_usage.get('rec_yards', 0) or previous_usage.get('avg_rec_yards', 0) * prev_games
    prev_rush_tds = previous_usage.get('rush_tds', 0) or previous_usage.get('avg_rush_tds', 0) * prev_games
    prev_rec_tds = previous_usage.get('rec_tds', 0) or previous_usage.get('avg_rec_tds', 0) * prev_games
    prev_snap_share = previous_usage.get('snap_share', 0) or previous_usage.get('avg_off_snap_pct', 0)

    # Role changes
    carries_delta = role_change.get('carries_delta', 0)
    targets_delta = role_change.get('targets_delta', 0)
    snap_share_delta = role_change.get('snap_share_delta', 0.1)

    # Calculate efficiency (with fallbacks)
    ypc = (prev_rush_yards / prev_carries) if prev_carries > 0 else 4.2
    yards_per_target = (prev_rec_yards / prev_targets) if prev_targets > 0 else 8.5
    catch_rate = (prev_receptions / prev_targets) if prev_targets > 0 else 0.65
    rush_td_rate = (prev_rush_tds / prev_carries) if prev_carries > 0 else 0.04
    rec_td_rate = (prev_rec_tds / prev_targets) if prev_targets > 0 else 0.06

    # Apply diminishing returns for large increases
    volume_increase_pct = (carries_delta + targets_delta) / max(prev_carries + prev_targets, 1)
    efficiency_penalty = min(volume_increase_pct * 0.05, 0.10)  # Max 10% efficiency drop

    ypc_adjusted = ypc * (1 - efficiency_penalty)
    yards_per_target_adjusted = yards_per_target * (1 - efficiency_penalty)

    # Projected stats (assume 16 games)
    proj_games = 16
    proj_carries = prev_carries + carries_delta
    proj_targets = prev_targets + targets_delta
    proj_receptions = proj_targets * catch_rate
    proj_snap_share = min(prev_snap_share + snap_share_delta, 0.95)

    proj_rush_yards = proj_carries * ypc_adjusted
    proj_rec_yards = proj_targets * yards_per_target_adjusted
    proj_rush_tds = proj_carries * rush_td_rate
    proj_rec_tds = proj_targets * rec_td_rate

    # Fantasy points
    ppr_total = (
        proj_rush_yards * 0.1 +
        proj_rush_tds * 6 +
        proj_rec_yards * 0.1 +
        proj_rec_tds * 6 +
        proj_receptions * 1
    )
    half_ppr_total = ppr_total - (proj_receptions * 0.5)
    standard_total = ppr_total - proj_receptions

    return {
        "projected_usage": {
            "games": proj_games,
            "carries": int(proj_carries),
            "targets": int(proj_targets),
            "receptions": int(proj_receptions),
            "snap_share": round(proj_snap_share, 3)
        },
        "projected_stats": {
            "rush_yards": int(proj_rush_yards),
            "rush_tds": round(proj_rush_tds, 1),
            "rec_yards": int(proj_rec_yards),
            "rec_tds": round(proj_rec_tds, 1),
            "pass_yards": 0,
            "pass_tds": 0,
            "interceptions": 0
        },
        "fantasy_points": {
            "ppr_total": round(ppr_total, 1),
            "ppr_ppg": round(ppr_total / proj_games, 1),
            "half_ppr_total": round(half_ppr_total, 1),
            "half_ppr_ppg": round(half_ppr_total / proj_games, 1),
            "standard_total": round(standard_total, 1),
            "standard_ppg": round(standard_total / proj_games, 1)
        },
        "efficiency_adjustments": {
            "yards_per_touch_change": round(-efficiency_penalty * ypc, 2),
            "td_rate_change": 0.0
        },
        "confidence": 60,  # Medium confidence for fallback
        "notes": "Fallback projection (LLM unavailable). Basic math applied with diminishing returns."
    }
