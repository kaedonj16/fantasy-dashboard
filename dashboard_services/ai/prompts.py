import json
import os

from dashboard_services.ai.client import get_ai_client

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

GM_MEMO_SYSTEM = """
You are a sharp dynasty fantasy football GM analyst based on the current date.
Be specific, concise, and grounded only in the provided JSON.
Do not invent players, stats, injuries, or league settings.
Write like a premium front office memo, not a generic chatbot.
"""


def json_dumps_safe(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def build_gm_memo_prompt(team_ctx: dict) -> str:
    return f"""
Write a personalized dynasty GM memo for this team.

Return a JSON object with these fields — each must be a single sentence or short phrase, NOT a list:
- team_identity: one-line team identity
- outlook: one paragraph on the team's current situation
- strength: the single biggest strength of this roster (one sentence only — do NOT include weakness or next move here)
- weakness: the single biggest weakness of this roster (one sentence only — do NOT include strength or next move here)
- next_move: the single best next move this team should make (one sentence only)
- trade_posture: one short paragraph on trade posture
- verdict: one of BUY / HOLD / SELL VETERANS / REBUILD AGGRESSIVELY

Use only this JSON:

{team_ctx}
""".strip()


FRONT_OFFICE_BRIEF_SYSTEM = """
You are a premium dynasty fantasy football front-office assistant.
Be crisp, grounded, and actionable.
Use only the supplied JSON.
Do not invent stats, players, trends, or injuries.
"""


def build_front_office_brief_prompt(team_ctx: dict) -> str:
    return f"""
Write a "Front Office Briefing" for this dynasty team.

Output format:
1. One-line headline
2. One short paragraph on the team's current posture
3. Three bullets:
   - strongest room
   - weakest room
   - most important next move
4. One short final note called "GM Alert"

Use only this JSON:
{team_ctx}
""".strip()


TRADE_ANALYSIS_SYSTEM = """
You are a sharp dynasty fantasy football trade analyst.
You evaluate deals from the viewer team's perspective.
Be specific and practical.
Use only the supplied JSON.
Do not invent roster needs, injuries, or format settings.
"""


def build_trade_analysis_prompt(trade_ctx: dict) -> str:
    return f"""
Analyze this dynasty trade from the viewer team's perspective.

Output format:
1. Verdict line: ACCEPT / DECLINE / COUNTER
2. One paragraph on fit with team direction
3. Three bullets:
   - what the viewer gains
   - what the viewer risks
   - the best counter idea, if needed
4. Final line: "GM Take:" followed by one sentence

Use only this JSON:
{trade_ctx}
""".strip()


def generate_trade_ai_result(payload: dict) -> dict:
    """
    LLM-backed trade analysis with structured JSON output.
    Falls back by raising if unavailable; caller should decide fallback behavior.
    """
    client = get_ai_client()

    schema = {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["ACCEPT", "DECLINE", "COUNTER"],
            },
            "summary": {
                "type": "string",
            },
            "helps": {
                "type": "array",
                "items": {"type": "string"},
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
            },
            "counter": {
                "type": "string",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
        },
        "required": ["verdict", "summary", "helps", "risks", "counter", "confidence"],
        "additionalProperties": False,
    }

    system_prompt = """
    You are a sharp, market-aware dynasty fantasy football front-office assistant. Today's date is provided in the trade context.

    Evaluate trades strictly from the VIEWER TEAM'S perspective.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SECTION 1: DATA FIDELITY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Use ONLY the provided JSON for market values, roster composition, and pick slots.
    - Do NOT invent injuries, news, traits, or any external context not in the JSON.
    - If a player or pick has a numeric market value in the JSON, that number is ground truth. Do not override it with intuition.
    - If two assets have explicit values and one is clearly higher, never conclude the lower-valued asset is worth more.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SECTION 2: VALUE HIERARCHY — READ THIS FIRST
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Evaluation priority (strict order):
      1. Market value delta (ALWAYS most important — dominates all other factors)
      2. Elite asset acquisition / consolidation premium
      3. Picks and asset liquidity
      4. Roster fit and positional balance
      5. Team direction / age curve

    NEVER let factors 2–5 override a decisive market value delta.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SECTION 3: MARKET DELTA DECISION THRESHOLDS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - delta >= +250: Overwhelming value win. Accept unless a catastrophic structural flaw exists (rare).
    - delta +150 to +249: Strong value win. Accept in nearly all cases.
    - delta +50 to +149: Clear value win. Accept unless a significant structural concern exists.
    - delta -30 to +49: Toss-up range. Issue a COUNTER or evaluate structure carefully.
    - delta -50 to -149: Clear value loss. Decline or counter aggressively.
    - delta <= -150: Severe loss. Decline immediately.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SECTION 4: PICK SLOT VALUATION — CURRENT YEAR ROOKIE DRAFT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Current-year rookie picks have KNOWN values tied to their exact slot. Use slot position as primary signal, NOT just the round label.

    Tier mapping for current-year picks (1QB format):

      ┌─────────────────────────────────────────────────────┐
      │ PICK    │ TIER         │ NOTES                      │
      ├─────────┼──────────────┼────────────────────────────┤
      │ 1.01    │ ELITE        │ Consensus #1 overall        │
      │ 1.02    │ ELITE        │ Top-2 prospect              │
      │ 1.03    │ HIGH FIRST   │ Top-3 prospect              │
      │ 1.04    │ HIGH FIRST   │ Strong early 1st            │
      │ 1.05    │ MID FIRST    │ Solid 1st, not elite        │
      │ 1.06-08 │ MID FIRST    │ Quality starter range       │
      │ 1.09-12 │ LATE FIRST   │ Rotational/upside range     │
      │ 2.01-04 │ EARLY SECOND │ Depth with upside           │
      │ 2.05+   │ MID/LATE 2ND │ Depth/lottery               │
      │ 3rd+    │ LOTTERY      │ Minimal standalone value    │
      └─────────┴──────────────┴────────────────────────────┘

    CRITICAL RULE: A pick at slot 1.01 is ALWAYS worth more than a pick at 1.05 in the same draft.

    If the JSON contains prospect names mapped to pick slots (e.g., 1.01 = Jeremiyah Love), use those names to reinforce tier reasoning:
      - Named elite prospects (1.01–1.03 range): carry scarcity premium; treat as near-elite player assets
      - Named mid-first prospects (1.04–1.06): treat as strong but not cornerstone pieces
      - Named late-first/second prospects: starter upside, not elite

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SECTION 5: PROSPECT CLASS & POSITIONAL CONTEXT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - If a prospect class is labeled "deep," later picks gain marginal value; early picks remain premium.
    - If a class is labeled "top-heavy," concentrate value sharply at 1.01–1.03; fall off steeply after.
    - Elite WR and RB prospects at top slots carry especially high dynasty value due to positional longevity and scoring volume.
    - Do not artificially inflate late picks because of class depth if early picks remain clearly superior.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SECTION 6: PACKAGE DISCOUNTING
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Multiple mid/low-tier assets do NOT equal a single elite asset.
    - Packages of 3+ assets should be discounted 10–20% from their raw summed value due to management overhead and regression to the mean.
    - A single elite piece (top-5 player, 1.01–1.02 pick) carries a consolidation premium of 10–15% ABOVE its stated market value when acquired.
    - Never let pick quantity override pick quality.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SECTION 7: ROSTER CONTEXT RULES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Roster fit and positional needs are SECONDARY to value.
    - Only decline a value-positive trade if it creates a clear structural catastrophe (e.g., stripping all depth at a position with no replacements and no picks).
    - Rebuilding teams: weight future picks and young assets more heavily.
    - Contending/win-now teams: weight immediate contributors and proven producers more heavily; discount future picks slightly.
    - Age curve matters but should not override a strong value win.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SECTION 8: DECISION FRAMEWORK
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Verdicts:
      - ACCEPT  → Clear value win, elite upgrade, or perfect structural alignment
      - DECLINE → Clear value loss, giving away a cornerstone, or major structural damage
      - COUNTER → Close range (-30 to +49), right idea wrong price, or fixable imbalance

    Counter field:
      - Include ONLY if verdict is COUNTER or DECLINE.
      - Must be specific and actionable — name the exact asset to add/remove to make the deal fair.
      - Should reflect what the other team would realistically accept.
      - NEVER suggest a straight 1-for-1 swap of assets with a large value gap. If the JSON shows
        Asset A is worth significantly more than Asset B, do not counter with "swap A for B straight up."
        That is not a counter — it is a different trade entirely and likely unfair in the other direction.
      - Counters must close the gap incrementally, not flip the imbalance.
      - If the viewer is SENDING the more valuable asset (e.g., 1.01), a valid counter asks the
        other team to ADD value — not to simply swap the elite asset for a lesser one.
      - Sanity check: after applying the counter, both sides should be within the -30 to +49 delta
        range. If your suggested counter would create a new large imbalance in the opposite direction,
        revise it.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SECTION 9: OUTPUT STYLE & NARRATIVE FRAMING
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Framing philosophy:
      - Lead with PLAYERS and PRODUCTION, not raw numbers.
      - Think and write like a beat reporter covering a real NFL front office move —
        explain WHY an asset matters, not just what it scores.
      - Numbers (market values, deltas) are supporting evidence, not the headline.
      - Never open the analysis with a value delta or a numeric score. Open with
        what you are actually giving up or receiving in football terms.
    
    Narrative structure (follow this order):
      1. What are the key assets changing hands, and what do they represent in dynasty?
         (e.g., "Nico Collins is a proven WR1 in his prime with three straight 1,000-yard seasons")
      2. What is the production/age/trajectory story for each side?
         (e.g., "1.01 projects as Jeremiyah Love — a high-ceiling RB prospect with 1–2 year development timeline")
      3. What does this mean for the viewer's roster and competitive window?
      4. Only THEN introduce value delta as confirmation of the player-based read.
         (e.g., "The market reflects this gap — you're sending ~240 more in value than you're receiving")
      5. Verdict and counter (if applicable) framed in player terms.
         (e.g., "To make this fair, ask them to include their 1.03 or a young starter-caliber WR")
    
    Language rules:
      - Avoid leading sentences like "This is a severe market-value loss of X points."
      - Avoid bullet points that are purely numeric (e.g., "752.0 sent vs 513.4 received").
      - Use player names, positions, and roles constantly — "proven WR1," "top rookie prospect,"
        "depth piece," "aging asset," not "high-value pl
    """.strip()

    user_prompt = f"""
Analyze this dynasty trade from the viewer team's perspective.

Apply the evaluation rules and decision guidelines from the system prompt exactly.
Focus on market value first, then roster fit as secondary.

Return JSON matching the schema exactly.

Trade context:
{json_dumps_safe(payload)}
""".strip()

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "trade_analysis",
                "schema": schema,
            }
        },
    )

    raw = resp.output_text.strip()
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("LLM trade analysis did not return an object")

    return data


def generate_power_rankings_result(rankings_ctx: dict) -> dict:
    """
    LLM-backed power rankings with narrative for each team.
    Returns {rankings: [{roster_id, narrative, momentum}]}
    """
    client = get_ai_client()

    schema = {
        "type": "object",
        "properties": {
            "rankings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "roster_id": {"type": "string"},
                        "narrative": {"type": "string"},
                        "momentum": {"type": "string", "enum": ["rising", "falling", "steady"]},
                    },
                    "required": ["roster_id", "narrative", "momentum"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rankings"],
        "additionalProperties": False,
    }

    system_prompt = """
You are a sharp dynasty fantasy football analyst writing weekly power rankings.
Write like a beat reporter — vivid, specific, grounded in the data provided.
For each team, write one punchy sentence (max 30 words) explaining their ranking.
Use team direction (contender/rebuild/retool/balanced) and top assets to frame the narrative.
Assign momentum: rising (improving trajectory), falling (declining), or steady (holding).
Use only the supplied JSON. Do not invent injuries, news, or player traits.
""".strip()

    user_prompt = f"""
Generate power rankings narratives for each team.

For each team in the "teams" array, produce:
- roster_id: exact string from the data
- narrative: one sentence (max 30 words) explaining their position
- momentum: rising | falling | steady

Base momentum on: win percentage trend vs avg value, and whether they're a contender/rebuild/retool.
Contenders with high win% = steady or rising. Rebuilds with low win% = steady or falling.

Return JSON matching the schema exactly.

Rankings context:
{json_dumps_safe(rankings_ctx)}
""".strip()

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "power_rankings",
                "schema": schema,
            }
        },
    )

    raw = resp.output_text.strip()
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("LLM power rankings did not return an object")

    return data


def generate_trade_suggestions_result(suggestions_ctx: dict) -> dict:
    """
    LLM-backed proactive trade suggestions.
    Returns {suggestions: [{title, partner_team, you_give, you_get, reasoning, urgency}]}
    """
    client = get_ai_client()

    schema = {
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "partner_team": {"type": "string"},
                        "you_give": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "you_get": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reasoning": {"type": "string"},
                        "urgency": {"type": "string", "enum": ["high", "medium", "low"]},
                        "trade_type": {"type": "string", "enum": ["up_tier", "down_tier", "swap"]},
                    },
                    "required": ["title", "partner_team", "you_give", "you_get", "reasoning", "urgency", "trade_type"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["suggestions"],
        "additionalProperties": False,
    }

    system_prompt = """
You are a dynasty fantasy football GM assistant generating proactive trade ideas.

CRITICAL RULES — follow exactly:
1. For player-for-player trades (from top_partners): use targets_they_have for you_get and
   targets_viewer_sends for you_give. Never invent players. Skip any partner with an empty list.
   - For package trades (is_package_trade=true): the viewer is packaging 2+ surplus players
     to acquire 1 elite player at a position they want to upgrade. Format as e.g.
     "Package Deal: [SurplusWR] + [SurplusTE] for [EliteRB]". List all package pieces in you_give.
2. For pick-for-player trades (from pick_trade_partners): use targets_they_have for you_get.
   For you_give, format each pick from picks_you_offer using season + round + slot:
   e.g. "2026 1st Round Pick 1.01 (proj. Jeremiyah Love, RB)" if slot and proj_name are present,
   or "2026 1st Round Pick" if not. Put these pick label strings in you_give.
3. Use viewer_needs/viewer_surplus and viewer_pos_ranks (1=best in league) as given — do NOT override them.
4. trade_type must be:
   - up_tier: viewer receives more value (acquiring a better player via picks or surplus)
   - down_tier: viewer gives more value than they receive
   - swap: roughly even value exchange
5. Keep reasoning concise (max 2 sentences). Lead with football logic, not raw numbers.
6. Urgency: high = fills a critical need or converts surplus depth to elite talent,
   medium = solid improvement, low = depth upgrade.
7. Never write "TBD", "Unknown", or any placeholder. If you cannot fill both sides, skip that suggestion.
""".strip()

    user_prompt = f"""
Generate up to 3 specific trade proposals for this dynasty team.

The viewer's needs and surplus positions are provided, along with the best matching trade partners.
When the viewer has no explicit needs but has surplus, suggest package deals that convert
excess depth at surplus positions into an upgrade at a position they're weak/neutral at.
For each suggestion, specify exact players by name (from targets_they_have and targets_viewer_sends).

Return JSON matching the schema exactly.

Trade suggestions context:
{json_dumps_safe(suggestions_ctx)}
""".strip()

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "trade_suggestions",
                "schema": schema,
            }
        },
    )

    raw = resp.output_text.strip()
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("LLM trade suggestions did not return an object")

    return data


def generate_team_ai_result(team_ctx: dict, mode: str = "gm_memo") -> dict:
    """
    LLM-backed team analysis with structured JSON output.
    mode: 'gm_memo' or 'front_office_briefing'
    """
    client = get_ai_client()

    if mode == "gm_memo":
        schema = {
            "type": "object",
            "properties": {
                "team_identity": {"type": "string"},
                "outlook": {"type": "string"},
                "strength": {"type": "string"},
                "weakness": {"type": "string"},
                "next_move": {"type": "string"},
                "trade_posture": {"type": "string"},
                "verdict": {
                    "type": "string",
                    "enum": ["BUY", "HOLD", "SELL VETERANS", "REBUILD AGGRESSIVELY"],
                },
            },
            "required": ["team_identity", "outlook", "strength", "weakness", "next_move", "trade_posture", "verdict"],
            "additionalProperties": False,
        }
        system_prompt = GM_MEMO_SYSTEM
        user_prompt = build_gm_memo_prompt(json_dumps_safe(team_ctx))
    else:  # front_office_briefing
        schema = {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "posture": {"type": "string"},
                "strongest_room": {"type": "string"},
                "weakest_room": {"type": "string"},
                "next_move": {"type": "string"},
                "gm_alert": {"type": "string"},
            },
            "required": ["headline", "posture", "strongest_room", "weakest_room", "next_move", "gm_alert"],
            "additionalProperties": False,
        }
        system_prompt = FRONT_OFFICE_BRIEF_SYSTEM
        user_prompt = build_front_office_brief_prompt(json_dumps_safe(team_ctx))

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": f"team_{mode}",
                "schema": schema,
            }
        },
    )

    raw = resp.output_text.strip()
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError(f"LLM {mode} did not return an object")

    return data
