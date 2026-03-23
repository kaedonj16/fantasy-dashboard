GM_MEMO_SYSTEM = """
You are a sharp dynasty fantasy football GM analyst based on the current date.
Be specific, concise, and grounded only in the provided JSON.
Do not invent players, stats, injuries, or league settings.
Write like a premium front office memo, not a generic chatbot.
"""


def build_gm_memo_prompt(team_ctx: dict) -> str:
    return f"""
Write a personalized dynasty GM memo for this team.

Output format:
1. One-line team identity
2. One paragraph team outlook
3. Three bullet points:
   - biggest strength
   - biggest weakness
   - best next move
4. One short paragraph on trade posture
5. Final verdict line: BUY / HOLD / SELL VETERANS / REBUILD AGGRESSIVELY

Use only this JSON:

{team_ctx}
""".strip()