"""Weekly standings-seed series for the team modal's Seed Movement chart.

Pure computation (pandas only) so it can be unit-tested without the app.
Seed is the standings rank through each finalized week by cumulative
(wins, points-for), the same ordering used for week-over-week movement.
"""
from __future__ import annotations


def seed_series_for(df_weekly, owner) -> list:
    """Return [(week, seed)] for one owner across finalized weeks.

    Best-effort: any failure (or no usable data) returns [].
    """
    try:
        cols = getattr(df_weekly, "columns", [])
        fin = df_weekly[df_weekly["finalized"] == True] if "finalized" in cols else df_weekly
        if fin is None or fin.empty:
            return []
        if "win" not in cols or "points" not in cols:
            return []
        weeks = sorted(int(w) for w in fin["week"].unique())
        out = []
        for wk in weeks:
            sub = fin[fin["week"] <= wk]
            agg = (
                sub.groupby("owner")
                .agg(W=("win", "sum"), PF=("points", "sum"))
                .reset_index()
                .sort_values(by=["W", "PF"], ascending=[False, False])
                .reset_index(drop=True)
            )
            seeds = {str(r["owner"]): i + 1 for i, r in agg.iterrows()}
            s = seeds.get(str(owner))
            if s is not None:
                out.append((wk, s))
        return out
    except Exception:
        return []
