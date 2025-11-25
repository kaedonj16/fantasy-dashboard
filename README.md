# Tactical Blueprint League Report

This pack includes a single Python script that:
- pulls Sleeper league data (users/rosters/matchups),
- computes PF, PA, AVG, MAX, MIN, STD, Win%,
- generates four visuals + a weekly scoreboard,
- compiles everything into a PDF using a **Tactical Blueprint** theme.

## Quick Start

1) Create and activate a virtual environment.
2) Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3) Run the report (replace with your league ID and current week number):

```bash
python3 app.oy
```

Outputs in the same folder:
- `weekly_points.csv`
- `league_summary.csv`
- `pf_vs_pa.png`
- `scores_boxplot.png`
- `scores_linegraph.png`
- `standardized_radar.png`
- `scoreboard_week{N}.png` (latest week)
- `Tactical_Blueprint_League_Report.pdf`

> Note: This script calls Sleeper's public API. Make sure your machine has internet access.
