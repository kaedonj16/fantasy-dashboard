# Breakout Detection Engine - Integration Guide

## Overview

The Breakout Detection Engine identifies fantasy football breakout candidates using a multi-component scoring system that analyzes opportunity, competition, team environment, player readiness, and role trajectory.

See full documentation at: BREAKOUT_RESULTS_SUMMARY.md

---

## Quick Start

### 1. Database Setup

```bash
DATABASE_URL="postgresql://user@localhost:5432/brfantasy" psql -f data_building/breakout_engine/setup_database.sql
```

### 2. Calculate Scores

```bash
python3 -m data_building.breakout_engine.calculate_breakouts_with_real_data
```

### 3. View Results

```bash
python3 -m data_building.breakout_engine.display_results --summary --min-score 40
python3 -m data_building.breakout_engine.analyze_results --top-n 20
```

### 4. Access API

```bash
python3 app.py
# Then visit: http://localhost:5000/api/breakout/candidates
```

---

## API Endpoints

- `GET /api/breakout/candidates` - All candidates
- `GET /api/breakout/candidates/{position}` - Filter by position  
- `GET /api/breakout/player/{player_id}` - Player detail
- `GET /api/breakout/statistics` - Aggregate stats
- `GET /api/breakout/team/{team}` - Team roster situation

See full API reference in main README.

---

## Scheduling

```bash
# Cron job (daily 3 AM)
0 3 * * * python3 -m data_building.breakout_engine.scheduler --cron

# Daemon mode
python3 -m data_building.breakout_engine.scheduler --daemon

# Run now
python3 -m data_building.breakout_engine.scheduler --run-now
```

---

## Files

- `core.py` - Main BreakoutEngine
- `components.py` - Component calculations
- `breakout_api.py` - REST API endpoints
- `scheduler.py` - Automated jobs
- `setup_database.sql` - Database schema
