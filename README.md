# Fantasy Dashboard

A comprehensive fantasy football analytics dashboard featuring advanced breakout detection, AI-powered insights, and real-time data processing.

## Overview

The Fantasy Dashboard is a sophisticated web application that provides:

- **Breakout Detection Engine** - Multi-component scoring system to identify fantasy football breakout candidates
- **AI-Powered Analysis** - Automated trade suggestions, roster grades, and strategic insights
- **Real-time Data Processing** - Live NFL data integration with ESPN and Sleeper APIs
- **Advanced Analytics** - Player projections, historical analysis, and trend detection
- **Interactive Web Interface** - Modern dashboard with responsive design

## Features

### 🔥 Breakout Detection Engine
- Multi-factor scoring (opportunity, competition, team environment, player readiness, role trajectory)
- Position-specific analysis for QB, RB, WR, TE
- Automated daily updates with scheduling
- RESTful API endpoints for integration

### 🤖 AI-Powered Insights
- Trade suggestions powered by machine learning
- Roster grading and optimization recommendations
- Power rankings with automated analysis
- Historical recap generation

### 📊 Advanced Analytics
- Player value projections and trend analysis
- Team roster situation analysis
- Historical performance tracking
- Advanced metrics and efficiency calculations

### 🌐 Web Dashboard
- Modern, responsive interface
- Real-time data updates
- Interactive charts and visualizations
- Mobile-friendly design

## Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL database
- OpenAI API key (for AI features)

### Installation

1. **Clone and setup environment**
```bash
git clone <repository-url>
cd fantasy-dashboard
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup database**
```bash
# Create PostgreSQL database
createdb brfantasy

# Set environment variable
export DATABASE_URL="postgresql://username@localhost:5432/brfantasy"

# Run migrations
python3 -m data_building.breakout_engine.setup_database
```

4. **Configure environment**
```bash
# Copy .env template and configure
cp .env.example .env
# Edit .env with your API keys and database URL
```

### Running the Application

**Development mode:**
```bash
python app.py
```

**Production mode:**
```bash
gunicorn app:app
```

The dashboard will be available at `http://localhost:5000`

## API Endpoints

### Breakout Detection
- `GET /api/breakout/candidates` - All breakout candidates
- `GET /api/breakout/candidates/{position}` - Filter by position
- `GET /api/breakout/player/{player_id}` - Player details
- `GET /api/breakout/statistics` - Aggregate statistics
- `GET /api/breakout/team/{team}` - Team roster situation

### Player Data
- `GET /api/players` - Player search and information
- `GET /api/players/{player_id}` - Specific player data
- `GET /api/players/values` - Dynasty values and rankings

### League Management
- `GET /api/leagues/{league_id}` - League information
- `GET /api/leagues/{league_id}/rosters` - Team rosters
- `GET /api/leagues/{league_id}/matchups` - Weekly matchups

## Breakout Engine Usage

### Calculate Breakout Scores
```bash
python3 -m data_building.breakout_engine.calculate_breakouts_with_real_data --season 2026
```

### View Results
```bash
python3 -m data_building.breakout_engine.display_results --summary --min-score 40
python3 -m data_building.breakout_engine.analyze_results --top-n 20
```

### Automated Scheduling
```bash
# Daily cron job (3 AM)
0 3 * * * python3 -m data_building.breakout_engine.scheduler --cron

# Run immediately
python3 -m data_building.breakout_engine.scheduler --run-now
```

## Project Structure

```
fantasy-dashboard/
├── app.py                          # Main Flask application
├── dashboard_services/             # API services and endpoints
│   ├── breakout_api.py            # Breakout detection API
│   ├── api.py                     # Core API functions
│   └── ai/                        # AI-powered features
├── data_building/                  # Data processing and analysis
│   ├── breakout_engine/           # Breakout detection system
│   ├── external_data/             # External data sources
│   └── rookie_pipeline/           # Rookie analysis pipeline
├── cache/                         # Cached data storage
├── migrations/                     # Database migrations
├── static/                         # Static web assets
├── utils/                          # Utility functions
└── scripts/                        # Maintenance scripts
```

## Configuration

### Environment Variables
- `DATABASE_URL` - PostgreSQL connection string
- `OPENAI_API_KEY` - OpenAI API key for AI features
- `RENDER_API_KEY` - Render deployment key (if applicable)

### Database Setup
The application uses PostgreSQL for data storage. Key tables:
- `breakout_opportunity_scores` - Breakout candidate scores
- `player_values` - Dynasty player values
- `roster_changes` - Team roster transactions
- `vacated_opportunity` - Team opportunity analysis

## Development

### Adding New Features
1. Create feature branch
2. Add tests in appropriate test directory
3. Update documentation
4. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document functions with docstrings
- Maintain test coverage

## Deployment

### Render (Recommended)
1. Connect repository to Render
2. Configure environment variables
3. Deploy web service

### Docker
```bash
docker build -t fantasy-dashboard .
docker run -p 5000:5000 fantasy-dashboard
```

## Monitoring & Maintenance

### Daily Tasks
- Breakout score calculations (automated)
- Player data updates
- Cache cleanup

### Weekly Tasks
- Performance monitoring
- Error log review
- Data quality checks

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check existing GitHub issues
- Review documentation in `/docs` directory
- Contact maintainers

---

**Recent Updates:**
- ✅ Fixed breakout engine name resolution issue
- ✅ Enhanced player data caching
- ✅ Added AI-powered trade suggestions
- ✅ Improved mobile responsiveness
