# NFL 2025 Season Prediction Model

A comprehensive data analysis and machine learning project for predicting NFL game outcomes and analyzing team performance during the 2025 season.

## Video Demonstration

**Watch the project walkthrough:** [NFL Prediction Analysis - Code Walkthrough](https://www.loom.com/share/116dc9a0a8714f498500d5e21e687c3d?sid=790066d2-b1dd-435b-816b-70e92c2e9c30)

This 4-5 minute video demonstrates:
- Software execution and analysis pipeline
- Code structure and implementation details
- Dataset exploration and analysis methods
- Results and visualizations
- Key findings and insights

## Project Overview

This project uses publicly available NFL data from nflverse to answer key questions about team performance, betting spread coverage, and offensive efficiency correlations with wins.

### Core Questions

1. **Which teams are more likely to cover the spread in the 2025 season?**
   - Analyzes home underdogs vs. home favorites
   - Calculates cover rates and identifies trends

2. **Which offensive stats correlate best with wins in 2025?**
   - Examines passing efficiency, rushing yards, EPA metrics
   - Identifies key performance indicators for winning teams

### Stretch Challenges

- Thursday vs Sunday game total score analysis
- Machine learning models for game outcome prediction
- Interactive visualizations and dashboards

## Technology Stack

- **Language:** Python 3.9+
- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Data Source:** nflreadpy (nflverse API)
- **Machine Learning:** scikit-learn
- **Development:** Jupyter Notebook

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nfl-predict
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
nfl-predict/
├── data/
│   ├── raw/              # Raw data from nflreadpy
│   └── processed/        # Cleaned and processed data
├── src/
│   ├── data_loader.py    # Data loading and preparation
│   ├── analysis.py       # Statistical analysis functions
│   ├── visualization.py  # Plotting and charting functions
│   └── models.py         # Machine learning models
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_spread_analysis.ipynb
│   ├── 03_offensive_stats.ipynb
│   └── 04_ml_predictions.ipynb
├── outputs/
│   ├── figures/          # Generated visualizations
│   └── reports/          # Analysis reports
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── nfl-predict.md       # Project specification
```

## Usage

### Running the Analysis

1. Load and explore the data:
```python
from src.data_loader import load_nfl_data

schedules, betting_lines, pbp_data = load_nfl_data(season=2025)
```

2. Analyze spread coverage:
```python
from src.analysis import analyze_spread_coverage

cover_rates = analyze_spread_coverage(schedules, betting_lines)
```

3. Examine offensive stat correlations:
```python
from src.analysis import correlate_offensive_stats

correlations = correlate_offensive_stats(pbp_data, schedules)
```

4. Generate visualizations:
```python
from src.visualization import plot_cover_rates, plot_stat_correlations

plot_cover_rates(cover_rates)
plot_stat_correlations(correlations)
```

### Using Jupyter Notebooks

Start Jupyter:
```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and open the analysis notebooks in order.

## Data Sources

### Dataset Description
This project uses the **NFL 2024 Season Dataset** from [nflverse](https://github.com/nflverse), accessed via the `nflreadpy` Python package. The dataset is free, publicly available, and includes comprehensive NFL game data.

**Dataset Components:**
- **Game Schedules** (nfl.import_schedules): 285 games with dates, scores, and results
- **Betting Lines** (nfl.import_betting_lines): Point spreads and over/under lines for all games
- **Play-by-Play Data** (nfl.import_pbp_data): Over 48,000 plays with EPA (Expected Points Added), success rates, and detailed play information
- **Time Period:** 2024 NFL Regular Season (Weeks 1-18)

### Data Analysis Methods

The project demonstrates multiple data analysis techniques:

1. **Filtering**
   - Completed games only (excluding future games)
   - Offensive plays (pass/run plays only)
   - Home games by team for spread analysis

2. **Sorting**
   - Teams ranked by cover rates
   - Correlations sorted by predictive strength
   - Feature importance ranked by model weights

3. **Aggregation**
   - Average EPA (Expected Points Added) per play
   - Team-level statistics from play-by-play data
   - Cover rates calculated by team and favorite/underdog status
   - Count of games by day of week (Thursday vs Sunday)

4. **Data Conversion**
   - Binary win/loss outcomes from game scores
   - Spread coverage calculations (margin + spread > 0)
   - Date parsing for day-of-week analysis
   - Feature scaling for machine learning models

## Key Findings

### Question 1: Which teams cover the spread most frequently?
**Analysis:** Examined 285 home games from 2024 season, separating home favorites from home underdogs.

**Top 5 Teams by Cover Rate (Home Games):**
1. Kansas City Chiefs (KC): 100% (8/8 games)
2. Philadelphia Eagles (PHI): 100% (9/9 games)
3. Buffalo Bills (BUF): 100% (8/8 games)
4. Detroit Lions (DET): 88.9% (8/9 games)
5. Minnesota Vikings (MIN): 88.9% (8/9 games)

**Insight:** Elite teams with strong home-field advantage consistently beat expectations, making them reliable bets when playing at home.

### Question 2: Which offensive statistics best predict wins?
**Analysis:** Calculated correlation coefficients between 9 offensive metrics and game outcomes across 500+ team-game performances.

**Top Predictive Metrics:**
1. Total EPA: 0.512 correlation
2. Average EPA per play: 0.511 correlation
3. Passing EPA: 0.507 correlation
4. Average Pass Yards: 0.444 correlation
5. Average Yards per Play: 0.399 correlation

**Insight:** Expected Points Added (EPA) is the strongest predictor of wins. Teams that gain more expected points per play, especially through passing, win more games. Traditional yardage metrics are less predictive than efficiency metrics.

### Stretch Challenge: Thursday vs Sunday Scoring
**Analysis:** Compared total points scored in Thursday Night Football vs Sunday games.

**Results:**
- Thursday games: 44.68 points/game (19 games)
- Sunday games: 45.90 points/game (229 games)
- Difference: -1.21 points

**Insight:** Thursday games average slightly lower scoring, possibly due to shorter rest periods, but the difference is minimal (2.7% lower).

## Development Timeline

- **Week 1:** Data gathering, initial analysis (spread coverage, stat correlations)
- **Week 2:** Visualization implementation, ML model development
- **Week 3:** Final report, documentation, and project finalization

## Contributing

This is an academic/personal project. Contributions and suggestions are welcome.

## License

MIT License

## Acknowledgments

- nflverse community for providing accessible NFL data
- Python data science community for excellent tools and libraries

