# NFL 2025 Season Prediction Model

A comprehensive data analysis and machine learning project for predicting NFL game outcomes and analyzing team performance during the 2025 season.

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

All data is sourced from [nflverse](https://github.com/nflverse) via the `nflreadpy` Python package:
- Game schedules
- Betting lines and spreads
- Play-by-play data
- Team statistics

## Key Findings

*(To be populated as analysis progresses)*

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

