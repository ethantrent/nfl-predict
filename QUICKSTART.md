# Quick Start Guide

Get started with the NFL 2025 Season Prediction Model in just a few steps.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (optional, for cloning)

## Installation (5 minutes)

### Step 1: Set up the environment

Open PowerShell and navigate to the project directory:

```powershell
cd nfl-predict
```

### Step 2: Create virtual environment

```powershell
python -m venv venv
```

### Step 3: Activate virtual environment

```powershell
.\venv\Scripts\Activate
```

You should see `(venv)` in your command prompt.

### Step 4: Install dependencies

```powershell
pip install -r requirements.txt
```

This will install all required packages including pandas, matplotlib, nflreadpy, and scikit-learn.

## Running the Analysis

### Option 1: Run Complete Analysis Pipeline

Execute the main script to run the entire analysis:

```powershell
python main.py
```

This will:
1. Load 2025 NFL season data (or 2024 if 2025 not available)
2. Analyze spread coverage for home favorites vs underdogs
3. Correlate offensive statistics with wins
4. Analyze Thursday vs Sunday game scoring
5. Train machine learning models (Logistic Regression & Random Forest)
6. Generate visualizations and save to `outputs/figures/`
7. Create summary report in `outputs/reports/`

**Expected runtime:** 2-5 minutes depending on data size

### Option 2: Interactive Exploration with Jupyter

For interactive data exploration:

```powershell
jupyter notebook
```

Navigate to `notebooks/01_data_exploration.ipynb` and run the cells.

## Viewing Results

After running the analysis, check these locations:

### Visualizations
- `outputs/figures/cover_rates.png` - Spread coverage analysis
- `outputs/figures/stat_correlations.png` - Offensive stats vs wins
- `outputs/figures/thursday_vs_sunday.png` - Day-of-week scoring comparison
- `outputs/figures/comprehensive_dashboard.png` - All-in-one dashboard

### Reports
- `outputs/reports/analysis_summary.txt` - Text summary of findings
- `outputs/reports/cover_analysis.csv` - Detailed cover rate data
- `outputs/reports/correlation_analysis.csv` - Correlation coefficients
- `outputs/reports/model_comparison.csv` - ML model performance

### Logs
- `nfl_analysis.log` - Detailed execution log

## Example Usage

### Python Script

```python
from src.data_loader import load_nfl_data
from src.analysis import analyze_spread_coverage

# Load data
schedules, betting_lines, pbp = load_nfl_data(2024)

# Analyze spread coverage
results = analyze_spread_coverage(schedules, betting_lines)

# View top teams
print(results.head(10))
```

### Command Line

```powershell
# Run tests
python -m pytest tests/

# Run specific module
python -c "from src.data_loader import load_schedules; print(load_schedules(2024).shape)"
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'nflreadpy'"

**Solution:** Make sure virtual environment is activated and dependencies are installed:
```powershell
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### "No data available for 2025 season"

**Solution:** The 2025 season data may not be available yet. Edit `main.py` and change:
```python
SEASON = 2024  # Use 2024 data for now
```

### "Permission denied" errors

**Solution:** Run PowerShell as Administrator or ensure you have write permissions in the project directory.

### Import errors

**Solution:** Ensure you're running from the project root directory:
```powershell
cd C:\Users\ethan\nfl-predict
python main.py
```

## Next Steps

1. **Explore the data:** Open Jupyter notebooks to interactively explore the NFL data
2. **Customize analysis:** Modify `main.py` to focus on specific teams or metrics
3. **Extend models:** Add new features or try different ML algorithms in `src/models.py`
4. **Create visualizations:** Use `src/visualization.py` to create custom charts

## Key Findings Preview

After running the analysis, you'll discover:

✓ Which teams consistently beat the spread
✓ Whether passing or rushing efficiency correlates more with wins
✓ If Thursday games have different scoring patterns than Sunday games
✓ Machine learning model accuracy for predicting game outcomes

## Support

For issues or questions:
1. Check the main `README.md` for detailed documentation
2. Review the `nfl-predict.md` specification document
3. Check the execution logs in `nfl_analysis.log`

Happy analyzing! 🏈📊

