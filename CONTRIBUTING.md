# Contributing to NFL 2025 Season Prediction Model

Thank you for your interest in contributing to this project! This document provides guidelines for contributions.

## Project Structure

```
nfl-predict/
├── src/              # Source code modules
│   ├── data_loader.py    # Data loading functions
│   ├── analysis.py       # Statistical analysis
│   ├── visualization.py  # Plotting and charts
│   └── models.py         # Machine learning models
├── tests/            # Unit tests
├── notebooks/        # Jupyter notebooks
├── data/            # Data storage (gitignored)
├── outputs/         # Generated results (gitignored)
└── main.py          # Main execution script
```

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```powershell
   git clone https://github.com/YOUR_USERNAME/nfl-predict.git
   cd nfl-predict
   ```

3. Set up development environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   pip install -r requirements.txt
   ```

4. Create a feature branch:
   ```powershell
   git checkout -b feature/your-feature-name
   ```

## Code Style

This project follows these conventions:

### Python Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular (single responsibility)
- Use type hints where appropriate

### Documentation
- Document all public functions with Google-style docstrings
- Include parameter types and return types
- Provide usage examples for complex functions

Example:
```python
def analyze_spread_coverage(schedules: pd.DataFrame, 
                           betting_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which teams cover the spread more frequently.
    
    Args:
        schedules: Game schedules DataFrame with game_id, teams, scores
        betting_lines: Betting lines DataFrame with spread_line column
        
    Returns:
        DataFrame with cover rates by team and favorite/underdog status
        
    Example:
        >>> schedules, betting = load_nfl_data(2024)
        >>> results = analyze_spread_coverage(schedules, betting)
        >>> print(results.head())
    """
    # Implementation...
```

### Commit Messages
Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements

Example: `feat: add playoff game filtering to spread analysis`

## Testing

All new features should include tests:

1. Write tests in `tests/` directory
2. Use unittest framework
3. Run tests before committing:
   ```powershell
   python -m pytest tests/
   ```

Example test:
```python
import unittest
from src.analysis import calculate_game_margin

class TestAnalysis(unittest.TestCase):
    def test_calculate_margin(self):
        # Test implementation
        pass
```

## Adding New Features

### New Analysis Functions
1. Add function to appropriate module in `src/`
2. Include comprehensive docstring
3. Add unit tests
4. Update main.py if needed for pipeline integration
5. Document in README.md

### New Visualizations
1. Add function to `src/visualization.py`
2. Follow existing plotting style (seaborn whitegrid)
3. Include save_path parameter
4. Add usage example

### New ML Models
1. Add to `src/models.py`
2. Follow existing model pattern (train/evaluate/compare)
3. Include cross-validation
4. Document hyperparameters

## Pull Request Process

1. Update documentation for any new features
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md with changes
5. Create pull request with clear description

Pull request template:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests passing
```

## Areas for Contribution

Looking for ideas? Consider these areas:

### Analysis Enhancements
- Additional betting market analysis (totals, moneylines)
- Weather impact on game outcomes
- Home field advantage quantification
- Divisional game analysis

### Visualization Improvements
- Interactive dashboards (plotly/dash)
- Animated season progression charts
- Team-specific deep-dive reports

### Model Improvements
- Neural network implementations
- Ensemble methods
- Time series forecasting for season outcomes
- Feature engineering enhancements

### Data Pipeline
- Automated data updates
- Additional data sources integration
- Data validation and quality checks
- Caching mechanisms

### Documentation
- Video tutorials
- More notebook examples
- API documentation
- Use case examples

## Code Review Process

All contributions will be reviewed for:
- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Compatibility with existing features

## Questions?

Feel free to open an issue for:
- Feature requests
- Bug reports
- Questions about implementation
- Suggestions for improvements

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

