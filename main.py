# nfl-predict/main.py
"""
Main execution script for NFL 2025 Season Prediction Model.
Runs complete analysis pipeline and generates outputs.
"""

import pandas as pd
import logging
from pathlib import Path

from src.data_loader import load_nfl_data
from src.analysis import (
    analyze_spread_coverage,
    correlate_offensive_stats,
    analyze_thursday_vs_sunday,
    aggregate_team_performance,
    generate_summary_report
)
from src.visualization import (
    plot_cover_rates,
    plot_stat_correlations,
    plot_thursday_vs_sunday,
    plot_correlation_heatmap,
    create_comprehensive_dashboard
)
from src.models import (
    prepare_features,
    split_data,
    train_logistic_regression,
    train_random_forest,
    evaluate_model,
    get_feature_importance,
    compare_models
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nfl_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Execute complete NFL analysis pipeline.
    """
    logger.info("=" * 80)
    logger.info("NFL 2025 SEASON PREDICTION MODEL - ANALYSIS PIPELINE")
    logger.info("=" * 80)
    
    # Configuration
    SEASON = 2024  # Use 2024 data; change to 2025 when available
    OUTPUT_DIR = Path('outputs')
    FIGURES_DIR = OUTPUT_DIR / 'figures'
    REPORTS_DIR = OUTPUT_DIR / 'reports'
    
    # Ensure output directories exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========== STEP 1: LOAD DATA ==========
    logger.info("\nSTEP 1: Loading NFL Data...")
    try:
        schedules, betting_lines, pbp_data = load_nfl_data(SEASON)
        logger.info(f"✓ Successfully loaded {SEASON} season data")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # ========== STEP 2: SPREAD COVERAGE ANALYSIS ==========
    logger.info("\nSTEP 2: Analyzing Spread Coverage...")
    try:
        cover_analysis = analyze_spread_coverage(schedules, betting_lines)
        cover_analysis.to_csv(REPORTS_DIR / 'cover_analysis.csv', index=False)
        logger.info("✓ Spread coverage analysis complete")
        
        # Generate visualization
        plot_cover_rates(
            cover_analysis,
            save_path=FIGURES_DIR / 'cover_rates.png'
        )
    except Exception as e:
        logger.error(f"Spread coverage analysis failed: {e}")
        cover_analysis = pd.DataFrame()
    
    # ========== STEP 3: OFFENSIVE STATS CORRELATION ==========
    logger.info("\nSTEP 3: Correlating Offensive Stats with Wins...")
    try:
        correlation_analysis = correlate_offensive_stats(pbp_data, schedules)
        correlation_analysis.to_csv(REPORTS_DIR / 'correlation_analysis.csv', index=False)
        logger.info("✓ Correlation analysis complete")
        
        # Generate visualization
        plot_stat_correlations(
            correlation_analysis,
            save_path=FIGURES_DIR / 'stat_correlations.png'
        )
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        correlation_analysis = pd.DataFrame()
    
    # ========== STEP 4: THURSDAY VS SUNDAY ANALYSIS ==========
    logger.info("\nSTEP 4: Analyzing Thursday vs Sunday Scoring...")
    try:
        thursday_analysis = analyze_thursday_vs_sunday(schedules)
        logger.info("✓ Day-of-week analysis complete")
        
        # Generate visualization
        plot_thursday_vs_sunday(
            thursday_analysis,
            save_path=FIGURES_DIR / 'thursday_vs_sunday.png'
        )
    except Exception as e:
        logger.error(f"Thursday/Sunday analysis failed: {e}")
        thursday_analysis = {}
    
    # ========== STEP 5: COMPREHENSIVE DASHBOARD ==========
    logger.info("\nSTEP 5: Creating Comprehensive Dashboard...")
    try:
        if not cover_analysis.empty and not correlation_analysis.empty and thursday_analysis:
            create_comprehensive_dashboard(
                cover_analysis,
                correlation_analysis,
                thursday_analysis,
                save_path=FIGURES_DIR / 'comprehensive_dashboard.png'
            )
            logger.info("✓ Dashboard created")
    except Exception as e:
        logger.error(f"Dashboard creation failed: {e}")
    
    # ========== STEP 6: MACHINE LEARNING MODELS ==========
    logger.info("\nSTEP 6: Training Machine Learning Models...")
    try:
        # Prepare data for ML
        team_stats = aggregate_team_performance(pbp_data)
        
        # Create game results for merging
        game_results = schedules[['game_id', 'home_team', 'away_team', 
                                  'home_score', 'away_score']].copy()
        game_results = game_results.dropna()
        
        # Create both home and away records
        home_results = game_results.copy()
        home_results['team'] = home_results['home_team']
        home_results['won'] = (home_results['home_score'] > home_results['away_score']).astype(int)
        
        away_results = game_results.copy()
        away_results['team'] = away_results['away_team']
        away_results['won'] = (away_results['away_score'] > away_results['home_score']).astype(int)
        
        all_results = pd.concat([
            home_results[['game_id', 'team', 'won']],
            away_results[['game_id', 'team', 'won']]
        ])
        
        # Prepare features
        ml_data = prepare_features(team_stats, all_results)
        
        # Split data
        X_train, X_test, y_train, y_test, scaler, feature_names = split_data(ml_data)
        
        # Train Logistic Regression
        lr_model = train_logistic_regression(X_train, y_train)
        lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        lr_importance = get_feature_importance(lr_model, feature_names)
        
        # Train Random Forest
        rf_model = train_random_forest(X_train, y_train)
        rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        rf_importance = get_feature_importance(rf_model, feature_names)
        
        # Compare models
        comparison = compare_models([lr_results, rf_results])
        comparison.to_csv(REPORTS_DIR / 'model_comparison.csv', index=False)
        
        logger.info("✓ Machine learning models trained and evaluated")
        
    except Exception as e:
        logger.error(f"Machine learning pipeline failed: {e}")
    
    # ========== STEP 7: GENERATE SUMMARY REPORT ==========
    logger.info("\nSTEP 7: Generating Summary Report...")
    try:
        if not cover_analysis.empty and not correlation_analysis.empty and thursday_analysis:
            report = generate_summary_report(
                cover_analysis,
                correlation_analysis,
                thursday_analysis
            )
            
            # Save report
            with open(REPORTS_DIR / 'analysis_summary.txt', 'w') as f:
                f.write(report)
            
            print("\n" + report)
            logger.info("✓ Summary report generated")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
    
    # ========== COMPLETION ==========
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS PIPELINE COMPLETE")
    logger.info(f"Outputs saved to: {OUTPUT_DIR.absolute()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

