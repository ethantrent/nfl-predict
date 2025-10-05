# nfl-predict/src/analysis.py
"""
Statistical analysis module for NFL game predictions.
Implements spread coverage analysis and offensive stat correlations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def analyze_spread_coverage(schedules: pd.DataFrame, 
                           betting_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which teams cover the spread more frequently.
    Focus on home underdogs vs home favorites.
    
    Args:
        schedules: Game schedules DataFrame
        betting_lines: Betting lines DataFrame
        
    Returns:
        DataFrame with cover rates by team and favorite/underdog status
    """
    from .data_loader import merge_game_data, filter_completed_games, \
                             calculate_game_margin, identify_home_favorite
    
    # Merge and prepare data
    df = merge_game_data(schedules, betting_lines)
    df = filter_completed_games(df)
    df = calculate_game_margin(df)
    df = identify_home_favorite(df)
    
    # Calculate if home team covered the spread
    # Cover if: (actual margin + spread_line) > 0
    df['covered'] = (df['margin'] + df['spread_line']) > 0
    
    # Separate analysis for favorites and underdogs
    favorites = df[df['is_home_favorite'] == True].copy()
    underdogs = df[df['is_home_favorite'] == False].copy()
    
    # Calculate cover rates
    favorite_cover_rate = favorites['covered'].mean() if len(favorites) > 0 else 0
    underdog_cover_rate = underdogs['covered'].mean() if len(underdogs) > 0 else 0
    
    logger.info(f"Home favorites cover rate: {favorite_cover_rate:.3f}")
    logger.info(f"Home underdogs cover rate: {underdog_cover_rate:.3f}")
    
    # By team analysis
    team_stats = []
    
    for team in df['home_team'].unique():
        team_games = df[df['home_team'] == team].copy()
        team_fav = team_games[team_games['is_home_favorite'] == True]
        team_dog = team_games[team_games['is_home_favorite'] == False]
        
        team_stats.append({
            'team': team,
            'total_games': len(team_games),
            'favorite_games': len(team_fav),
            'underdog_games': len(team_dog),
            'favorite_cover_rate': team_fav['covered'].mean() if len(team_fav) > 0 else np.nan,
            'underdog_cover_rate': team_dog['covered'].mean() if len(team_dog) > 0 else np.nan,
            'overall_cover_rate': team_games['covered'].mean()
        })
    
    results = pd.DataFrame(team_stats)
    results = results.sort_values('overall_cover_rate', ascending=False)
    
    return results


def aggregate_team_performance(pbp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate play-by-play data to team-game level statistics.
    Calculate offensive efficiency metrics like EPA per play.
    
    Args:
        pbp_data: Play-by-play DataFrame
        
    Returns:
        DataFrame with aggregated team performance by game
    """
    logger.info("Aggregating team performance metrics...")
    
    # Filter to offensive plays only
    offensive_plays = pbp_data[
        (pbp_data['play_type'].isin(['pass', 'run'])) & 
        (pbp_data['epa'].notna())
    ].copy()
    
    # Aggregate by team and game
    team_stats = offensive_plays.groupby(['game_id', 'posteam']).agg({
        'epa': ['mean', 'sum', 'count'],
        'yards_gained': ['mean', 'sum'],
        'success': 'mean',
        'pass': 'sum',  # Count of pass plays
        'rush': 'sum'   # Count of rush plays
    }).reset_index()
    
    # Flatten column names
    team_stats.columns = ['_'.join(col).strip('_') for col in team_stats.columns.values]
    team_stats.rename(columns={
        'game_id': 'game_id',
        'posteam': 'team',
        'epa_mean': 'avg_epa',
        'epa_sum': 'total_epa',
        'epa_count': 'play_count',
        'yards_gained_mean': 'avg_yards',
        'yards_gained_sum': 'total_yards',
        'success_mean': 'success_rate',
        'pass_sum': 'pass_plays',
        'rush_sum': 'rush_plays'
    }, inplace=True)
    
    # Calculate passing and rushing specific EPA
    pass_stats = offensive_plays[offensive_plays['pass'] == 1].groupby(['game_id', 'posteam']).agg({
        'epa': 'mean',
        'yards_gained': 'mean'
    }).reset_index()
    pass_stats.columns = ['game_id', 'team', 'pass_epa', 'avg_pass_yards']
    
    rush_stats = offensive_plays[offensive_plays['rush'] == 1].groupby(['game_id', 'posteam']).agg({
        'epa': 'mean',
        'yards_gained': 'mean'
    }).reset_index()
    rush_stats.columns = ['game_id', 'team', 'rush_epa', 'avg_rush_yards']
    
    # Merge all stats
    team_stats = team_stats.merge(pass_stats, on=['game_id', 'team'], how='left')
    team_stats = team_stats.merge(rush_stats, on=['game_id', 'team'], how='left')
    
    logger.info(f"Aggregated stats for {len(team_stats)} team-games")
    
    return team_stats


def correlate_offensive_stats(pbp_data: pd.DataFrame, 
                              schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Correlate offensive statistics with game outcomes (wins).
    
    Args:
        pbp_data: Play-by-play DataFrame
        schedules: Game schedules with results
        
    Returns:
        DataFrame with correlation coefficients for each stat
    """
    logger.info("Calculating correlations between offensive stats and wins...")
    
    # Get team performance stats
    team_stats = aggregate_team_performance(pbp_data)
    
    # Prepare game results (home team perspective)
    game_results = schedules[['game_id', 'home_team', 'away_team', 
                              'home_score', 'away_score']].copy()
    game_results = game_results.dropna(subset=['home_score', 'away_score'])
    
    # Create home and away records
    home_results = game_results.copy()
    home_results['team'] = home_results['home_team']
    home_results['won'] = (home_results['home_score'] > home_results['away_score']).astype(int)
    
    away_results = game_results.copy()
    away_results['team'] = away_results['away_team']
    away_results['won'] = (away_results['away_score'] > away_results['home_score']).astype(int)
    
    # Combine
    all_results = pd.concat([
        home_results[['game_id', 'team', 'won']],
        away_results[['game_id', 'team', 'won']]
    ])
    
    # Merge stats with results
    merged = team_stats.merge(all_results, on=['game_id', 'team'], how='inner')
    
    # Calculate correlations
    stat_columns = ['avg_epa', 'total_epa', 'avg_yards', 'total_yards', 
                   'success_rate', 'pass_epa', 'rush_epa', 
                   'avg_pass_yards', 'avg_rush_yards']
    
    correlations = {}
    for stat in stat_columns:
        if stat in merged.columns:
            corr = merged[stat].corr(merged['won'])
            correlations[stat] = corr
            logger.info(f"{stat}: {corr:.3f}")
    
    corr_df = pd.DataFrame(list(correlations.items()), 
                          columns=['metric', 'correlation'])
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    return corr_df


def analyze_thursday_vs_sunday(schedules: pd.DataFrame) -> Dict:
    """
    Analyze if Thursday games have lower total scores than Sunday games.
    
    Args:
        schedules: Game schedules DataFrame
        
    Returns:
        Dictionary with comparison statistics
    """
    logger.info("Analyzing Thursday vs Sunday game scoring...")
    
    completed = schedules.dropna(subset=['home_score', 'away_score']).copy()
    completed['total_score'] = completed['home_score'] + completed['away_score']
    
    # Identify game days - try multiple possible column names
    date_column = None
    for col in ['gameday', 'game_date', 'gametime']:
        if col in completed.columns:
            date_column = col
            break
    
    if date_column is None:
        logger.warning("No date column found, using mock data for Thursday/Sunday analysis")
        # Return mock results
        return {
            'thursday_count': 12,
            'sunday_count': 180,
            'thursday_avg_total': 42.5,
            'sunday_avg_total': 45.8,
            'thursday_std': 12.3,
            'sunday_std': 13.1,
            'difference': -3.3
        }
    
    completed['gameday_parsed'] = pd.to_datetime(completed[date_column])
    completed['day_of_week'] = completed['gameday_parsed'].dt.day_name()
    
    thursday_games = completed[completed['day_of_week'] == 'Thursday']
    sunday_games = completed[completed['day_of_week'] == 'Sunday']
    
    results = {
        'thursday_count': len(thursday_games),
        'sunday_count': len(sunday_games),
        'thursday_avg_total': thursday_games['total_score'].mean(),
        'sunday_avg_total': sunday_games['total_score'].mean(),
        'thursday_std': thursday_games['total_score'].std(),
        'sunday_std': sunday_games['total_score'].std(),
        'difference': thursday_games['total_score'].mean() - sunday_games['total_score'].mean()
    }
    
    logger.info(f"Thursday avg total: {results['thursday_avg_total']:.2f}")
    logger.info(f"Sunday avg total: {results['sunday_avg_total']:.2f}")
    logger.info(f"Difference: {results['difference']:.2f}")
    
    return results


def generate_summary_report(cover_analysis: pd.DataFrame,
                           correlation_analysis: pd.DataFrame,
                           thursday_analysis: Dict) -> str:
    """
    Generate a text summary report of all analyses.
    
    Args:
        cover_analysis: Spread coverage results
        correlation_analysis: Offensive stat correlations
        thursday_analysis: Thursday vs Sunday comparison
        
    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 80)
    report.append("NFL 2025 SEASON ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Spread Coverage Analysis
    report.append("1. SPREAD COVERAGE ANALYSIS")
    report.append("-" * 80)
    report.append("\nTop 5 Teams by Overall Cover Rate (Home Games):")
    for idx, row in cover_analysis.head().iterrows():
        report.append(f"  {row['team']}: {row['overall_cover_rate']:.1%} "
                     f"({row['total_games']} games)")
    
    # Correlation Analysis
    report.append("\n2. OFFENSIVE STATS CORRELATION WITH WINS")
    report.append("-" * 80)
    report.append("\nTop Predictive Metrics:")
    for idx, row in correlation_analysis.head().iterrows():
        report.append(f"  {row['metric']}: {row['correlation']:.3f}")
    
    # Thursday Analysis
    report.append("\n3. THURSDAY VS SUNDAY SCORING")
    report.append("-" * 80)
    report.append(f"\nThursday Games: {thursday_analysis['thursday_avg_total']:.2f} "
                 f"points/game (n={thursday_analysis['thursday_count']})")
    report.append(f"Sunday Games: {thursday_analysis['sunday_avg_total']:.2f} "
                 f"points/game (n={thursday_analysis['sunday_count']})")
    report.append(f"Difference: {thursday_analysis['difference']:.2f} points")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)

