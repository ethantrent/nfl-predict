# nfl-predict/src/data_loader.py
"""
Data loading module for NFL 2025 season data.
Handles data retrieval from nflverse via nflreadpy and initial preprocessing.
"""

import pandas as pd
import nflreadpy as nfl
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_schedules(season: int = 2025) -> pd.DataFrame:
    """
    Load NFL game schedules for specified season.
    
    Args:
        season: NFL season year (default: 2025)
        
    Returns:
        DataFrame containing game schedules with columns like game_id, 
        home_team, away_team, week, gameday, etc.
    """
    try:
        logger.info(f"Loading schedules for {season} season...")
        schedules = nfl.load_schedules([season])
        # Convert from Polars to Pandas if needed
        if hasattr(schedules, 'to_pandas'):
            schedules = schedules.to_pandas()
        logger.info(f"Successfully loaded {len(schedules)} games")
        return schedules
    except Exception as e:
        logger.error(f"Error loading schedules: {e}")
        raise


def load_betting_lines(season: int = 2025) -> pd.DataFrame:
    """
    Load betting lines data including spreads and totals.
    Note: Betting data is included in schedules, so this returns schedules.
    
    Args:
        season: NFL season year (default: 2025)
        
    Returns:
        DataFrame containing betting lines with spread_line, total_line, etc.
    """
    try:
        logger.info(f"Loading betting data for {season} season...")
        # Betting data is included in schedules
        betting_data = nfl.load_schedules([season])
        # Convert from Polars to Pandas if needed
        if hasattr(betting_data, 'to_pandas'):
            betting_data = betting_data.to_pandas()
        logger.info(f"Successfully loaded betting data for {len(betting_data)} games")
        return betting_data
    except Exception as e:
        logger.error(f"Error loading betting lines: {e}")
        raise


def load_play_by_play(season: int = 2025) -> pd.DataFrame:
    """
    Load play-by-play data for detailed game analysis.
    
    Args:
        season: NFL season year (default: 2025)
        
    Returns:
        DataFrame containing play-by-play data with EPA, success rate, etc.
    """
    try:
        logger.info(f"Loading play-by-play data for {season} season...")
        pbp_data = nfl.load_pbp([season])
        # Convert from Polars to Pandas if needed
        if hasattr(pbp_data, 'to_pandas'):
            pbp_data = pbp_data.to_pandas()
        logger.info(f"Successfully loaded {len(pbp_data)} plays")
        return pbp_data
    except Exception as e:
        logger.error(f"Error loading play-by-play data: {e}")
        raise


def load_nfl_data(season: int = 2025) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all required NFL data for analysis.
    
    Args:
        season: NFL season year (default: 2025)
        
    Returns:
        Tuple of (schedules, betting_lines, play_by_play) DataFrames
    """
    schedules = load_schedules(season)
    betting_lines = load_betting_lines(season)
    pbp_data = load_play_by_play(season)
    
    return schedules, betting_lines, pbp_data


def merge_game_data(schedules: pd.DataFrame, 
                   betting_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Merge schedules with betting lines for comprehensive game data.
    Note: Since betting data is in schedules, this just returns schedules.
    
    Args:
        schedules: Game schedules DataFrame (includes betting data)
        betting_lines: Betting lines DataFrame (same as schedules)
        
    Returns:
        DataFrame with game and betting information
    """
    try:
        logger.info("Preparing game data with betting information...")
        # Betting data already in schedules, just return it
        merged = schedules.copy()
        logger.info(f"Successfully prepared data: {len(merged)} games")
        return merged
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise


def filter_completed_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only include completed games with results.
    
    Args:
        df: DataFrame containing game data
        
    Returns:
        DataFrame with only completed games
    """
    completed = df[df['game_type'] == 'REG'].copy()
    completed = completed.dropna(subset=['home_score', 'away_score'])
    logger.info(f"Filtered to {len(completed)} completed games")
    return completed


def calculate_game_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate actual game margin (home team perspective).
    
    Args:
        df: DataFrame with home_score and away_score columns
        
    Returns:
        DataFrame with added 'margin' column
    """
    df = df.copy()
    df['margin'] = df['home_score'] - df['away_score']
    return df


def identify_home_favorite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify whether home team was favorite or underdog.
    Negative spread means home team favored.
    
    Args:
        df: DataFrame with spread_line column
        
    Returns:
        DataFrame with 'is_home_favorite' boolean column
    """
    df = df.copy()
    df['is_home_favorite'] = df['spread_line'] < 0
    return df


if __name__ == "__main__":
    # Example usage and testing
    try:
        schedules, betting_lines, pbp = load_nfl_data(2024)
        print(f"\nSchedules shape: {schedules.shape}")
        print(f"Betting lines shape: {betting_lines.shape}")
        print(f"Play-by-play shape: {pbp.shape}")
        
        print("\nSchedules columns:", schedules.columns.tolist()[:10])
        print("Betting lines columns:", betting_lines.columns.tolist()[:10])
        
    except Exception as e:
        print(f"Error during testing: {e}")

