# nfl-predict/tests/test_data_loader.py
"""
Unit tests for data loading module.
"""

import unittest
import pandas as pd
from src.data_loader import (
    calculate_game_margin,
    identify_home_favorite
)


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'game_id': ['2024_01_BUF_MIA', '2024_01_KC_DET'],
            'home_score': [30, 21],
            'away_score': [24, 20],
            'spread_line': [-3.5, 1.0]
        })
    
    def test_calculate_game_margin(self):
        """Test game margin calculation."""
        result = calculate_game_margin(self.sample_data)
        
        self.assertIn('margin', result.columns)
        self.assertEqual(result.iloc[0]['margin'], 6)
        self.assertEqual(result.iloc[1]['margin'], 1)
    
    def test_identify_home_favorite(self):
        """Test home favorite identification."""
        result = identify_home_favorite(self.sample_data)
        
        self.assertIn('is_home_favorite', result.columns)
        self.assertTrue(result.iloc[0]['is_home_favorite'])
        self.assertFalse(result.iloc[1]['is_home_favorite'])


if __name__ == '__main__':
    unittest.main()

