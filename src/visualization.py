# nfl-predict/src/visualization.py
"""
Visualization module for NFL game analysis.
Creates charts and graphs for spread coverage, correlations, and trends.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_cover_rates(cover_analysis: pd.DataFrame, 
                     save_path: Optional[str] = None) -> None:
    """
    Plot cover rates for home favorites vs underdogs by team.
    
    Args:
        cover_analysis: DataFrame with cover rate statistics
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Overall cover rates
    top_teams = cover_analysis.nlargest(10, 'overall_cover_rate')
    
    axes[0].barh(top_teams['team'], top_teams['overall_cover_rate'], 
                 color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Cover Rate', fontsize=12)
    axes[0].set_title('Top 10 Teams by Overall Cover Rate (Home Games)', 
                      fontsize=14, fontweight='bold')
    axes[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
    axes[0].legend()
    
    # Plot 2: Favorite vs Underdog comparison
    comparison_data = cover_analysis[
        cover_analysis['favorite_games'] >= 3
    ].head(10).copy()
    
    x = range(len(comparison_data))
    width = 0.35
    
    axes[1].bar([i - width/2 for i in x], 
                comparison_data['favorite_cover_rate'], 
                width, label='As Favorite', color='green', alpha=0.7)
    axes[1].bar([i + width/2 for i in x], 
                comparison_data['underdog_cover_rate'], 
                width, label='As Underdog', color='orange', alpha=0.7)
    
    axes[1].set_xlabel('Team', fontsize=12)
    axes[1].set_ylabel('Cover Rate', fontsize=12)
    axes[1].set_title('Cover Rate: Favorites vs Underdogs', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparison_data['team'], rotation=45, ha='right')
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cover rate plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_stat_correlations(correlation_df: pd.DataFrame,
                          save_path: Optional[str] = None) -> None:
    """
    Plot correlation coefficients between offensive stats and wins.
    
    Args:
        correlation_df: DataFrame with metric-correlation pairs
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort and plot
    corr_sorted = correlation_df.sort_values('correlation', ascending=True)
    
    colors = ['red' if x < 0 else 'green' for x in corr_sorted['correlation']]
    
    ax.barh(corr_sorted['metric'], corr_sorted['correlation'], 
            color=colors, alpha=0.7)
    ax.set_xlabel('Correlation with Winning', fontsize=12)
    ax.set_title('Offensive Stats Correlation with Game Outcomes', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for idx, row in corr_sorted.iterrows():
        ax.text(row['correlation'], idx, f" {row['correlation']:.3f}", 
               va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_thursday_vs_sunday(thursday_stats: Dict,
                           save_path: Optional[str] = None) -> None:
    """
    Visualize Thursday vs Sunday game scoring comparison.
    
    Args:
        thursday_stats: Dictionary with Thursday vs Sunday statistics
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar chart comparison
    days = ['Thursday', 'Sunday']
    avg_scores = [thursday_stats['thursday_avg_total'], 
                  thursday_stats['sunday_avg_total']]
    std_scores = [thursday_stats['thursday_std'], 
                  thursday_stats['sunday_std']]
    
    axes[0].bar(days, avg_scores, color=['purple', 'blue'], alpha=0.7, 
               yerr=std_scores, capsize=10)
    axes[0].set_ylabel('Average Total Points', fontsize=12)
    axes[0].set_title('Average Game Scoring: Thursday vs Sunday', 
                      fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (day, score) in enumerate(zip(days, avg_scores)):
        axes[0].text(i, score + 2, f'{score:.1f}', 
                    ha='center', fontsize=11, fontweight='bold')
    
    # Plot 2: Summary statistics
    axes[1].axis('off')
    summary_text = f"""
    THURSDAY GAMES
    ─────────────────
    Games: {thursday_stats['thursday_count']}
    Avg Total: {thursday_stats['thursday_avg_total']:.2f}
    Std Dev: {thursday_stats['thursday_std']:.2f}
    
    SUNDAY GAMES
    ─────────────────
    Games: {thursday_stats['sunday_count']}
    Avg Total: {thursday_stats['sunday_avg_total']:.2f}
    Std Dev: {thursday_stats['sunday_std']:.2f}
    
    DIFFERENCE
    ─────────────────
    {thursday_stats['difference']:.2f} points
    ({abs(thursday_stats['difference']/thursday_stats['sunday_avg_total']*100):.1f}% difference)
    """
    
    axes[1].text(0.1, 0.5, summary_text, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Thursday vs Sunday plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_correlation_heatmap(team_stats: pd.DataFrame,
                            save_path: Optional[str] = None) -> None:
    """
    Create heatmap of correlations between different performance metrics.
    
    Args:
        team_stats: DataFrame with aggregated team statistics
        save_path: Optional path to save the figure
    """
    # Select numeric columns for correlation
    numeric_cols = ['avg_epa', 'avg_yards', 'success_rate', 
                   'pass_epa', 'rush_epa', 'avg_pass_yards', 'avg_rush_yards']
    
    available_cols = [col for col in numeric_cols if col in team_stats.columns]
    
    corr_matrix = team_stats[available_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Performance Metrics Correlation Heatmap', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation heatmap to {save_path}")
        plt.close()
    else:
        plt.show()


def create_comprehensive_dashboard(cover_analysis: pd.DataFrame,
                                  correlation_df: pd.DataFrame,
                                  thursday_stats: Dict,
                                  save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive dashboard with all key visualizations.
    
    Args:
        cover_analysis: Spread coverage results
        correlation_df: Offensive stat correlations
        thursday_stats: Thursday vs Sunday comparison
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Cover rates
    ax1 = fig.add_subplot(gs[0, :])
    top_teams = cover_analysis.nlargest(10, 'overall_cover_rate')
    ax1.barh(top_teams['team'], top_teams['overall_cover_rate'], 
            color='steelblue', alpha=0.8)
    ax1.set_xlabel('Cover Rate')
    ax1.set_title('Top 10 Teams by Cover Rate', fontweight='bold')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: Correlations
    ax2 = fig.add_subplot(gs[1, 0])
    corr_sorted = correlation_df.sort_values('correlation', ascending=True)
    colors = ['red' if x < 0 else 'green' for x in corr_sorted['correlation']]
    ax2.barh(corr_sorted['metric'], corr_sorted['correlation'], 
            color=colors, alpha=0.7)
    ax2.set_xlabel('Correlation')
    ax2.set_title('Stats vs Wins Correlation', fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    # Plot 3: Thursday vs Sunday
    ax3 = fig.add_subplot(gs[1, 1])
    days = ['Thursday', 'Sunday']
    avg_scores = [thursday_stats['thursday_avg_total'], 
                  thursday_stats['sunday_avg_total']]
    ax3.bar(days, avg_scores, color=['purple', 'blue'], alpha=0.7)
    ax3.set_ylabel('Average Total Points')
    ax3.set_title('Thursday vs Sunday Scoring', fontweight='bold')
    for i, score in enumerate(avg_scores):
        ax3.text(i, score + 1, f'{score:.1f}', ha='center', fontweight='bold')
    
    # Plot 4: Summary text
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    summary = f"""
    KEY FINDINGS
    ════════════════════════════════════════════════════════════════════════
    
    Best Cover Team: {cover_analysis.iloc[0]['team']} ({cover_analysis.iloc[0]['overall_cover_rate']:.1%})
    
    Top Predictive Stat: {correlation_df.iloc[0]['metric']} (r={correlation_df.iloc[0]['correlation']:.3f})
    
    Thursday vs Sunday: Thursday games score {abs(thursday_stats['difference']):.1f} points 
    {'less' if thursday_stats['difference'] < 0 else 'more'} on average
    """
    
    ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
            fontsize=12, ha='center', va='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    fig.suptitle('NFL 2025 Season Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comprehensive dashboard to {save_path}")
        plt.close()
    else:
        plt.show()

