"""
Exploratory Data Analysis and Feature Engineering
Generates comprehensive EDA plots and creates features for ML models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

def perform_eda():
    """Perform comprehensive EDA and save plots"""
    
    # Load data
    df = pd.read_csv('data/epl_combined.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Match outcome distribution
    plt.figure(figsize=(10, 6))
    outcome_counts = df['FTR'].value_counts()
    colors = ['#9c88ff', '#d1c4e9', '#f3e8ff']
    plt.pie(outcome_counts.values, labels=['Home Win', 'Away Win', 'Draw'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Match Outcome Distribution', fontsize=16, fontweight='bold')
    plt.savefig('outputs/match_outcomes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Goals analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Average goals home vs away
    home_goals = df.groupby('HomeTeam')['FTHG'].mean().sort_values(ascending=False)
    away_goals = df.groupby('AwayTeam')['FTAG'].mean().sort_values(ascending=False)
    
    axes[0,0].bar(range(len(home_goals[:10])), home_goals[:10].values, color='#9c88ff')
    axes[0,0].set_title('Top 10 Home Scoring Teams', fontweight='bold')
    axes[0,0].set_xticks(range(len(home_goals[:10])))
    axes[0,0].set_xticklabels(home_goals[:10].index, rotation=45)
    
    axes[0,1].bar(range(len(away_goals[:10])), away_goals[:10].values, color='#d1c4e9')
    axes[0,1].set_title('Top 10 Away Scoring Teams', fontweight='bold')
    axes[0,1].set_xticks(range(len(away_goals[:10])))
    axes[0,1].set_xticklabels(away_goals[:10].index, rotation=45)
    
    # Goals distribution
    axes[1,0].hist(df['FTHG'], bins=range(0, 8), alpha=0.7, color='#9c88ff', label='Home Goals')
    axes[1,0].hist(df['FTAG'], bins=range(0, 8), alpha=0.7, color='#d1c4e9', label='Away Goals')
    axes[1,0].set_title('Goals Distribution', fontweight='bold')
    axes[1,0].legend()
    
    # Total goals per match
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    axes[1,1].hist(df['TotalGoals'], bins=range(0, 12), color='#f3e8ff', edgecolor='black')
    axes[1,1].set_title('Total Goals per Match Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/goals_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Shots and cards analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Shots analysis
    axes[0,0].scatter(df['HS'], df['FTHG'], alpha=0.6, color='#9c88ff')
    axes[0,0].set_xlabel('Home Shots')
    axes[0,0].set_ylabel('Home Goals')
    axes[0,0].set_title('Home Shots vs Goals', fontweight='bold')
    
    axes[0,1].scatter(df['HST'], df['FTHG'], alpha=0.6, color='#d1c4e9')
    axes[0,1].set_xlabel('Home Shots on Target')
    axes[0,1].set_ylabel('Home Goals')
    axes[0,1].set_title('Home Shots on Target vs Goals', fontweight='bold')
    
    # Cards analysis
    card_data = df[['HY', 'AY', 'HR', 'AR']].mean()
    axes[1,0].bar(card_data.index, card_data.values, color=['#ffeb3b', '#ffeb3b', '#f44336', '#f44336'])
    axes[1,0].set_title('Average Cards per Match', fontweight='bold')
    axes[1,0].set_ylabel('Average Cards')
    
    # Fouls vs Cards
    axes[1,1].scatter(df['HF'], df['HY'], alpha=0.6, color='#9c88ff', label='Home')
    axes[1,1].scatter(df['AF'], df['AY'], alpha=0.6, color='#d1c4e9', label='Away')
    axes[1,1].set_xlabel('Fouls')
    axes[1,1].set_ylabel('Yellow Cards')
    axes[1,1].set_title('Fouls vs Yellow Cards', fontweight='bold')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/shots_cards_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = ['HTHG', 'HTAG', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 
                   'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    corr_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Team performance analysis
    team_stats = calculate_team_stats(df)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top scoring teams
    top_scorers = team_stats.nlargest(10, 'GoalsFor')
    axes[0,0].barh(range(len(top_scorers)), top_scorers['GoalsFor'], color='#9c88ff')
    axes[0,0].set_yticks(range(len(top_scorers)))
    axes[0,0].set_yticklabels(top_scorers.index)
    axes[0,0].set_title('Top 10 Scoring Teams', fontweight='bold')
    
    # Best defensive teams
    best_defense = team_stats.nsmallest(10, 'GoalsAgainst')
    axes[0,1].barh(range(len(best_defense)), best_defense['GoalsAgainst'], color='#d1c4e9')
    axes[0,1].set_yticks(range(len(best_defense)))
    axes[0,1].set_yticklabels(best_defense.index)
    axes[0,1].set_title('Best Defensive Teams (Fewest Goals Conceded)', fontweight='bold')
    
    # Win percentage
    axes[1,0].bar(range(len(team_stats[:10])), team_stats['WinPercentage'][:10], color='#f3e8ff')
    axes[1,0].set_xticks(range(len(team_stats[:10])))
    axes[1,0].set_xticklabels(team_stats.index[:10], rotation=45)
    axes[1,0].set_title('Win Percentage by Team', fontweight='bold')
    
    # Points vs Goal difference
    axes[1,1].scatter(team_stats['GoalDifference'], team_stats['Points'], 
                     s=100, alpha=0.7, color='#9c88ff')
    axes[1,1].set_xlabel('Goal Difference')
    axes[1,1].set_ylabel('Points')
    axes[1,1].set_title('Points vs Goal Difference', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/team_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save team stats
    team_stats.to_csv('outputs/team_statistics.csv')
    
    print("EDA complete! All plots saved to outputs/ directory")
    return df, team_stats

def calculate_team_stats(df):
    """Calculate comprehensive team statistics"""
    
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_stats = []
    
    for team in teams:
        home_matches = df[df['HomeTeam'] == team]
        away_matches = df[df['AwayTeam'] == team]
        
        # Goals
        goals_for = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
        goals_against = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
        
        # Wins, draws, losses
        home_wins = len(home_matches[home_matches['FTR'] == 'H'])
        away_wins = len(away_matches[away_matches['FTR'] == 'A'])
        home_draws = len(home_matches[home_matches['FTR'] == 'D'])
        away_draws = len(away_matches[away_matches['FTR'] == 'D'])
        
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_matches = len(home_matches) + len(away_matches)
        total_losses = total_matches - total_wins - total_draws
        
        # Points (3 for win, 1 for draw)
        points = total_wins * 3 + total_draws
        
        # Other stats
        win_percentage = (total_wins / total_matches * 100) if total_matches > 0 else 0
        goal_difference = goals_for - goals_against
        
        team_stats.append({
            'Team': team,
            'Matches': total_matches,
            'Wins': total_wins,
            'Draws': total_draws,
            'Losses': total_losses,
            'GoalsFor': goals_for,
            'GoalsAgainst': goals_against,
            'GoalDifference': goal_difference,
            'Points': points,
            'WinPercentage': win_percentage
        })
    
    team_df = pd.DataFrame(team_stats).set_index('Team')
    return team_df.sort_values('Points', ascending=False)

def create_features(df):
    """Create additional features for ML models"""
    
    # Team form features (last 5 matches)
    df_with_features = df.copy()
    
    # Add season and month
    df_with_features['Season'] = df_with_features['Date'].dt.year
    df_with_features['Month'] = df_with_features['Date'].dt.month
    
    # Goal difference features
    df_with_features['GoalDifference'] = df_with_features['FTHG'] - df_with_features['FTAG']
    df_with_features['HTGoalDifference'] = df_with_features['HTHG'] - df_with_features['HTAG']
    
    # Shot efficiency
    df_with_features['HomeShotEfficiency'] = df_with_features['FTHG'] / (df_with_features['HS'] + 1)
    df_with_features['AwayShotEfficiency'] = df_with_features['FTAG'] / (df_with_features['AS'] + 1)
    
    # Shots on target ratio
    df_with_features['HomeSOTRatio'] = df_with_features['HST'] / (df_with_features['HS'] + 1)
    df_with_features['AwaySOTRatio'] = df_with_features['AST'] / (df_with_features['AS'] + 1)
    
    return df_with_features

if __name__ == '__main__':
    print("Starting EDA and feature engineering...")
    df, team_stats = perform_eda()
    df_features = create_features(df)
    
    # Save enhanced dataset
    df_features.to_csv('data/epl_features.csv', index=False)
    print(f"Enhanced dataset saved with {len(df_features.columns)} features")
    print("EDA and feature engineering complete!")