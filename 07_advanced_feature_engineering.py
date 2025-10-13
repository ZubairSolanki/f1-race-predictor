"""
Advanced Feature Engineering for F1 Podium Prediction
Creates momentum, trend, and circuit-specific features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("🔧 F1 ADVANCED FEATURE ENGINEERING")
print("=" * 80)

# Load data
print("📥 Loading data...")
df = pd.read_csv('f1_complete_data.csv')
df['race_date'] = pd.to_datetime(df['race_date'])
df = df.sort_values(['year', 'race_date', 'driver_abbreviation'])

print(f"✅ Loaded {len(df)} entries from {df['race_name'].nunique()} races\n")

# ============================================
# 1. DRIVER MOMENTUM FEATURES
# ============================================
print("🏎️  Creating Driver Momentum Features...")

def calculate_recent_performance(group, window=5):
    """Calculate rolling performance metrics for each driver"""

    # Sort by date
    group = group.sort_values('race_date')

    # Recent podium rate (last N races)
    group['recent_podiums_5'] = group['on_podium'].rolling(window=5, min_periods=1).sum()
    group['recent_podiums_3'] = group['on_podium'].rolling(window=3, min_periods=1).sum()

    # Recent points average
    group['recent_points_avg_5'] = group['points_scored'].rolling(window=5, min_periods=1).mean()
    group['recent_points_avg_3'] = group['points_scored'].rolling(window=3, min_periods=1).mean()

    # Recent finish position average (lower is better)
    group['recent_finish_avg_5'] = group['finish_position'].rolling(window=5, min_periods=1).mean()
    group['recent_finish_avg_3'] = group['finish_position'].rolling(window=3, min_periods=1).mean()

    # Win streak (consecutive podiums)
    group['is_hot_streak'] = (group['on_podium'].rolling(window=3, min_periods=1).sum() >= 2).astype(int)

    # Trend: Is performance improving?
    group['performance_trend'] = group['points_scored'].rolling(window=5, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )

    return group

# Apply to each driver
df = df.groupby('driver_abbreviation', group_keys=False).apply(calculate_recent_performance)

print("  ✅ Driver momentum features created")

# ============================================
# 2. TEAM PERFORMANCE FEATURES
# ============================================
print("🏁 Creating Team Performance Features...")

def calculate_team_performance(group):
    """Calculate team-level performance metrics"""

    group = group.sort_values('race_date')

    # Team podium rate (last N races)
    group['team_recent_podiums_5'] = group['on_podium'].rolling(window=5, min_periods=1).sum()

    # Team average finish
    group['team_avg_finish_5'] = group['finish_position'].rolling(window=5, min_periods=1).mean()

    # Team points in last 5 races
    group['team_points_last_5'] = group['points_scored'].rolling(window=5, min_periods=1).sum()

    return group

# Apply to each team
df = df.groupby('team_name', group_keys=False).apply(calculate_team_performance)

print("  ✅ Team performance features created")

# ============================================
# 3. SEASON CONTEXT FEATURES
# ============================================
print("📊 Creating Season Context Features...")

# Season progress (what % through the season)
season_race_counts = df.groupby('year')['race_date'].transform('rank', method='dense')
season_total_races = df.groupby('year')['race_date'].transform('nunique')
df['season_progress'] = season_race_counts / season_total_races

# Cumulative season points
df['season_points_cumulative'] = df.groupby(['year', 'driver_abbreviation'])['points_scored'].cumsum()

# Championship position (within each year)
df['championship_position'] = df.groupby(['year', 'race_date'])['season_points_cumulative'].rank(
    ascending=False, method='min'
)

# Gap to leader
leader_points = df.groupby(['year', 'race_date'])['season_points_cumulative'].transform('max')
df['points_gap_to_leader'] = leader_points - df['season_points_cumulative']

print("  ✅ Season context features created")

# ============================================
# 4. CIRCUIT-SPECIFIC FEATURES
# ============================================
print("🏟️  Creating Circuit-Specific Features...")

def calculate_circuit_history(group):
    """Calculate driver performance at specific circuits"""

    group = group.sort_values('race_date')

    # Historical podiums at this circuit
    group['circuit_historical_podiums'] = group['on_podium'].expanding().sum() - group['on_podium']

    # Historical avg finish at this circuit
    group['circuit_historical_avg_finish'] = group['finish_position'].expanding().mean().shift(1)

    # Races at this circuit (experience)
    group['circuit_experience'] = group.groupby('circuit_name').cumcount()

    return group

# Apply per driver per circuit
df = df.groupby(['driver_abbreviation', 'circuit_name'], group_keys=False).apply(calculate_circuit_history)

# Fill NaN values for first race at circuit
df['circuit_historical_avg_finish'] = df['circuit_historical_avg_finish'].fillna(10)
df['circuit_historical_podiums'] = df['circuit_historical_podiums'].fillna(0)

print("  ✅ Circuit-specific features created")

# ============================================
# 5. QUALIFYING PERFORMANCE FEATURES
# ============================================
print("🚦 Creating Qualifying Features...")

# Grid position advantage (starting in top 10)
df['starts_top10'] = (df['grid_position'] <= 10).astype(int)
df['starts_top5'] = (df['grid_position'] <= 5).astype(int)
df['starts_front_row'] = (df['grid_position'] <= 2).astype(int)

# Historical qualifying performance
df['recent_grid_avg_5'] = df.groupby('driver_abbreviation')['grid_position'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)

# Grid position improvement/decline
df['qualifying_trend'] = df.groupby('driver_abbreviation')['grid_position'].transform(
    lambda x: x.rolling(window=5, min_periods=2).apply(
        lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
    )
)

print("  ✅ Qualifying features created")

# ============================================
# 6. RACE STRATEGY FEATURES
# ============================================
print("⚙️  Creating Race Strategy Features...")

# Tire strategy effectiveness
df['tire_strategy_score'] = (
    df['soft_tire_laps'] * 1.2 +  # Soft tires are faster
    df['medium_tire_laps'] * 1.0 +
    df['hard_tire_laps'] * 0.9
) / df['total_laps_completed']

# Pit stop efficiency (fewer stops can be better)
df['pit_stop_efficiency'] = 1 / (df['num_pit_stops'] + 1)

# Consistency score (lower variance is better)
df['consistency_score'] = 1 / (df['lap_time_variance'] + 0.001)

print("  ✅ Strategy features created")

# ============================================
# 7. ENCODE CATEGORICAL VARIABLES
# ============================================
print("🔢 Encoding Categorical Variables...")

# Driver encoding (frequency-based)
driver_win_rate = df.groupby('driver_abbreviation')['on_podium'].mean()
df['driver_historical_podium_rate'] = df['driver_abbreviation'].map(driver_win_rate)

# Team encoding
team_win_rate = df.groupby('team_name')['on_podium'].mean()
df['team_historical_podium_rate'] = df['team_name'].map(team_win_rate)

# Circuit difficulty (based on average finish variance)
circuit_difficulty = df.groupby('circuit_name')['finish_position'].std()
df['circuit_difficulty'] = df['circuit_name'].map(circuit_difficulty)

print("  ✅ Categorical encoding complete")

# ============================================
# SAVE ENGINEERED FEATURES
# ============================================
print("\n💾 Saving Engineered Dataset...")

# Fill any remaining NaN values
df = df.fillna({
    'performance_trend': 0,
    'qualifying_trend': 0,
    'circuit_difficulty': df['circuit_difficulty'].mean(),
    'avg_speed': df['avg_speed'].mean(),
    'max_speed': df['max_speed'].mean(),
})

# Save
output_file = 'f1_data_with_features.csv'
df.to_csv(output_file, index=False)

print(f"✅ Saved to: {output_file}")
print(f"   Total entries: {len(df)}")
print(f"   Total features: {len(df.columns)}")

# ============================================
# FEATURE SUMMARY
# ============================================
print("\n" + "=" * 80)
print("📋 FEATURE ENGINEERING SUMMARY")
print("=" * 80)

feature_categories = {
    'Basic Features': ['grid_position', 'avg_lap_time', 'fastest_lap_time', 'num_pit_stops'],
    'Driver Momentum': ['recent_podiums_5', 'recent_points_avg_5', 'is_hot_streak', 'performance_trend'],
    'Team Performance': ['team_recent_podiums_5', 'team_avg_finish_5', 'team_points_last_5'],
    'Season Context': ['season_progress', 'championship_position', 'points_gap_to_leader'],
    'Circuit History': ['circuit_historical_podiums', 'circuit_historical_avg_finish', 'circuit_experience'],
    'Qualifying': ['starts_top10', 'recent_grid_avg_5', 'qualifying_trend'],
    'Strategy': ['tire_strategy_score', 'pit_stop_efficiency', 'consistency_score'],
    'Historical Rates': ['driver_historical_podium_rate', 'team_historical_podium_rate'],
}

for category, features in feature_categories.items():
    available = [f for f in features if f in df.columns]
    print(f"\n{category}:")
    for feature in available:
        print(f"  • {feature}")

print("\n" + "=" * 80)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("🚀 Next: Train ML models to predict podium finishes!")
print("=" * 80)

# Show top performing drivers in 2025
print("\n🏆 2025 TOP PERFORMERS (based on engineered features):")
df_2025 = df[df['year'] == 2025].groupby('driver_abbreviation').agg({
    'on_podium': 'sum',
    'recent_podiums_5': 'last',
    'recent_points_avg_5': 'last',
    'driver_historical_podium_rate': 'last'
}).sort_values('on_podium', ascending=False).head(10)

df_2025.columns = ['Podiums_2025', 'Recent_Podiums_L5', 'Avg_Points_L5', 'Career_Podium_Rate']
print(df_2025.round(2))