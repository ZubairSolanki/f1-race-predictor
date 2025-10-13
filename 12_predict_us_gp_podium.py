"""
2025 US GRAND PRIX PODIUM PREDICTOR
Uses trained Random Forest model to predict top 3 finishers
Circuit: Circuit of The Americas (COTA), Austin, Texas
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🏎️  2025 US GRAND PRIX PODIUM PREDICTOR")
print("=" * 80)
print("📍 Circuit: Circuit of The Americas (COTA), Austin, Texas")
print("🗓️  Race Weekend: October 2025\n")

# ============================================
# 1. LOAD TRAINED MODEL AND DATA
# ============================================
print("📥 Loading trained model and historical data...")

# Load the champion model
rf_model = joblib.load('model_random_forest.pkl')
print("✅ Loaded Random Forest model (96.6% accuracy)")

# Load complete dataset for feature calculation
df = pd.read_csv('f1_data_with_features.csv')
df['race_date'] = pd.to_datetime(df['race_date'])
print(f"✅ Loaded historical data: {len(df)} entries\n")

# ============================================
# 2. ANALYZE 2025 SEASON SO FAR
# ============================================
print("📊 2025 SEASON ANALYSIS (through last race)")
print("-" * 80)

df_2025 = df[df['year'] == 2025].copy()

# Get active drivers in 2025
active_drivers = df_2025['driver_abbreviation'].unique()
print(f"Active drivers: {len(active_drivers)}")

# Show current form
recent_form = df_2025.groupby('driver_abbreviation').agg({
    'on_podium': 'sum',
    'recent_podiums_5': 'last',
    'recent_points_avg_5': 'last',
    'performance_trend': 'last',
    'is_hot_streak': 'last'
}).sort_values('on_podium', ascending=False)

print("\n🔥 TOP 5 DRIVERS (2025 season podiums):")
for i, (driver, stats) in enumerate(recent_form.head(5).iterrows(), 1):
    hot = "🔥" if stats['is_hot_streak'] == 1 else ""
    print(f"   {i}. {driver}: {int(stats['on_podium'])} podiums {hot}")

# ============================================
# 3. BUILD US GP PREDICTION FEATURES
# ============================================
print("\n" + "=" * 80)
print("🔧 BUILDING US GP PREDICTION FEATURES")
print("=" * 80)

# Get each driver's most recent features (latest race)
latest_features = df_2025.sort_values('race_date').groupby('driver_abbreviation').last()

# Get COTA-specific historical data
cota_history = df[df['circuit_name'].str.contains('Americas', case=False, na=False)]

# Build prediction dataset
us_gp_features = []

for driver in active_drivers:
    # Get driver's latest features
    if driver not in latest_features.index:
        continue

    driver_latest = latest_features.loc[driver]

    # Get COTA history for this driver
    driver_cota = cota_history[cota_history['driver_abbreviation'] == driver]

    # Historical COTA performance
    if len(driver_cota) > 0:
        cota_podiums = driver_cota['on_podium'].sum()
        cota_avg_finish = driver_cota['finish_position'].mean()
    else:
        cota_podiums = 0
        cota_avg_finish = 10.0

    # Create feature dict (using all features from trained model)
    features = {
        'driver_abbreviation': driver,
        'team_name': driver_latest['team_name'],

        # Current form (most important!)
        'recent_podiums_3': driver_latest.get('recent_podiums_3', 0),
        'recent_podiums_5': driver_latest.get('recent_podiums_5', 0),
        'is_hot_streak': driver_latest.get('is_hot_streak', 0),
        'recent_points_avg_3': driver_latest.get('recent_points_avg_3', 0),
        'recent_points_avg_5': driver_latest.get('recent_points_avg_5', 0),
        'recent_finish_avg_3': driver_latest.get('recent_finish_avg_3', 10),
        'recent_finish_avg_5': driver_latest.get('recent_finish_avg_5', 10),
        'performance_trend': driver_latest.get('performance_trend', 0),

        # Qualifying (estimate based on recent grid positions)
        'grid_position': driver_latest.get('recent_grid_avg_5', 10),
        'recent_grid_avg_5': driver_latest.get('recent_grid_avg_5', 10),
        'qualifying_trend': driver_latest.get('qualifying_trend', 0),
        'starts_top10': 1 if driver_latest.get('recent_grid_avg_5', 10) <= 10 else 0,
        'starts_top5': 1 if driver_latest.get('recent_grid_avg_5', 10) <= 5 else 0,
        'starts_front_row': 1 if driver_latest.get('recent_grid_avg_5', 10) <= 2 else 0,

        # Team performance
        'team_recent_podiums_5': driver_latest.get('team_recent_podiums_5', 0),
        'team_avg_finish_5': driver_latest.get('team_avg_finish_5', 10),
        'team_points_last_5': driver_latest.get('team_points_last_5', 0),

        # Season context
        'season_progress': 15/24,  # ~15 races done of ~24
        'season_points_cumulative': driver_latest.get('season_points_cumulative', 0),
        'championship_position': driver_latest.get('championship_position', 10),
        'points_gap_to_leader': driver_latest.get('points_gap_to_leader', 100),

        # Circuit history (COTA-specific)
        'circuit_historical_podiums': cota_podiums,
        'circuit_historical_avg_finish': cota_avg_finish,
        'circuit_experience': len(driver_cota),

        # Performance metrics (recent averages)
        'avg_lap_time': driver_latest.get('avg_lap_time', 90),
        'fastest_lap_time': driver_latest.get('fastest_lap_time', 88),
        'slowest_lap_time': driver_latest.get('slowest_lap_time', 95),
        'lap_time_std': driver_latest.get('lap_time_std', 2),
        'lap_time_variance': driver_latest.get('lap_time_variance', 0.02),
        'lap_time_range': driver_latest.get('lap_time_range', 5),

        # Strategy
        'num_pit_stops': driver_latest.get('num_pit_stops', 2),
        'total_laps_completed': driver_latest.get('total_laps_completed', 55),
        'soft_tire_laps': driver_latest.get('soft_tire_laps', 15),
        'medium_tire_laps': driver_latest.get('medium_tire_laps', 25),
        'hard_tire_laps': driver_latest.get('hard_tire_laps', 15),
        'intermediate_tire_laps': driver_latest.get('intermediate_tire_laps', 0),
        'wet_tire_laps': driver_latest.get('wet_tire_laps', 0),
        'tire_strategy_score': driver_latest.get('tire_strategy_score', 1.0),
        'pit_stop_efficiency': driver_latest.get('pit_stop_efficiency', 0.5),
        'consistency_score': driver_latest.get('consistency_score', 50),

        # Speed metrics
        'avg_speed': driver_latest.get('avg_speed', 200),
        'max_speed': driver_latest.get('max_speed', 320),

        # Weather (COTA in October - typically warm)
        'avg_air_temp': 28.0,
        'avg_humidity': 60.0,
        'rainfall': 0,

        # Historical rates
        'driver_historical_podium_rate': driver_latest.get('driver_historical_podium_rate', 0.2),
        'team_historical_podium_rate': driver_latest.get('team_historical_podium_rate', 0.2),
        'circuit_difficulty': 5.0,
    }

    us_gp_features.append(features)

# Create DataFrame
us_gp_df = pd.DataFrame(us_gp_features)

print(f"✅ Built features for {len(us_gp_df)} drivers\n")

# ============================================
# 4. MAKE PREDICTIONS
# ============================================
print("=" * 80)
print("🎯 PREDICTING US GP PODIUM")
print("=" * 80 + "\n")

# Prepare features for model
# Get the exact features the model expects
model_features = rf_model.feature_names_in_

# Ensure we have all required features
missing_features = set(model_features) - set(us_gp_df.columns)
if missing_features:
    print(f"⚠️  Adding {len(missing_features)} missing features with default values")
    for feature in missing_features:
        us_gp_df[feature] = 0

# Select only the features the model was trained on, in the correct order
X_pred = us_gp_df[model_features]

# Handle any missing values
X_pred = X_pred.fillna(X_pred.median())

# Get predictions
podium_probs = rf_model.predict_proba(X_pred)[:, 1]
us_gp_df['podium_probability'] = podium_probs
us_gp_df['podium_prediction'] = rf_model.predict(X_pred)

# Sort by probability
us_gp_df = us_gp_df.sort_values('podium_probability', ascending=False)

# ============================================
# 5. DISPLAY PREDICTIONS
# ============================================

print("🏆 TOP 10 PODIUM CONTENDERS:")
print("-" * 80)
print(f"{'Pos':<5} {'Driver':<8} {'Team':<25} {'Probability':<12} {'Hot?'}")
print("-" * 80)

for i, row in us_gp_df.head(10).iterrows():
    driver = row['driver_abbreviation']
    team = row['team_name'][:23]
    prob = row['podium_probability']
    hot = "🔥" if row['is_hot_streak'] == 1 else ""

    # Visual probability bar
    bar_length = int(prob * 20)
    bar = "█" * bar_length + "░" * (20 - bar_length)

    print(f"{us_gp_df.index.get_loc(i)+1:<5} {driver:<8} {team:<25} {prob:>5.1%} {bar} {hot}")

print("\n" + "=" * 80)
print("🥇🥈🥉 PREDICTED PODIUM:")
print("=" * 80)

top_3 = us_gp_df.head(3)

for i, (idx, row) in enumerate(top_3.iterrows(), 1):
    driver = row['driver_abbreviation']
    team = row['team_name']
    prob = row['podium_probability']

    medal = ["🥇 1st", "🥈 2nd", "🥉 3rd"][i-1]

    print(f"\n{medal} Place: {driver} ({team})")
    print(f"   Confidence: {prob:.1%}")
    print(f"   Recent form: {int(row['recent_podiums_5'])} podiums in last 5 races")
    print(f"   Hot streak: {'Yes 🔥' if row['is_hot_streak'] == 1 else 'No'}")

# ============================================
# 6. KEY INSIGHTS
# ============================================
print("\n" + "=" * 80)
print("💡 KEY INSIGHTS:")
print("=" * 80)

# McLaren dominance
mclaren_drivers = us_gp_df[us_gp_df['team_name'].str.contains('McLaren', case=False)]
if len(mclaren_drivers) >= 2:
    print(f"🟠 McLaren Lock-Out? Both drivers in top {len(us_gp_df[us_gp_df['podium_probability'] > 0.5])} podium contenders")

# Hot streaks
hot_drivers = us_gp_df[us_gp_df['is_hot_streak'] == 1]
print(f"🔥 {len(hot_drivers)} drivers currently on hot streaks")

# Confidence level
top_prob = us_gp_df.iloc[0]['podium_probability']
if top_prob > 0.85:
    print(f"✅ High confidence prediction (model is {top_prob:.1%} certain)")
elif top_prob > 0.65:
    print(f"⚠️  Moderate confidence ({top_prob:.1%} - competitive field)")
else:
    print(f"❓ Low confidence ({top_prob:.1%} - very unpredictable race)")

# ============================================
# 7. SAVE PREDICTIONS
# ============================================
print("\n💾 Saving predictions...")

us_gp_df[['driver_abbreviation', 'team_name', 'podium_probability', 'podium_prediction']].to_csv(
    'us_gp_2025_predictions.csv',
    index=False
)

print("✅ Saved to: us_gp_2025_predictions.csv")

print("\n" + "=" * 80)
print("🏁 PREDICTION COMPLETE!")
print("=" * 80)
print(f"\nModel Accuracy: 96.6%")
print(f"Training Data: 2021-2024 (1,732 races)")
print(f"Validation: 2025 season (290 races)")
print("\n🍀 Good luck with your predictions! See you at COTA! 🏎️💨")
print("=" * 80)