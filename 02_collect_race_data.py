"""
F1 Data Collection Pipeline
Collects 4 years of F1 race data for podium prediction
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import time
import os

# Create cache directory if it doesn't exist
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"📁 Created cache directory: {cache_dir}\n")

# Enable cache
fastf1.Cache.enable_cache(cache_dir)

print("🏎️  F1 PODIUM PREDICTOR - DATA COLLECTION")
print("=" * 80)
print("📦 Collecting 4 years of race data...\n")

# ============================================
# CONFIGURATION
# ============================================
YEARS = [2021, 2022, 2023, 2024]  # 4 years of data
DATABASE_NAME = 'f1_data.db'

# ============================================
# HELPER FUNCTIONS
# ============================================

def collect_race_data(year, race_name):
    """
    Collects data for a single race.
    Returns a DataFrame with features for each driver.
    """
    try:
        print(f"  📥 Loading {year} {race_name}...", end=" ")

        # Load the race session
        session = fastf1.get_session(year, race_name, 'R')
        session.load()

        # Get race results
        results = session.results

        # Get all laps
        laps = session.laps

        # Get weather data (if available)
        try:
            weather = session.weather_data
            avg_temp = weather['AirTemp'].mean() if len(weather) > 0 else None
            avg_humidity = weather['Humidity'].mean() if len(weather) > 0 else None
            rainfall = weather['Rainfall'].any() if len(weather) > 0 else False
        except:
            avg_temp = None
            avg_humidity = None
            rainfall = False

        # Collect data for each driver
        race_data = []

        for idx, driver_result in results.iterrows():
            driver_abbr = driver_result['Abbreviation']

            # Skip if driver didn't participate
            if pd.isna(driver_abbr):
                continue

            # Get driver's laps
            driver_laps = laps.pick_driver(driver_abbr)

            if len(driver_laps) == 0:
                continue

            # Convert lap times to seconds
            driver_laps['LapTimeSeconds'] = driver_laps['LapTime'].dt.total_seconds()

            # Remove invalid laps (pit stops, yellow flags, etc.)
            valid_laps = driver_laps[driver_laps['LapTimeSeconds'].notna()]

            if len(valid_laps) == 0:
                continue

            # ============================================
            # FEATURE ENGINEERING
            # ============================================

            # Basic driver info
            driver_number = driver_result['DriverNumber']
            team_name = driver_result['TeamName']
            grid_position = driver_result['GridPosition']
            finish_position = driver_result['Position']

            # Target variable: Is this driver on the podium? (Position 1, 2, or 3)
            on_podium = 1 if finish_position <= 3 else 0

            # Lap time features
            avg_lap_time = valid_laps['LapTimeSeconds'].mean()
            fastest_lap_time = valid_laps['LapTimeSeconds'].min()
            lap_time_std = valid_laps['LapTimeSeconds'].std()

            # Tire strategy features
            tire_compounds = valid_laps['Compound'].value_counts().to_dict()
            num_pit_stops = len(valid_laps[valid_laps['PitInTime'].notna()])

            # Consistency metric (lower is better)
            lap_time_variance = lap_time_std / avg_lap_time if avg_lap_time > 0 else 0

            # Track position features
            total_laps = len(valid_laps)

            # Create feature dictionary
            features = {
                'year': year,
                'race_name': race_name,
                'race_date': session.date,
                'circuit_name': session.event['Location'],
                'driver_abbreviation': driver_abbr,
                'driver_number': driver_number,
                'team_name': team_name,
                'grid_position': grid_position,
                'finish_position': finish_position,
                'on_podium': on_podium,  # TARGET VARIABLE
                'avg_lap_time': avg_lap_time,
                'fastest_lap_time': fastest_lap_time,
                'lap_time_std': lap_time_std,
                'lap_time_variance': lap_time_variance,
                'num_pit_stops': num_pit_stops,
                'total_laps_completed': total_laps,
                'soft_tire_laps': tire_compounds.get('SOFT', 0),
                'medium_tire_laps': tire_compounds.get('MEDIUM', 0),
                'hard_tire_laps': tire_compounds.get('HARD', 0),
                'avg_air_temp': avg_temp,
                'avg_humidity': avg_humidity,
                'rainfall': rainfall,
            }

            race_data.append(features)

        print(f"✅ ({len(race_data)} drivers)")
        return pd.DataFrame(race_data)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def save_to_database(df, db_name):
    """
    Saves DataFrame to SQLite database.
    """
    conn = sqlite3.connect(db_name)
    df.to_sql('race_data', conn, if_exists='append', index=False)
    conn.close()
    print(f"💾 Saved {len(df)} records to database")


# ============================================
# MAIN DATA COLLECTION LOOP
# ============================================

all_race_data = []
total_races = 0
successful_races = 0

for year in YEARS:
    print(f"\n📅 YEAR {year}")
    print("-" * 80)

    # Get the race schedule for this year
    try:
        schedule = fastf1.get_event_schedule(year)

        # Filter for actual races (not testing, sprints, etc.)
        races = schedule[schedule['EventFormat'] != 'testing']

        for idx, race in races.iterrows():
            race_name = race['EventName']

            # Skip if race hasn't happened yet
            if pd.notna(race['EventDate']) and race['EventDate'] > datetime.now():
                continue

            total_races += 1

            # Collect data for this race
            race_df = collect_race_data(year, race_name)

            if race_df is not None and len(race_df) > 0:
                all_race_data.append(race_df)
                successful_races += 1

            # Be nice to the API - small delay between requests
            time.sleep(1)

    except Exception as e:
        print(f"  ❌ Error loading {year} schedule: {e}")
        continue

# ============================================
# COMBINE AND SAVE ALL DATA
# ============================================

if len(all_race_data) > 0:
    print("\n" + "=" * 80)
    print("📊 DATA COLLECTION SUMMARY")
    print("=" * 80)

    # Combine all races into one DataFrame
    final_df = pd.concat(all_race_data, ignore_index=True)

    print(f"Total races processed: {successful_races}/{total_races}")
    print(f"Total driver entries: {len(final_df)}")
    print(f"Podium finishes: {final_df['on_podium'].sum()}")
    print(f"Non-podium finishes: {(final_df['on_podium'] == 0).sum()}")
    print(f"\nFeatures collected: {len(final_df.columns)}")

    # Save to SQLite database
    print(f"\n💾 Saving to database: {DATABASE_NAME}")
    conn = sqlite3.connect(DATABASE_NAME)
    final_df.to_sql('race_data', conn, if_exists='replace', index=False)
    conn.close()

    # Save as CSV backup
    csv_filename = 'f1_race_data.csv'
    final_df.to_csv(csv_filename, index=False)
    print(f"💾 Backup saved to CSV: {csv_filename}")

    # Show sample data
    print("\n📋 Sample Data (first 5 rows):")
    print("-" * 80)
    print(final_df.head())

    print("\n✅ DATA COLLECTION COMPLETE!")
    print("\n🚀 Next step: Feature engineering and model training!")

else:
    print("\n❌ No data collected. Please check your internet connection.")