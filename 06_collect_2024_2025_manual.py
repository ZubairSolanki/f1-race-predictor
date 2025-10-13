"""
Manual Collection for 2024 & 2025 Data
Bypasses buggy get_event_schedule() by directly loading known races
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import time
import os
import warnings

warnings.filterwarnings('ignore')

cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

print("🏎️  MANUAL 2024-2025 DATA COLLECTION")
print("=" * 80)
print("📦 Collecting missing 2024 & 2025 races...\n")

# ============================================
# MANUAL RACE LISTS (Known to exist)
# ============================================

RACES_2023_REMAINING = [
    'Singapore', 'Japan', 'Qatar', 'United States',
    'Mexico City', 'São Paulo', 'Las Vegas', 'Abu Dhabi'
]

RACES_2024 = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China',
    'Miami', 'Emilia Romagna', 'Monaco', 'Canada', 'Spain',
    'Austria', 'Great Britain', 'Hungary', 'Belgium', 'Netherlands',
    'Italy', 'Azerbaijan', 'Singapore', 'United States', 'Mexico City',
    'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi'
]

RACES_2025 = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China',
    'Miami', 'Emilia Romagna', 'Monaco', 'Canada', 'Spain',
    'Austria', 'Great Britain', 'Hungary', 'Belgium', 'Netherlands',
    'Italy', 'Azerbaijan', 'Singapore'
    # US GP hasn't happened yet - we're predicting it!
]


# ============================================
# HELPER FUNCTION
# ============================================

def collect_race_data(year, race_name):
    """
    Collects comprehensive data for a single race.
    """
    try:
        session = fastf1.get_session(year, race_name, 'R')
        session.load()

        results = session.results
        laps = session.laps

        # Weather data
        try:
            weather = session.weather_data
            avg_temp = weather['AirTemp'].mean() if len(weather) > 0 else None
            avg_humidity = weather['Humidity'].mean() if len(weather) > 0 else None
            rainfall = weather['Rainfall'].any() if len(weather) > 0 else False
        except:
            avg_temp = None
            avg_humidity = None
            rainfall = False

        race_data = []

        for idx, driver_result in results.iterrows():
            driver_abbr = driver_result['Abbreviation']

            if pd.isna(driver_abbr):
                continue

            # Get driver's laps
            try:
                driver_laps = laps.pick_drivers(driver_abbr)
            except:
                try:
                    driver_laps = laps.pick_driver(driver_abbr)
                except:
                    continue

            if len(driver_laps) == 0:
                continue

            driver_laps = driver_laps.copy()
            driver_laps['LapTimeSeconds'] = driver_laps['LapTime'].dt.total_seconds()
            valid_laps = driver_laps[driver_laps['LapTimeSeconds'].notna()]

            if len(valid_laps) == 0:
                continue

            # Feature engineering
            driver_number = driver_result['DriverNumber']
            team_name = driver_result['TeamName']
            grid_position = driver_result['GridPosition']
            finish_position = driver_result['Position']

            if pd.isna(finish_position):
                finish_position = 99

            on_podium = 1 if finish_position <= 3 else 0

            # Lap time features
            avg_lap_time = valid_laps['LapTimeSeconds'].mean()
            fastest_lap_time = valid_laps['LapTimeSeconds'].min()
            slowest_lap_time = valid_laps['LapTimeSeconds'].max()
            lap_time_std = valid_laps['LapTimeSeconds'].std()
            lap_time_variance = lap_time_std / avg_lap_time if avg_lap_time > 0 else 0
            lap_time_range = slowest_lap_time - fastest_lap_time

            # Tire strategy
            tire_compounds = valid_laps['Compound'].value_counts().to_dict()
            num_pit_stops = len(valid_laps[valid_laps['PitInTime'].notna()])
            total_laps = len(valid_laps)

            # Speed metrics
            try:
                avg_speed = valid_laps['SpeedI1'].mean()
                max_speed = valid_laps['SpeedI1'].max()
            except:
                avg_speed = None
                max_speed = None

            features = {
                'year': year,
                'race_name': session.event['EventName'],
                'race_date': session.date,
                'circuit_name': session.event['Location'],
                'country': session.event.get('Country', None),
                'driver_abbreviation': driver_abbr,
                'driver_number': driver_number,
                'team_name': team_name,
                'grid_position': grid_position if not pd.isna(grid_position) else 99,
                'finish_position': finish_position,
                'on_podium': on_podium,
                'points_scored': driver_result.get('Points', 0),
                'avg_lap_time': avg_lap_time,
                'fastest_lap_time': fastest_lap_time,
                'slowest_lap_time': slowest_lap_time,
                'lap_time_std': lap_time_std,
                'lap_time_variance': lap_time_variance,
                'lap_time_range': lap_time_range,
                'num_pit_stops': num_pit_stops,
                'total_laps_completed': total_laps,
                'soft_tire_laps': tire_compounds.get('SOFT', 0),
                'medium_tire_laps': tire_compounds.get('MEDIUM', 0),
                'hard_tire_laps': tire_compounds.get('HARD', 0),
                'intermediate_tire_laps': tire_compounds.get('INTERMEDIATE', 0),
                'wet_tire_laps': tire_compounds.get('WET', 0),
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'avg_air_temp': avg_temp,
                'avg_humidity': avg_humidity,
                'rainfall': rainfall,
            }

            race_data.append(features)

        return pd.DataFrame(race_data)

    except Exception as e:
        raise e


# ============================================
# COLLECT MISSING DATA
# ============================================

all_new_data = []
successful = 0
failed = 0

# Collect remaining 2023 races
print("📅 YEAR 2023 (Missing Races)")
print("-" * 80)
for i, race_name in enumerate(RACES_2023_REMAINING, 1):
    print(f"  {i:2d}. {race_name:35s} ", end="", flush=True)
    try:
        race_df = collect_race_data(2023, race_name)
        if race_df is not None and len(race_df) > 0:
            all_new_data.append(race_df)
            print(f"✅ ({len(race_df):2d} drivers)")
            successful += 1
        else:
            print(f"❌ No data")
            failed += 1
    except Exception as e:
        print(f"❌ {str(e)[:40]}")
        failed += 1
    time.sleep(2)

# Collect all 2024 races
print(f"\n📅 YEAR 2024")
print("-" * 80)
for i, race_name in enumerate(RACES_2024, 1):
    print(f"  {i:2d}. {race_name:35s} ", end="", flush=True)
    try:
        race_df = collect_race_data(2024, race_name)
        if race_df is not None and len(race_df) > 0:
            all_new_data.append(race_df)
            print(f"✅ ({len(race_df):2d} drivers)")
            successful += 1
        else:
            print(f"❌ No data")
            failed += 1
    except Exception as e:
        print(f"❌ {str(e)[:40]}")
        failed += 1
    time.sleep(2)

# Collect all 2025 races
print(f"\n📅 YEAR 2025")
print("-" * 80)
for i, race_name in enumerate(RACES_2025, 1):
    print(f"  {i:2d}. {race_name:35s} ", end="", flush=True)
    try:
        race_df = collect_race_data(2025, race_name)
        if race_df is not None and len(race_df) > 0:
            all_new_data.append(race_df)
            print(f"✅ ({len(race_df):2d} drivers)")
            successful += 1
        else:
            print(f"❌ No data")
            failed += 1
    except Exception as e:
        print(f"❌ {str(e)[:40]}")
        failed += 1
    time.sleep(2)

# ============================================
# MERGE WITH EXISTING DATA
# ============================================

print("\n" + "=" * 80)
print("📊 MERGING WITH EXISTING DATA")
print("=" * 80)

if len(all_new_data) > 0:
    # Load existing data
    try:
        existing_df = pd.read_csv('f1_complete_data.csv')
        print(f"✅ Loaded existing data: {len(existing_df)} entries")
    except:
        existing_df = pd.DataFrame()
        print("⚠️  No existing data found, starting fresh")

    # Combine new data
    new_df = pd.concat(all_new_data, ignore_index=True)
    print(f"✅ Collected new data: {len(new_df)} entries")

    # Merge (remove duplicates based on year + race + driver)
    if len(existing_df) > 0:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=['year', 'race_name', 'driver_abbreviation'],
            keep='last'
        )
    else:
        combined_df = new_df

    print(f"✅ Total combined data: {len(combined_df)} entries")

    # Summary by year
    print(f"\n📅 FINAL DATASET BY YEAR:")
    year_summary = combined_df.groupby('year').agg({
        'race_name': 'nunique',
        'driver_abbreviation': 'count',
        'on_podium': 'sum'
    })
    year_summary.columns = ['Races', 'Entries', 'Podiums']
    print(year_summary.to_string())

    # Show 2025 races
    if 2025 in combined_df['year'].values:
        print(f"\n🏁 2025 RACES (for US GP prediction):")
        races_2025 = combined_df[combined_df['year'] == 2025].groupby('race_name')['race_date'].first().sort_values()
        for race, date in races_2025.items():
            winner = combined_df[(combined_df['year'] == 2025) & (combined_df['race_name'] == race) & (
                        combined_df['finish_position'] == 1)]['driver_abbreviation'].values
            winner_name = winner[0] if len(winner) > 0 else 'N/A'
            print(f"   {date.strftime('%Y-%m-%d')} - {race:35s} Winner: {winner_name}")

    # Save
    print(f"\n💾 SAVING COMPLETE DATASET:")
    combined_df.to_csv('f1_complete_data.csv', index=False)
    print(f"   ✅ CSV: f1_complete_data.csv")

    conn = sqlite3.connect('f1_complete_data.db')
    combined_df.to_sql('race_data', conn, if_exists='replace', index=False)
    conn.close()
    print(f"   ✅ Database: f1_complete_data.db")

    # Final stats
    print("\n" + "=" * 80)
    print("🎯 COLLECTION COMPLETE!")
    print(f"   New races collected: {successful}/{successful + failed}")
    print(f"   Total entries: {len(combined_df)}")
    print(f"   Years: {sorted(combined_df['year'].unique())}")
    print(f"   Total races: {combined_df['race_name'].nunique()}")
    print("=" * 80)
    print("\n🚀 Ready to build features and train the model!")

else:
    print("\n❌ No new data collected")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")