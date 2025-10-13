"""
BULLETPROOF F1 Data Collection - 2021 through 2025
Collects ALL data needed to predict 2025 US Grand Prix podium
Includes retry logic and detailed error reporting
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import time
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

print("🏎️  F1 PODIUM PREDICTOR - BULLETPROOF DATA COLLECTION")
print("=" * 80)
print("📦 Collecting 5 YEARS of data (2021-2025) for US GP prediction...")
print("⏱️  This will take 20-30 minutes. Grab a coffee! ☕\n")

# ============================================
# CONFIGURATION
# ============================================
YEARS = [2021, 2022, 2023, 2024, 2025]  # All 5 years!
DATABASE_NAME = 'f1_complete_data.db'
CSV_NAME = 'f1_complete_data.csv'
MAX_RETRIES = 2  # Retry failed races


# ============================================
# HELPER FUNCTION
# ============================================

def collect_race_data(year, race_name, retry_attempt=0):
    """
    Collects comprehensive data for a single race.
    Returns DataFrame with engineered features.
    """
    try:
        # Load the race session
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

            # Get driver's laps (try new API, fallback to old)
            try:
                driver_laps = laps.pick_drivers(driver_abbr)
            except:
                try:
                    driver_laps = laps.pick_driver(driver_abbr)
                except:
                    continue

            if len(driver_laps) == 0:
                continue

            # Convert lap times to seconds
            driver_laps = driver_laps.copy()
            driver_laps['LapTimeSeconds'] = driver_laps['LapTime'].dt.total_seconds()

            # Remove invalid laps
            valid_laps = driver_laps[driver_laps['LapTimeSeconds'].notna()]

            if len(valid_laps) == 0:
                continue

            # ============================================
            # FEATURE ENGINEERING
            # ============================================

            driver_number = driver_result['DriverNumber']
            team_name = driver_result['TeamName']
            grid_position = driver_result['GridPosition']
            finish_position = driver_result['Position']

            # Handle DNS (Did Not Start), DNF (Did Not Finish), DSQ (Disqualified)
            if pd.isna(finish_position):
                finish_position = 99  # Mark as non-finisher

            # TARGET: Podium finish (1st, 2nd, or 3rd)
            on_podium = 1 if finish_position <= 3 else 0

            # Lap time features
            avg_lap_time = valid_laps['LapTimeSeconds'].mean()
            fastest_lap_time = valid_laps['LapTimeSeconds'].min()
            slowest_lap_time = valid_laps['LapTimeSeconds'].max()
            lap_time_std = valid_laps['LapTimeSeconds'].std()

            # Consistency metrics
            lap_time_variance = lap_time_std / avg_lap_time if avg_lap_time > 0 else 0
            lap_time_range = slowest_lap_time - fastest_lap_time

            # Tire strategy
            tire_compounds = valid_laps['Compound'].value_counts().to_dict()
            num_pit_stops = len(valid_laps[valid_laps['PitInTime'].notna()])

            # Track position metrics
            total_laps = len(valid_laps)

            # Speed metrics (if available)
            try:
                avg_speed = valid_laps['SpeedI1'].mean()
                max_speed = valid_laps['SpeedI1'].max()
            except:
                avg_speed = None
                max_speed = None

            # Create feature dictionary
            features = {
                'year': year,
                'race_name': race_name,
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

                # Lap time features
                'avg_lap_time': avg_lap_time,
                'fastest_lap_time': fastest_lap_time,
                'slowest_lap_time': slowest_lap_time,
                'lap_time_std': lap_time_std,
                'lap_time_variance': lap_time_variance,
                'lap_time_range': lap_time_range,

                # Strategy features
                'num_pit_stops': num_pit_stops,
                'total_laps_completed': total_laps,
                'soft_tire_laps': tire_compounds.get('SOFT', 0),
                'medium_tire_laps': tire_compounds.get('MEDIUM', 0),
                'hard_tire_laps': tire_compounds.get('HARD', 0),
                'intermediate_tire_laps': tire_compounds.get('INTERMEDIATE', 0),
                'wet_tire_laps': tire_compounds.get('WET', 0),

                # Speed features
                'avg_speed': avg_speed,
                'max_speed': max_speed,

                # Weather features
                'avg_air_temp': avg_temp,
                'avg_humidity': avg_humidity,
                'rainfall': rainfall,
            }

            race_data.append(features)

        return pd.DataFrame(race_data)

    except Exception as e:
        if retry_attempt < MAX_RETRIES:
            print(f" [Retry {retry_attempt + 1}]", end="")
            time.sleep(3)
            return collect_race_data(year, race_name, retry_attempt + 1)
        else:
            raise e


# ============================================
# MAIN COLLECTION LOOP
# ============================================

all_race_data = []
race_details = []  # Track each race's status

for year in YEARS:
    print(f"\n📅 YEAR {year}")
    print("-" * 80)

    try:
        # Get race schedule
        schedule = fastf1.get_event_schedule(year)

        # Get current date to filter completed races
        now = datetime.now()

        year_races = 0
        year_success = 0

        for idx, race in schedule.iterrows():
            race_name = race['EventName']

            # Skip if race hasn't happened yet
            if pd.notna(race['EventDate']) and race['EventDate'] > now:
                continue

            # Skip testing/sprint qualifying sessions
            if 'test' in race_name.lower() or race.get('EventFormat') == 'testing':
                continue

            year_races += 1

            # Show progress
            print(f"  {year_races:2d}. {race_name:40s} ", end="", flush=True)

            # Collect data
            try:
                race_df = collect_race_data(year, race_name)

                if race_df is not None and len(race_df) > 0:
                    all_race_data.append(race_df)
                    year_success += 1
                    print(f"✅ ({len(race_df):2d} drivers)")

                    race_details.append({
                        'year': year,
                        'race': race_name,
                        'status': 'SUCCESS',
                        'drivers': len(race_df)
                    })
                else:
                    print(f"❌ No data")
                    race_details.append({
                        'year': year,
                        'race': race_name,
                        'status': 'NO_DATA',
                        'drivers': 0
                    })

            except Exception as e:
                print(f"❌ {str(e)[:40]}")
                race_details.append({
                    'year': year,
                    'race': race_name,
                    'status': 'FAILED',
                    'drivers': 0,
                    'error': str(e)[:50]
                })

            # Be nice to the API
            time.sleep(1.5)

        print(f"\n  Year {year} Summary: {year_success}/{year_races} races collected")

    except Exception as e:
        print(f"  ❌ Error loading {year} schedule: {e}")
        continue

# ============================================
# SAVE RESULTS & ANALYSIS
# ============================================

print("\n" + "=" * 80)
print("📊 DATA COLLECTION COMPLETE!")
print("=" * 80)

if len(all_race_data) > 0:
    # Combine all races
    final_df = pd.concat(all_race_data, ignore_index=True)

    # Summary statistics
    total_races = len([r for r in race_details if r['status'] == 'SUCCESS'])
    failed_races = [r for r in race_details if r['status'] != 'SUCCESS']

    print(f"\n✅ COLLECTION SUMMARY:")
    print(f"   Total races attempted: {len(race_details)}")
    print(f"   Successfully collected: {total_races}")
    print(f"   Failed: {len(failed_races)}")
    print(f"   Total driver entries: {len(final_df)}")
    print(f"   Podium finishes: {final_df['on_podium'].sum()}")
    print(f"   Non-podium finishes: {(final_df['on_podium'] == 0).sum()}")

    # Year breakdown
    print(f"\n📅 DATA BY YEAR:")
    year_summary = final_df.groupby('year').agg({
        'race_name': 'nunique',
        'driver_abbreviation': 'count',
        'on_podium': 'sum'
    })
    year_summary.columns = ['Races', 'Entries', 'Podiums']
    print(year_summary.to_string())

    # Expected vs Actual
    print(f"\n🔍 DATA QUALITY CHECK:")
    for year in YEARS:
        year_data = final_df[final_df['year'] == year]
        races = year_data['race_name'].nunique()
        entries = len(year_data)
        expected = races * 20  # ~20 drivers per race
        completeness = (entries / expected * 100) if expected > 0 else 0
        print(
            f"   {year}: {entries:4d} entries from {races:2d} races (Expected ~{expected:4d}, {completeness:.0f}% complete)")

    # Most recent 2025 races
    if 2025 in final_df['year'].values:
        print(f"\n🏁 MOST RECENT 2025 RACES (for US GP prediction):")
        recent_2025 = final_df[final_df['year'] == 2025].groupby('race_name')['race_date'].first().sort_values(
            ascending=False).head(5)
        for race, date in recent_2025.items():
            winner = \
            final_df[(final_df['year'] == 2025) & (final_df['race_name'] == race) & (final_df['finish_position'] == 1)][
                'driver_abbreviation'].values
            winner_name = winner[0] if len(winner) > 0 else 'N/A'
            print(f"   • {date.strftime('%Y-%m-%d')} - {race:35s} Winner: {winner_name}")

    # Save to database
    print(f"\n💾 SAVING DATA:")
    conn = sqlite3.connect(DATABASE_NAME)
    final_df.to_sql('race_data', conn, if_exists='replace', index=False)
    conn.close()
    print(f"   ✅ Database: {DATABASE_NAME}")

    # Save CSV backup
    final_df.to_csv(CSV_NAME, index=False)
    print(f"   ✅ CSV: {CSV_NAME}")

    # Show failed races
    if failed_races:
        print(f"\n⚠️  FAILED RACES ({len(failed_races)}):")
        for race in failed_races[:10]:
            print(f"   • {race['year']} {race['race']}: {race['status']}")
        if len(failed_races) > 10:
            print(f"   ... and {len(failed_races) - 10} more")

    # Final stats
    print("\n" + "=" * 80)
    print("🎯 DATASET READY!")
    print(f"   📊 Total entries: {len(final_df)}")
    print(f"   📈 Features per entry: {len(final_df.columns)}")
    print(f"   🏎️  Years covered: {sorted(final_df['year'].unique())}")
    print(f"   🏁 Races collected: {final_df['race_name'].nunique()}")
    print(f"   👤 Unique drivers: {final_df['driver_abbreviation'].nunique()}")
    print(f"   🏆 Ready to predict 2025 US Grand Prix podium!")
    print("=" * 80)
    print("\n🚀 NEXT STEP: Run verification script to analyze the data!")

else:
    print("\n❌ No data collected. Please check errors above.")