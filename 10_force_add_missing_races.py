"""
FORCE ADD Italy and Singapore races to CSV
Ensures they're properly saved
"""

import fastf1
import pandas as pd
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')
fastf1.Cache.enable_cache('f1_cache')

print("🔧 FORCE ADDING MISSING RACES")
print("=" * 80)


def collect_race_data(year, race_name):
    """Collects data for a single race"""
    try:
        print(f"📥 Loading {year} {race_name}... ", end="", flush=True)

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
            avg_temp, avg_humidity, rainfall = None, None, False

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

            # All features
            driver_number = driver_result['DriverNumber']
            team_name = driver_result['TeamName']
            grid_position = driver_result['GridPosition']
            finish_position = driver_result['Position']

            if pd.isna(finish_position):
                finish_position = 99

            on_podium = 1 if finish_position <= 3 else 0

            avg_lap_time = valid_laps['LapTimeSeconds'].mean()
            fastest_lap_time = valid_laps['LapTimeSeconds'].min()
            slowest_lap_time = valid_laps['LapTimeSeconds'].max()
            lap_time_std = valid_laps['LapTimeSeconds'].std()
            lap_time_variance = lap_time_std / avg_lap_time if avg_lap_time > 0 else 0
            lap_time_range = slowest_lap_time - fastest_lap_time

            tire_compounds = valid_laps['Compound'].value_counts().to_dict()
            num_pit_stops = len(valid_laps[valid_laps['PitInTime'].notna()])
            total_laps = len(valid_laps)

            try:
                avg_speed = valid_laps['SpeedI1'].mean()
                max_speed = valid_laps['SpeedI1'].max()
            except:
                avg_speed, max_speed = None, None

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

        df = pd.DataFrame(race_data)
        print(f"✅ {len(df)} drivers")
        return df

    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        return None


# ============================================
# LOAD EXISTING DATA
# ============================================

print("📂 Loading existing CSV...")
df_existing = pd.read_csv('f1_complete_data.csv')
print(f"   Current entries: {len(df_existing)}")

# Check what's missing
df_existing['race_date'] = pd.to_datetime(df_existing['race_date'])
df_2025 = df_existing[df_existing['year'] == 2025]

has_italy = 'Italian Grand Prix' in df_2025['race_name'].values
has_singapore = 'Singapore Grand Prix' in df_2025['race_name'].values

print(f"   Italy GP: {'✅ Present' if has_italy else '❌ Missing'}")
print(f"   Singapore GP: {'✅ Present' if has_singapore else '❌ Missing'}")

# ============================================
# COLLECT MISSING RACES
# ============================================

new_races = []

if not has_italy:
    print("\n📥 Collecting Italy GP...")
    italy_df = collect_race_data(2025, 'Italy')
    if italy_df is not None and len(italy_df) > 0:
        new_races.append(italy_df)
        time.sleep(2)
else:
    print("\n✅ Italy GP already in dataset")

if not has_singapore:
    print("\n📥 Collecting Singapore GP...")
    singapore_df = collect_race_data(2025, 'Singapore')
    if singapore_df is not None and len(singapore_df) > 0:
        new_races.append(singapore_df)
        time.sleep(2)
else:
    print("\n✅ Singapore GP already in dataset")

# ============================================
# MERGE AND SAVE
# ============================================

if len(new_races) > 0:
    print("\n" + "=" * 80)
    print("💾 MERGING AND SAVING")
    print("=" * 80)

    # Combine new races
    df_new = pd.concat(new_races, ignore_index=True)
    print(f"✅ New races collected: {len(df_new)} entries")

    # Remove existing Italy/Singapore data (if any) to avoid duplicates
    df_cleaned = df_existing[
        ~((df_existing['year'] == 2025) &
          (df_existing['race_name'].isin(['Italian Grand Prix', 'Singapore Grand Prix'])))
    ]

    # Merge
    df_final = pd.concat([df_cleaned, df_new], ignore_index=True)
    df_final = df_final.sort_values(['year', 'race_date', 'driver_abbreviation'])

    print(f"✅ Final dataset: {len(df_final)} entries")

    # Verify 2025 data
    df_2025_final = df_final[df_final['year'] == 2025]
    print(f"\n📊 2025 RACES: {df_2025_final['race_name'].nunique()} total")

    # Show recent races
    print(f"\n🏁 MOST RECENT 2025 RACES:")
    recent = df_2025_final.groupby('race_name')['race_date'].first().sort_values(ascending=False).head(5)
    for race, date in recent.items():
        winner = df_2025_final[(df_2025_final['race_name'] == race) & (df_2025_final['finish_position'] == 1)][
            'driver_abbreviation'].values
        winner_name = winner[0] if len(winner) > 0 else 'N/A'
        print(f"   {date.strftime('%Y-%m-%d')} - {race:40s} Winner: {winner_name}")

    # Driver wins
    print(f"\n🏆 2025 DRIVER WINS:")
    wins = df_2025_final[df_2025_final['finish_position'] == 1]['driver_abbreviation'].value_counts()
    for driver, count in wins.items():
        print(f"   {driver}: {count} wins")

    # SAVE
    print(f"\n💾 SAVING...")
    df_final.to_csv('f1_complete_data.csv', index=False)
    print(f"   ✅ CSV saved: f1_complete_data.csv")

    import sqlite3

    conn = sqlite3.connect('f1_complete_data.db')
    df_final.to_sql('race_data', conn, if_exists='replace', index=False)
    conn.close()
    print(f"   ✅ Database saved: f1_complete_data.db")

    print("\n" + "=" * 80)
    print("✅ DATA COMPLETE!")
    print("🔄 Now run: python3 07_advanced_feature_engineering.py")
    print("=" * 80)

else:
    print("\n✅ All critical races already in dataset!")
    print("📊 Data is ready for feature engineering.")