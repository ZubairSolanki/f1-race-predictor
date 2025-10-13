"""
Add Missing 2025 Races: Italy (Verstappen win) and Singapore (Russell win)
Critical for accurate US GP prediction!
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

fastf1.Cache.enable_cache('f1_cache')

print("🏎️  ADDING CRITICAL MISSING RACES")
print("=" * 80)
print("Adding: Italy GP (VER win) & Singapore GP (RUS win)\n")


# ============================================
# FUNCTION TO COLLECT RACE DATA
# ============================================

def collect_race_data(year, race_name):
    """Collects data for a single race"""
    try:
        print(f"📥 Loading {year} {race_name}...", end=" ", flush=True)

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

        print(f"✅ ({len(race_data)} drivers)")
        winner = [r for r in race_data if r['finish_position'] == 1]
        if winner:
            print(f"   🏆 Winner: {winner[0]['driver_abbreviation']}")

        return pd.DataFrame(race_data)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


# ============================================
# COLLECT THE TWO MISSING RACES
# ============================================

missing_races = []

# Try Italy
italy_df = collect_race_data(2025, 'Italy')
if italy_df is not None and len(italy_df) > 0:
    missing_races.append(italy_df)

# Try Singapore
singapore_df = collect_race_data(2025, 'Singapore')
if singapore_df is not None and len(singapore_df) > 0:
    missing_races.append(singapore_df)

# ============================================
# MERGE WITH EXISTING DATA
# ============================================

if len(missing_races) > 0:
    print("\n" + "=" * 80)
    print("📊 MERGING WITH EXISTING DATA")
    print("=" * 80)

    # Load existing data
    existing_df = pd.read_csv('f1_complete_data.csv')
    print(f"✅ Current data: {len(existing_df)} entries")

    # Add new races
    new_df = pd.concat(missing_races, ignore_index=True)
    print(f"✅ New data: {len(new_df)} entries")

    # Combine and remove duplicates
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(
        subset=['year', 'race_name', 'driver_abbreviation'],
        keep='last'
    )

    print(f"✅ Combined total: {len(combined_df)} entries")

    # Show 2025 summary
    print(f"\n📅 2025 SEASON SUMMARY:")
    df_2025 = combined_df[combined_df['year'] == 2025]
    races_2025 = df_2025.groupby('race_name').agg({
        'race_date': 'first',
        'driver_abbreviation': 'count'
    }).sort_values('race_date')

    print(f"Total 2025 races: {len(races_2025)}")
    print(f"Total 2025 entries: {len(df_2025)}")

    # Show recent winners
    print(f"\n🏁 MOST RECENT 2025 RACES:")
    recent = df_2025.groupby('race_name')['race_date'].first().sort_values(ascending=False).head(5)
    for race, date in recent.items():
        winner = df_2025[(df_2025['race_name'] == race) & (df_2025['finish_position'] == 1)][
            'driver_abbreviation'].values
        winner_name = winner[0] if len(winner) > 0 else 'N/A'
        print(f"   {date.strftime('%Y-%m-%d')} - {race:35s} Winner: {winner_name}")

    # Driver 2025 wins
    print(f"\n🏆 2025 WINS BY DRIVER:")
    wins_2025 = df_2025[df_2025['finish_position'] == 1]['driver_abbreviation'].value_counts()
    for driver, wins in wins_2025.head(10).items():
        print(f"   {driver}: {wins} wins")

    # Save
    print(f"\n💾 SAVING UPDATED DATASET:")
    combined_df.to_csv('f1_complete_data.csv', index=False)
    print(f"   ✅ CSV: f1_complete_data.csv")

    import sqlite3

    conn = sqlite3.connect('f1_complete_data.db')
    combined_df.to_sql('race_data', conn, if_exists='replace', index=False)
    conn.close()
    print(f"   ✅ Database: f1_complete_data.db")

    print("\n" + "=" * 80)
    print("✅ CRITICAL RACES ADDED!")
    print("🔄 Now re-run feature engineering (script 07) to update momentum features!")
    print("=" * 80)

else:
    print("\n❌ Could not collect the missing races")
    print("Your existing data is still valid, but predictions may be slightly less accurate")