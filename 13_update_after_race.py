"""
Update Data After New Race
Run this script after each race to add new data
"""

import fastf1
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

fastf1.Cache.enable_cache('f1_cache')

print("🏁 F1 DATA UPDATER - Add Latest Race")
print("=" * 80)

# ============================================
# CONFIGURATION
# ============================================

# CHANGE THESE AFTER EACH RACE:
YEAR = 2025
RACE_NAME = "United States"  # Change to: "Mexico City", "Brazil", etc.

print(f"📥 Fetching: {YEAR} {RACE_NAME} Grand Prix\n")


# ============================================
# COLLECT NEW RACE DATA
# ============================================

def collect_race_data(year, race_name):
    """Collects data for a single race"""
    try:
        print(f"Loading {year} {race_name}...", end=" ", flush=True)

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
        print(f"❌ {str(e)}")
        return None


# Collect the new race
new_race_df = collect_race_data(YEAR, RACE_NAME)

if new_race_df is not None and len(new_race_df) > 0:

    # Load existing data
    existing_df = pd.read_csv('f1_complete_data.csv')
    print(f"\n✅ Current data: {len(existing_df)} entries")

    # Check if race already exists
    existing_df['race_date'] = pd.to_datetime(existing_df['race_date'])
    race_exists = ((existing_df['year'] == YEAR) &
                   (existing_df['race_name'] == new_race_df['race_name'].iloc[0])).any()

    if race_exists:
        print(f"⚠️  Race already in database. Replacing old data...")
        existing_df = existing_df[~((existing_df['year'] == YEAR) &
                                    (existing_df['race_name'] == new_race_df['race_name'].iloc[0]))]

    # Add new race
    combined_df = pd.concat([existing_df, new_race_df], ignore_index=True)
    combined_df = combined_df.sort_values(['year', 'race_date', 'driver_abbreviation'])

    print(f"✅ Updated data: {len(combined_df)} entries")

    # Show race results
    print(f"\n🏁 {RACE_NAME} RESULTS:")
    print("-" * 80)
    podium = new_race_df[new_race_df['on_podium'] == 1].sort_values('finish_position')
    for _, driver in podium.iterrows():
        pos = int(driver['finish_position'])
        medal = ["🥇", "🥈", "🥉"][pos - 1]
        print(f"   {medal} {driver['driver_abbreviation']} ({driver['team_name']})")

    # Save
    combined_df.to_csv('f1_complete_data.csv', index=False)
    print(f"\n✅ Saved to: f1_complete_data.csv")

    import sqlite3

    conn = sqlite3.connect('f1_complete_data.db')
    combined_df.to_sql('race_data', conn, if_exists='replace', index=False)
    conn.close()
    print(f"✅ Saved to: f1_complete_data.db")

    print("\n" + "=" * 80)
    print("✅ DATA UPDATED!")
    print("=" * 80)
    print("\n🔄 NEXT STEPS:")
    print("   1. Run: python3 07_advanced_feature_engineering.py")
    print("   2. Run: python3 12_predict_us_gp_podium.py")
    print("      (Update RACE_NAME to next race!)")
    print("=" * 80)

else:
    print("\n❌ Failed to collect race data")
    print("   Check that the race name is correct")
    print("   Try: 'United States', 'Mexico City', 'Brazil', 'Las Vegas', etc.")