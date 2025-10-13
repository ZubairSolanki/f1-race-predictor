"""
Diagnose FastF1 API availability and check what data we can access
"""

import fastf1
import pandas as pd
from datetime import datetime
import time

fastf1.Cache.enable_cache('f1_cache')

print("🔍 FASTF1 API DIAGNOSTIC")
print("=" * 80)
print(f"FastF1 version: {fastf1.__version__}")
print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
print()

# ============================================
# Test 1: Check individual years
# ============================================
print("📅 TESTING YEAR-BY-YEAR ACCESS")
print("-" * 80)

for year in [2023, 2024, 2025]:
    print(f"\nYear {year}:")
    try:
        # Try to get schedule
        schedule = fastf1.get_event_schedule(year)
        print(f"  ✅ Schedule loaded: {len(schedule)} events")

        # Show completed races
        now = datetime.now()
        completed = schedule[schedule['EventDate'] < now]
        print(f"  ✅ Completed races: {len(completed)}")

        # Show some race names
        if len(completed) > 0:
            recent_races = completed['EventName'].tail(3).tolist()
            print(f"  📋 Recent races: {', '.join(recent_races)}")

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")

# ============================================
# Test 2: Try loading specific recent races
# ============================================
print("\n\n🏁 TESTING SPECIFIC RACE ACCESS")
print("-" * 80)

# Try some recent races that definitely happened
test_races = [
    (2024, 'Bahrain', 'Bahrain GP 2024'),
    (2024, 'Monaco', 'Monaco GP 2024'),
    (2025, 'Australia', 'Australia GP 2025 (if happened)'),
    (2025, 'Bahrain', 'Bahrain GP 2025 (if happened)'),
]

for year, location, description in test_races:
    print(f"\n{description}:")
    try:
        session = fastf1.get_session(year, location, 'R')
        session.load()
        results = session.results
        print(f"  ✅ SUCCESS! {len(results)} drivers loaded")
        print(f"  🏆 Winner: {results.iloc[0]['Abbreviation']} ({results.iloc[0]['TeamName']})")
    except Exception as e:
        print(f"  ❌ Failed: {str(e)}")

    time.sleep(2)  # Be nice to the API

# ============================================
# Test 3: Alternative method - Manual race list
# ============================================
print("\n\n🔧 TRYING ALTERNATIVE APPROACH")
print("-" * 80)
print("Instead of get_event_schedule(), trying direct race access...\n")

# Known 2024 races (we can hardcode these)
known_2024_races = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China',
    'Miami', 'Emilia Romagna', 'Monaco', 'Canada', 'Spain',
    'Austria', 'Great Britain', 'Hungary', 'Belgium', 'Netherlands',
    'Italy', 'Azerbaijan', 'Singapore', 'United States', 'Mexico',
    'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi'
]

successful_2024 = []
failed_2024 = []

print("Testing 2024 races (first 3 only for speed)...")
for race_name in known_2024_races[:3]:  # Test first 3
    try:
        session = fastf1.get_session(2024, race_name, 'R')
        session.load()
        successful_2024.append(race_name)
        print(f"  ✅ {race_name}")
    except Exception as e:
        failed_2024.append(race_name)
        print(f"  ❌ {race_name}: {str(e)[:50]}...")
    time.sleep(1)

print(f"\nSuccessful: {len(successful_2024)}/{len(known_2024_races[:3])}")

# ============================================
# Summary
# ============================================
print("\n\n📊 DIAGNOSTIC SUMMARY")
print("=" * 80)
print("\n💡 RECOMMENDATIONS:")

print("\n1. If 2024 races loaded successfully:")
print("   → We can collect 2023-2024 data using manual race list")
print("   → Bypass the buggy get_event_schedule() function")

print("\n2. If 2025 races loaded successfully:")
print("   → Great! We can get current season data")

print("\n3. If nothing worked:")
print("   → Might be API issues")
print("   → Try again later or use alternative data source")

print("\n🚀 NEXT STEP: If races loaded, I'll create a fixed data collection script!")