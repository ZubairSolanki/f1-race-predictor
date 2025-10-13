"""
Verify what races are actually in the CSV file
"""

import pandas as pd

print("🔍 VERIFYING CSV DATA")
print("=" * 80)

# Load the CSV
df = pd.read_csv('f1_complete_data.csv')

print(f"Total entries in CSV: {len(df)}")
print(f"Years in CSV: {sorted(df['year'].unique())}\n")

# Check 2025 races
print("📅 2025 RACES IN CSV:")
print("-" * 80)

df['race_date'] = pd.to_datetime(df['race_date'])
df_2025 = df[df['year'] == 2025].copy()

if len(df_2025) > 0:
    # Get unique races with dates and winners
    races_2025 = df_2025.groupby('race_name')['race_date'].first().sort_values()

    print(f"Total 2025 races found: {len(races_2025)}\n")

    for i, (race, date) in enumerate(races_2025.items(), 1):
        # Find winner
        winner = df_2025[(df_2025['race_name'] == race) & (df_2025['finish_position'] == 1)][
            'driver_abbreviation'].values
        winner_name = winner[0] if len(winner) > 0 else 'N/A'
        print(f"{i:2d}. {date.strftime('%Y-%m-%d')} - {race:40s} Winner: {winner_name}")

    # Check specifically for Italy and Singapore
    print(f"\n🔍 CHECKING FOR CRITICAL RACES:")
    print("-" * 80)

    has_italy = 'Italian Grand Prix' in df_2025['race_name'].values
    has_singapore = 'Singapore Grand Prix' in df_2025['race_name'].values

    print(f"Italy GP (Verstappen): {'✅ FOUND' if has_italy else '❌ MISSING'}")
    print(f"Singapore GP (Russell): {'✅ FOUND' if has_singapore else '❌ MISSING'}")

    # Show driver wins
    print(f"\n🏆 2025 WINS BY DRIVER:")
    print("-" * 80)
    wins_2025 = df_2025[df_2025['finish_position'] == 1]['driver_abbreviation'].value_counts()
    for driver, wins in wins_2025.items():
        print(f"   {driver}: {wins} wins")

    # Summary
    print(f"\n📊 2025 DATA SUMMARY:")
    print(f"   Total races: {len(races_2025)}")
    print(f"   Total entries: {len(df_2025)}")
    print(f"   Expected entries: ~{len(races_2025) * 20}")
    print(f"   Completeness: {len(df_2025) / (len(races_2025) * 20) * 100:.0f}%")

    # Data quality check
    if not has_italy or not has_singapore:
        print("\n⚠️  WARNING: CRITICAL RACES MISSING!")
        print("   Italy and/or Singapore GPs are not in the CSV.")
        print("   This will affect prediction accuracy for US GP.")
        print("\n🔧 SOLUTION: Run the script below to force-add these races.")
    else:
        print("\n✅ ALL CRITICAL RACES PRESENT!")
        print("   Data is ready for feature engineering.")

else:
    print("❌ No 2025 data found in CSV!")

print("\n" + "=" * 80)