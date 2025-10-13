"""
F1 Data Explorer - Your First Step!
This script fetches F1 race data and shows you what's available.
"""

import fastf1
import pandas as pd

# Enable FastF1 cache (saves data locally so we don't re-download)
fastf1.Cache.enable_cache('f1_cache')

print("🏎️  Welcome to F1 Data Exploration!\n")

# ============================================
# STEP 1: Load a Race Session
# ============================================
print("📥 Loading 2024 Bahrain Grand Prix Race data...")

# Get the race session
# Year, Grand Prix name, Session type (Race, Qualifying, Practice, etc.)
session = fastf1.get_session(2024, 'Bahrain', 'R')  # 'R' = Race
session.load()  # Actually download and load the data

print(f"✅ Loaded: {session.event['EventName']} - {session.name}")
print(f"📅 Date: {session.date}\n")

# ============================================
# STEP 2: Explore Race Results
# ============================================
print("🏁 Race Results (Top 10):")
print("-" * 80)

# Get the results for all drivers
results = session.results

# Show interesting columns
columns_to_show = ['Position', 'DriverNumber', 'Abbreviation', 'TeamName', 'Points']
print(results[columns_to_show].head(10))
print()

# ============================================
# STEP 3: Look at Lap Times
# ============================================
print("⏱️  Analyzing Lap Times...")

# Get all laps driven in the race
laps = session.laps

print(f"Total laps recorded: {len(laps)}")
print(f"Columns available: {list(laps.columns)}\n")

# Let's look at lap times for the winner
winner = results.iloc[0]['Abbreviation']  # Get winner's abbreviation (e.g., 'VER')
winner_laps = laps.pick_driver(winner)

print(f"📊 {winner}'s lap times (first 10 laps):")
print("-" * 80)
print(winner_laps[['LapNumber', 'LapTime', 'Compound', 'TyreLife']].head(10))
print()

# ============================================
# STEP 4: Quick Statistics
# ============================================
print("📈 Quick Statistics:")
print("-" * 80)

# Convert LapTime to seconds for analysis
winner_laps['LapTimeSeconds'] = winner_laps['LapTime'].dt.total_seconds()

print(f"Fastest lap: {winner_laps['LapTimeSeconds'].min():.3f} seconds")
print(f"Average lap: {winner_laps['LapTimeSeconds'].mean():.3f} seconds")
print(f"Slowest lap: {winner_laps['LapTimeSeconds'].max():.3f} seconds")
print()

# ============================================
# STEP 5: Tire Strategy
# ============================================
print("🔴 Tire Strategy Analysis:")
print("-" * 80)

# Group laps by tire compound
tire_stats = winner_laps.groupby('Compound')['LapTimeSeconds'].agg(['count', 'mean'])
print(tire_stats)
print()

print("🎉 Data exploration complete!")
print("\n💡 What you learned:")
print("   - How to load F1 race sessions")
print("   - Access race results and lap times")
print("   - Explore tire compounds and strategies")
print("   - Basic data analysis with pandas")
print("\n🚀 Next: We'll start collecting multiple races for our ML model!")