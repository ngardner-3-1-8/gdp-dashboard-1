import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import pytz
from dateutil.parser import parse
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from ortools.linear_solver import pywraplp
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import itertools
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options # Make sure this is present!
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import os
import json
import sqlite3
import polars as pl 
import nflreadpy as nfl
import random
import csv
from typing import Optional
from typing import Dict, List, Any
from sklearn.feature_selection import RFE
from scipy.stats import percentileofscore
import warnings


# 1. Get current date
today = datetime.datetime.now()
current_cal_year = today.year 

# 2. Initial Year Logic based on Month (User Rule)
# If Jan-May (< 6), assume we are finishing the previous season.
target_year = current_cal_year - 1 if today.month < 5 else current_cal_year

schedule_df = pd.read_csv(f"nfl-schedules/schedule_{target_year}.csv")
    
        
schedule_df = pd.read_csv(f"nfl-schedules/schedule_{target_year}.csv")
schedule_df['gameday'] = pd.to_datetime(schedule_df['gameday'])
schedule_df = schedule_df[schedule_df['game_type'] == 'REG']

first_game_date = schedule_df['gameday'].min()

# 3. Calculate Important Dates Automatically
def get_thanksgiving(year):
    # 4th Thursday in November
    c = calendar.monthcalendar(year, 11)
    thursdays = [row[calendar.THURSDAY] for row in c if row[calendar.THURSDAY] != 0]
    return datetime.datetime(year, 11, thursdays[3])

thanksgiving_date = get_thanksgiving(target_year)
black_friday = thanksgiving_date + datetime.timedelta(days=1)
christmas_day = datetime.datetime(target_year, 12, 25)
boxing_day = datetime.datetime(target_year, 12, 26)

if today <= first_game_date:
    starting_week = 1
else:# We find the latest game that has happened to determine "current" week
    games_played = schedule_df[schedule_df['gameday'] <= today]
    last_played_week = int(games_played['week'].max())
    if not games_played.empty:
        standard_nfl_week = int(games_played['week'].max())
        
        # ADJUST FOR CIRCA SPECIAL WEEKS
        # Start with standard week
        starting_week = standard_nfl_week + 1
        if today >= black_friday:
            starting_week += 1
        if today >= boxing_day:
            starting_week += 1
        
    # Bound check: If season is over (e.g. Week 22), cap it or handle as needed
    if starting_week > 19: 
        starting_week = 19

# 5. Final Assignment to your variables
current_year = target_year
starting_year = target_year

current_year_plus_1 = current_year + 1

season_start_date = first_game_date - 1

thanksgiving_reset_date = black_friday_date + 1 #THIS DATE IS INCLUDED IN THE RESET. SO IF THERE ARE GAMES ON THIS DATE, THEY WILL HAVE A WEEK ADDED
christmas_reset_date = boxing_day

NUM_WEEKS_TO_KEEP = starting_week - 1
current_year_plus_1 = current_year + 1 #current_year + 1

circa_2020_entries = 1373
circa_2021_entries = 4071
circa_2022_entries = 6106
circa_2023_entries = 9234
circa_2024_entries = 14221
circa_2025_entries = 18718
circa_2026_entries = 24000

circa_total_entries = 18718
splash_big_splash_total_entries = 16337
splash_4_for_4_total_entries = 10000
splash_for_the_fans_total_entries = 8382
splash_ship_it_nation_total_entries = 10114
splash_high_roller_total_entries = 1004
splash_rotowire_total_entries = 9048
splash_walkers_25_total_entries = 36501
splash_bloody_total_entries = 5000
dk_total_entries = 20000

MP_PRESEASON_RANKS = {
    'Arizona Cardinals': 0.075,
    'Atlanta Falcons': -0.71,
    'Baltimore Ravens': 6.69,
    'Buffalo Bills': 4.795,
    'Carolina Panthers': -5.25,
    'Chicago Bears': -1.575,
    'Cincinnati Bengals': 1.31,
    'Cleveland Browns': -4.705,
    'Dallas Cowboys': -0.615,
    'Denver Broncos': 2.05,
    'Detroit Lions': 4.305,
    'Green Bay Packers': 3.535,
    'Houston Texans': 0.035,
    'Indianapolis Colts': -2.265,
    'Jacksonville Jaguars': -1.825,
    'Kansas City Chiefs': 4.395,
    'Las Vegas Raiders': -2.35,
    'Los Angeles Chargers': 0.935,
    'Los Angeles Rams': 1.29,
    'Miami Dolphins': 0.66,
    'Minnesota Vikings': 0.27,
    'New England Patriots': -1.995,
    'New Orleans Saints': -6.145,
    'New York Giants': -2.84,
    'New York Jets': -3.725,
    'Philadelphia Eagles': 4.905,
    'Pittsburgh Steelers': -0.565,
    'San Francisco 49ers': 3.325,
    'Seattle Seahawks': -0.13,
    'Tampa Bay Buccaneers': 1.025,
    'Tennessee Titans': -4.36,
    'Washington Commanders': 1.45
}

GSF_PRESEASON_RANKS = {
    'Arizona Cardinals': 0.075,
    'Atlanta Falcons': -0.71,
    'Baltimore Ravens': 6.69,
    'Buffalo Bills': 4.795,
    'Carolina Panthers': -5.25,
    'Chicago Bears': -1.575,
    'Cincinnati Bengals': 1.31,
    'Cleveland Browns': -4.705,
    'Dallas Cowboys': -0.615,
    'Denver Broncos': 2.05,
    'Detroit Lions': 4.305,
    'Green Bay Packers': 3.535,
    'Houston Texans': 0.035,
    'Indianapolis Colts': -2.265,
    'Jacksonville Jaguars': -1.825,
    'Kansas City Chiefs': 4.395,
    'Las Vegas Raiders': -2.35,
    'Los Angeles Chargers': 0.935,
    'Los Angeles Rams': 1.29,
    'Miami Dolphins': 0.66,
    'Minnesota Vikings': 0.27,
    'New England Patriots': -1.995,
    'New Orleans Saints': -6.145,
    'New York Giants': -2.84,
    'New York Jets': -3.725,
    'Philadelphia Eagles': 4.905,
    'Pittsburgh Steelers': -0.565,
    'San Francisco 49ers': 3.325,
    'Seattle Seahawks': -0.13,
    'Tampa Bay Buccaneers': 1.025,
    'Tennessee Titans': -4.36,
    'Washington Commanders': 1.45
}

mp_current_ranks = {
    'Arizona Cardinals' : -5.6,
    'Atlanta Falcons' : -1.61,
    'Baltimore Ravens' : 5,
    'Buffalo Bills' : 5.24,
    'Carolina Panthers' : -5.07,
    'Chicago Bears' : -1.68,
    'Cincinnati Bengals' : -6.02,
    'Cleveland Browns' : -8.97,
    'Dallas Cowboys' : 0.33,
    'Denver Broncos' : 3.31,
    'Detroit Lions' : 4.55,
    'Green Bay Packers' : 4.69,
    'Houston Texans' : -1.23,
    'Indianapolis Colts' : 3.95,
    'Jacksonville Jaguars' : 1.02,
    'Kansas City Chiefs' : 6.47,
    'Las Vegas Raiders' : -5.91,
    'Los Angeles Chargers' : 0.68,
    'Los Angeles Rams' : 7.26,
    'Miami Dolphins' : -1.34,
    'Minnesota Vikings' : -0.87,
    'New England Patriots' : 0.28,
    'New Orleans Saints' : -7.09,
    'New York Giants' : -5.86,
    'New York Jets' : -3.07,
    'Philadelphia Eagles' : 5.73,
    'Pittsburgh Steelers' : 1.1,
    'San Francisco 49ers' : 4.49,
    'Seattle Seahawks' : 8.34,
    'Tampa Bay Buccaneers' : 1.3,
    'Tennessee Titans' : -7.24,
    'Washington Commanders' : -2.04
}
import pandas as pd
import os

# 1. Define the file path based on your existing variables
ratings_file = f"nfl-power-ratings/nfl_power_ratings_blended_week_{standard_nfl_week}_{target_year}.csv"

# 2. Check if the file exists before trying to read it
if os.path.exists(ratings_file):
    print(f"Loading ratings from {ratings_file}")
    ratings_df = pd.read_csv(ratings_file)
    
    # 3. Create a helper function to get the rating safely
    def get_team_rating(team_abbr):
        # Look for the team in the 'team' or 'off_team' column (check your CSV header)
        # We use .iloc[0] to get the value from the matching row
        try:
            # Change 'team' to 'off_team' if that is the name of your team column
            rating = ratings_df.loc[ratings_df['team'] == team_abbr, 'MP_Rating'].values[0]
            return float(rating)
        except (IndexError, KeyError):
            print(f"Warning: Could not find rating for {team_abbr}. Defaulting to 0.")
            return 0.0

    # 4. Build your dictionary dynamically
    team_ratings_dict = {
        'Arizona Cardinals' : get_team_rating("ARI"),
        'Atlanta Falcons' : get_team_rating("ATL"),
        'Baltimore Ravens' : get_team_rating("BAL"),
        'Buffalo Bills' : get_team_rating("BUF"),
        'Carolina Panthers' : get_team_rating("CAR"),
        'Chicago Bears' : get_team_rating("CHI"),
        'Cincinnati Bengals' : get_team_rating("CIN"),
        'Cleveland Browns' : get_team_rating("CLE"),
        'Dallas Cowboys' : get_team_rating("DAL"),
        'Denver Broncos' : get_team_rating("DEN"),
        'Detroit Lions' : get_team_rating("DET"),
        'Green Bay Packers' : get_team_rating("GB"),
        'Houston Texans' : get_team_rating("HOU"),
        'Indianapolis Colts' : get_team_rating("IND"),
        'Jacksonville Jaguars' : get_team_rating("JAX"),
        'Kansas City Chiefs' : get_team_rating("KC"),
        'Las Vegas Raiders' : get_team_rating("LV"),
        'Los Angeles Chargers' : get_team_rating("LAC"),
        'Los Angeles Rams' : get_team_rating("LAR"),
        'Miami Dolphins' : get_team_rating("MIA"),
        'Minnesota Vikings' : get_team_rating("MIN"),
        'New England Patriots' : get_team_rating("NE"),
        'New Orleans Saints' : get_team_rating("NO"),
        'New York Giants' : get_team_rating("NYG"),
        'New York Jets' : get_team_rating("NYJ"),
        'Philadelphia Eagles' : get_team_rating("PHI"),
        'Pittsburgh Steelers' : get_team_rating("PIT"),
        'San Francisco 49ers' : get_team_rating("SF"),
        'Seattle Seahawks' : get_team_rating("SEA"),
        'Tampa Bay Buccaneers' : get_team_rating("TB"),
        'Tennessee Titans' : get_team_rating("TEN"),
        'Washington Commanders' : get_team_rating("WAS")
    }
else:
    print(f"Error: {ratings_file} not found. Hardcoded fallback or empty dict required.")
    team_ratings_dict = {}

print("Dynamic Team Ratings Loaded Successfully.")

gsf_current_ranks = {
    'Arizona Cardinals' : -5.6,
    'Atlanta Falcons' : -1.61,
    'Baltimore Ravens' : 5,
    'Buffalo Bills' : 5.24,
    'Carolina Panthers' : -5.07,
    'Chicago Bears' : -1.68,
    'Cincinnati Bengals' : -6.02,
    'Cleveland Browns' : -8.97,
    'Dallas Cowboys' : 0.33,
    'Denver Broncos' : 3.31,
    'Detroit Lions' : 4.55,
    'Green Bay Packers' : 4.69,
    'Houston Texans' : -1.23,
    'Indianapolis Colts' : 3.95,
    'Jacksonville Jaguars' : 1.02,
    'Kansas City Chiefs' : 6.47,
    'Las Vegas Raiders' : -5.91,
    'Los Angeles Chargers' : 0.68,
    'Los Angeles Rams' : 7.26,
    'Miami Dolphins' : -1.34,
    'Minnesota Vikings' : -0.87,
    'New England Patriots' : 0.28,
    'New Orleans Saints' : -7.09,
    'New York Giants' : -5.86,
    'New York Jets' : -3.07,
    'Philadelphia Eagles' : 5.73,
    'Pittsburgh Steelers' : 1.1,
    'San Francisco 49ers' : 4.49,
    'Seattle Seahawks' : 8.34,
    'Tampa Bay Buccaneers' : 1.3,
    'Tennessee Titans' : -7.24,
    'Washington Commanders' : -2.04
}

CUSTOM_RANKS = {
    'Arizona Cardinals' : 0,
    'Atlanta Falcons' : 0,
    'Baltimore Ravens' : 0,
    'Buffalo Bills' : 0,
    'Carolina Panthers' : 0,
    'Chicago Bears' : 0,
    'Cincinnati Bengals' : 0,
    'Cleveland Browns' : 0,
    'Dallas Cowboys' : 0,
    'Denver Broncos' : 0,
    'Detroit Lions' : 0,
    'Green Bay Packers' : 0,
    'Houston Texans' : 0,
    'Indianapolis Colts' : 0,
    'Jacksonville Jaguars' : 0,
    'Kansas City Chiefs' : 0,
    'Las Vegas Raiders' : 0,
    'Los Angeles Chargers' : 0,
    'Los Angeles Rams' : 0,
    'Miami Dolphins' : 0,
    'Minnesota Vikings' : 0,
    'New England Patriots' : 0,
    'New Orleans Saints' : 0,
    'New York Giants' : 0,
    'New York Jets' : 0,
    'Philadelphia Eagles' : 0,
    'Pittsburgh Steelers' : 0,
    'San Francisco 49ers' : 0,
    'Seattle Seahawks' : 0,
    'Tampa Bay Buccaneers' : 0,
    'Tennessee Titans' : 0,
    'Washington Commanders' : 0
}
    
# --------------------------------------------------------------------------
# --- 3. HOME FIELD ADVANTAGE (STATIC DEFAULTS) ---
# Used if the user selects 'Default' in the UI for home field advantage.
# These values are divided by 2 from the input as they appear to be half-points.
# --------------------------------------------------------------------------
DEFAULT_HOME_ADVANTAGE = {
    'Arizona Cardinals': 1.5/2,
    'Atlanta Falcons': 2.3/2,
    'Baltimore Ravens': 3.8/2,
    'Buffalo Bills': 3.6/2,
    'Carolina Panthers': 1.9/2,
    'Chicago Bears': 1.5/2,
    'Cincinnati Bengals': 2.1/2,
    'Cleveland Browns': 1.3/2,
    'Dallas Cowboys': 3.7/2,
    'Denver Broncos': 2.6/2,
    'Detroit Lions': 2.1/2,
    'Green Bay Packers': 3.8/2,
    'Houston Texans': 1.9/2,
    'Indianapolis Colts': 2.6/2,
    'Jacksonville Jaguars': 1.4/2,
    'Kansas City Chiefs': 3.8/2,
    'Las Vegas Raiders': 1.4/2,
    'Los Angeles Chargers': 2.6/2,
    'Los Angeles Rams': 2.6/2,
    'Miami Dolphins': 2.3/2,
    'Minnesota Vikings': 3.1/2,
    'New England Patriots': 3.9/2,
    'New Orleans Saints': 3.1/2,
    'New York Giants': 1.1/2,
    'New York Jets': 1.2/2,
    'Philadelphia Eagles': 3.3/2,
    'Pittsburgh Steelers': 3.5/2,
    'San Francisco 49ers': 3.6/2,
    'Seattle Seahawks': 2.6/2,
    'Tampa Bay Buccaneers': 2.0/2,
    'Tennessee Titans': 2.1/2,
    'Washington Commanders': 1.3/2
}

# --------------------------------------------------------------------------
# --- 4. AWAY ADJUSTMENT (STATIC DEFAULTS) ---
# Used if the user selects 'Default' in the UI for away adjustment.
# These values are divided by 2 from the input as they appear to be half-points.
# --------------------------------------------------------------------------
DEFAULT_AWAY_ADJ = {
    'Arizona Cardinals': -0.3/2,
    'Atlanta Falcons': 0.2/2,
    'Baltimore Ravens': -1.5/2,
    'Buffalo Bills': -1.1/2,
    'Carolina Panthers': 0.5/2,
    'Chicago Bears': 1.0/2,
    'Cincinnati Bengals': -0.2/2,
    'Cleveland Browns': 1.5/2,
    'Dallas Cowboys': -1.2/2,
    'Denver Broncos': 0.6/2,
    'Detroit Lions': 0.7/2,
    'Green Bay Packers': -0.1/2,
    'Houston Texans': 0.7/2,
    'Indianapolis Colts': -0.2/2,
    'Jacksonville Jaguars': 1.6/2,
    'Kansas City Chiefs': -1.6/2,
    'Las Vegas Raiders': -0.3/2,
    'Los Angeles Chargers': -0.8/2,
    'Los Angeles Rams': 1.3/2,
    'Miami Dolphins': 1.4/2,
    'Minnesota Vikings': -0.5/2,
    'New England Patriots': -1.8/2,
    'New Orleans Saints': -1.6/2,
    'New York Giants': 0.9/2,
    'New York Jets': 1.9/2,
    'Philadelphia Eagles': -0.2/2,
    'Pittsburgh Steelers': -0.2/2,
    'San Francisco 49ers': -1.1/2,
    'Seattle Seahawks': -0.4/2,
    'Tampa Bay Buccaneers': -0.1/2,
    'Tennessee Titans': 0.4/2,
    'Washington Commanders': 0.6/2
}

# --------------------------------------------------------------------------
# --- 5. STADIUM INFO (STATIC) ---
# Static stadium and location data. This list is *merged* with the dynamic
# rank/advantage values inside the collect_schedule_travel_ranking_data function.
# The structure is: [Stadium Name, Lat, Lon, Timezone, Division]
# --------------------------------------------------------------------------
STADIUM_INFO = {
    'Arizona Cardinals': ['State Farm Stadium', 33.5277, -112.262608, 'America/Denver', 'NFC West'],
    'Atlanta Falcons': ['Mercedes-Benz Stadium', 33.7489, -84.3880, 'America/New_York', 'NFC South'],
    'Baltimore Ravens': ['M&T Bank Stadium', 39.2789, -76.6228, 'America/New_York', 'AFC North'],
    'Buffalo Bills': ['Highmark Stadium', 42.7725, -78.7877, 'America/New_York', 'AFC East'],
    'Carolina Panthers': ['Bank of America Stadium', 35.2258, -80.8528, 'America/New_York', 'NFC South'],
    'Chicago Bears': ['Soldier Field', 41.8623, -87.6167, 'America/Chicago', 'NFC North'],
    'Cincinnati Bengals': ['Paycor Stadium', 39.0955, -84.5165, 'America/New_York', 'AFC North'],
    'Cleveland Browns': ['FirstEnergy Stadium', 41.5061, -81.6994, 'America/New_York', 'AFC North'],
    'Dallas Cowboys': ['AT&T Stadium', 32.7369, -97.0826, 'America/Chicago', 'NFC East'],
    'Denver Broncos': ['Empower Field at Mile High', 39.7648, -105.0076, 'America/Denver', 'AFC West'],
    'Detroit Lions': ['Ford Field', 42.3395, -83.0450, 'America/Detroit', 'NFC North'],
    'Green Bay Packers': ['Lambeau Field', 44.5013, -88.0622, 'America/Chicago', 'NFC North'],
    'Houston Texans': ['NRG Stadium', 29.6847, -95.4093, 'America/Chicago', 'AFC South'],
    'Indianapolis Colts': ['Lucas Oil Stadium', 39.7601, -86.1638, 'America/Indiana/Indianapolis', 'AFC South'],
    'Jacksonville Jaguars': ['TIAA Bank Field', 30.3239, -81.6554, 'America/New_York', 'AFC South'],
    'Kansas City Chiefs': ['GEHA Field at Arrowhead Stadium', 39.0489, -94.4839, 'America/Chicago', 'AFC West'],
    'Las Vegas Raiders': ['Allegiant Stadium', 36.1080, -115.1578, 'America/Los_Angeles', 'AFC West'],
    'Los Angeles Chargers': ['SoFi Stadium', 33.9535, -118.3395, 'America/Los_Angeles', 'AFC West'],
    'Los Angeles Rams': ['SoFi Stadium', 33.9535, -118.3395, 'America/Los_Angeles', 'NFC West'],
    'Miami Dolphins': ['Hard Rock Stadium', 25.9602, -80.2384, 'America/New_York', 'AFC East'],
    'Minnesota Vikings': ['U.S. Bank Stadium', 44.9738, -93.2575, 'America/Chicago', 'NFC North'],
    'New England Patriots': ['Gillette Stadium', 42.0628, -71.2687, 'America/New_York', 'AFC East'],
    'New Orleans Saints': ['Caesars Superdome', 29.9507, -90.0813, 'America/Chicago', 'NFC South'],
    'New York Giants': ['MetLife Stadium', 40.8136, -74.0744, 'America/New_York', 'NFC East'],
    'New York Jets': ['MetLife Stadium', 40.8136, -74.0744, 'America/New_York', 'AFC East'],
    'Philadelphia Eagles': ['Lincoln Financial Field', 39.9008, -75.1675, 'America/New_York', 'NFC East'],
    'Pittsburgh Steelers': ['Acrisure Stadium', 40.4468, -80.0158, 'America/New_York', 'AFC North'],
    'San Francisco 49ers': ['Levi\'s Stadium', 37.4031, -121.9702, 'America/Los_Angeles', 'NFC West'],
    'Seattle Seahawks': ['Lumen Field', 47.5952, -122.3316, 'America/Los_Angeles', 'NFC West'],
    'Tampa Bay Buccaneers': ['Raymond James Stadium', 27.9759, -82.5033, 'America/New_York', 'NFC South'],
    'Tennessee Titans': ['Nissan Stadium', 36.1664, -86.7716, 'America/Chicago', 'AFC South'],
    'Washington Commanders': ['FedExField', 38.9077, -76.8645, 'America/New_York', 'NFC East']
}

ALL_TEAMS = list(STADIUM_INFO.keys())


def collect_schedule_travel_ranking_data(pd, config: dict, schedule_rows):
# Get the user's custom rankings from the config
    custom_rankings = config.get('team_rankings', {})

    stadiums = {}
    for team, info in STADIUM_INFO.items():
        # info = [Stadium Name, Lat, Lon, Timezone, Division]
        
        # 1. Get Preseason Rank (from global static dict)
        preseason_rank = PRESEASON_RANKS.get(team, 0)
        
        # 2. Get Current/Custom Rank (from config or default)
        user_rank = custom_rankings.get(team, 'Default')
        if user_rank == 'Default':
            current_rank = massey_peabody_rankings.get(team, 0)
        else:
            current_rank = float(user_rank) # Ensure it's a number
        
        # 3. Get Home Advantage (from global static dict)
        #    (Your config doesn't store this, so we use default)
        home_adv = DEFAULT_HOME_ADVANTAGE.get(team, 0)
        
        # 4. Get Away Adjustment (from global static dict)
        #    (Your config doesn't store this, so we use default)
        away_adj = DEFAULT_AWAY_ADJ.get(team, 0)
        
        # Build the list in the format your code expects [cite: 25-28, 116]
        stadiums[team] = [
            info[0], # Stadium Name
            info[1], # Lat
            info[2], # Lon
            info[3], # Timezone
            info[4], # Division
            preseason_rank,  # 5: Preseason Rank
            current_rank,    # 6: Current Rank
            home_adv,        # 7: Home Advantage
            away_adj         # 8: Away Adjustment
        ]
    data = []
    # Initialize a variable to hold the last valid date and week
    last_date = None
    start_date_str = season_start_date
    start_date = parse(start_date_str)
    week = 1
    # Initialize a dictionary to store the last game date for each team
    last_game = {}
    last_away_game = {}
    # Initialize dictionaries to store cumulative rest advantage for each team
    cumulative_advantage = {}
    # 0: Stadium | 1: Lattitude | 2: Longitude | 3: Timezone | 4: Division | 5: Preseason Average points better than Average Team (Used for Spread and Odds Calculation) | 6: Current Average points better than Average Team (Used for Spread and Odds Calculation) | 7: Home Advantage | 8: Reduction of Home Advantage when Away Team #Calculated here: https://nfllines.com/nfl-2023-home-field-advantage-values/
# default_data.py
# Contains static, non-user-configurable data for NFL teams and stadiums.


    #Get the distances traveled
    def haversine(lat1, lon1, lat2, lon2):
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Differences
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Radius of earth in kilometers. Use 3956 for miles
        r = 3956

        return c * r

    #Get the timzone differences
    def calculate_hours_difference(tz1, tz2):
        # This function calculates the difference in hours between two timezones
        tz1_offset = pytz.timezone(tz1).utcoffset(pd.Timestamp.now()).total_seconds() / 3600
        tz2_offset = pytz.timezone(tz2).utcoffset(pd.Timestamp.now()).total_seconds() / 3600
        return tz1_offset - tz2_offset

    # Loop through each row in the table
    for schedule_row in schedule_rows:
    #for schedule_row in schedule_table.find_all('tr'):
        # Check if the row has a white background
        if schedule_row.get('bgcolor') == '#ffffff':
            # Find all columns in the row
            cols = schedule_row.find_all('td')
    #        print(cols)
            # Get the text from each column and strip leading/trailing whitespaces
            cols_text = []
            actual_stadium = []
            for col in cols:
                text = BeautifulSoup(col.get_text(strip=True), 'html.parser').text
                if " ¹" in text:
                    actual_stadium = "London, UK"
                    text = text.replace(" ¹", "")
                else:
                    actual_stadium = ''
                if " *" in text:
                    text = text.replace(" *", "")
                cols_text.append(text)
            # If the date field is not blank, update last_date and check if it's a new week
            if cols_text[0].strip() != '':
                # Parse the date
                date_str = cols_text[0]
                date = parse(date_str)
                if date.month == 1:
                    # Change the year to 2025
                    date = date.replace(year=2026)
                else:
                    date = date.replace(year=2025)
                # Adjust week for games on or after November 30th
                #if date >= pd.Timestamp('2024-11-30'):
                    #week += 1
                # Adjust week for games on or after December 27th
                #if date >= pd.Timestamp('2024-12-27'):
                    #week += 1
                # Calculate the difference in days
                days_diff = (date - start_date).days

                # Calculate the week number
                week = 1 + days_diff // 7

                # Update cols_text with the week information
                #cols_text.insert(0, f'Week {week}')

                # Rest of your existing logic (rest days, advantage, etc.)

                # Update last_date
                last_date = date
            # If the date field is blank and last_date is not None, use last_date
            elif last_date is not None:
                cols_text[0] = last_date.strftime('%a %b %d')
            # Add week to the start of cols_text
            cols_text.insert(0, f'Week {week}')

            # Calculate rest days for away team and add it to cols_text
            away_team = cols_text[3]
            home_team = cols_text[4]
            if away_team in last_game:
                away_rest_days = (last_date - last_game[away_team]).days
            else:
                away_rest_days = 0
            if home_team in last_game:
                home_rest_days = (last_date - last_game[home_team]).days
            else:
                home_rest_days = 0   
            # Calculate rest advantage for both teams and add it to cols_text
            if isinstance(away_rest_days, int) and isinstance(home_rest_days, int):
                away_advantage = away_rest_days - home_rest_days
                home_advantage = home_rest_days - away_rest_days

                # Update cumulative rest advantage for both teams regardless of whether they are home or away this game.
                cumulative_advantage[away_team] = cumulative_advantage.get(away_team, 0) + away_advantage
                cumulative_advantage[home_team] = cumulative_advantage.get(home_team, 0) + home_advantage

            else:
                away_advantage = 'NA'
                home_advantage = 'NA'        

            cols_text.extend([away_rest_days, home_rest_days, away_advantage, home_advantage,
                              cumulative_advantage.get(away_team, 'NA'), 
                              cumulative_advantage.get(home_team, 'NA')])  

            # Update last game date for both teams regardless of whether they are home or away this game.
            last_game[away_team] = last_date
            last_game[home_team] = last_date

            # Check if the current game is an away game in the next week after the last away game
            back_to_back_away = False
            if away_team in last_away_game and last_away_game[away_team] == week - 1:
                back_to_back_away = True
            # Update the last away game week for the away team
            last_away_game[away_team] = week


            data.append(cols_text + [actual_stadium, back_to_back_away])
    #        print(cols_text)
    df = pd.DataFrame(data, columns=['Week', 'Date', 'Time', 'Away Team', 'Home Team', 
                                     'Away Team Weekly Rest', 'Home Team Weekly Rest', 
                                     'Weekly Away Rest Advantage', 'Weekly Home Rest Advantage',
                                     'Away Cumulative Rest Advantage', 'Home Cumulative Rest Advantage','Actual Stadium', 'Back to Back Away Games'])


    df['Date'] = df['Date'].str.replace(r'(\w+)\s(\w+)\s(\d+)', r'\2 \3, 2025', regex=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    # Adjust January games to 2025 in the DataFrame
    df['Date'] = df['Date'].apply(lambda x: x.replace(year=2026) if x.month == 1 else x)
    df['Week_Num'] = df['Week'].str.replace('Week ', '').astype(int)
    df['Week'] = df['Week'].str.replace('Week ', '', regex=False).astype(int)

    selected_contest = config['selected_contest']
    if selected_contest == 'Circa':
        df.loc[df['Date'] >= pd.to_datetime(thanksgiving_reset_date), 'Week'] += 1
        df.loc[df['Date'] >= pd.to_datetime(christmas_reset_date), 'Week'] += 1
        df.loc[df['Date'] >= pd.to_datetime(thanksgiving_reset_date), 'Week_Num'] += 1
        df.loc[df['Date'] >= pd.to_datetime(christmas_reset_date), 'Week_Num'] += 1


    # Convert 'Week' back to string format if needed
    df['Week'] = 'Week ' + df['Week'].astype(str)
    df['Away Team Current Week Cumulative Rest Advantage'] = pd.to_numeric(df['Away Cumulative Rest Advantage'], errors='coerce').fillna(0) - pd.to_numeric(df['Home Cumulative Rest Advantage'], errors='coerce').fillna(0)
    df['Home Team Current Week Cumulative Rest Advantage'] = pd.to_numeric(df['Home Cumulative Rest Advantage'], errors='coerce').fillna(0) - pd.to_numeric(df['Away Cumulative Rest Advantage'], errors='coerce').fillna(0)
    df['Away Team Division'] = df['Away Team'].map(lambda team: stadiums[team][4] if team in stadiums else 'NA')
    df['Away Stadium'] = df['Away Team'].map(lambda team: stadiums[team][0] if team in stadiums else 'NA')
    df['Away Stadium Latitude'] = df['Away Team'].map(lambda team: stadiums[team][1] if team in stadiums else 'NA')
    df['Away Stadium Longitude'] = df['Away Team'].map(lambda team: stadiums[team][2] if team in stadiums else 'NA')
    df['Away Stadium TimeZone'] = df['Away Team'].map(lambda team: stadiums[team][3] if team in stadiums else 'NA')

    df['Home Team Division'] = df['Home Team'].map(lambda team: stadiums[team][4] if team in stadiums else 'NA')
    df['Home Stadium'] = df['Home Team'].map(lambda team: stadiums[team][0] if team in stadiums else 'NA')
    df['Home Stadium Latitude'] = df['Home Team'].map(lambda team: stadiums[team][1] if team in stadiums else 'NA')
    df['Home Stadium Longitude'] = df['Home Team'].map(lambda team: stadiums[team][2] if team in stadiums else 'NA')
    df['Home Stadium TimeZone'] = df['Home Team'].map(lambda team: stadiums[team][3] if team in stadiums else 'NA')
    df.loc[df['Actual Stadium'] == '', 'Actual Stadium'] = df['Home Stadium']

    df['Away Team Previous Opponent'] = 'BYE'
    df['Home Team Previous Opponent'] = 'BYE'
    df['Away Team Previous Location'] = 'BYE'
    df['Home Team Previous Location'] = 'BYE'
    df['Away Team Next Opponent'] = 'BYE'
    df['Home Team Next Opponent'] = 'BYE'
    df['Away Team Next Location'] = 'BYE'
    df['Home Team Next Location'] = 'BYE'

    team_last_opponent = {}
    team_last_location = {}

    for index, row in df.iterrows():
        away_team = row['Away Team']
        home_team = row['Home Team']
        week_num = row['Week_Num']
        away_stadium = row['Actual Stadium']
        home_stadium = row['Actual Stadium']
        
        # Check if its not the first week
        if week_num > 1:
            # Get the previous opponents from the dictionary
            if away_team in team_last_opponent:
                df.loc[index, 'Away Team Previous Opponent'] = team_last_opponent[away_team]
            if home_team in team_last_opponent:
                 df.loc[index, 'Home Team Previous Opponent'] = team_last_opponent[home_team]
            
            # Get the previous locations from the dictionary
            if away_team in team_last_location:
                df.loc[index, 'Away Team Previous Location'] = team_last_location[away_team]
            if home_team in team_last_location:
                 df.loc[index, 'Home Team Previous Location'] = team_last_location[home_team]
        elif week_num == 1:
            df.loc[index, 'Away Team Previous Opponent'] = 'Preseason'
            df.loc[index, 'Home Team Previous Opponent'] = 'Preseason'
            df.loc[index, 'Away Team Previous Location'] = 'Preseason'
            df.loc[index, 'Home Team Previous Location'] = 'Preseason'
    
        # Update team last opponent dictionary
        team_last_opponent[home_team] = away_team
        team_last_opponent[away_team] = home_team
       
        # Update team last location dictionary
        team_last_location[home_team] = home_stadium
        team_last_location[away_team] = away_stadium
    

    team_next_opponent = {}
    team_next_location = {}

    # Iterate through the DataFrame in reverse order
    for index in reversed(df.index):
        row = df.loc[index]
        away_team = row['Away Team']
        home_team = row['Home Team']
        week_num = row['Week_Num']
        away_stadium = row['Actual Stadium']
        home_stadium = row['Actual Stadium']
        
        # Check if its not the last week
        if week_num < df['Week_Num'].max():
            # Get the previous opponents from the dictionary
            if away_team in team_next_opponent:
                df.loc[index, 'Away Team Next Opponent'] = team_next_opponent[away_team]
            if home_team in team_next_opponent:
                 df.loc[index, 'Home Team Next Opponent'] = team_next_opponent[home_team]
            
            # Get the previous locations from the dictionary
            if away_team in team_next_location:
                df.loc[index, 'Away Team Next Location'] = team_next_location[away_team]
            if home_team in team_next_location:
                 df.loc[index, 'Home Team Next Location'] = team_next_location[home_team]
        else:
            df.loc[index, 'Away Team Next Opponent'] = "Playoffs?"
            df.loc[index, 'Away Team Next Opponent'] = "Playoffs?"
            df.loc[index, 'Away Team Next Location'] = "Playoffs?"
            df.loc[index, 'Home Team Next Location'] = "Playoffs?"

        # Update team next opponent dictionary
        team_next_opponent[home_team] = away_team
        team_next_opponent[away_team] = home_team
       
        # Update team next location dictionary
        team_next_location[home_team] = home_stadium
        team_next_location[away_team] = away_stadium
    #df['Home Team'] = df['Home Team'].str.replace(' *', '')
    #df.to_csv('test.csv', index=False)


    # Add new columns to the DataFrame
    df['Actual Stadium Latitude'] = np.where(df['Actual Stadium'] == 'London, UK', 51.555973, df['Home Stadium Latitude'])
    df['Actual Stadium Longitude'] = np.where(df['Actual Stadium'] == 'London, UK', -0.279672, df['Home Stadium Longitude'])
    df['Actual Stadium TimeZone'] = np.where(df['Actual Stadium'] == 'London, UK', 'Europe/London', df['Home Stadium TimeZone'])

    df['Away Stadium Latitude'] = pd.to_numeric(df['Away Stadium Latitude'])
    df['Away Stadium Longitude'] = pd.to_numeric(df['Away Stadium Longitude'])
    df['Actual Stadium Latitude'] = pd.to_numeric(df['Actual Stadium Latitude'])
    df['Actual Stadium Longitude'] = pd.to_numeric(df['Actual Stadium Longitude'])
    df['Home Stadium Latitude'] = pd.to_numeric(df['Home Stadium Latitude'])
    df['Home Stadium Longitude'] = pd.to_numeric(df['Home Stadium Longitude'])

    df['Away Travel Distance'] = df.apply(lambda row: round(haversine(row['Away Stadium Latitude'], row['Away Stadium Longitude'], row['Actual Stadium Latitude'], row['Actual Stadium Longitude'])), axis=1)
    df['Home Travel Distance'] = df.apply(lambda row: round(haversine(row['Home Stadium Latitude'], row['Home Stadium Longitude'], row['Actual Stadium Latitude'], row['Actual Stadium Longitude'])), axis=1)

    df['Away Travel Advantage'] =  df['Home Travel Distance'] - df['Away Travel Distance']
    df['Home Travel Advantage'] =  df['Away Travel Distance'] - df['Home Travel Distance']

    # Apply the function to your DataFrame
    df['Away Timezone Change'] = df.apply(lambda row: calculate_hours_difference(row['Away Stadium TimeZone'], row['Actual Stadium TimeZone']), axis=1)
    df['Home Timezone Change'] = df.apply(lambda row: calculate_hours_difference(row['Home Stadium TimeZone'], row['Actual Stadium TimeZone']), axis=1)

    # Initialize empty lists for storing last game timezones
    last_game_timezones_away = []
    last_game_timezones_home = []

    # Initialize dictionary for storing last game timezone for each team
    last_game_timezone = {}

    # Iterate over DataFrame rows
    for i, row in df.iterrows():
        # Get current away team, home team and actual stadium timezone
        away_team = row['Away Team']
        home_team = row['Home Team']
        actual_stadium_timezone = row['Actual Stadium TimeZone']

        # Check if this is not the away team's first game
        if away_team in last_game_timezone:
            # If not, append last game's actual stadium timezone to list
            last_game_timezones_away.append(last_game_timezone[away_team])
        else:
            # If it is, append None (or any other value indicating no previous game)
            last_game_timezones_away.append(None)

        # Check if this is not the home team's first game
        if home_team in last_game_timezone:
            # If not, append last game's actual stadium timezone to list
            last_game_timezones_home.append(last_game_timezone[home_team])
        else:
            # If it is, append None (or any other value indicating no previous game)
            last_game_timezones_home.append(None)

        # Update last game's actual stadium timezone for current away and home teams
        last_game_timezone[away_team] = actual_stadium_timezone
        last_game_timezone[home_team] = actual_stadium_timezone

    # Add new columns to DataFrame
    df['Away Previous Game Actual Stadium TimeZone'] = last_game_timezones_away
    df['Home Previous Game Actual Stadium TimeZone'] = last_game_timezones_home

    # Add new column to DataFrame
    df['Home Previous Game Actual Stadium TimeZone'] = last_game_timezones_home
    df['Away Weekly Timezone Difference'] = df.apply(lambda row: calculate_hours_difference(row['Away Previous Game Actual Stadium TimeZone'], row['Actual Stadium TimeZone']) if pd.notnull(row['Away Previous Game Actual Stadium TimeZone']) and row['Away Previous Game Actual Stadium TimeZone'].strip() != '' else None, axis=1)
    df['Home Weekly Timezone Difference'] = df.apply(lambda row: calculate_hours_difference(row['Home Previous Game Actual Stadium TimeZone'], row['Actual Stadium TimeZone']) if pd.notnull(row['Home Previous Game Actual Stadium TimeZone']) and row['Home Previous Game Actual Stadium TimeZone'].strip() != '' else None, axis=1)

    df['Adjusted Away Timezone Change'] = df.apply(lambda row: 0 if row['Away Previous Game Actual Stadium TimeZone'] == row['Actual Stadium TimeZone'] and row['Actual Stadium'] != row['Away Stadium'] else calculate_hours_difference(row['Away Stadium TimeZone'], row['Actual Stadium TimeZone']), axis=1)
    df['Adjusted Home Timezone Change'] = df.apply(lambda row: 0 if row['Home Previous Game Actual Stadium TimeZone'] == row['Actual Stadium TimeZone'] and row['Actual Stadium'] != row['Home Stadium'] else calculate_hours_difference(row['Home Stadium TimeZone'], row['Actual Stadium TimeZone']), axis=1)

    df['Away Timezone Advantage'] = df.apply(lambda row: 0 if row['Adjusted Away Timezone Change'] == 0 else row['Adjusted Away Timezone Change'] - row['Adjusted Home Timezone Change'], axis=1)
    df['Home Timezone Advantage'] = df.apply(lambda row: 0 if row['Adjusted Home Timezone Change'] == 0 else row['Adjusted Home Timezone Change'] - row['Adjusted Away Timezone Change'], axis=1)

    #df['Away Timezone Advantage'] = (df['Away Timezone Change'] - df['Home Timezone Change'])
    #df['Home Timezone Advantage'] = (df['Home Timezone Change'] - df['Away Timezone Change'])

    df['Away Team Preseason Rank'] = df['Away Team'].map(lambda team: stadiums[team][5] if team in stadiums else 'NA')
    df['Home Team Preseason Rank'] = df['Home Team'].map(lambda team: stadiums[team][5] if team in stadiums else 'NA')

    df['Preseason Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Preseason Rank'] > row['Home Team Preseason Rank'] else (row['Home Team'] if row['Away Team Preseason Rank'] < row['Home Team Preseason Rank'] else 'Tie'), axis=1)
    df['Preseason Difference'] = abs(df['Away Team Preseason Rank'] - df['Home Team Preseason Rank'])

    df['Away Team Adjusted Preseason Rank'] = df['Away Team'].map(lambda team: stadiums[team][5]) + np.where((df['Away Travel Advantage'] < -100) & (df['Home Stadium'] == df['Actual Stadium']), -.125, 0) - pd.to_numeric(df['Away Timezone Advantage']*.25, errors='coerce').fillna(0)-pd.to_numeric(df['Weekly Away Rest Advantage'], errors='coerce').fillna(0)-.125*df['Away Team Current Week Cumulative Rest Advantage'] - np.where((df['Away Team'].map(lambda team: stadiums[team][0])) != df['Home Team'].map(lambda team: stadiums[team][0]), df['Away Team'].map(lambda team: stadiums[team][8]), 0)
    df['Home Team Adjusted Preseason Rank'] = df['Home Team'].map(lambda team: stadiums[team][5]) - np.where((df['Away Travel Advantage'] < -100) & (df['Home Stadium'] == df['Actual Stadium']), .125, 0) - pd.to_numeric(df['Home Timezone Advantage']*.25, errors='coerce').fillna(0)-pd.to_numeric(df['Weekly Home Rest Advantage'], errors='coerce').fillna(0)-.125*df['Home Team Current Week Cumulative Rest Advantage'] + np.where((df['Away Team'].map(lambda team: stadiums[team][0])) != df['Home Team'].map(lambda team: stadiums[team][0]), df['Home Team'].map(lambda team: stadiums[team][7]), 0)

    df['Adjusted Preseason Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Adjusted Preseason Rank'] > row['Home Team Adjusted Preseason Rank'] else (row['Home Team'] if row['Away Team Adjusted Preseason Rank'] < row['Home Team Adjusted Preseason Rank'] else 'Tie'), axis=1)
    df['Adjusted Preseason Difference'] = abs(df['Away Team Adjusted Preseason Rank'] - df['Home Team Adjusted Preseason Rank'])

    df['Away Team Current Rank'] = df['Away Team'].map(lambda team: stadiums[team][6] if team in stadiums else 'NA')
    df['Home Team Current Rank'] = df['Home Team'].map(lambda team: stadiums[team][6] if team in stadiums else 'NA')

    df['Current Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Current Rank'] > row['Home Team Current Rank'] else (row['Home Team'] if row['Away Team Current Rank'] < row['Home Team Current Rank'] else 'Tie'), axis=1)
    df['Current Difference'] = abs(df['Away Team Current Rank'] - df['Home Team Current Rank'])

    df['Away Team Adjusted Current Rank'] = df['Away Team'].map(lambda team: stadiums[team][6]) + np.where((df['Away Travel Advantage'] < -400) & (df['Home Stadium'] == df['Actual Stadium']), -.125, 0) - pd.to_numeric(df['Away Timezone Advantage'] * .1, errors='coerce').fillna(0) + pd.to_numeric(df['Weekly Away Rest Advantage'], errors='coerce').fillna(0) * .125 + df['Away Team Current Week Cumulative Rest Advantage'] * .0625 + np.where((df['Away Team'].map(lambda team: stadiums[team][0])) != df['Home Team'].map(lambda team: stadiums[team][0]), df['Away Team'].map(lambda team: stadiums[team][8]), 0)
    df['Home Team Adjusted Current Rank'] = df['Home Team'].map(lambda team: stadiums[team][6]) + np.where((df['Away Travel Advantage'] < -400) & (df['Home Stadium'] == df['Actual Stadium']), .125, 0) + pd.to_numeric(df['Home Timezone Advantage'] * .1, errors='coerce').fillna(0) + pd.to_numeric(df['Weekly Home Rest Advantage'], errors='coerce').fillna(0) * .125 + df['Home Team Current Week Cumulative Rest Advantage'] * .0625 + np.where((df['Away Team'].map(lambda team: stadiums[team][0])) != df['Home Team'].map(lambda team: stadiums[team][0]), df['Home Team'].map(lambda team: stadiums[team][7]), 0)

    df['Adjusted Current Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Adjusted Current Rank'] > row['Home Team Adjusted Current Rank'] else (row['Home Team'] if row['Away Team Adjusted Current Rank'] < row['Home Team Adjusted Current Rank'] else 'Tie'), axis=1)
    df['Adjusted Current Difference'] = abs(df['Away Team Adjusted Current Rank'] - df['Home Team Adjusted Current Rank'])

    df['Same Winner?'] = df.apply(lambda row: 'Same' if row['Preseason Winner'] == row['Adjusted Preseason Winner'] == row['Current Winner'] == row['Adjusted Current Winner'] else 'Different', axis=1)
    df['Same Adjusted Preseason Winner?'] = df.apply(lambda row: 'Same' if row['Adjusted Preseason Winner'] == row['Adjusted Current Winner'] else 'Different', axis=1)
    df['Same Current and Adjusted Current Winner?'] = df.apply(lambda row: 'Same' if row['Current Winner'] == row['Adjusted Current Winner'] else 'Different', axis=1)

    
    df['Thursday Night Game'] = 'False'
    df["Thursday Night Game"] = df.apply(lambda row: 'True' if (row['Date'].weekday() == 3) and (row['Date'] != pd.to_datetime(thanksgiving_date)) and (row['Date'] != pd.to_datetime(boxing_day_date)) and (row['Date'] != pd.to_datetime(christmas_date))  else row["Thursday Night Game"], axis =1)


    df['Home Team Winner?'] = df.apply(lambda row: 'Home Team' if row['Adjusted Current Winner'] == row['Home Team'] else 'Away Team', axis=1)
    #df['Divisional Matchup?'] = df.apply(lambda row: 'Divisional' if row['Home Team Division'] == row['Away Team Division'] else 'Non-divisional', axis=1)
    df['Divisional Matchup?'] = (df['Home Team Division'] == df['Away Team Division']).astype(int)


    # Create "HT 3 games in 10 days" and "AT 3 games in 10 Days" columns with default "No"
    df['Home Team 3 games in 10 days'] = 'No'
    df['Away Team 3 games in 10 days'] = 'No'

    # Convert 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the home and away teams
        home_team = row['Home Team']
        away_team = row['Away Team']
        game_date = row['Date']

        # Calculate the start date of the 10-day window
        ten_days_ago = game_date - pd.Timedelta(days=10)

        # Get the previous 10 days of games for the home team (regardless of home/away)
        home_team_games = df[
            ((df['Home Team'] == home_team) | (df['Away Team'] == home_team)) &
            (df['Date'] >= ten_days_ago) & (df['Date'] <= game_date) 
        ].sort_values('Date', ascending=False).head(10)

        # Get the previous 10 days of games for the away team (regardless of home/away)
        away_team_games = df[
            ((df['Home Team'] == away_team) | (df['Away Team'] == away_team)) &
            (df['Date'] >= ten_days_ago) & (df['Date'] <= game_date)
        ].sort_values('Date', ascending=False).head(10)

        # Check if home team has played 3 games in the last 10 days (regardless of home/away)
        if len(home_team_games) >= 3:
            df.loc[index, 'Home Team 3 games in 10 days'] = 'Yes'

        # Check if away team has played 3 games in the last 10 days (regardless of home/away)
        if len(away_team_games) >= 3:
            df.loc[index, 'Away Team 3 games in 10 days'] = 'Yes'

    # Create "HT 4 games in 17 days" and "AT 4 games in 17 Days" columns with default "No"
    df['Home Team 4 games in 17 days'] = 'No'
    df['Away Team 4 games in 17 days'] = 'No'

    # Convert 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the home and away teams
        home_team = row['Home Team']
        away_team = row['Away Team']
        game_date = row['Date']

        # Calculate the start date of the 10-day window
        seventeen_days_ago = game_date - pd.Timedelta(days=17)

        # Get the previous 10 days of games for the home team (regardless of home/away)
        home_team_games = df[
            ((df['Home Team'] == home_team) | (df['Away Team'] == home_team)) &
            (df['Date'] >= seventeen_days_ago) & (df['Date'] <= game_date) 
        ].sort_values('Date', ascending=False).head(17)

        # Get the previous 10 days of games for the away team (regardless of home/away)
        away_team_games = df[
            ((df['Home Team'] == away_team) | (df['Away Team'] == away_team)) &
            (df['Date'] >= seventeen_days_ago) & (df['Date'] <= game_date)
        ].sort_values('Date', ascending=False).head(17)

        # Check if home team has played 3 games in the last 10 days (regardless of home/away)
        if len(home_team_games) >= 4:
            df.loc[index, 'Home Team 4 games in 17 days'] = 'Yes'

        # Check if away team has played 3 games in the last 10 days (regardless of home/away)
        if len(away_team_games) >= 4:
            df.loc[index, 'Away Team 4 games in 17 days'] = 'Yes'


    # Convert 'NA' to NaN
    df['Away Team Weekly Rest'] = df['Away Team Weekly Rest'].replace('NA', 0)
    df['Home Team Weekly Rest'] = df['Home Team Weekly Rest'].replace('NA', 0)

    # Convert to integers
    df['Away Team Weekly Rest'] = pd.to_numeric(df['Away Team Weekly Rest'], errors='coerce')
    df['Home Team Weekly Rest'] = pd.to_numeric(df['Home Team Weekly Rest'], errors='coerce')        

    df['Away Team Short Rest'] = 'No'
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the home and away teams
        home_team_rest = row['Home Team Weekly Rest']
        away_team_rest = row['Away Team Weekly Rest']
        game_date = row['Date']

        # Check for short rest and rest disadvantage
        if (away_team_rest < 7) and (away_team_rest < home_team_rest):
            # Update the 'Away Team Short Rest' for the specific row
            df.loc[index, 'Away Team Short Rest'] = 'Yes'
    
    def get_backup_nfl_odds():
        """
        Fetches odds from nfl_data_py (nflverse) as a fallback.
        Useful for past games or when the main API is down.
        """
        try:
            print("Fetching backup odds from nflreadpy...")
            
            # 1. Determine the current NFL season
            # If it's Jan/Feb, we are technically in the previous calendar year's season
            now = datetime.now()
            season = now.year if now.month > 3 else now.year - 1
            
            # 2. Load Schedule and Team Data
            df_schedule = nfl.load_schedules([season])
            df_teams = nfl.load_teams()
            
            # Create a mapping from Abbreviation (KC) to Full Name (Kansas City Chiefs)
            # to match The Odds API format
            team_map = dict(zip(df_teams['team_abbr'], df_teams['team_name']))
            
            formatted_games = []
            
            # 3. Iterate and Format
            for index, row in df_schedule.iterrows():
                # Skip games that don't have lines/odds yet
                if pd.isna(row['home_moneyline']) or pd.isna(row['gametime']):
                    continue
    
                # Format Time: nflreadpy times are typically strings in Eastern Time already
                # Combine gameday (YYYY-MM-DD) and gametime (HH:MM)
                game_time_str = f"{row['gameday']} {row['gametime']}"
                try:
                    dt_obj = datetime.strptime(game_time_str, '%Y-%m-%d %H:%M')
                    # Format to your specific style: "8:20 pm ET"
                    formatted_time = dt_obj.strftime('%I:%M %p ET').replace('AM ET', 'am').replace('PM ET', 'pm').lstrip('0')
                except ValueError:
                    formatted_time = row['gametime'] # Fallback if parsing fails
    
                # Calculate Spreads
                # nflreadpy 'spread_line' is usually the Home Team's spread
                home_spread = row['spread_line']
                # Away spread is typically the inverse
                away_spread = -1 * home_spread if home_spread is not None else None
    
                formatted_games.append({
                    'Time': formatted_time,
                    'Away Team': team_map.get(row['away_team'], row['away_team']),
                    'Away Odds': row['away_moneyline'], # nflreadpy already provides American odds
                    'Home Team': team_map.get(row['home_team'], row['home_team']),
                    'Home Odds': row['home_moneyline'], # nflreadpy already provides American odds
                    'Away Spread': away_spread,
                    'Home Spread': home_spread
                })
    
            return pd.DataFrame(formatted_games)
    
        except Exception as e:
            print(f"Backup data fetch failed: {e}")
            return pd.DataFrame()
    
    def get_full_season_odds(api_key):
        """
        Generates a full season view:
        1. Fetches the ENTIRE season schedule from nflreadpy (Past & Future).
        2. Fetches LIVE odds from The Odds API.
        3. Merges them: Updates the nflreadpy schedule with live API data where available.
        """
        
        # ---------------------------------------------------------
        # STEP 1: Get the "Base" Schedule (Past + Future) from nflreadpy
        # ---------------------------------------------------------
        print("Fetching full season schedule from nflreadpy...")
        
        # Determine season (if currently Jan/Feb 2025, we want the 2024 season)
        now = datetime.now()
        season = now.year if now.month > 3 else now.year - 1
        
        try:
            # 1. Load data (returns Polars DataFrame)
            df_schedule_polars = nfl.load_schedules([season])
            df_teams_polars = nfl.load_teams()
        
            # 2. Convert to Pandas to use .iterrows()
            df_schedule = df_schedule_polars.to_pandas()
            df_teams = df_teams_polars.to_pandas()
        except Exception as e:
            st.error(f"Error fetching nflreadpy data: {e}")
            return pd.DataFrame()
    
        # Create mapping: Abbr (KC) -> Full Name (Kansas City Chiefs) to match Odds API
        team_map = dict(zip(df_teams['team_abbr'], df_teams['team_name']))
        
        base_games = []
    
        for index, row in df_schedule.iterrows():
            # Map abbreviations to full names
            home_full = team_map.get(row['home_team'], row['home_team'])
            away_full = team_map.get(row['away_team'], row['away_team'])
            
            # Format Time
            try:
                # Combine gameday and gametime
                game_time_str = f"{row['gameday']} {row['gametime']}"
                dt_obj = datetime.strptime(game_time_str, '%Y-%m-%d %H:%M')
                formatted_time = dt_obj.strftime('%I:%M %p ET').replace('AM ET', 'am').replace('PM ET', 'pm').lstrip('0')
            except:
                formatted_time = str(row['gameday']) # Fallback
    
            # Handle Spreads (nflreadpy is usually Home relative)
            # If Spread is -3.0, Home is favored by 3.
            home_spread = row['spread_line']
            away_spread = -1 * home_spread if home_spread is not None else None
    
            # Build the row
            base_games.append({
                'Match_ID': f"{home_full} vs {away_full}", # Unique Key for merging
                'Time': formatted_time,
                'Away Team': away_full,
                'Away Odds': row['away_moneyline'],
                'Home Team': home_full,
                'Home Odds': row['home_moneyline'],
                'Away Spread': away_spread,
                'Home Spread': home_spread,
                'Source': 'Historical (nflreadpy)' # Tag source for debugging
            })
        
        df_base = pd.DataFrame(base_games)
    
        # ---------------------------------------------------------
        # STEP 2: Get the "Live" Data from The Odds API
        # ---------------------------------------------------------
        print("Fetching live odds from API...")
        
        live_games = []
        
        # API Config
        SPORT = 'americanfootball_nfl'
        REGIONS = 'us'
        MARKETS = 'h2h,spreads'
        ODDS_FORMAT = 'decimal'
        DATE_FORMAT = 'iso'
        url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={api_key}&regions={REGIONS}&markets={MARKETS}&oddsFormat={ODDS_FORMAT}&dateFormat={DATE_FORMAT}'
    
        try:
            response = requests.get(url)
            if response.status_code == 200:
                odds_data = response.json()
                eastern_tz = pytz.timezone('America/New_York')
    
                for event in odds_data:
                    home_team = event['home_team']
                    away_team = event['away_team']
                    
                    # Time Formatting
                    utc_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
                    east_time = utc_time.astimezone(eastern_tz)
                    formatted_time = east_time.strftime('%I:%M %p ET').replace('AM ET', 'am').replace('PM ET', 'pm').lstrip('0')
    
                    # Odds Aggregation
                    game_odds = {'home': [], 'away': [], 'home_spread': [], 'away_spread': []}
                    for bookmaker in event['bookmakers']:
                        for market in bookmaker['markets']:
                            if market['key'] == 'h2h':
                                for outcome in market['outcomes']:
                                    if outcome['name'] == home_team: game_odds['home'].append(outcome['price'])
                                    elif outcome['name'] == away_team: game_odds['away'].append(outcome['price'])
                            elif market['key'] == 'spreads':
                                for outcome in market['outcomes']:
                                    if outcome['name'] == home_team: game_odds['home_spread'].append(outcome['point'])
                                    elif outcome['name'] == away_team: game_odds['away_spread'].append(outcome['point'])
    
                    # Averages
                    avg_home = sum(game_odds['home'])/len(game_odds['home']) if game_odds['home'] else None
                    avg_away = sum(game_odds['away'])/len(game_odds['away']) if game_odds['away'] else None
                    avg_home_spread = sum(game_odds['home_spread'])/len(game_odds['home_spread']) if game_odds['home_spread'] else None
                    avg_away_spread = sum(game_odds['away_spread'])/len(game_odds['away_spread']) if game_odds['away_spread'] else None
    
                    # Convert Decimal to American
                    def dec_to_amer(dec):
                        if not dec: return None
                        if dec >= 2.0: return round((dec - 1) * 100)
                        else: return round(-100 / (dec - 1))
    
                    live_games.append({
                        'Match_ID': f"{home_team} vs {away_team}",
                        'Time': formatted_time,
                        'Away Team': away_team,
                        'Away Odds': dec_to_amer(avg_away),
                        'Home Team': home_team,
                        'Home Odds': dec_to_amer(avg_home),
                        'Away Spread': avg_away_spread,
                        'Home Spread': avg_home_spread,
                        'Source': 'Live API'
                    })
        except Exception as e:
            print(f"API failed ({e}), relying solely on backup data.")
    
        # ---------------------------------------------------------
        # STEP 3: Merge - Overwrite Base with Live Data
        # ---------------------------------------------------------
        
        if live_games:
            df_live = pd.DataFrame(live_games)
            
            # Iterate through live games and update the base dataframe
            # We match on "Match_ID" (Home vs Away)
            for index, row in df_live.iterrows():
                match_id = row['Match_ID']
                
                # Find matching index in df_base
                mask = df_base['Match_ID'] == match_id
                
                if mask.any():
                    # Update specific columns
                    cols_to_update = ['Time', 'Away Odds', 'Home Odds', 'Away Spread', 'Home Spread', 'Source']
                    df_base.loc[mask, cols_to_update] = row[cols_to_update].values
                else:
                    # Optional: If for some reason the game isn't in nflreadpy (rare), append it
                    # df_base = pd.concat([df_base, pd.DataFrame([row])], ignore_index=True)
                    pass
    
        # Drop the Match_ID helper column before returning
        df_base = df_base.drop(columns=['Match_ID'])
        
        return df_base
    
    # ---------------------------------------------------------
    # Usage in Streamlit
    # ---------------------------------------------------------
    API_KEY = '34671f7aeaa8f4fbee2398163f2f45d3'# Replace with actual key
    
    if API_KEY != 'YOUR_API_KEY':
        # Fetch Data
        live_api_odds_df = get_full_season_odds(API_KEY)
        
        st.subheader("Full Season Odds (Historical + Live)")
        
        # Optional: Highlight the Source column so you see which are Live vs Historical
        st.dataframe(live_api_odds_df)
    else:
        st.warning("Please enter your API Key")
	
    def add_odds_to_main_csv():
        """
        Adds odds data to the main DataFrame, prioritizing DraftKings data if available and complete.
        If DraftKings data is missing or incomplete for a game, it overrides with internal calculations.
    
        Args:
            df (pd.DataFrame): The main DataFrame to which odds will be added.
            live_api_odds_df (pd.DataFrame): DataFrame containing live odds scraped from DraftKings.
            # ... (all preseason_X_rank, X_rank, X_home_adv, X_away_adj parameters for each team)
    
        Returns:
            pd.DataFrame: The updated DataFrame with odds.
        """
    
        # 0: Spread | 1: Favorite Odds| 2: Underdog Odds
        odds = {
            0: [-110, -110], .5: [-116, -104], 1: [-122, 101], 1.5: [-128, 105], 2: [-131, 108],
            2.5: [-142, 117], 3: [-164, 135], 3.5: [-191, 156], 4: [-211, 171], 4.5: [-224, 181],
            5: [-234, 188], 5.5: [-244, 195], 6: [-261, 208], 6.5: [-282, 224], 7: [-319, 249],
            7.5: [-346, 268], 8: [-366, 282], 8.5: [-397, 302], 9: [-416, 314], 9.5: [-436, 327],
            10: [-483, 356], 10.5: [-538, 389], 11: [-567, 406], 11.5: [-646, 450], 12: [-660, 458],
            12.5: [-675, 466], 13: [-729, 494], 13.5: [-819, 539], 14: [-890, 573], 14.5: [-984, 615],
            15: [-1134, 677], 15.5: [-1197, 702], 16: [-1266, 728], 16.5: [-1267, 728], 17: [-1381, 769],
            17.5: [-1832, 906], 18: [-2149, 986], 18.5: [-2590, 1079], 19: [-3245, 1190], 19.5: [-4323, 1324],
            20: [-4679, 1359], 20.5: [-5098, 1396], 21: [-5597, 1434], 21.5: [-6000, 1500], 22: [-6500, 1600],
            22.5: [-7000, 1650], 23: [-7500, 1700], 23.5: [-8000, 1750], 24: [-8500, 1800], 24.5: [-9000, 1850],
            25: [-9500, 1900], 25.5: [-10000, 2000], 26: [-10000, 2000], 26.5: [-10000, 2000], 27: [-10000, 2000],
            27.5: [-10000, 2000], 28: [-10000, 2000], 28.5: [-10000, 2000], 29: [-10000, 2000], 29.5: [-10000, 2000],
            30: [-10000, 2000]
        }
    
        # Create a copy of the main DataFrame to work with, avoiding modification of the original
        csv_df = df.copy()
    
        # Initialize columns that will be populated by DraftKings data or overridden with internal data
        csv_df['Home Team Moneyline'] = np.nan
        csv_df['Away Team Moneyline'] = np.nan
        csv_df['Favorite'] = np.nan
        csv_df['Underdog'] = np.nan
        csv_df['Home Team Sportsbook Spread'] = np.nan
        csv_df['Away Team Sportsbook Spread'] = np.nan
        
        # Attempt to update CSV data with scraped odds from DraftKings
        # This block only executes if live_api_odds_df is not empty
        if not live_api_odds_df.empty:
            for index, row in csv_df.iterrows():
                # Find a matching row in the scraped DraftKings data
                matching_row = live_api_odds_df[
                    (live_api_odds_df['Away Team'] == row['Away Team']) & 
                    (live_api_odds_df['Home Team'] == row['Home Team'])
                ]
                if not matching_row.empty:
                    # If a match is found, apply DraftKings moneyline odds
                    csv_df.loc[index, 'Away Team Moneyline'] = matching_row.iloc[0]['Away Odds']
                    csv_df.loc[index, 'Home Team Moneyline'] = matching_row.iloc[0]['Home Odds']
                    csv_df.loc[index, 'Away Team Sportsbook Spread'] = matching_row.iloc[0]['Away Spread']
                    csv_df.loc[index, 'Home Team Sportsbook Spread'] = matching_row.iloc[0]['Home Spread']					
                    
                    # Determine Favorite/Underdog based on DraftKings odds
                    # Assuming odds <= -110 typically indicates the favorite
                    if matching_row.iloc[0]['Home Odds'] <= -110:
                        csv_df.loc[index, 'Favorite'] = csv_df.loc[index, 'Home Team']
                        csv_df.loc[index, 'Underdog'] = csv_df.loc[index, 'Away Team']
                    else:
                        csv_df.loc[index, 'Favorite'] = csv_df.loc[index, 'Away Team']
                        csv_df.loc[index, 'Underdog'] = csv_df.loc[index, 'Home Team']
    
        # Calculate internal data for all rows. These values will be used to override
        # any missing or incomplete DraftKings data.
        csv_df['Adjusted Home Points'] = csv_df['Home Team Adjusted Current Rank']
        csv_df['Adjusted Away Points'] = csv_df['Away Team Adjusted Current Rank']
    
        csv_df['Preseason Spread'] = abs(csv_df['Away Team Adjusted Preseason Rank'] - csv_df['Home Team Adjusted Preseason Rank'])

        # Determine Favorite and Underdog
		
        missing_odds_mask = pd.isna(csv_df['Favorite'])
        
        # Use a vectorized approach to conditionally update only those rows.
        csv_df.loc[missing_odds_mask, 'Favorite'] = csv_df[missing_odds_mask].apply(
            lambda row: row['Home Team'] if row['Home Team Adjusted Current Rank'] >= row['Away Team Adjusted Current Rank'] else row['Away Team'], axis=1
        )
        
        csv_df.loc[missing_odds_mask, 'Underdog'] = csv_df[missing_odds_mask].apply(
            lambda row: row['Home Team'] if row['Home Team Adjusted Current Rank'] < row['Away Team Adjusted Current Rank'] else row['Away Team'], axis=1
        )

        # Adjust Spread based on Favorite
        csv_df['Adjusted Spread'] = abs(csv_df['Away Team Adjusted Current Rank'] - csv_df['Home Team Adjusted Current Rank'])
    
        # Helper function to get moneyline based on calculated spread and internal odds dictionary
        def get_moneyline(row, odds, team_type):
            """
            Calculates moneyline based on a team's adjusted spread and the predefined odds dictionary.
            Finds the closest spread in the dictionary if an exact match is not found.
            """
            spread = round(row['Adjusted Spread'] * 2) / 2
            
            # Find the closest spread in the odds dictionary to handle non-exact matches
            closest_spread = min(odds.keys(), key=lambda k: abs(k - spread))
            
            moneyline_tuple = odds[closest_spread] # Use the moneyline values for the closest spread
            
            # Determine which moneyline (favorite or underdog) applies to the current team
            if team_type == 'home':
                if row['Favorite'] == row['Home Team']:
                    return moneyline_tuple[0] # Favorite odds
                else:
                    return moneyline_tuple[1] # Underdog odds
            elif team_type == 'away':
                if row['Favorite'] == row['Away Team']:
                    return moneyline_tuple[0] # Favorite odds
                else:
                    return moneyline_tuple[1] # Underdog odds
            return np.nan # Should not be reached under normal circumstances
    
        # Calculate internal moneyline values for all games
        csv_df['Internal Home Team Moneyline'] = csv_df.apply(
            lambda row: get_moneyline(row, odds, 'home'), axis=1
        )
        csv_df['Internal Away Team Moneyline'] = csv_df.apply(
            lambda row: get_moneyline(row, odds, 'away'), axis=1
        )
        override_condition = pd.isna(csv_df['Away Team Moneyline']) | pd.isna(csv_df['Home Team Moneyline'])
        csv_df['No Live Odds Available - Internal Rankings Used for Moneyline Calculation'] = np.where(override_condition, 'True', 'False')
        overridden_games_df = csv_df[override_condition]
        overridden_games_df = csv_df[override_condition][['Home Team', 'Away Team', 'Actual Stadium', 'Date', 'Home Team Moneyline', 'Away Team Moneyline', 'Internal Home Team Moneyline', 'Internal Away Team Moneyline', 'Home Team Sportsbook Spread', 'Away Team Sportsbook Spread', 'Home Team Adjusted Current Rank', 'Away Team Adjusted Current Rank']].copy()

        for index, row in overridden_games_df.iterrows():
            # Override Moneyline if DraftKings data was missing (still NaN)
            if pd.isna(row['Away Team Moneyline']) or row['Away Team Moneyline'] is None:
                overridden_games_df.loc[index, 'Away Team Moneyline'] = row['Internal Away Team Moneyline']
            if pd.isna(row['Home Team Moneyline']) or row['Home Team Moneyline'] is None:
                overridden_games_df.loc[index, 'Home Team Moneyline'] = row['Internal Home Team Moneyline']
            if pd.isna(row['Home Team Sportsbook Spread']) or row['Home Team Sportsbook Spread'] is None:
                overridden_games_df.loc[index, 'Home Team Sportsbook Spread'] = row['Away Team Adjusted Current Rank'] - row['Home Team Adjusted Current Rank']
            if pd.isna(row['Away Team Sportsbook Spread']) or row['Away Team Sportsbook Spread'] is None:
                overridden_games_df.loc[index, 'Away Team Sportsbook Spread'] = row['Home Team Adjusted Current Rank'] - row['Away Team Adjusted Current Rank']
#        st.subheader('Games with Unavailable Live Odds')
#        st.write('This dataframe contains the games where live odds from the Live Odds API were unavailable. This will likely happen for lookahead lines and future weeks')
#        st.write(overridden_games_df)
        st.write('')
        st.write('')
        st.write('')
        csv_df['Internal Home Team Spread'] = csv_df['Away Team Adjusted Current Rank'] - csv_df['Home Team Adjusted Current Rank']
        csv_df['Internal Away Team Spread'] = csv_df['Home Team Adjusted Current Rank'] - csv_df['Away Team Adjusted Current Rank']
        # Iterate through the DataFrame to apply overrides and calculate implied/fair odds
        for index, row in csv_df.iterrows():
            # Override Moneyline if DraftKings data was missing (still NaN)
            if pd.isna(row['Away Team Moneyline']):
                csv_df.loc[index, 'Away Team Moneyline'] = row['Internal Away Team Moneyline']
            if pd.isna(row['Home Team Moneyline']):
                csv_df.loc[index, 'Home Team Moneyline'] = row['Internal Home Team Moneyline']
            if pd.isna(row['Home Team Sportsbook Spread']) or row['Home Team Sportsbook Spread'] is None:
                csv_df.loc[index, 'Home Team Sportsbook Spread'] = row['Internal Home Team Spread']
            if pd.isna(row['Away Team Sportsbook Spread']) or row['Away Team Sportsbook Spread'] is None:
                csv_df.loc[index, 'Away Team Sportsbook Spread'] = row['Internal Away Team Spread']		            
            # Override Favorite/Underdog if not set by DraftKings (i.e., still NaN)
            if pd.isna(row['Favorite']) or row['Favorite'] is None:
                # Determine Favorite and Underdog based on internal ranks
                if row['Home Team Adjusted Current Rank'] >= row['Away Team Adjusted Current Rank']:
                    csv_df.loc[index, 'Favorite'] = row['Home Team']
                    csv_df.loc[index, 'Underdog'] = row['Away Team']
                else:
                    csv_df.loc[index, 'Favorite'] = row['Away Team']
                    csv_df.loc[index, 'Underdog'] = row['Home Team']

    
            # Calculate Implied Odds for the final (potentially overridden) moneyline
            away_moneyline = csv_df.loc[index, 'Away Team Moneyline']
            home_moneyline = csv_df.loc[index, 'Home Team Moneyline']
    
            # Handle potential NaN values before calculating implied odds
            if pd.isna(away_moneyline):
                csv_df.loc[index, 'Away Team Implied Odds to Win'] = np.nan
            elif away_moneyline > 0:
                csv_df.loc[index, 'Away Team Implied Odds to Win'] = 100 / (away_moneyline + 100)
            else:
                csv_df.loc[index, 'Away Team Implied Odds to Win'] = abs(away_moneyline) / (abs(away_moneyline) + 100)
            
            if pd.isna(home_moneyline):
                csv_df.loc[index, 'Home team Implied Odds to Win'] = np.nan
            elif home_moneyline > 0:
                csv_df.loc[index, 'Home team Implied Odds to Win'] = 100 / (home_moneyline + 100)
            else:
                csv_df.loc[index, 'Home team Implied Odds to Win'] = abs(home_moneyline) / (abs(home_moneyline) + 100)
            
            # Calculate Implied Odds for Internal Moneyline (always calculated regardless of DK data)
            internal_away_moneyline = row['Internal Away Team Moneyline']
            internal_home_moneyline = row['Internal Home Team Moneyline']
    
            if pd.isna(internal_away_moneyline):
                csv_df.loc[index, 'Internal Away Team Implied Odds to Win'] = np.nan
            elif internal_away_moneyline > 0:
                csv_df.loc[index, 'Internal Away Team Implied Odds to Win'] = 100 / (internal_away_moneyline + 100)
            else:
                csv_df.loc[index, 'Internal Away Team Implied Odds to Win'] = abs(internal_away_moneyline) / (abs(internal_away_moneyline) + 100)
    
            if pd.isna(internal_home_moneyline):
                csv_df.loc[index, 'Internal Home team Implied Odds to Win'] = np.nan
            elif internal_home_moneyline > 0:
                csv_df.loc[index, 'Internal Home team Implied Odds to Win'] = 100 / (internal_home_moneyline + 100)
            else:
                csv_df.loc[index, 'Internal Home team Implied Odds to Win'] = abs(internal_home_moneyline) / (abs(internal_home_moneyline) + 100)
    
            # Calculate Fair Odds for the final (potentially overridden) moneyline
            away_implied_odds = csv_df.loc[index, 'Away Team Implied Odds to Win']
            home_implied_odds = csv_df.loc[index, 'Home team Implied Odds to Win']
            
            # Ensure sum is not zero or NaN before division
            if pd.isna(away_implied_odds) or pd.isna(home_implied_odds) or (away_implied_odds + home_implied_odds) == 0:
                csv_df.loc[index, 'Away Team Fair Odds'] = np.nan
                csv_df.loc[index, 'Home Team Fair Odds'] = np.nan
            else:
                csv_df.loc[index, 'Away Team Fair Odds'] = away_implied_odds / (away_implied_odds + home_implied_odds)
                csv_df.loc[index, 'Home Team Fair Odds'] = home_implied_odds / (away_implied_odds + home_implied_odds)
    
            # Calculate Fair Odds for Internal Moneyline (always calculated)
            internal_away_implied_odds = csv_df.loc[index, 'Internal Away Team Implied Odds to Win']
            internal_home_implied_odds = csv_df.loc[index, 'Internal Home team Implied Odds to Win']
            
            if pd.isna(internal_away_implied_odds) or pd.isna(internal_home_implied_odds) or (internal_away_implied_odds + internal_home_implied_odds) == 0:
                csv_df.loc[index, 'Internal Away Team Fair Odds'] = np.nan
                csv_df.loc[index, 'Internal Home Team Fair Odds'] = np.nan
            else:
                csv_df.loc[index, 'Internal Away Team Fair Odds'] = internal_away_implied_odds / (internal_away_implied_odds + internal_home_implied_odds)
                csv_df.loc[index, 'Internal Home Team Fair Odds'] = internal_home_implied_odds / (internal_away_implied_odds + internal_home_implied_odds)
    
            # Round all calculated odds to 4 decimal places
            for col in ['Away Team Implied Odds to Win', 'Home team Implied Odds to Win',
                        'Away Team Fair Odds', 'Home Team Fair Odds',
                        'Internal Away Team Implied Odds to Win', 'Internal Home team Implied Odds to Win',
                        'Internal Away Team Fair Odds', 'Internal Home Team Fair Odds']:
                if not pd.isna(csv_df.loc[index, col]): # Only round if not NaN
                    csv_df.loc[index, col] = round(csv_df.loc[index, col], 4)
    
        main_df_with_odds_df = csv_df
        return main_df_with_odds_df
    
    schedule_df_with_odds_df = add_odds_to_main_csv()
    
    df = schedule_df_with_odds_df
        
            

    # Calculate expected win advantage for away team
    df["Away Team Expected Win Advantage"] = round(df["Away Team Fair Odds"] - 0.5,4)

    # Calculate expected win advantage for home team
    df["Home Team Expected Win Advantage"] = round(df["Home Team Fair Odds"] - 0.5,4)

    # Initialize an empty dictionary to store team information
    team_dict = {}

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        week = row["Week"]
        away_team = row["Away Team"]
        home_team = row["Home Team"]    
        away_odds = row["Away Team Expected Win Advantage"]
        home_odds = row["Home Team Expected Win Advantage"]

        # Create a nested dictionary for each team if not already present
        if away_team not in team_dict:
            team_dict[away_team] = {}
        if home_team not in team_dict:
            team_dict[home_team] = {}

        # Populate the nested dictionary with game details and odds
        team_dict[away_team][week] = {"Opponent": home_team, "Home/Away": "Away", "Win Odds": away_odds}
        team_dict[home_team][week] = {"Opponent": away_team, "Home/Away": "Home", "Win Odds": home_odds}

    # Calculate cumulative win percentage for each team
    for team, games in team_dict.items():
        for week, details in games.items():
            opponent = details["Opponent"]
            home_away = details["Home/Away"]
            win_odds = details["Win Odds"]

            # Get the remaining weeks for the team
            remaining_weeks = [w for w in games.keys() if int(w.split()[1]) > int(week.split()[1])]

            #print(remaining_weeks)

            # Calculate cumulative win percentage
            if remaining_weeks:
                cumulative_win_odds = sum(team_dict[team][w]["Win Odds"] for w in remaining_weeks)
                cumulative_win_percentage = cumulative_win_odds/len(remaining_weeks)
            else:
                cumulative_win_percentage = 0  # Set to 0 for week 18

            # Add the cumulative win percentage to the dictionary
            team_dict[team][week]["Cumulative Win Percentage"] = cumulative_win_percentage


    # Initialize empty lists for cumulative win percentages
    away_cumulative_win_percentages = []
    home_cumulative_win_percentages = []

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        week = row["Week"]
        away_team = row["Away Team"]
        home_team = row["Home Team"]

        # Get cumulative win percentages from your dictionary
        away_cumulative_win_percentage = team_dict.get(away_team, {}).get(week, {}).get("Cumulative Win Percentage", 0)
        home_cumulative_win_percentage = team_dict.get(home_team, {}).get(week, {}).get("Cumulative Win Percentage", 0)

        # Append to the lists
        away_cumulative_win_percentages.append(away_cumulative_win_percentage)
        home_cumulative_win_percentages.append(home_cumulative_win_percentage)

    # Add new columns to the DataFrame
    df["Away Team Cumulative Win Percentage"] = away_cumulative_win_percentages
    df["Home Team Cumulative Win Percentage"] = home_cumulative_win_percentages


    # Get unique week values
    unique_weeks = df["Week"].unique()

    # Create an empty DataFrame to store the consolidated data
    consolidated_df = pd.DataFrame()

    # Calculate the maximum cumulative win percentage for each week
    max_cumulative_win_percentage = {}
    for week in unique_weeks:
        week_df = df[df["Week"] == week]
        # Calculate the maximum, using `0` as default if week_df is empty
        if week_df.empty:
            max_val = 0
        else:
            max_val = max(week_df["Away Team Cumulative Win Percentage"].max(),
                         week_df["Home Team Cumulative Win Percentage"].max())

        # Check if the calculated max_val is NaN and replace with 1 if so
        if pd.isna(max_val):
            max_cumulative_win_percentage[week] = 1
        else:
            max_cumulative_win_percentage[week] = max_val

    # Calculate the minimum cumulative win percentage for each week
    min_cumulative_win_percentage = {}
    for week in unique_weeks:
        week_df = df[df["Week"] == week]
        # Calculate the maximum, using `0` as default if week_df is empty
        if week_df.empty:
            min_val = 0
        else:
            min_val = min(week_df["Away Team Cumulative Win Percentage"].min(),
                         week_df["Home Team Cumulative Win Percentage"].min())

        # Check if the calculated max_val is NaN and replace with 1 if so
        if pd.isna(min_val):
            min_cumulative_win_percentage[week] = 0
        else:
            min_cumulative_win_percentage[week] = min_val
    
    # Calculate the range of cumulative win percentages for each week
    range_cumulative_win_percentage = {}
    for week in unique_weeks:
        range_cumulative_win_percentage[week] = max_cumulative_win_percentage[week] - min_cumulative_win_percentage[week]
        if range_cumulative_win_percentage[week] == 0:
            range_cumulative_win_percentage[week]=1
        if pd.isna(range_cumulative_win_percentage[week]):
            range_cumulative_win_percentage[week] = 1
            
    # Define a function to calculate the star rating
    def calculate_star_rating(cumulative_win_percentage, week):
        # Normalize the cumulative win percentage to a scale of 0 to 1
        if pd.isna(cumulative_win_percentage):
            cumulative_win_percentage = 0.0  # Return 0 for NaN inputs
            print("Cumulative Win % is error")
        if pd.isna(min_cumulative_win_percentage[week]):
            min_cumulative_win_percentage[week] = 0.0
            print("Minimum Cumulative Win % is error")
        if pd.isna(range_cumulative_win_percentage[week]):
            range_cumulative_win_percentage[week] = 1.0
            print("Range Cumulative Win % is error")
        try:
            normalized_percentage = (cumulative_win_percentage - min_cumulative_win_percentage[week]) / range_cumulative_win_percentage[week]
            # Assign stars linearly based on the normalized percentage
            return round(10 * normalized_percentage) / 2
        except ZeroDivisionError:
            return 0.0

    # Apply the function to create the new columns for each week

    for week in unique_weeks:
        week_df = df[df["Week"] == week]

        # Calculate star ratings first
        # Assuming calculate_star_rating is defined elsewhere and returns a number
        week_df["Away Team Star Rating"] = week_df["Away Team Cumulative Win Percentage"].apply(lambda x: calculate_star_rating(x, week))
        week_df["Home Team Star Rating"] = week_df["Home Team Cumulative Win Percentage"].apply(lambda x: calculate_star_rating(x, week))

        # Mark Thanksgiving Favorites
        # Find Week 13 games and winners
        week13_df = df[df["Week"] == "Week 13"]
        week13_winners = week13_df["Favorite"].unique()
        
        # Create new columns and mark Thanksgiving Favorites (returning 1 or 0)
        week_df["Away Team Thanksgiving Favorite"] = week_df.apply(
            lambda row: 1
            if (1 <= int(row["Week"].replace("Week ", "")) <= 12) and (row["Away Team"] in week13_winners)
            else 0,
            axis=1,
        )

        week_df["Home Team Thanksgiving Favorite"] = week_df.apply(
            lambda row: 1
            if (1 <= int(row["Week"].replace("Week ", "")) <= 12) and (row["Home Team"] in week13_winners)
            else 0,
            axis=1,
        )

        # Mark Christmas Favorites
        # Find Week 18 games and winners
        week18_df = df[df["Week"] == "Week 18"]
        week18_winners = week18_df["Favorite"].unique()

        # Create new columns and mark Christmas Favorites (returning 1 or 0)
        week_df["Away Team Christmas Favorite"] = week_df.apply(
            lambda row: 1
            if (1 <= int(row["Week"].replace("Week ", "")) <= 17) and (row["Away Team"] in week18_winners)
            else 0,
            axis=1,
        )
        week_df["Home Team Christmas Favorite"] = week_df.apply(
            lambda row: 1
            if (1 <= int(row["Week"].replace("Week ", "")) <= 17) and (row["Home Team"] in week18_winners)
            else 0,
            axis=1,
        )
        
        # Mark Thanksgiving Underdogs
        week13_df = df[df["Week"] == "Week 13"]
        week13_underdogs = week13_df["Underdog"].unique() # Changed variable name to underdogs
        

        # Create new columns and mark Thanksgiving Underdogs (returning 1 or 0)
        week_df["Away Team Thanksgiving Underdog"] = week_df.apply(
            lambda row: 1
            if (1 <= int(row["Week"].replace("Week ", "")) <= 12) and (row["Away Team"] in week13_underdogs)
            else 0,
            axis=1,
        )

        week_df["Home Team Thanksgiving Underdog"] = week_df.apply(
            lambda row: 1
            if (1 <= int(row["Week"].replace("Week ", "")) <= 12) and (row["Home Team"] in week13_underdogs)
            else 0,
            axis=1,
        )

        # Mark Christmas Underdogs
        # Find Week 18 games and underdogs
        week18_df = df[df["Week"] == "Week 18"]
        week18_underdogs = week18_df["Underdog"].unique() # Changed variable name to underdogs

        # Create new columns and mark Christmas Underdogs (returning 1 or 0)
        week_df["Away Team Christmas Underdog"] = week_df.apply(
            lambda row: 1
            if (1 <= int(row["Week"].replace("Week ", "")) <= 17) and (row["Away Team"] in week18_underdogs)
            else 0,
            axis=1,
        )
        week_df["Home Team Christmas Underdog"] = week_df.apply(
            lambda row: 1
            if (1 <= int(row["Week"].replace("Week ", "")) <= 17) and (row["Home Team"] in week18_underdogs)
            else 0,
            axis=1,
        )
        consolidated_df = pd.concat([consolidated_df, week_df])

    # Corrected logic for Pre-Holiday weeks using integer columns
    # We use boolean logic with | (OR) and & (AND) and then call .astype(int) 
    # to convert the resulting True/False Series to 1/0.
    
    # NOTE: You may need to ensure 'Week_Num' is also integer/numeric here if it wasn't already.
    
    consolidated_df['Away Team Pre Thanksgiving'] = (
        (consolidated_df['Away Team Thanksgiving Favorite'].astype(bool) | consolidated_df['Away Team Thanksgiving Underdog'].astype(bool))
        & (consolidated_df['Week_Num'] < thanksgiving_week)
    ).astype(int)
    
    consolidated_df['Home Team Pre Thanksgiving'] = (
        (consolidated_df['Home Team Thanksgiving Favorite'].astype(bool) | consolidated_df['Home Team Thanksgiving Underdog'].astype(bool))
        & (consolidated_df['Week_Num'] < thanksgiving_week)
    ).astype(int)
    
    consolidated_df['Away Team Pre Christmas'] = (
        (consolidated_df['Away Team Christmas Favorite'].astype(bool) | consolidated_df['Away Team Christmas Underdog'].astype(bool))
        & (consolidated_df['Week_Num'] < christmas_week)
    ).astype(int)
    
    consolidated_df['Home Team Pre Christmas'] = (
        (consolidated_df['Home Team Christmas Favorite'].astype(bool) | consolidated_df['Home Team Christmas Underdog'].astype(bool))
        & (consolidated_df['Week_Num'] < christmas_week)
    ).astype(int)
    # Create the 'Divisional Matchup Boolean' column
    consolidated_df["Divisional Matchup Boolean"] = 0

    # Set values based on 'Divisional Matchup?' column
    consolidated_df.loc[consolidated_df["Divisional Matchup?"] == True, "Divisional Matchup Boolean"] = 1
    def scrape_data(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "lxml")
        table_rows = soup.find_all("tr")
    
        data = []
        for row in table_rows:
            columns = row.find_all("td")
            if len(columns) >= 5:
                ev, win_pct, pick_pct, team, opponent = columns[:5]
                rest = columns[5:]
                future_value_cell = rest[-1] if rest else None
    
                if future_value_cell:
                    div_tag = future_value_cell.find("div")
                    if div_tag and "style" in div_tag.attrs:
                        style_attr = div_tag["style"]
                        width_match = re.search(r"width:\s*(\d+)px", style_attr)
                        star_rating = int(width_match.group(1)) / 16 if width_match else 0
                    else:
                        star_rating = 0
                else:
                    star_rating = 0
    
                data.append({
                    "EV": ev.text,
                    "Win %": win_pct.text,
                    "Pick %": pick_pct.text,
                    "Team": team.text,
                    "Opponent": opponent.text,
                    "Future Value (Stars)": star_rating
                })
    
        return data
    
    
    def scrape_all_data(starting_year, current_year_plus_1, config):
        all_data = []
        base_url = "https://www.survivorgrid.com/{year}/{week}"
    
        total_iterations = (current_year_plus_1 - starting_year) * 18
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_week = config.get("starting_week")
        completed = 0
        for year in range(starting_year, current_year_plus_1):
            for week in range(1, start_week + 1):
                url = base_url.format(year=year, week=week)
                status_text.text(f"🔄 Scraping data for {year} Week {week} ...")
                week_data = scrape_data(url)
    
                for row in week_data:
                    row["Year"] = year
                    row["Week"] = f"Week {week}"
                    all_data.append(row)
    
                completed += 1
                progress_bar.progress(completed / total_iterations)
                time.sleep(2)  # Delay between requests
    
        progress_bar.progress(1.0)
        status_text.text("✅ Data scraping complete!")
    
        return all_data
    st.write("Collecting Live Public Pick Percentages...")
    all_data = scrape_all_data(starting_year, current_year_plus_1, config)

    st.success(f"Scraping complete! Retrieved {len(all_data)} rows.")
    
    # Convert the list of dictionaries to a DataFrame
    public_pick_df = pd.DataFrame(all_data)
    
    # Cleanup the scraped data
    public_pick_df['Team'] = public_pick_df['Team'].str.replace(r'\s\(L\)', '', regex=True)
    public_pick_df['Team'] = public_pick_df['Team'].str.replace(r'\s\(W\)', '', regex=True)
    public_pick_df['Opponent'] = public_pick_df['Opponent'].str.replace('@', '', regex=True)
    public_pick_df['Opponent'] = public_pick_df['Opponent'].str.replace(r'[\t\n\+\-]', '', regex=True)
    public_pick_df['Opponent'] = (
        public_pick_df['Opponent']
        .str.strip() # Strip whitespace
        .str[:3]      # Get the first 3 characters
        # Use regex to replace the 3rd character (index 2) with an empty string ('')
        # if the 3rd character is a digit (\d).
        .str.replace(r'^(.{2})\d$', r'\1', regex=True)
    )
    
    public_pick_df = public_pick_df[public_pick_df['Opponent'] != 'BYE']
    
    public_pick_df = public_pick_df.drop_duplicates()
    
    public_pick_df.to_csv("Raw Pick Data.csv", index = False)
    
    # ==============================================================================
    # SECTION 2: API DATA COLLECTION (REPLACED BY nflreadpy)
    # ==============================================================================
    
    print(f"\nFetching NFL schedule and game results using nflreadpy from {starting_year} to {current_year}...")
    
    # Load the schedule data.
    # The object returned here is a Polars DataFrame.
    schedule_data_pl = nfl.load_schedules(list(range(starting_year, current_year + 1)))
    # --- Data Processing using POLARS FILTERING ---
    
    # Filter 1: Exclude in-season future games (those with game_id ending in _XX)
    # Use the .filter() method and the Polars `~` (NOT) operator
    schedule_data_pl = schedule_data_pl.filter(
        ~pl.col('game_id').str.contains(r'\_[0-9]{2}$')
    )
    
    # Filter 2: Filter only Regular Season games
    schedule_data_pl = schedule_data_pl.filter(
        pl.col('game_type') == 'REG'
    )
    
    # CONVERT TO PANDAS DATAFRAME BEFORE PROCEEDING
    completed_games = schedule_data_pl.to_pandas()
    
    
    # --- Data Processing to Match Your Old API Output Structure (Now back in Pandas) ---
    
    # Prepare columns for Winner/Loser determination and abbreviation mapping
    # This part is now safe because `completed_games` is a Pandas DataFrame
    completed_games.rename(columns={
        'gameday': 'Calendar Date',
        'week': 'Week', 
        'home_team': 'Home Team',
        'away_team': 'Away Team',
        'home_score': 'Home Score',
        'away_score': 'Away Score'
    }, inplace=True)
    
    # Function to determine winner/loser
    def determine_result(row):
        home_score = row['Home Score']
        away_score = row['Away Score']
        if home_score > away_score:
            return row['Home Team'], row['Away Team'], home_score, away_score
        elif away_score > home_score:
            return row['Away Team'], row['Home Team'], away_score, home_score
        else:
            # Note: nflreadpy data handles ties by having equal scores
            return 'Tie', 'Tie', home_score, home_score
    
    # Apply the function
    results = completed_games.apply(determine_result, axis=1, result_type='expand')
    results.columns = ['Winner/tie', 'Loser/tie', 'PtsW', 'PtsL']
    
    # Merge the results back
    df_nflreadpy_schedule = pd.concat([completed_games, results], axis=1)
    
    # Select and reorder columns to match your original script's output
    df_api_schedule = df_nflreadpy_schedule[[
        'season', 'Week', 'Calendar Date', 'Home Team', 'Away Team', 'Winner/tie', 'Loser/tie', 'PtsW', 'PtsL'
    ]].copy()
    
    # Rename the season column to Year
    df_api_schedule.rename(columns={'season': 'Year'}, inplace=True)
    
    # Drop any rows with NaN in critical columns (e.g., games not fully recorded)
    df_api_schedule.dropna(subset=['Winner/tie', 'Loser/tie'], inplace=True)
    
    # Convert to string and clean up data types
    df_api_schedule['Week'] = df_api_schedule['Week'].astype(int)
    
    df_api_schedule['Calendar Date'] = pd.to_datetime(df_api_schedule['Calendar Date'], errors='coerce')
    df_api_schedule['Calendar Date'] = df_api_schedule['Calendar Date'].dt.strftime('%Y-%m-%d')
    

    df_api_schedule['Home Team'] = df_api_schedule['Home Team'].replace('LA', 'LAR')
    df_api_schedule['Home Team'] = df_api_schedule['Home Team'].replace('WSH', 'WAS')
    df_api_schedule['Away Team'] = df_api_schedule['Away Team'].replace('LA', 'LAR')
    df_api_schedule['Away Team'] = df_api_schedule['Away Team'].replace('WSH', 'WAS')
    
    df_api_schedule = df_api_schedule.drop_duplicates()
    
    df_api_schedule.to_csv("df_api_schedule.csv", index = False)
    # ==============================================================================
    # SECTION 3: DATA CLEANING AND MERGE (ADJUSTED FOR nflreadpy COLUMN NAMES)
    # ==============================================================================
    
    # Your 'teams' dictionary for mapping is now **redundant for the schedule data**
    # since nflreadpy already uses the abbreviations (e.g., ARI, BAL) that your
    # web-scraped data uses. This simplifies the code significantly!        
    
    # Existing cleanup of the scraped data
    public_pick_df = public_pick_df.replace(r'\u00A0\(W\)', '', regex=True)
    public_pick_df = public_pick_df.replace(r'\u00A0\(L\)', '', regex=True)
    public_pick_df = public_pick_df.replace(r'\u00A0\(tie\)', '', regex=True)
    public_pick_df = public_pick_df.replace(r'\u00A0\(PPD\)', '', regex=True)
    public_pick_df = public_pick_df.replace('--', '0.0%', regex=True)
    # Select the desired columns
    public_pick_df = public_pick_df[['EV', 'Win %', 'Pick %', 'Team', 'Opponent', 'Future Value (Stars)', 'Year', 'Week']]
    
    # Convert to numeric
    public_pick_df['Win %'] = pd.to_numeric(public_pick_df['Win %'].str.rstrip('%')) / 100
    public_pick_df['Pick %'] = pd.to_numeric(public_pick_df['Pick %'].str.rstrip('%')) / 100
    public_pick_df['Pick %'].fillna(0.0, inplace=True)
    public_pick_df['Public Pick %'] = public_pick_df['Pick %']
    
    # Convert 'Week' to integer representing the week number
    public_pick_df['Week'] = public_pick_df['Week'].str.replace('Week ', '').astype(int)

    # df['Week'] = pd.to_numeric(df['Week']) # This is now redundant after astype(int)
    
    # Use your existing 'teams' dictionary for *Division* mapping (still needed)
    teams2 = {
        # ... (Keep your original 'teams' dictionary here for Division mapping)
        'ARI': ['Arizona Cardinals', 'State Farm Stadium', 33.5277, -112.262608, 'America/Denver', 'NFC West'],
        'ATL': ['Atlanta Falcons', 'Mercedez-Benz Stadium', 33.757614, -84.400972, 'America/New_York', 'NFC South'],
        'BAL': ['Baltimore Ravens', 'M&T Stadium', 39.277969, -76.622767, 'America/New_York', 'AFC North'],
        'BUF': ['Buffalo Bills', 'Highmark Stadium', 42.773739, -78.786978, 'America/New_York', 'AFC East'],
        'CAR': ['Carolina Panthers', 'Bank of America Stadium', 35.225808, -80.852861, 'America/New_York', 'NFC South'],
        'CHI': ['Chicago Bears', 'Soldier Field', 41.862306, -87.616672, 'America/Chicago', 'NFC North'],
        'CIN': ['Cincinnati Bengals', 'Paycor Stadium', 39.095442, -84.516039, 'America/New_York', 'AFC North'],
        'CLE': ['Cleveland Browns', 'Cleveland Browns Stadium', 41.506022, -81.699564, 'America/New_York', 'AFC North'],
        'DAL': ['Dallas Cowboys', 'AT&T Stadium', 32.747778, -97.092778, 'America/Chicago', 'NFC East'],
        'DEN': ['Denver Broncos', 'Empower Field at Mile High', 39.743936, -105.020097, 'America/Denver', 'AFC West'],
        'DET': ['Detroit Lions', 'Ford Field', 42.340156, -83.045808, 'America/New_York', 'NFC North'],
        'GB': ['Green Bay Packers', 'Lambeau Field', 44.501306, -88.062167, 'America/Chicago', 'NFC North'],
        'HOU': ['Houston Texans', 'NRG Stadium', 29.684781, -95.410956, 'America/Chicago', 'AFC South'],
        'IND': ['Indianapolis Colts', 'Lucas Oil Stadium', 39.760056, -86.163806, 'America/New_York', 'AFC South'],
        'JAX': ['Jacksonville Jaguars', 'Everbank Stadium', 30.323925, -81.637356, 'America/New_York', 'AFC South'],
        'KC': ['Kansas City Chiefs', 'Arrowhead Stadium', 39.048786, -94.484566, 'America/Chicago', 'AFC West'],
        'LV': ['Las Vegas Raiders', 'Allegiant Stadium', 36.090794, -115.183952, 'America/Los_Angeles', 'AFC West'],
        'LAC': ['Los Angeles Chargers', 'SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'AFC West'],
        'LAR': ['Los Angeles Rams', 'SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'NFC West'],
#        'LA': ['Los Angeles Rams', 'SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'NFC West'],
        'MIA': ['Miami Dolphins', 'Hard Rock Stadium', 25.957919, -80.238842, 'America/New_York', 'AFC East'],
        'MIN': ['Minnesota Vikings', 'U.S Bank Stadium', 44.973881, -93.258094, 'America/Chicago', 'NFC North'],
        'NE': ['New England Patriots', 'Gillette Stadium', 42.090925, -71.26435, 'America/New_York', 'AFC East'],
        'NO': ['New Orleans Saints', 'Caesars Superdome', 29.950931, -90.081364, 'America/Chicago', 'NFC South'],
        'NYG': ['New York Giants', 'MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'NFC East'],
        'NYJ': ['New York Jets', 'MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'AFC East'],
        'PHI': ['Philadelphia Eagles', 'Lincoln Financial Field', 39.900775, -75.167453, 'America/New_York', 'NFC East'],
        'PIT': ['Pittsburgh Steelers', 'Acrisure Stadium', 40.446786, -80.015761, 'America/New_York', 'AFC North'],
        'SF': ['San Francisco 49ers', 'Levi\'s Stadium', 37.713486, -122.386256, 'America/Los_Angeles', 'NFC West'],
        'SEA': ['Seattle Seahawks', 'Lumen Field', 47.595153, -122.331625, 'America/Los_Angeles', 'NFC West'],
        'TB': ['Tampa Bay Buccaneers', 'Raymomd James Stadium', 27.975967, -82.50335, 'America/New_York', 'NFC South'],
        'TEN': ['Tennessee Titans', 'Nissan Stadium', 36.166461, -86.771289, 'America/Chicago', 'AFC South'],
        'WAS': ['Washington Commanders', 'FedExField', 38.907697, -76.864517, 'America/New_York', 'NFC East'],
#        'WSH': ['Washington Commanders', 'FedExField', 38.907697, -76.864517, 'America/New_York', 'NFC East']
    }
    
    # Division mapping
    public_pick_df['Team'] = public_pick_df['Team'].replace('WSH', 'WAS')
    public_pick_df['Opponent'] = public_pick_df['Opponent'].replace('WSH', 'WAS')
    public_pick_df['Team Division'] = public_pick_df['Team'].map(lambda team: teams2.get(team, ['', '', '', '', '', ''])[5])
    public_pick_df['Opponent Division'] = public_pick_df['Opponent'].map(lambda opponent: teams2.get(opponent, ['', '', '', '', '', ''])[5])
    public_pick_df['Divisional Matchup?'] = (public_pick_df['Team Division'] == public_pick_df['Opponent Division']).astype(int)

    st.write('PUBLIC PICK DF')
    st.write(public_pick_df)    

    # Load the historical data from the file created by nflreadpy
    away_data_df = df_api_schedule
    away_data_df['Calendar Date'] = pd.to_datetime(away_data_df['Calendar Date'])
    
    # Initialization of new columns
    public_pick_df['Away Team'] = 0
    public_pick_df[['Availability', 'Calculated Current Week Alive Entries', 'Calculated Current Week Picks', 'Winning Team']] = [0,0,0,0]
    public_pick_df['Calendar Date'] = pd.NaT
    
    # Merge the dataframes directly (replacing the slow apply/lambda functions)
    
    # 1. Merge to get HOME/AWAY/WINNER
    merged_schedule = pd.merge(
        public_pick_df,
        away_data_df[['Year', 'Week', 'Home Team', 'Away Team', 'Winner/tie']],
        left_on=['Year', 'Week', 'Team'],
        right_on=['Year', 'Week', 'Home Team'],
        how='left',
        suffixes=('', '_home') # Suffix for Home/Away columns when 'Team' is Home
    )
    
    # Rename the column from the first merge to avoid a name conflict
    merged_schedule = merged_schedule.rename(columns={'Away Team_home': 'Opponent_from_home_merge'})
    
    
    # Merge again for when 'Team' is the Away Team
    merged_schedule = pd.merge(
        merged_schedule,
        away_data_df[['Year', 'Week', 'Home Team', 'Away Team', 'Winner/tie']],
        left_on=['Year', 'Week', 'Team'],
        right_on=['Year', 'Week', 'Away Team'],
        how='left',
        suffixes=('_home', '_away') # Suffix for Home/Away columns when 'Team' is Away
    )
    
    merged_schedule = merged_schedule.drop_duplicates(
        subset=['Year', 'Week', 'Team'],
        keep='first'
    ).reset_index(drop=True)
    
    
    # Populate 'Away Team' (binary) and 'Winning Team' (binary)
    public_pick_df['Away Team'] = (
        merged_schedule['Away Team_away'].notna()
    ).astype(int).values
    
    
    # Winning Team Logic:
    # The team is the winner if it matches the 'Winner/tie' column from either merge
    public_pick_df['Winning Team'] = (
        (merged_schedule['Winner/tie_home'] == merged_schedule['Team']) | 
        (merged_schedule['Winner/tie_away'] == merged_schedule['Team'])
    ).fillna(0).astype(int).values
    
    # 2. Merge to get Calendar Date (using the cleaner merge logic from your original script)
    home_dates = away_data_df[['Year', 'Week', 'Home Team', 'Calendar Date']].copy()
    home_dates.rename(columns={'Home Team': 'Team_schedule', 'Calendar Date': 'Matched_Date'}, inplace=True)
    away_dates = away_data_df[['Year', 'Week', 'Away Team', 'Calendar Date']].copy()
    away_dates.rename(columns={'Away Team': 'Team_schedule', 'Calendar Date':'Matched_Date'}, inplace=True)
    
    
    
    schedule_lookup = pd.concat([home_dates, away_dates]).drop_duplicates(
        subset=['Year', 'Week', 'Team_schedule']
    ).reset_index(drop=True)
    
    schedule_lookup['Team_schedule'] = schedule_lookup['Team_schedule'].replace('LA', 'LAR')
    # Merge with the lookup table for the date
    merged_for_calendar_date = pd.merge(
        public_pick_df.reset_index(), # Reset index to avoid merge issues
        schedule_lookup,
        left_on=['Year', 'Week', 'Team'],
        right_on=['Year', 'Week', 'Team_schedule'],
        how='left'
    )
    public_pick_df['Calendar Date'] = merged_for_calendar_date.set_index('index')['Matched_Date'].values
    # Assuming your conversion worked, or you fix it like we discussed:
    public_pick_df['Calendar Date'] = pd.to_datetime(public_pick_df['Calendar Date'], format='%Y-%m-%d')
    #df['Calendar Date_String'] = df['Calendar Date'].dt.strftime('%m/%d/%Y')
    
    # Drop rows where 'Team Division' or 'Opponent Division' is an empty string
    public_pick_df = public_pick_df[public_pick_df['Team Division'] != '']
    public_pick_df = public_pick_df[public_pick_df['Opponent Division'] != '']
    
    public_pick_df = public_pick_df[public_pick_df['Year'] == current_year]
    
    public_pick_df = public_pick_df.drop_duplicates()
    
    public_pick_df['Calendar Date'] = pd.to_datetime(public_pick_df['Calendar Date'], format='%Y-%m-%d')
    
    # ... (The final date manipulation logic remains the same)
    pre_circa_dates = {2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019}
    is_not_in_pre_circa = ~public_pick_df['Year'].isin(pre_circa_dates)
    public_pick_df = public_pick_df[is_not_in_pre_circa]
    
    # Final date manipulation (e.g., correcting Thanksgiving/Christmas week numbers)
    # NOTE: The df.loc assignments must be run *after* the Calendar Date is populated.
    
    # For Year 2025
    condition_2025_date = (public_pick_df['Year'] == 2025) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2025-11-29'))
    public_pick_df.loc[condition_2025_date, 'Week'] += 1
    condition_2025_week = (public_pick_df['Year'] == 2025) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2025-12-26'))
    public_pick_df.loc[condition_2025_week, 'Week'] += 1
    
    # For Year 2024
    condition_2024_date = (public_pick_df['Year'] == 2024) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2024-11-30'))
    public_pick_df.loc[condition_2024_date, 'Week'] += 1
    condition_2024_week = (public_pick_df['Year'] == 2024) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2024-12-27'))
    public_pick_df.loc[condition_2024_week, 'Week'] += 1
    
    # For Year 2023
    condition_2023_date = (public_pick_df['Year'] == 2023) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2023-11-25'))
    public_pick_df.loc[condition_2023_date, 'Week'] += 1
    condition_2023_week = (public_pick_df['Year'] == 2023) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2023-12-25'))
    public_pick_df.loc[condition_2023_week, 'Week'] += 1
    
    # For Year 2022
    condition_2022_date = (public_pick_df['Year'] == 2022) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2022-11-25'))
    public_pick_df.loc[condition_2022_date, 'Week'] += 1
    condition_2022_week = (public_pick_df['Year'] == 2022) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2022-12-25'))
    public_pick_df.loc[condition_2022_week, 'Week'] += 1
    
    # For Year 2021
    condition_2021_date = (public_pick_df['Year'] == 2021) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2021-11-26'))
    public_pick_df.loc[condition_2021_date, 'Week'] += 1
    
    condition_2021_week = (public_pick_df['Year'] == 2021) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2021-12-26'))
    public_pick_df.loc[condition_2021_week, 'Week'] += 1
    
    # For Year 2020
    condition_2020_date = (public_pick_df['Year'] == 2020) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2020-11-27'))
    public_pick_df.loc[condition_2020_date, 'Week'] += 1
    
    public_pick_df['EV'] = 0
  
    public_pick_df = public_pick_df.drop_duplicates()

    st.write(public_pick_df)
    
    # ==============================================================================
    # SECTION 4: POPULATE week_df WITH PUBLIC PICK DATA
    # ==============================================================================
    
    # This assumes 'week_df' already exists in your environment, as mentioned.
    
    print("Creating reverse team map for lookup...")
    # Create a reverse map: {"Carolina Panthers": "CAR", "Chicago Bears": "CHI", ...}
    # This is VITAL for linking week_df (full names) to public_pick_df (abbreviations)
    try:
        team_name_to_abbr_map = {details[0]: abbr for abbr, details in teams2.items()}
    except NameError:
        print("CRITICAL ERROR: 'teams' dictionary not defined. Cannot create lookup map.")
        # Handle this error, perhaps by exiting
        team_name_to_abbr_map = {}
    
    def get_public_pick_percent(row, team_type):
        """
        Looks up the public pick percentage from 'public_pick_df' for a team.
        
        'row' is a row from week_df.
        'team_type' is either 'home' or 'away'.
        """
        
        # 1. Get week number (e.g., "Week 10" -> 10)
        week_num = row["Week_Num"]
        
        # 2. Get the full team name and identify if we seek a home or away team
        if team_type == 'home':
            team_name = row["Home Team"]
            is_away_flag = 0 # The 'Away Team' flag in public_pick_df should be 0
        elif team_type == 'away':
            team_name = row["Away Team"]
            is_away_flag = 1 # The 'Away Team' flag in public_pick_df should be 1
        else:
            return np.nan # Invalid team_type

        # 3. Convert the full team name ("Carolina Panthers") to its abbreviation ("CAR")
        team_abbr = team_name_to_abbr_map.get(team_name)
        
        if not team_abbr:
            # print(f"Warning: Could not find abbreviation for {team_name}")
            return np.nan # Team name not in our map

        # 4. Find the matching row in public_pick_df
        # We filter by the integer week, the team abbreviation, and the home/away flag
        match = public_pick_df[
            (public_pick_df["Week"] == week_num) &
            (public_pick_df["Team"] == team_abbr) &
            (public_pick_df["Away Team"] == is_away_flag)
        ]

        # 5. Return the value if found, otherwise return NaN
        if not match.empty:
            # .values[0] gets the first (and should be only) matching value
            return match["Public Pick %"].values[0]
        else:
            # No match found in public_pick_df for this team/week
            return np.nan
    
    # --- Apply the function to your week_df ---
    
    print("Populating 'Away Team Public Pick %' in week_df...")
    consolidated_df["Away Team Public Pick %"] = consolidated_df.apply(
        lambda row: get_public_pick_percent(row, 'away'),
        axis=1
    )
    
    print("Populating 'Home Team Public Pick %' in week_df...")
    consolidated_df["Home Team Public Pick %"] = consolidated_df.apply(
        lambda row: get_public_pick_percent(row, 'home'),
        axis=1
    )

    print("Finished populating public pick percentages.")

    # Save the consolidated DataFrame to a single CSV file

    if selected_contest == 'Circa':
        consolidated_csv_file = "nfl_schedule_circa.csv"
    elif selected_contest == 'Splash Sports':
        consolidated_csv_file = "nfl_schedule_splash.csv"
    else:
        consolidated_csv_file = "nfl_schedule_dk.csv"
    consolidated_df.to_csv(consolidated_csv_file, index=False)    
    collect_schedule_travel_ranking_data_nfl_schedule_df = consolidated_df
    
    return collect_schedule_travel_ranking_data_nfl_schedule_df

#def get_live_contest_data():
#	if selected_contest = 'Circa':
#	elif selected_contest = 'Splash Sports':
#		if subcontest = '':
#		elif subcontest = '':
#		elif subcontest = '':
#		elif subcontest = '':
#		elif subcontest = '':
#		elif subcontest = '':
#		elif subcontest = '':
#	else:
