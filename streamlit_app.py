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

starting_week = 2
starting_year = 2025 #Can go as far back as 2010 if you need to collect all new data. You shouldn't need to change this though
current_year = 2025
current_year_plus_1 = current_year + 1 #current_year + 1
thanksgiving_week = 13
christmas_week = 18

thanksgiving_date = '2025-11-27'
black_friday_date = '2025-11-28'
christmas_date = '2025-12-25'
boxing_day_date = '2025-12-26'

thanksgiving_reset_date = '2025-11-29' #THIS DATE IS INCLUDED IN THE RESET. SO IF THERE ARE GAMES ON THIS DATE, THEY WILL HAVE A WEEK ADDED
christmas_reset_date = '2025-12-26'
season_start_date = 'September 3, 2025' # MUST BE A WEDNESDAY

circa_2020_entries = 1373
circa_2021_entries = 4071
circa_2022_entries = 6106
circa_2023_entries = 9234
circa_2024_entries = 14221
circa_2025_entries = 18718
# ==============================================================================
# SECTION 1: SURVIVORGRID.COM SCRAPING (UNCHANGED - nflreadpy CANNOT DO THIS)
# ==============================================================================

NUM_WEEKS_TO_KEEP = starting_week - 1
current_year_plus_1 = current_year + 1 #current_year + 1

circa_total_entries = 18714
splash_big_splash_total_entries = 16337
splash_4_for_4_total_entries = 10000
splash_for_the_fans_total_entries = 8382
splash_ship_it_nation_total_entries = 10114
splash_high_roller_total_entries = 1004
splash_rotowire_total_entries = 9048
splash_walkers_25_total_entries = 36501
splash_bloody_total_entries = 5000
dk_total_entries = 20000

# --------------------------------------------------------------------------
# --- 1. DEFAULT TEAM RANKS (Baseline) ---
# Used if the user selects 'Default' in the UI for their current rank setting.
# --------------------------------------------------------------------------
DEFAULT_RANKS = {
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

# --------------------------------------------------------------------------
# --- 2. PRE-SEASON RANKINGS (STATIC) ---
# This is a static reference for pre-season ranking data.
# --------------------------------------------------------------------------
PRESEASON_RANKS = {
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


def get_schedule(config: dict):
    print("Gathering Schedule Data...")
    # Make a request to the website
    r = requests.get('https://www.fftoday.com/nfl/schedule.php')
    r_html = r.text

    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(r_html, 'html.parser')

    # Find the table in the HTML
    table = soup.find('table', attrs={'width': '80%', 'border': '0', 'cellpadding': '0', 'cellspacing': '0'})

    # Find all rows in the table with a white background
    rows = table.find_all('tr', attrs={'bgcolor': '#ffffff'})
    rows = rows[:272]
    print("Schedule Data Retrieved")
    return table, rows

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
            current_rank = DEFAULT_RANKS.get(team, 0)
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
        'WSH': ['Washington Commanders', 'FedExField', 38.907697, -76.864517, 'America/New_York', 'NFC East']
    }
    
    # Division mapping
    public_pick_df['Team Division'] = public_pick_df['Team'].map(lambda team: teams2.get(team, ['', '', '', '', '', ''])[5])
    public_pick_df['Opponent Division'] = public_pick_df['Opponent'].map(lambda opponent: teams2.get(opponent, ['', '', '', '', '', ''])[5])
    public_pick_df['Divisional Matchup?'] = (public_pick_df['Team Division'] == public_pick_df['Opponent Division']).astype(int)
    
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

# --- Helper function used in the main logic (Moved from the bottom) ---
def get_expected_availability(team_name, availability_dict: Dict):
    """
    Calculates the expected availability percentage for a team.
    Normalizes the input 'team_name' to Full Name to ensure it matches the dictionary keys.
    """
    # 1. Define the mapping (Abbr -> Full Name)
    # Ensure this matches the keys used in your availability_dict
    team_name_map = {
        "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
        "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
        "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
        "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
        "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
        "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
        "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
        "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
        "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
        "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
        "TEN": "Tennessee Titans", "WAS": "Washington Commanders", "WSH": "Washington Commanders"
    }

    # 2. Normalize the lookup key
    # If team_name is "ARI", this becomes "Arizona Cardinals"
    # If team_name is "Arizona Cardinals", this stays "Arizona Cardinals"
    full_team_name = team_name_map.get(team_name, team_name)

    # 3. Perform Lookup
    # Try looking up the full name first
    availability = availability_dict.get(full_team_name)
    
    # Optional fallback: If not found, try the original abbreviation
    if availability is None:
        availability = availability_dict.get(team_name)

    # 4. Handle missing/invalid values
    if availability is None:
        return 1.0
    elif availability <= -0.01:
        return 1.0      
    else:
        return float(availability)

# Mock for st.write/st.success (assuming Streamlit is used for output)
def st_write(message):
    print(message)
    
def st_success(message):
    print(f"SUCCESS: {message}")

# --- Main Function ---
def get_predicted_pick_percentages(config: dict, schedule_df: pd.DataFrame):
    """
    Calculates predicted pick percentages for each team in each week,
    adjusting for team availability based on previous expected picks.
    """

    selected_contest = config['selected_contest'] 
    subcontest = config['subcontest'] 
    starting_week = config['starting_week'] 
    current_week_entries = config['current_week_entries'] 
    week_requiring_two_selections = config.get('weeks_two_picks', []) 
    week_requiring_three_selections = config.get('weeks_three_picks', []) 
    team_availability = config.get('team_availabilities', {}) 
    custom_pick_percentages = config.get('pick_percentages', {})
    
    # Define features related to holiday games (ensure consistency with training/prediction)
    # The actual presence of these columns depends on your data loading/feature engineering elsewhere.
    holiday_cols = ['Thanksgiving Favorite', 'Thanksgiving Underdog', 'Christmas Favorite', 'Christmas Underdog', 'Pre Thanksgiving', 'Pre Christmas']

    # Load your historical data (Replace dummy paths with your actual file loading logic if necessary)
    if selected_contest == 'Circa':
        df = pd.read_csv('Circa_historical_data.csv')
    elif selected_contest == 'Splash Sports':
        df = pd.read_csv('DK_historical_data.csv')
    else:
        df = pd.read_csv('DK_historical_data.csv')

    df.rename(columns={"Week": "Date"}, inplace=True)
    df['Pick %'].fillna(0.0, inplace=True)

    # --- Train Base Model (No Public Pick Data) ---
    st_write("Training base model (no public pick data)...")
    if selected_contest == 'Circa':
        # Removed holiday cols from base_features to avoid KeyError if they don't exist in historical data
        base_features = ['Win %', 'Future Value (Stars)', 'Date', 'Away Team', 'Divisional Matchup?', 'Week_Mean_WinPct', 'Week_Mean_FV', 'Week_Max_WinPct', 
						 'Week_Max_FV', 'Week_Min_WinPct', 'Week_Min_FV', 'Week_Std_WinPct', 'Week_Std_FV', 'Team_WinPct_RelativeToWeekMean', 'Team_FV_RelativeToWeekMean', 
						 'Team_WinPct_RelativeToTopTeam', 'Team_FV_RelativeToTopTeam', 'Win % Rank', 'Star Rating Rank','Num_Teams_This_Week', 'Rank_Density', 'FV_Rank_Density', 
						 'Future_Weeks_Top_Team', 'Future_Weeks_Over_80', 'Future_Weeks_70_80', 'Future_Weeks_60_70', 'Pre Christmas', 'Pre Thanksgiving', 'Christmas Underdog', 
						 'Christmas Favorite', 'Thanksgiving Underdog', 'Thanksgiving Favorite', 'thanksgiving_week', 'christmas_week', 'Thursday Night Game']
        # Add back only if guaranteed to exist in the historical data 'df'
        base_features.extend([col for col in holiday_cols if col in df.columns])
        base_features = list(set(base_features)) # Remove duplicates
    else:
        base_features = ['Win %', 'Future Value (Stars)', 'Date', 'Away Team', 'Divisional Matchup?', 'Week_Mean_WinPct', 'Week_Mean_FV', 'Week_Max_WinPct', 
						 'Week_Max_FV', 'Week_Min_WinPct', 'Week_Min_FV', 'Week_Std_WinPct', 'Week_Std_FV', 'Team_WinPct_RelativeToWeekMean', 'Team_FV_RelativeToWeekMean', 
						 'Team_WinPct_RelativeToTopTeam', 'Team_FV_RelativeToTopTeam', 'Win % Rank', 'Star Rating Rank','Num_Teams_This_Week', 'Rank_Density',
						 'FV_Rank_Density',  'Future_Weeks_Top_Team', 'Future_Weeks_Over_80', 'Future_Weeks_70_80', 'Future_Weeks_60_70', 'Thursday Night Game']
        
    X = df[base_features].fillna(0) # Fill NA for training data
    y = df['Pick %']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # This is your original model, now renamed 'base'
    rf_model_base = RandomForestRegressor(n_estimators=50, random_state=0)
    rf_model_base.fit(X_train, y_train)

    y_pred = rf_model_base.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st_write(f"Base Model Mean Absolute Error: {mae:.2f}")

    # --- Train Enhanced Model (WITH Public Pick Data) ---
    st_write("Training enhanced model (with public pick data)...")
    
    assumed_public_pick_col = 'Public Pick %' 
    rf_model_enhanced = None
    
    if assumed_public_pick_col not in df.columns:
        st_write(f"Warning: Historical data does not contain '{assumed_public_pick_col}'. Cannot train enhanced model.")
    else:
        enhanced_features = base_features + [assumed_public_pick_col]
        
        # Filter historical data to rows WHERE public pick data was available
        df_enhanced = df.dropna(subset=[assumed_public_pick_col])
        
        if df_enhanced.empty:
            st_write("Warning: No historical data found with public pick %. Enhanced model will not be trained.")
        else:
            st_write(f"Training enhanced model on {len(df_enhanced)} historical samples.")
            X_enhanced = df_enhanced[enhanced_features].fillna(0) # Fill NA for training data
            y_enhanced = df_enhanced['Pick %']
            
            # Train/test split for the enhanced model
            X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_enhanced, y_enhanced, test_size=0.2, random_state=42)
            
            rf_model_enhanced = RandomForestRegressor(n_estimators=50, random_state=0)
            rf_model_enhanced.fit(X_train_e, y_train_e)
            
            y_pred_e = rf_model_enhanced.predict(X_test_e)
            mae_e = mean_absolute_error(y_test_e, y_pred_e)
            st_write(f"Enhanced Model Mean Absolute Error: {mae_e:.2f}")

    st_write("Starting week-by-week pick percentage predictions...")
    
    # 1. Load full schedule and copy
    nfl_schedule_df = schedule_df.copy()

    nfl_schedule_df['Week_Num'] = pd.to_numeric(
            nfl_schedule_df['Week_Num'], 
            errors='coerce'
        ).fillna(-1).astype(int)

    if current_week_entries >= 0:
        nfl_schedule_df.loc[nfl_schedule_df['Week_Num'] == starting_week, 'Total Remaining Entries at Start of Week'] = current_week_entries
    else:
        # Handle the -1 (auto-estimate) case based on contest
        if selected_contest == 'Circa':
            default_entries = circa_total_entries # Example
        elif selected_contest == 'Splash Sports':
            if subcontest == "The Big Splash ($150 Entry)":
                default_entries = splash_big_splash_total_entries
            elif subcontest == "4 for 4 ($50 Entry)":
                default_entries = splash_4_for_4_total_entries
            elif subcontest == "Free RotoWire (Free Entry)":
                default_entries = splash_rotowire_total_entries
            elif subcontest == "For the Fans ($40 Entry)":
                default_entries = splash_for_the_fans_total_entries
            elif subcontest == "Walker's Ultimate Survivor ($25 Entry)":
                default_entries = splash_walkers_25_total_entries
            elif subcontest == "Ship It Nation ($25 Entry)":
                default_entries = splash_ship_it_nation_total_entries
            elif subcontest == "High Roller ($1000 Entry)":
                default_entries = splash_high_roller_total_entries
            elif subcontest == "Week 9 Bloody Survivor ($100 Entry)":
                default_entries = splash_bloody_total_entries
            else:
                default_entries = 20000
        else: # DraftKings
             default_entries = 20000 # Example
        nfl_schedule_df.loc[nfl_schedule_df['Week_Num'] == starting_week, 'Total Remaining Entries at Start of Week'] = default_entries
    # --- End POOL SIZE LOGIC ---

    # Ensure 'Total Remaining Entries at Start of Week' has been correctly initialized
    # If the entry size is not set, the simulation will break.
    if nfl_schedule_df.loc[nfl_schedule_df['Week_Num'] == starting_week, 'Total Remaining Entries at Start of Week'].empty:
         st_write(f"Error: 'Total Remaining Entries' not set for starting week {starting_week}. Assuming {default_entries}.")
         nfl_schedule_df.loc[nfl_schedule_df['Week_Num'] == starting_week, 'Total Remaining Entries at Start of Week'] = default_entries
    
    max_week = nfl_schedule_df['Week_Num'].max() # Get max week from the data itself
    
    # 2. Initialize 'used' dictionary (U_prev_week)
    S_at_sw = nfl_schedule_df[nfl_schedule_df['Week_Num'] == starting_week]['Total Remaining Entries at Start of Week'].iloc[0]
    U_prev_week: Dict[str, float] = {}
    
    # Get all unique teams
    all_teams_series = pd.unique(nfl_schedule_df[['Home Team', 'Away Team']].values.ravel('K'))
    all_teams = [team for team in all_teams_series if pd.notna(team)] 
    
    if S_at_sw > 0:
        for team in all_teams:
            avail_percent = get_expected_availability(team, team_availability) 
            implied_used_count = S_at_sw * (1.0 - avail_percent)
            U_prev_week[team] = max(0.0, min(implied_used_count, S_at_sw))
    else:
        st_write(f"Warning: S_at_sw is 0. Initializing U_prev_week to all zeros.")
        for team in all_teams:
            U_prev_week[team] = 0.0
    
    # 3. Initialize all columns you will calculate in the loop
    calc_cols = [
        'Home Team Expected Availability', 'Away Team Expected Availability',
        'Home Pick %', 'Away Pick %', 'Expected Home Team Survivors', 
        'Expected Away Team Survivors', 'Expected Home Team Eliminations', 
        'Expected Away Team Eliminations'
    ]
    for col in calc_cols:
        nfl_schedule_df[col] = np.nan

    # Loop through each week, starting from your defined starting week
    for current_week in range(starting_week, int(max_week) + 1):
        st_write(f"\n--- 🏈 Processing Week {current_week} of {max_week}---")
        current_week_mask = nfl_schedule_df['Week_Num'] == current_week
        if not current_week_mask.any():
            st_write(f"Skipping week {current_week} (no data found).")
            continue

        # --- A. GET TOTAL ENTRIES (S_w) ---
        S_w = nfl_schedule_df.loc[current_week_mask, 'Total Remaining Entries at Start of Week'].iloc[0]
        if pd.isna(S_w) or S_w <= 0:
            st_write(f"Warning: 0 or NaN entries for Week {current_week}. Stopping sequential calculation.")
            break

        # --- B. CALCULATE & SET *THIS* WEEK'S AVAILABILITY ---
        for team in all_teams:
            unavailable_count = U_prev_week.get(team, 0.0)
            # (S_w - unavailable_count) is the number of remaining entries who CAN pick this team
            # We divide by S_w to get the percentage of the remaining pool who can pick this team
            team_avail_percent = (S_w - unavailable_count) / S_w
            team_avail_percent = max(0.0, min(1.0, team_avail_percent)) # Clamp between 0 and 1

            
            # Set it in the main dataframe (only for the games this team is playing in this week)
            nfl_schedule_df.loc[current_week_mask & (nfl_schedule_df['Home Team'] == team), 'Home Team Expected Availability'] = team_avail_percent
            nfl_schedule_df.loc[current_week_mask & (nfl_schedule_df['Away Team'] == team), 'Away Team Expected Availability'] = team_avail_percent

        # --- C. PREPARE & PREDICT *THIS* WEEK'S PICKS ---
        new_df = nfl_schedule_df.loc[current_week_mask].copy()
        # Select all columns needed for prediction features
        selected_columns = [
            'Week', 'Away Team', 'Home Team', 'Away Team Fair Odds', 'Home Team Fair Odds', 
            'Away Team Star Rating', 'Home Team Star Rating', 'Divisional Matchup Boolean', 
            'Away Team Public Pick %', 'Home Team Public Pick %', 
            'Away Team Expected Availability', 'Home Team Expected Availability', 
			'Away Team Thanksgiving Favorite', 'Away Team Thanksgiving Underdog', 
			'Home Team Thanksgiving Favorite', 'Home Team Thanksgiving Underdog', 
			'Away Team Christmas Favorite', 'Away Team Christmas Underdog',
			'Home Team Christmas Favorite', 'Home Team Christmas Underdog',
			'Away Team Pre Thanksgiving', 'Away Team Pre Christmas',
			'Home Team Pre Thanksgiving', 'Home Team Pre Christmas', 'Thursday Night Game'
        ]
        
        # Ensure only valid columns are selected
        new_df = new_df[[col for col in selected_columns if col in new_df.columns]].copy()
        new_df = new_df.rename(columns={'Week': 'Date'})

	

        # Check if public pick data is available for this week's predictions
        # Note: This check relies on 'Home Team Public Pick %' not being NaN
        public_picks_available = (new_df['Home Team Public Pick %'].notna().any())
        
        # --- Create away_df and home_df (Feature Engineering) ---
        # Helper function to rename columns consistently for prediction
        def create_pick_df(df_in, team_type_1, team_type, opponent_type_1, opponent_type, is_away):
            df_out = df_in.rename(columns={
                f'{team_type_1} Team': 'Team', 
                f'{opponent_type} Team': 'Opponent', 
                f'{team_type} Fair Odds': 'Win %', 
                f'{team_type} Star Rating': 'Future Value (Stars)', 
                'Divisional Matchup Boolean': 'Divisional Matchup?',
                f'{team_type} Expected Availability': 'Availability', 
                f'{team_type} Public Pick %': 'Public Pick %',
				f'{team_type} Thanksgiving Favorite': 'Thanksgiving Favorite',
				f'{team_type} Thanksgiving Underdog': 'Thanksgiving Underdog',
				f'{team_type} Christmas Favorite': 'Christmas Favorite',
				f'{team_type} Christmas Underdog': 'Christmas Underdog',
				f'{team_type} Pre Thanksgiving': 'Pre Thanksgiving',
				f'{team_type} Pre Christmas': 'Pre Christmas'
            }).drop(columns=[f'{opponent_type_1} Fair Odds', f'{opponent_type_1} Star Rating', f'{opponent_type_1} Public Pick %', f'{opponent_type_1} Expected Availability'])
            
            df_out['Home/Away'] = 'Away' if is_away else 'Home'
            df_out['Away Team'] = 1 if is_away else 0
            df_out['Date'] = current_week
            return df_out.copy()

        away_df = create_pick_df(new_df, 'Away', 'Away Team', 'Home Team', 'Home', True)
        home_df = create_pick_df(new_df, 'Home', 'Home Team', 'Away Team', 'Away', False)

        # 3. CONCATENATE and NORMALIZE PICKS
        pick_predictions_df = pd.concat([away_df, home_df], ignore_index=True)

        # ==============================================================================
        # SECTION 4: NEW FEATURE ENGINEERING (RANKS AND RELATIVE STATS)
        # ==============================================================================
        print("\n⚙️ Starting Section 4: Feature Engineering (Ranks and Relative Stats)...")
        
        # Define group keys for weekly calculations
        group_keys = ['Date']
        
        # 1. Calculate Weekly Win % Stats
        # Using .transform() to broadcast the group-level stats to every row in that group
        print("  Calculating weekly Win % statistics (mean, max, min, std)...")
        pick_predictions_df['Week_Mean_WinPct'] = pick_predictions_df.groupby(group_keys)['Win %'].transform('mean')
        pick_predictions_df['Week_Max_WinPct'] = pick_predictions_df.groupby(group_keys)['Win %'].transform('max')
        pick_predictions_df['Week_Min_WinPct'] = pick_predictions_df.groupby(group_keys)['Win %'].transform('min')
        pick_predictions_df['Week_Std_WinPct'] = pick_predictions_df.groupby(group_keys)['Win %'].transform('std')
        
        print("  Calculating weekly Future Value statistics (mean, max, min, std)...")
        pick_predictions_df['Week_Mean_FV'] = pick_predictions_df.groupby(group_keys)['Future Value (Stars)'].transform('mean')
        pick_predictions_df['Week_Max_FV'] = pick_predictions_df.groupby(group_keys)['Future Value (Stars)'].transform('max')
        pick_predictions_df['Week_Min_FV'] = pick_predictions_df.groupby(group_keys)['Future Value (Stars)'].transform('min')
        pick_predictions_df['Week_Std_FV'] = pick_predictions_df.groupby(group_keys)['Future Value (Stars)'].transform('std')
        
        # Fill NaN for Std on weeks with only one game (if any)
        pick_predictions_df['Week_Std_WinPct'] = pick_predictions_df['Week_Std_WinPct'].fillna(0)
        
        # Fill NaN for Std on weeks with only one game (if any)
        pick_predictions_df['Week_Std_FV'] = pick_predictions_df['Week_Std_FV'].fillna(0)
        
        # 2. Calculate Team-Specific Relative Stats
        print("  Calculating team-relative Win % stats...")
        pick_predictions_df['Team_WinPct_RelativeToWeekMean'] = pick_predictions_df['Win %'] - pick_predictions_df['Week_Mean_WinPct']
        
        # 2. Calculate Team-Specific Relative Stats
        print("  Calculating team-relative Future Value stats...")
        pick_predictions_df['Team_FV_RelativeToWeekMean'] = pick_predictions_df['Future Value (Stars)'] - pick_predictions_df['Week_Mean_FV']
        
        # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
        pick_predictions_df['Team_WinPct_RelativeToTopTeam'] = pick_predictions_df['Win %'] / pick_predictions_df['Week_Max_WinPct']
        pick_predictions_df['Team_WinPct_RelativeToTopTeam'] = pick_predictions_df['Team_WinPct_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Handle potential division by zero if Max_Win is 0 (unlikely, but safe)
        pick_predictions_df['Team_FV_RelativeToTopTeam'] = pick_predictions_df['Future Value (Stars)'] / pick_predictions_df['Week_Max_FV']
        pick_predictions_df['Team_FV_RelativeToTopTeam'] = pick_predictions_df['Team_FV_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                                  
        
        # 3. Calculate Ranks (Win % and Star Rating)
        # .rank(ascending=False) means the highest value gets rank 1 (e.g., "best")
        print("  Calculating Win % and Star Rating ranks...")
        pick_predictions_df['Win % Rank'] = pick_predictions_df.groupby(group_keys)['Win %'].rank(ascending=False, method='min')
        pick_predictions_df['Star Rating Rank'] = pick_predictions_df.groupby(group_keys)['Future Value (Stars)'].rank(ascending=False, method='min')
        
        # 4. Calculate Rank Density
        # First, get the number of teams (games) in each week
        print("  Calculating Rank Density...")
        pick_predictions_df['Num_Teams_This_Week'] = pick_predictions_df.groupby(group_keys)['Team'].transform('count')
        
        # This normalizes the rank based on the number of available teams that week
        pick_predictions_df['Rank_Density'] = pick_predictions_df['Win % Rank'] / pick_predictions_df['Num_Teams_This_Week']
        
        pick_predictions_df['FV_Rank_Density'] = pick_predictions_df['Star Rating Rank'] / pick_predictions_df['Num_Teams_This_Week']
        
        print("✅ Feature engineering complete.")
        # print(df[['Year', 'Week', 'Team', 'Win %', 'Win % Rank', 'Rank_Density', 'Num_Teams_This_Week']].head())


		# ------------------------------------------------------------------------------
        # NEW SECTION: Future Value & Holiday Features
        # ------------------------------------------------------------------------------
        
        # A. Holiday Booleans
        # Convert existing holiday specific columns into a single boolean "Is Holiday Game?"
        # (Checks if either Favorite or Underdog status is > 0)
        pick_predictions_df['christmas_week'] = (
            pick_predictions_df['Date'] == christmas_week).astype(int)

        pick_predictions_df['thanksgiving_week'] = (
            pick_predictions_df['Date'] == thanksgiving_week).astype(int)

        if selected_contest == 'Circa':
            pick_predictions_df['Calendar Date'] = pd.to_datetime(pick_predictions_df['Calendar Date'])

            # Create the "Thursday Night Game" column
            # Logic:
            # 1. Day of week is Thursday (dt.dayofweek == 3; Monday is 0, Sunday is 6)
            # 2. christmas_week is 0
            # 3. thanksgiving_week is 0
            pick_predictions_df['Thursday Night Game'] = (
                (pick_predictions_df['Calendar Date'].dt.dayofweek == 3) & 
                (pick_predictions_df['christmas_week'] == 0) & 
                (pick_predictions_df['thanksgiving_week'] == 0)
            ).astype(int) # Convert boolean (True/False) to integer (1/0)
        else:
            pick_predictions_df['Calendar Date'] = pd.to_datetime(pick_predictions_df['Calendar Date'])
            pick_predictions_df['Thursday Night Game'] = (
                (pick_predictions_df['Calendar Date'].dt.dayofweek == 3)
            ).astype(int)            


        # B. Current Week Relative Strength
        # "Win Percentage of the team minus the win percentage of the Top team that week."
        # Note: 'Week_Max_WinPct' was calculated in Section 4
        pick_predictions_df['WinPct_Diff_From_Top'] = pick_predictions_df['Win %'] - pick_predictions_df['Week_Max_WinPct']

        # C. Future Schedule Analysis (The "Look-ahead" counts)
        # We need to look at nfl_schedule_df for all weeks GREATER than current_week
        future_schedule = nfl_schedule_df[nfl_schedule_df['Week_Num'] > current_week].copy()

        if not future_schedule.empty:
            # 1. Flatten the future schedule to a simple (Team, Week, WinPct) format
            fut_home = future_schedule[['Home Team', 'Home Team Fair Odds', 'Week_Num']].rename(
                columns={'Home Team': 'Team', 'Home Team Fair Odds': 'WinPct'}
            )
            fut_away = future_schedule[['Away Team', 'Away Team Fair Odds', 'Week_Num']].rename(
                columns={'Away Team': 'Team', 'Away Team Fair Odds': 'WinPct'}
            )
            fut_long = pd.concat([fut_home, fut_away], ignore_index=True)

            # 2. Identify if they are the "Top Team" in that future week
            # Group by Week to find the Max Win % for that specific future week
            weekly_max_series = fut_long.groupby('Week_Num')['WinPct'].transform('max')
            fut_long['Is_Top_Team'] = (fut_long['WinPct'] == weekly_max_series)

            total_future_weeks = future_schedule['Week_Num'].nunique()
            # 3. Calculate the counts per team
            # Create boolean columns for the criteria
            fut_long['Future_Weeks_Top_Team'] = fut_long['Is_Top_Team'].astype(int)
            fut_long['Future_Weeks_Over_80'] = (fut_long['WinPct'] > 0.80).astype(int)
            fut_long['Future_Weeks_70_80'] = ((fut_long['WinPct'] >= 0.70) & (fut_long['WinPct'] <= 0.80)).astype(int)
            fut_long['Future_Weeks_60_70'] = ((fut_long['WinPct'] >= 0.60) & (fut_long['WinPct'] < 0.70)).astype(int)

            # 4. Aggregate by Team (Summing the weeks)
            team_future_stats = fut_long.groupby('Team')[[
                'Future_Weeks_Top_Team', 
                'Future_Weeks_Over_80', 
                'Future_Weeks_70_80', 
                'Future_Weeks_60_70'
            ]].sum().reset_index()

            if total_future_weeks > 0:
                stat_cols = ['Future_Weeks_Top_Team', 'Future_Weeks_Over_80', 'Future_Weeks_70_80', 'Future_Weeks_60_70']
                team_future_stats[stat_cols] = team_future_stats[stat_cols] / total_future_weeks

            # 5. Merge these stats back into the current prediction dataframe
            pick_predictions_df = pick_predictions_df.merge(team_future_stats, on='Team', how='left')
            
            # Fill NaNs with 0 (for teams that might not have future games in the filtered set)
            pick_predictions_df[['Future_Weeks_Top_Team', 'Future_Weeks_Over_80', 'Future_Weeks_70_80', 'Future_Weeks_60_70']] = \
                pick_predictions_df[['Future_Weeks_Top_Team', 'Future_Weeks_Over_80', 'Future_Weeks_70_80', 'Future_Weeks_60_70']].fillna(0)

        else:
            # If no future weeks exist (last week of season), set all to 0
            pick_predictions_df['Future_Weeks_Top_Team'] = 0
            pick_predictions_df['Future_Weeks_Over_80'] = 0
            pick_predictions_df['Future_Weeks_70_80'] = 0
            pick_predictions_df['Future_Weeks_60_70'] = 0
        
        # ==============================================================================
        # END SECTION 4
        # ==============================================================================

        # --- Conditional Prediction (NOW on the combined, feature-rich dataframe) ---
        if public_picks_available and rf_model_enhanced:
            st_write("--- Predicting using ENHANCED model (with public pick data) ---")
            features = enhanced_features
            model = rf_model_enhanced
        else:
            st_write("--- Predicting using BASE model (no public pick data) ---")
            features = base_features
            model = rf_model_base
            
        # Ensure all required features exist, fill NA with 0
        predict_data_cols = [col for col in features if col in pick_predictions_df.columns]
        
        # Add any missing feature columns (e.g., if a holiday col wasn't in future data)
        for col in features:
            if col not in pick_predictions_df.columns:
                pick_predictions_df[col] = 0.0 # Default value for missing feature
                
        predict_data = pick_predictions_df[predict_data_cols].fillna(0) # Fill NA for prediction
        
        # Predict on the *entire* weekly dataframe at once
        st.write('TESTESTESTESTESTTESTTESTTESTTESTESTTESTTESTTESTT')
        st.write(pick_predictions_df)
        pick_predictions_df['Pick %'] = model.predict(predict_data)

        target_pick_sum = 1.0
        if current_week in week_requiring_two_selections:
            target_pick_sum = 2.0
        elif current_week in week_requiring_three_selections:
            target_pick_sum = 3.0
            
        # 2. Initial Normalization
        # Scale predictions so they sum to target BEFORE applying constraints
        # This ensures we are starting from the correct total volume
        current_sum = pick_predictions_df['Pick %'].sum()
        if current_sum > 0:
            pick_predictions_df['Pick %'] = pick_predictions_df['Pick %'] * (target_pick_sum / current_sum)
        else:
            # Handle rare case where model predicts 0 for everyone
            pick_predictions_df['Pick %'] = 0.0
        
        # 3. Iterative Water-Filling Loop
        max_iterations = 15 
        
        for i in range(max_iterations):
            # Ensure Availability is filled
            pick_predictions_df['Availability'] = pick_predictions_df['Availability'].fillna(0.0)
        
            # A. Find teams exceeding their availability
            over_cap_mask = pick_predictions_df['Pick %'] > pick_predictions_df['Availability']
            
            # Check if we are done (no violations and sum is correct)
            current_total_pick = pick_predictions_df['Pick %'].sum()
            
            # If no one is over the cap, we are likely done, unless the sum drifted due to floating point math
            if not over_cap_mask.any():
                break
                
            # B. Calculate the "Overflow" (The amount we must take away)
            # Sum of (Prediction - Availability) for all teams over the limit
            excess_prob = (pick_predictions_df.loc[over_cap_mask, 'Pick %'] - pick_predictions_df.loc[over_cap_mask, 'Availability']).sum()
            
            # C. Clamp the violators to their max availability
            pick_predictions_df.loc[over_cap_mask, 'Pick %'] = pick_predictions_df.loc[over_cap_mask, 'Availability']
            
            # D. Distribute Overflow to valid teams
            # We only distribute to teams that are NOT currently over the cap
            non_violator_mask = ~over_cap_mask
            sum_non_violators = pick_predictions_df.loc[non_violator_mask, 'Pick %'].sum()
            
            if sum_non_violators > 0:
                # Calculate the share for each team
                # Your Logic: Team_Share = Team_Pick / Sum_Of_Available_Teams
                # Example: Eagles (28%) / Sum (56%) = 0.5
                shares = pick_predictions_df.loc[non_violator_mask, 'Pick %'] / sum_non_violators
                
                # Add their share of the excess
                # Example: Eagles (28%) += 8% * 0.5
                pick_predictions_df.loc[non_violator_mask, 'Pick %'] += (excess_prob * shares)
            else:
                # If sum_non_violators is 0 (everyone is either capped or has 0% prediction),
                # we cannot distribute the excess. The specific math is impossible.
                # We break to avoid infinite loop.
                st_write(f"Warning Week {current_week}: Cannot redistribute excess {excess_prob:.4f}. Pool is saturated.")
                break
        
        # Final sanity clamp to ensure floating point math didn't push anyone 0.00001 over
        pick_predictions_df['Pick %'] = pick_predictions_df[['Pick %', 'Availability']].min(axis=1)

        # --- D. STORE PREDICTIONS & CALCULATE SURVIVORS ---
        
        # 4. Map the normalized 'Pick %' back to the main nfl_schedule_df
        for _, row in pick_predictions_df.iterrows():
            team = row['Team']
            pick_percent = row['Pick %']
            
            # Map Home Pick %
            nfl_schedule_df.loc[current_week_mask & (nfl_schedule_df['Home Team'] == team), 'Home Pick %'] = pick_percent
            
            # Map Away Pick %
            nfl_schedule_df.loc[current_week_mask & (nfl_schedule_df['Away Team'] == team), 'Away Pick %'] = pick_percent

        # 5. Calculate Survivors and Eliminations for this week
        nfl_schedule_df.loc[current_week_mask, 'Expected Home Team Survivors'] = \
            nfl_schedule_df.loc[current_week_mask, 'Home Pick %'] * \
            nfl_schedule_df.loc[current_week_mask, 'Home Team Fair Odds'] * S_w
            
        nfl_schedule_df.loc[current_week_mask, 'Expected Away Team Survivors'] = \
            nfl_schedule_df.loc[current_week_mask, 'Away Pick %'] * \
            nfl_schedule_df.loc[current_week_mask, 'Away Team Fair Odds'] * S_w
            
        nfl_schedule_df.loc[current_week_mask, 'Expected Home Team Eliminations'] = \
            nfl_schedule_df.loc[current_week_mask, 'Home Pick %'] * \
            (1.0 - nfl_schedule_df.loc[current_week_mask, 'Home Team Fair Odds']) * S_w
            
        nfl_schedule_df.loc[current_week_mask, 'Expected Away Team Eliminations'] = \
            nfl_schedule_df.loc[current_week_mask, 'Away Pick %'] * \
            (1.0 - nfl_schedule_df.loc[current_week_mask, 'Away Team Fair Odds']) * S_w
            
        # Calculate Total Survivors from this week
        week_df_rows = nfl_schedule_df[current_week_mask]
        total_survivors_this_week = week_df_rows['Expected Home Team Survivors'].sum() + week_df_rows['Expected Away Team Survivors'].sum()
        
        st_write(f"Total Entries Surviving Week {current_week}: {total_survivors_this_week:,.0f}")
        
        # --- E. UPDATE U_prev_week FOR *NEXT* WEEK'S ITERATION ---
        
        overall_survival_rate_this_week = 0.0
        if S_w > 0:
            overall_survival_rate_this_week = total_survivors_this_week / S_w

        U_next_week: Dict[str, float] = {}
        survivors_who_picked_team: Dict[str, float] = {}
        
        # Calculate survivors based on the team they picked (val1)
        for _, row in week_df_rows.iterrows():
            survivors_who_picked_team[row['Home Team']] = survivors_who_picked_team.get(row['Home Team'], 0.0) + row['Expected Home Team Survivors']
            survivors_who_picked_team[row['Away Team']] = survivors_who_picked_team.get(row['Away Team'], 0.0) + row['Expected Away Team Survivors']

        for team in all_teams:
            # val1: Survivors who picked this team in *this* week (and are therefore now unavailable)
            val1 = survivors_who_picked_team.get(team, 0.0)
            
            # val2: Survivors who had *already* used this team (U_prev_week) AND survived *this* week's overall rate.
            num_already_used_team = U_prev_week.get(team, 0.0)
            val2 = num_already_used_team * overall_survival_rate_this_week
            
            # The total used count for the next week
            U_next_week[team] = val1 + val2
            
            # Clamp values
            U_next_week[team] = max(0.0, min(U_next_week[team], total_survivors_this_week))

        # *** FEEDBACK LOOP ***
        # The "used" dictionary for the next loop is the one we just calculated
        U_prev_week = U_next_week.copy()
		
        # Set the next week's starting pool size based on this week's survivors
        next_week = current_week + 1
        nfl_schedule_df.loc[nfl_schedule_df['Week_Num'] == next_week, 'Total Remaining Entries at Start of Week'] = total_survivors_this_week

        
        st_write(f"Projected Pool Size for Week {next_week}: {total_survivors_this_week:,.0f}")
        
    # Create the boolean mask once, as it's used twice
        multiplier_mask = (selected_contest == 'Splash Sports') & \
                      (nfl_schedule_df['Week_Num'].isin(week_requiring_two_selections)) & \
    	              (subcontest != "Week 9 Bloody Survivor ($100 Entry)")
        multiplier_mask_3 = (selected_contest == 'Splash Sports') & \
                      (nfl_schedule_df['Week_Num'].isin(week_requiring_three_selections)) & \
    	              (subcontest == "Week 9 Bloody Survivor ($100 Entry)")
    	
        nfl_schedule_df['Home Expected Survival Rate'] = nfl_schedule_df['Home Team Fair Odds'] * nfl_schedule_df['Home Pick %']
        nfl_schedule_df.loc[multiplier_mask, 'Home Expected Survival Rate'] *= 0.65
        nfl_schedule_df.loc[multiplier_mask_3, 'Home Expected Survival Rate'] *= 0.35
        nfl_schedule_df['Home Expected Elimination Percent'] = nfl_schedule_df['Home Pick %'] - nfl_schedule_df['Home Expected Survival Rate']
        nfl_schedule_df['Away Expected Survival Rate'] = nfl_schedule_df['Away Team Fair Odds'] * nfl_schedule_df['Away Pick %']
        nfl_schedule_df.loc[multiplier_mask, 'Away Expected Survival Rate'] *= 0.65
        nfl_schedule_df.loc[multiplier_mask_3, 'Away Expected Survival Rate'] *= 0.35
        nfl_schedule_df['Away Expected Elimination Percent'] = nfl_schedule_df['Away Pick %'] - nfl_schedule_df['Away Expected Survival Rate']
        nfl_schedule_df['Expected Eliminated Entry Percent From Game'] = nfl_schedule_df['Home Expected Elimination Percent'] + nfl_schedule_df['Away Expected Elimination Percent']
        nfl_schedule_df['Expected Away Team Picks'] = nfl_schedule_df['Away Pick %'] * nfl_schedule_df['Total Remaining Entries at Start of Week']
        nfl_schedule_df['Expected Home Team Picks'] = nfl_schedule_df['Home Pick %'] * nfl_schedule_df['Total Remaining Entries at Start of Week']
    
    

        def assign_pick_percentages_from_config(row, custom_picks_config):
            home_team = row['Home Team']
            away_team = row['Away Team']
            week = row['Week'] # Assumes week is like "Week 1", "Week 2"
            week_num_str = str(week).replace('Week ', '')
            week_key = f"week_{week_num_str}"
    
            home_pick_percent = row.get('Home Pick %') # Default
            away_pick_percent = row.get('Away Pick %') # Default
    
            if week_key in custom_picks_config:
                week_overrides = custom_picks_config[week_key]
                
                # Check for Home Team override [cite: 638]
                if home_team in week_overrides:
                    user_override_value = week_overrides[home_team]
                    if user_override_value >= 0:
                        home_pick_percent = user_override_value
                        
                # Check for Away Team override [cite: 639]
                if away_team in week_overrides:
                    user_override_value = week_overrides[away_team]
                    if user_override_value >= 0: # Keep -1 logic [cite: 639]
                        away_pick_percent = user_override_value
    
            return pd.Series({'Home Pick %': home_pick_percent, 'Away Pick %': away_pick_percent})
                                                      
        # Get the single source of truth...
        custom_pick_percentages = config.get('pick_percentages', {})
        
        nfl_schedule_df[['Home Pick %', 'Away Pick %']] = nfl_schedule_df.apply(
            assign_pick_percentages_from_config,  # <-- CORRECT NAME
            axis=1, 
            args=(custom_pick_percentages,) # Pass the config dict
        )

    ####################################################################################################
    
    def run_monte_carlo_simulation(nfl_schedule_df, num_trials=1000):
        """
        Runs a Monte Carlo simulation to estimate the distribution of survivor
        pool outcomes, based on the 'Expected Value' pick percentages.
        
        This function is defined *inside* get_predicted_pick_percentages
        to access its scope (starting_week, max_week).
        """
        
        st.write(f"Running Monte Carlo Simulation with {num_trials:,} trials...")
        
        # Get all unique team names from the schedule
        all_teams_series = pd.unique(nfl_schedule_df[['Home Team', 'Away Team']].values.ravel('K'))
        all_teams = [team for team in all_teams_series if pd.notna(team)]
    
        # --- Use variables from the outer function's scope ---
        start_w = starting_week
        end_w = max_week
        # -----------------------------------------------------
    
        # Get the absolute starting pool size from the main DF
        initial_pool_size = nfl_schedule_df.loc[
            nfl_schedule_df['Week_Num'] == start_w,
            'Total Remaining Entries at Start of Week'
        ].iloc[0]
        
        if pd.isna(initial_pool_size) or initial_pool_size <= 0:
            st.write(f"Warning: Initial pool size for MC Sim is {initial_pool_size}. Defaulting to 1.")
            initial_pool_size = 1
        
        initial_pool_size = int(initial_pool_size)
    
        # Collect results for aggregation
        monte_results = []
        
        # Add a progress bar for Streamlit
        progress_bar = st.progress(0)
    
        # --- Run all trials ---
        for trial in range(num_trials):
            
            # Initialize this trial's state
            remaining_entries_sim = initial_pool_size
            week_records = []
            
            # --- Simulate each week sequentially for this trial ---
            for week in range(start_w, int(end_w) + 1):
                
                # If all entries are eliminated, stop this trial
                if remaining_entries_sim <= 0:
                    break
                    
                week_df = nfl_schedule_df[nfl_schedule_df['Week_Num'] == week].copy()
                if week_df.empty:
                    continue
    
                # --- 1. Robustness: Clean & Normalize Probabilities ---
                
                # Fill NaNs
                week_df[['Home Pick %', 'Away Pick %']] = week_df[['Home Pick %', 'Away Pick %']].fillna(0.0)
                week_df[['Home Team Fair Odds', 'Away Team Fair Odds']] = week_df[['Home Team Fair Odds', 'Away Team Fair Odds']].fillna(0.5)
    
                # Re-normalize pick percentages for this week
                total_pick_prob = week_df['Home Pick %'].sum() + week_df['Away Pick %'].sum()
                if total_pick_prob <= 0:
                    st.write(f"Warning: Zero pick prob in Wk {week}. Skipping sim.")
                    continue # Cannot distribute picks
                    
                week_df['Home Pick %'] = week_df['Home Pick %'] / total_pick_prob
                week_df['Away Pick %'] = week_df['Away Pick %'] / total_pick_prob
    
                # --- 2. Simulation Step 1: Distribute Picks ---
                # Create a single list of all possible choices (teams) and their probabilities
                # This is critical for using the correct (multinomial) distribution
                
                # Get all teams playing and their associated pick probabilities
                choices = list(week_df['Home Team']) + list(week_df['Away Team'])
                probs = list(week_df['Home Pick %']) + list(week_df['Away Pick %'])
                
                # Ensure probabilities sum perfectly to 1 for the simulation
                probs = np.array(probs) / np.sum(probs)
                
                # Simulate the picks:
                # This one call distributes all 'remaining_entries_sim' among all choices
                picks_array = np.random.multinomial(n=remaining_entries_sim, pvals=probs)
                
                # Map results back to the dataframe
                picks_dict = dict(zip(choices, picks_array))
                week_df['Home Picks'] = week_df['Home Team'].map(picks_dict).fillna(0).astype(int)
                week_df['Away Picks'] = week_df['Away Team'].map(picks_dict).fillna(0).astype(int)
    
                # --- 3. Simulation Step 2: Simulate Game Outcomes ---
                # CRITICAL FIX: Simulate game outcomes so that only one team can win.
                
                # Simulate home team win probability
                week_df['Home Wins'] = np.random.binomial(1, week_df['Home Team Fair Odds'])
                
                # Away team wins if home team *doesn't* (ignoring ties)
                week_df['Away Wins'] = 1 - week_df['Home Wins']
                
                # --- 4. Calculate Survivors & Eliminations ---
                home_survivors = (week_df['Home Picks'] * week_df['Home Wins']).sum()
                away_survivors = (week_df['Away Picks'] * week_df['Away Wins']).sum()
                
                survivors_this_week = home_survivors + away_survivors
                total_eliminations = remaining_entries_sim - survivors_this_week
                
                # Store week-level results for this trial
                week_records.append({
                    'Week_Num': week,
                    'Trial': trial,
                    'Eliminations': total_eliminations,
                    'Survivors': survivors_this_week
                })
                
                # --- 5. Feedback Loop for Next Week ---
                remaining_entries_sim = survivors_this_week
                
            # Add this trial's full weekly results to the main list
            monte_results.extend(week_records)
    
            # Update progress bar (e.g., every 100 trials to be efficient)
            if (trial + 1) % 100 == 0:
                progress_bar.progress((trial + 1) / num_trials)
    
        # --- Aggregation ---
        progress_bar.progress(1.0) # Complete the bar
        st.write("Simulation trials complete. Aggregating results...")
        
        if not monte_results:
            st.write("Warning: Monte Carlo simulation produced no results.")
            return pd.DataFrame() # Return empty frame
    
        monte_df = pd.DataFrame(monte_results)
        
        # Group by week and get summary statistics
        summary = monte_df.groupby('Week_Num').agg({
            'Eliminations': ['mean', 'std', 'median'],
            'Survivors': ['mean', 'std', 'median']
        }).reset_index()
        
        # Clean up the multi-index column names
        summary.columns = [
            'Week_Num', 
            'Avg Eliminations', 'Std Eliminations', 'Median Eliminations',
            'Avg Survivors', 'Std Survivors', 'Median Survivors'
        ]
        
        st_write("Monte Carlo simulation completed ✅")
        return summary

    ###################################################################################################

    # --- OPTIONAL: Run Monte Carlo after predictions ---
    monte_summary = run_monte_carlo_simulation(nfl_schedule_df, num_trials=1000)

    # Merge back into main dataframe for charting
    nfl_schedule_df = nfl_schedule_df.merge(
        monte_summary[['Week_Num', 'Avg Survivors', 'Avg Eliminations']],
        on='Week_Num',
        how='left'
    )
	# 1. Convert all 'object' columns to 'str' to handle mixed types
    for col in nfl_schedule_df.select_dtypes(include=['object']).columns:
        nfl_schedule_df[col] = nfl_schedule_df[col].astype(str).fillna('')

    # 2. Explicitly convert calculated columns to float
    float_cols = [
        'Home Team Expected Availability', 'Away Team Expected Availability',
        'Home Pick %', 'Away Pick %', 'Expected Home Team Survivors', 
        'Expected Away Team Survivors', 'Expected Home Team Eliminations', 
        'Expected Away Team Eliminations', 'Total Remaining Entries at Start of Week'
    ]
    
    for col in float_cols:
        if col in nfl_schedule_df.columns:
            # The errors='coerce' is a fallback, but simple .astype(float) is better
            # since we expect only numbers or NaNs at this point.
            nfl_schedule_df[col] = pd.to_numeric(nfl_schedule_df[col], errors='coerce') 

    if selected_contest == 'Circa':
        nfl_schedule_df.to_csv("Circa_Predicted_pick_percent.csv", index=False)
    elif selected_contest == 'Splash Sports':
        nfl_schedule_df.to_csv("Splash_Predicted_pick_percent.csv", index=False)
    else:
        nfl_schedule_df.to_csv("DK_Predicted_pick_percent.csv", index=False)
	
    return nfl_schedule_df


def calculate_ev(df, config: dict, use_cache=False):
    fav_qualifier = config.get('favored_qualifier', 'Live Sportsbook Odds (If Available)')
    use_live_sportsbook_odds = 1 if 'Live Sportsbook' in fav_qualifier else 0
    start_w = config['starting_week']
    end_w = config['ending_week']
    selected_c = config['selected_contest']
    def calculate_all_scenarios(week_df):
        num_games = len(week_df)
        teams = week_df['Home Team'].tolist() + week_df['Away Team'].tolist()
        num_teams = len(teams)

        all_outcomes_matrix = np.array(list(itertools.product(['Home Win', 'Away Win'], repeat=num_games)))
        num_scenarios = all_outcomes_matrix.shape[0]

        ev_df = pd.DataFrame(index=range(num_scenarios), columns=teams)
        scenario_weights = np.zeros(num_scenarios)

        # Vectorized calculations within the scenario loop
        for i in range(num_scenarios):
            outcome = all_outcomes_matrix[i]
            winning_teams = np.where(outcome == 'Home Win', week_df['Home Team'].values, week_df['Away Team'].values)  # Use .values for numpy array
            winning_team_indices = np.isin(teams, winning_teams)

            if use_live_sportsbook_odds == 1:
                winning_probs = np.where(outcome == 'Home Win', week_df['Home Team Fair Odds'].values, week_df['Away Team Fair Odds'].values) # Use .values for numpy array
            else:
                winning_probs = np.where(outcome == 'Home Win', week_df['Internal Home Team Fair Odds'].values, week_df['Internal Away Team Fair Odds'].values) # Use .values for numpy array

            scenario_weights[i] = np.prod(winning_probs)

            pick_percentages = np.where(outcome == 'Home Win', week_df['Home Pick %'].values, week_df['Away Pick %'].values) # Use .values for numpy array
            surviving_entries = np.sum(pick_percentages)

            ev_values = np.zeros(num_teams)
            ev_values[winning_team_indices] = 1 / surviving_entries if surviving_entries > 0 else 0
            ev_df.iloc[i] = ev_values

        weighted_avg_ev = (ev_df * scenario_weights[:, np.newaxis]).sum(axis=0) / scenario_weights.sum()
        return weighted_avg_ev, all_outcomes_matrix, scenario_weights  # Return weighted_avg_ev directly
    def get_pick_percentage(week_df, team_name):
        # Check if the team is a home team in any game this week
        if team_name in week_df['Home Team'].values:
            return week_df[week_df['Home Team'] == team_name]['Home Pick %'].iloc[0]
        # Check if the team is an away team
        elif team_name in week_df['Away Team'].values:
            return week_df[week_df['Away Team'] == team_name]['Away Pick %'].iloc[0]
        # Return 0 if the team is not found (this shouldn't happen with correct data)
        return 0.0
    def calculate_all_scenarios_two_picks(week_df):
        num_games = len(week_df)
        teams = week_df['Home Team'].tolist() + week_df['Away Team'].tolist()
        
        all_outcomes_matrix = np.array(list(itertools.product(['Home Win', 'Away Win'], repeat=num_games)))
        num_scenarios = all_outcomes_matrix.shape[0]
        
        # ... (logic to generate team pairs) ...
        team_pairs = list(itertools.combinations(teams, 2))
        
        # DataFrame to hold EV for each team pair in each scenario
        pair_ev_df = pd.DataFrame(0.0, index=range(num_scenarios), columns=team_pairs)
        scenario_weights = np.zeros(num_scenarios)
        
        for i in range(num_scenarios):
            outcome = all_outcomes_matrix[i]
            winning_teams = np.where(outcome == 'Home Win', week_df['Home Team'].values, week_df['Away Team'].values)
            
            # Calculate scenario weight based on fair odds
            winning_probs = np.where(outcome == 'Home Win', week_df['Home Team Fair Odds'].values, week_df['Away Team Fair Odds'].values)
            scenario_weights[i] = np.prod(winning_probs)
            
            # Calculate EV for each pair
            for pair in team_pairs:
                team1, team2 = pair
                if team1 in winning_teams and team2 in winning_teams:
                    # Estimate combined pick percentage by multiplying
                    pick_perc1 = get_pick_percentage(week_df, team1)
                    pick_perc2 = get_pick_percentage(week_df, team2)
                    
                    surviving_entries = pick_perc1 * pick_perc2
                    
                    if surviving_entries > 0:
                        pair_ev_df.loc[i, pair] = 1 / surviving_entries
        
        # Now, calculate the weighted average EV for each pair
        weighted_avg_pair_ev = (pair_ev_df.mul(scenario_weights, axis=0)).sum(axis=0) / scenario_weights.sum()
        
        # Consolidate pair EVs into single-team EVs
        weighted_avg_ev = pd.Series(0.0, index=teams)
        for pair, ev in weighted_avg_pair_ev.items():
            team1, team2 = pair
            weighted_avg_ev[team1] += ev
            weighted_avg_ev[team2] += ev
        
        return weighted_avg_ev, all_outcomes_matrix, scenario_weights


    st.write("Current Week Progress")  # Streamlit progress bar
    progress_bar = st.progress(0)

    all_weeks_ev = {} #Store the EV values for each week

    for week in tqdm(range(start_w, end_w), desc="Processing Weeks", leave=False):
        week_df = df[df['Week_Num'] == week].copy() # Use the 'df' argument
        # Check if the current week requires two picks
#        if week in week_requiring_two_selections:
            # Call a new function to handle two-pick calculations
#            weighted_avg_ev, all_outcomes, scenario_weights = calculate_all_scenarios_two_picks(week_df)
#        else:
            # Use the existing function for single-pick weeks
        weighted_avg_ev, all_outcomes, scenario_weights = calculate_all_scenarios(week_df)

        #Store the EV values for the current week
        all_weeks_ev[week] = weighted_avg_ev

        #More efficient way to set the EV values for the current week
        for team in week_df['Home Team'].unique():
            df.loc[(df['Week_Num'] == week) & (df['Home Team'] == team), 'Home Team EV'] = weighted_avg_ev[team]
        for team in week_df['Away Team'].unique():
            df.loc[(df['Week_Num'] == week) & (df['Away Team'] == team), 'Away Team EV'] = weighted_avg_ev[team]

        progress_percent = int((week / end_w) * 100)
        progress_bar.progress(progress_percent)

    if selected_contest == 'Circa':
        df.to_csv("NFL Schedule with full ev_circa.csv", index=False)
    elif selected_contest == 'Splash Sports':
        df.to_csv("NFL Schedule with full ev_splash.csv", index=False)
    else:
        df.to_csv("NFL Schedule with full ev_dk.csv", index=False)
    return df



def reformat_df(df: pd.DataFrame, config: dict):
    """
    Reformats the combined dataframe with EV into a team-centric,
    one-row-per-team-per-game format.
    """
    # --- Fix 1: Get selected_contest from the config dict ---
    selected_contest = config['selected_contest']
    
    # --- Fix 2: Use the passed 'df' argument instead of global 'full_df_with_ev' ---
    # Make a copy to avoid SettingWithCopyWarning on the original df
    df = df.copy() 

    df['ID'] = df.index + 1
    
    # --- Away Team Dataframe ---
    away_reformatted_df = df[['Week', 'Week_Num', 'Date', 'Time', 'Away Team', 'Home Team','Internal Away Team Spread', 'Away Team Sportsbook Spread', 'Away Team Weekly Rest', 'Weekly Away Rest Advantage', 'Away Cumulative Rest Advantage', 'Away Team Current Week Cumulative Rest Advantage', 'Actual Stadium', 'Back to Back Away Games', 'Away Team Previous Opponent', 'Away Team Previous Location', 'Away Previous Game Actual Stadium TimeZone','Away Weekly Timezone Difference', 'Away Team Next Opponent', 'Away Team Next Location', 'Away Travel Advantage', 'Away Timezone Change', 'Away Team Preseason Rank','Away Team Adjusted Preseason Rank', 'Away Team Current Rank', 'Away Team Adjusted Current Rank', 'Thursday Night Game', 'Divisional Matchup?', 'Away Team 3 games in 10 days', 'Away Team 4 games in 17 days', 'Away Team Short Rest', 'Away Team Moneyline', 'Favorite', 'Underdog', 'Adjusted Away Points', 'Adjusted Home Points', 'Internal Away Team Moneyline', 'Away Team Implied Odds to Win', 'Internal Away Team Implied Odds to Win', 'Away Team Fair Odds', 'Internal Away Team Fair Odds', 'Away Team Star Rating', 'Away Team Thanksgiving Favorite', 'Away Team Christmas Favorite', 'Away Team Thanksgiving Underdog', 'Away Team Christmas Underdog', 'Away Team Expected Availability','Away Pick %', 'Away Team EV', 'Total Remaining Entries at Start of Week', 'Away Expected Survival Rate', 'Away Expected Elimination Percent', 'Expected Away Team Picks', 'Expected Away Team Eliminations', 'Expected Away Team Survivors', 'Same Winner?', 'Same Current and Adjusted Current Winner?', 'Same Adjusted Preseason Winner?', 'Adjusted Current Winner', 'Adjusted Current Difference', 'Home Team Preseason Rank', 'Home Team Adjusted Preseason Rank', 'Home Team Current Rank', 'Home Team Adjusted Current Rank', 'ID', 'No Live Odds Available - Internal Rankings Used for Moneyline Calculation']]
    
    new_column_names_away = {
        'Week': 'Week',
		'Week_Num': 'Week_Num',
        'Date': 'Date',
        'Time': 'Time',
        'Away Team': 'Team',
        'Home Team': 'Opponent',
        'Away Team Weekly Rest': 'Weekly Rest',
        'Weekly Away Rest Advantage': 'Weekly Rest Advantage',
        'Away Cumulative Rest Advantage': 'Season-Long Rest Advantage',
        'Away Team Current Week Cumulative Rest Advantage': 'Season-Long Rest Advantage Including This Week',
        'Actual Stadium': 'Location',
        'Back to Back Away Games': 'Back to Back Away Games',
        'Away Team Previous Opponent': 'Previous Opponent',
        'Away Team Previous Location': 'Previous Game Location',
        'Away Previous Game Actual Stadium TimeZone': 'Previous Game Timezone',
        'Away Weekly Timezone Difference': 'Weekly Timezone Difference',
        'Away Team Next Opponent': 'Next Opponent',
        'Away Team Next Location': 'Next Game Location',
        'Away Travel Advantage': 'Travel Advantage',
        'Away Team Preseason Rank': 'Preseason Rank',
        'Away Team Adjusted Preseason Rank': 'Adjusted Preseason Rank',
        'Away Team Current Rank': 'Current Rank',
        'Away Team Adjusted Current Rank': 'Adjusted Current Rank',
        'Thursday Night Game': 'Thursday Night Game',
        'Divisional Matchup?': 'Divisional Matchup?',
        'Away Team 3 games in 10 days': '3 Games in 10 Days',
        'Away Team 4 games in 17 days': '4 Games in 17 Days',
        'Away Team Short Rest': 'Away Team Short Rest',
        'Away Team Moneyline': 'Moneyline Based on Sportsbook Odds',
		'Away Team Sportsbook Spread': 'Spread Based on Sportsbook Odds',
        'Internal Away Team Spread': 'Spread Based on Internal Rankings',
        'Favorite': 'Favorite',
        'Underdog': 'Underdog',
        'Adjusted Away Points': 'Adjusted Away Points',
        'Adjusted Home Points': 'Adjusted Home Points',
		'Adjusted Spread': 'Adjusted Spread',
        'Internal Away Team Moneyline': 'Moneyline Based on Internal Rankings',
        'Away Team Implied Odds to Win': 'Implied Odds Based on Sportsbook Odds',
        'Internal Away Team Implied Odds to Win': 'Implied Odds Based on Internal Rankings',
        'Away Team Fair Odds': 'Fair Odds Based on Sportsbook Odds',
        'Internal Away Team Fair Odds': 'Fair Odds Based on Internal Rankings',
        'Away Team Star Rating': 'Future Value',
        'Away Team Thanksgiving Favorite': 'Thanksgiving Favorite',
        'Away Team Christmas Favorite': 'Christmas Favorite',
        'Away Team Thanksgiving Underdog': 'Thanksgiving Underdog',
        'Away Team Christmas Underdog': 'Christmas Underdog',
        'Away Team Expected Availability': 'Expected Availability',
        'Away Pick %': 'Expected Pick Percent',
        'Away Team EV': 'Expected EV',
        'Total Remaining Entries at Start of Week': 'Total Remaining Entries at Start of Week',
        'Away Expected Survival Rate': 'Expected Survival Rate',
        'Away Expected Elimination Percent': 'Expected Contest Elimination Percent',
        'Expected Away Team Picks': 'Expected Picks',
        'Expected Away Team Eliminations': 'Expected Eliminations',
        'Expected Away Team Survivors': 'Expected Survivors',
        'Same Winner?': 'Same Winner?',
        'Same Current and Adjusted Current Winner?': 'Same Current and Adjusted Current Winner?',
        'Same Adjusted Preseason Winner?': 'Same Adjusted Preseason Winner?',
        'Adjusted Current Winner': 'Adjusted Current Winner',
        'Adjusted Current Difference': 'Adjusted Current Difference',
        'Home Team Preseason Rank': 'Opp Preseason Rank',
        'Home Team Adjusted Preseason Rank': 'Opp Adjusted Preseason Rank',
        'Home Team Current Rank': 'Opp Current Rank',
        'Home Team Adjusted Current Rank': 'Opp Adjusted Current Rank',
		'No Live Odds Available - Internal Rankings Used for Moneyline Calculation': 'No Live Odds Available - Internal Rankings Used for Moneyline Calculation'
    }
    away_reformatted_df = away_reformatted_df.rename(columns=new_column_names_away)

    away_reformatted_df['Date'] = pd.to_datetime(away_reformatted_df['Date']).dt.strftime('%m-%d-%Y')
    
    # ... (Rounding) ...
    away_reformatted_df['Expected EV'] = round(away_reformatted_df['Expected EV'], 4)
    away_reformatted_df['Fair Odds Based on Sportsbook Odds'] = round(away_reformatted_df['Fair Odds Based on Sportsbook Odds'], 4)
    away_reformatted_df['Fair Odds Based on Internal Rankings'] = round(away_reformatted_df['Fair Odds Based on Internal Rankings'], 4)
    away_reformatted_df['Expected Pick Percent'] = round(away_reformatted_df['Expected Pick Percent'], 4)
    away_reformatted_df['Expected Availability'] = round(away_reformatted_df['Expected Availability'], 4)
    away_reformatted_df['Expected Survival Rate'] = round(away_reformatted_df['Expected Survival Rate'], 4)
    away_reformatted_df['Expected Contest Elimination Percent'] = round(away_reformatted_df['Expected Contest Elimination Percent'], 4)
    away_reformatted_df['Expected Picks'] = round(away_reformatted_df['Expected Picks'])
    away_reformatted_df['Expected Eliminations'] = round(away_reformatted_df['Expected Eliminations'])
    away_reformatted_df['Expected Survivors'] = round(away_reformatted_df['Expected Survivors'])

    
    away_reformatted_df['Preseason Difference'] = round(away_reformatted_df['Preseason Rank'] - away_reformatted_df['Opp Preseason Rank'], 1)
    away_reformatted_df['Adjusted Preseason Difference'] = round(away_reformatted_df['Adjusted Preseason Rank'] - away_reformatted_df['Opp Adjusted Preseason Rank'], 1)
    away_reformatted_df['Current Difference'] = round(away_reformatted_df['Current Rank'] - away_reformatted_df['Opp Current Rank'], 1)
    away_reformatted_df['Adjusted Current Difference'] = round(away_reformatted_df['Adjusted Current Rank'] - away_reformatted_df['Opp Adjusted Current Rank'], 1)

    away_reformatted_df['Same Winner?'] = away_reformatted_df.apply(lambda row: 'Yes' if (row['Preseason Difference'] > 0 and row['Adjusted Preseason Difference'] > 0 and row['Current Difference'] > 0 and row['Adjusted Current Difference'] > 0) else 'No', axis=1)
#    away_reformatted_df['Week'] = away_reformatted_df['Week'].str.extract(r'(\d+)').astype(int)

    away_reformatted_df['Same Adjusted Preseason Winner?'] = away_reformatted_df.apply(lambda row: 'Yes' if (row['Adjusted Preseason Difference'] > 0 and row['Adjusted Current Difference'] > 0) else 'No', axis=1)
    away_reformatted_df['Same Current and Adjusted Current Winner?'] = away_reformatted_df.apply(lambda row: 'Yes' if (row['Current Difference'] > 0 and row['Adjusted Current Difference'] > 0) else 'No', axis=1)
    away_reformatted_df['Team Is Away'] = 'True'
    away_reformatted_df['Same Internal Ranking + Sportsbook Winner'] = away_reformatted_df.apply(lambda row: 'Yes' if (row['Fair Odds Based on Sportsbook Odds'] > 0.5 and row['Fair Odds Based on Internal Rankings'] > 0.5) else 'No', axis=1)
    
    new_order_away = ['Week', 'Week_Num', 'Date', 'Time', 'Team', 'Opponent', 'Team Is Away', 'Location', 'Expected EV', 'Moneyline Based on Sportsbook Odds', 'Moneyline Based on Internal Rankings', 'Spread Based on Sportsbook Odds', 'Spread Based on Internal Rankings', 'Fair Odds Based on Sportsbook Odds', 'Fair Odds Based on Internal Rankings', 'Preseason Rank', 'Adjusted Preseason Rank', 'Current Rank', 'Adjusted Current Rank', 'Preseason Difference', 'Adjusted Preseason Difference', 'Current Difference', 'Adjusted Current Difference','Expected Pick Percent', 'Expected Availability', 'Future Value', 'Weekly Rest', 'Weekly Rest Advantage', 'Season-Long Rest Advantage', 'Season-Long Rest Advantage Including This Week', 'Travel Advantage', 'Weekly Timezone Difference', 'Previous Opponent', 'Previous Game Location', 'Previous Game Timezone', 'Next Opponent', 'Next Game Location', 'Back to Back Away Games', 'Thursday Night Game', 'Divisional Matchup?', '3 Games in 10 Days', '4 Games in 17 Days', 'Away Team Short Rest', 'Thanksgiving Favorite', 'Christmas Favorite', 'Thanksgiving Underdog', 'Christmas Underdog', 'Total Remaining Entries at Start of Week', 'Expected Picks', 'Expected Survival Rate', 'Expected Contest Elimination Percent', 'Expected Eliminations', 'Expected Survivors', 'Adjusted Current Winner', 'Favorite', 'Underdog', 'Opp Preseason Rank', 'Opp Adjusted Preseason Rank', 'Opp Current Rank', 'Opp Adjusted Current Rank', 'Same Internal Ranking + Sportsbook Winner', 'Same Current and Adjusted Current Winner?', 'Same Adjusted Preseason Winner?', 'ID', 'No Live Odds Available - Internal Rankings Used for Moneyline Calculation']
    away_reformatted_df = away_reformatted_df.reindex(columns=new_order_away) # Use reindex to handle missing columns gracefully

    # --- Home Team Dataframe ---
    home_reformatted_df = df[['Week','Week_Num', 'Date', 'Time', 'Home Team', 'Away Team', 'Home Team Weekly Rest','Home Team Sportsbook Spread', 'Internal Home Team Spread', 'Weekly Home Rest Advantage', 'Home Cumulative Rest Advantage', 'Home Team Current Week Cumulative Rest Advantage', 'Actual Stadium', 'Back to Back Away Games', 'Home Team Previous Opponent', 'Home Team Previous Location', 'Home Previous Game Actual Stadium TimeZone','Home Weekly Timezone Difference', 'Home Team Next Opponent', 'Home Team Next Location', 'Home Travel Advantage', 'Home Timezone Change', 'Home Team Preseason Rank', 'Home Team Adjusted Preseason Rank', 'Home Team Current Rank', 'Home Team Adjusted Current Rank', 'Thursday Night Game', 'Divisional Matchup?', 'Home Team 3 games in 10 days', 'Home Team 4 games in 17 days', 'Away Team Short Rest', 'Home Team Moneyline', 'Favorite', 'Underdog', 'Adjusted Home Points', 'Adjusted Away Points', 'Internal Home Team Moneyline', 'Home team Implied Odds to Win', 'Internal Home team Implied Odds to Win', 'Home Team Fair Odds', 'Internal Home Team Fair Odds', 'Home Team Star Rating', 'Home Team Thanksgiving Favorite', 'Home Team Christmas Favorite', 'Home Team Thanksgiving Underdog', 'Home Team Christmas Underdog', 'Home Team Expected Availability','Home Pick %', 'Home Team EV', 'Total Remaining Entries at Start of Week', 'Home Expected Survival Rate', 'Home Expected Elimination Percent', 'Expected Home Team Picks', 'Expected Home Team Eliminations', 'Expected Home Team Survivors', 'Same Winner?', 'Same Current and Adjusted Current Winner?', 'Same Adjusted Preseason Winner?', 'Adjusted Current Winner', 'Adjusted Current Difference', 'Away Team Preseason Rank', 'Away Team Adjusted Preseason Rank', 'Away Team Current Rank', 'Away Team Adjusted Current Rank', 'ID', 'No Live Odds Available - Internal Rankings Used for Moneyline Calculation']]

    new_column_names_home = {
        'Week': 'Week',
		'Week_Num':'Week_Num',
        'Date': 'Date',
        'Time': 'Time',
        'Home Team': 'Team',
        'Away Team': 'Opponent',
        'Home Team Weekly Rest': 'Weekly Rest',
        'Weekly Home Rest Advantage': 'Weekly Rest Advantage',
        'Home Cumulative Rest Advantage': 'Season-Long Rest Advantage',
        'Home Team Current Week Cumulative Rest Advantage': 'Season-Long Rest Advantage Including This Week',
        'Actual Stadium': 'Location',
        'Back to Back Away Games': 'Back to Back Away Games',
        'Home Team Previous Opponent': 'Previous Opponent',
        'Home Team Previous Location': 'Previous Game Location',
        'Home Previous Game Actual Stadium TimeZone': 'Previous Game Timezone',
        'Home Weekly Timezone Difference': 'Weekly Timezone Difference',
        'Home Team Next Opponent': 'Next Opponent',
        'Home Team Next Location': 'Next Game Location',
        'Home Travel Advantage': 'Travel Advantage',
        'Home Team Preseason Rank': 'Preseason Rank',
        'Home Team Adjusted Preseason Rank': 'Adjusted Preseason Rank',
        'Home Team Current Rank': 'Current Rank',
        'Home Team Adjusted Current Rank': 'Adjusted Current Rank',
        'Thursday Night Game': 'Thursday Night Game',
        'Divisional Matchup?': 'Divisional Matchup?',
        'Home Team 3 games in 10 days': '3 Games in 10 Days',
        'Home Team 4 games in 17 days': '4 Games in 17 Days',
        'Away Team Short Rest': 'Away Team Short Rest',
        'Home Team Moneyline': 'Moneyline Based on Sportsbook Odds',
        'Favorite': 'Favorite',
        'Underdog': 'Underdog',
		'Home Team Sportsbook Spread': 'Spread Based on Sportsbook Odds',
        'Internal Home Team Spread': 'Spread Based on Internal Rankings',
        'Adjusted Home Points': 'Adjusted Home Points',
        'Adjusted Away Points': 'Adjusted Away Points',
		'Adjusted Spread': 'Adjusted Spread',
        'Internal Home Team Moneyline': 'Moneyline Based on Internal Rankings',
        'Home team Implied Odds to Win': 'Implied Odds Based on Sportsbook Odds',
        'Internal Home team Implied Odds to Win': 'Implied Odds Based on Internal Rankings',
        'Home Team Fair Odds': 'Fair Odds Based on Sportsbook Odds',
        'Internal Home Team Fair Odds': 'Fair Odds Based on Internal Rankings',
        'Home Team Star Rating': 'Future Value',
        'Home Team Thanksgiving Favorite': 'Thanksgiving Favorite',
        'Home Team Christmas Favorite': 'Christmas Favorite',
        'Home Team Thanksgiving Underdog': 'Thanksgiving Underdog',
        'Home Team Christmas Underdog': 'Christmas Underdog',
        'Home Team Expected Availability': 'Expected Availability',
        'Home Pick %': 'Expected Pick Percent',
        'Home Team EV': 'Expected EV',
        'Total Remaining Entries at Start of Week': 'Total Remaining Entries at Start of Week',
        'Home Expected Survival Rate': 'Expected Survival Rate',
        'Home Expected Elimination Percent': 'Expected Contest Elimination Percent',
        'Expected Home Team Picks': 'Expected Picks',
        'Expected Home Team Eliminations': 'Expected Eliminations',
        'Expected Home Team Survivors': 'Expected Survivors',
        'Same Winner?': 'Same Winner?',
        'Same Current and Adjusted Current Winner?': 'Same Current and Adjusted Current Winner?',
        'Same Adjusted Preseason Winner?': 'Same Adjusted Preseason Winner?',
        'Adjusted Current Winner': 'Adjusted Current Winner',
        'Adjusted Current Difference': 'Adjusted Current Difference',
        'Away Team Preseason Rank': 'Opp Preseason Rank',
        'Away Team Adjusted Preseason Rank': 'Opp Adjusted Preseason Rank',
        'Away Team Current Rank': 'Opp Current Rank',
        'Away Team Adjusted Current Rank': 'Opp Adjusted Current Rank',
		'No Live Odds Available - Internal Rankings Used for Moneyline Calculation': 'No Live Odds Available - Internal Rankings Used for Moneyline Calculation'
    }
    home_reformatted_df = home_reformatted_df.rename(columns=new_column_names_home)

    home_reformatted_df['Date'] = pd.to_datetime(home_reformatted_df['Date']).dt.strftime('%m-%d-%Y')
    home_reformatted_df['Away Team Short Rest'] = 'No'
    home_reformatted_df['Back to Back Away Games'] = 'False'

    # ... (Rounding) ...
    home_reformatted_df['Expected EV'] = round(home_reformatted_df['Expected EV'], 4)
    home_reformatted_df['Fair Odds Based on Sportsbook Odds'] = round(home_reformatted_df['Fair Odds Based on Sportsbook Odds'], 2)
    home_reformatted_df['Fair Odds Based on Internal Rankings'] = round(home_reformatted_df['Fair Odds Based on Internal Rankings'], 2)
    home_reformatted_df['Expected Pick Percent'] = round(home_reformatted_df['Expected Pick Percent'], 2)
    home_reformatted_df['Expected Availability'] = round(home_reformatted_df['Expected Availability'], 2)
    home_reformatted_df['Expected Survival Rate'] = round(home_reformatted_df['Expected Survival Rate'], 2)
    home_reformatted_df['Expected Contest Elimination Percent'] = round(home_reformatted_df['Expected Contest Elimination Percent'], 2)
    home_reformatted_df['Expected Picks'] = round(home_reformatted_df['Expected Picks'])
    home_reformatted_df['Expected Eliminations'] = round(home_reformatted_df['Expected Eliminations'])
    home_reformatted_df['Expected Survivors'] = round(home_reformatted_df['Expected Survivors'])

    
    home_reformatted_df['Preseason Difference'] = round(home_reformatted_df['Preseason Rank'] - home_reformatted_df['Opp Preseason Rank'], 1)
    home_reformatted_df['Adjusted Preseason Difference'] = round(home_reformatted_df['Adjusted Preseason Rank'] - home_reformatted_df['Opp Adjusted Preseason Rank'], 1)
    home_reformatted_df['Current Difference'] = round(home_reformatted_df['Current Rank'] - home_reformatted_df['Opp Current Rank'], 1)
    home_reformatted_df['Adjusted Current Difference'] = round(home_reformatted_df['Adjusted Current Rank'] - home_reformatted_df['Opp Adjusted Current Rank'], 1)

    home_reformatted_df['Same Winner?'] = home_reformatted_df.apply(lambda row: 'Yes' if (row['Preseason Difference'] > 0 and row['Adjusted Preseason Difference'] > 0 and row['Current Difference'] > 0 and row['Adjusted Current Difference'] > 0) else 'No', axis=1)
#    home_reformatted_df['Week'] = home_reformatted_df['Week'].str.extract(r'(\d+)').astype(int)

    home_reformatted_df['Same Adjusted Preseason Winner?'] = home_reformatted_df.apply(lambda row: 'Yes' if (row['Adjusted Preseason Difference'] > 0 and row['Adjusted Current Difference'] > 0) else 'No', axis=1)
    home_reformatted_df['Same Current and Adjusted Current Winner?'] = home_reformatted_df.apply(lambda row: 'Yes' if (row['Current Difference'] > 0 and row['Adjusted Current Difference'] > 0) else 'No', axis=1)
    home_reformatted_df['Team Is Away'] = 'False'
    home_reformatted_df['Same Internal Ranking + Sportsbook Winner'] = home_reformatted_df.apply(lambda row: 'Yes' if (row['Fair Odds Based on Sportsbook Odds'] > 0.5 and row['Fair Odds Based on Internal Rankings'] > 0.5) else 'No', axis=1)
    
    
    new_order_home = ['Week', 'Week_Num', 'Date', 'Time', 'Team', 'Opponent', 'Team Is Away', 'Location', 'Expected EV', 'Moneyline Based on Sportsbook Odds', 'Moneyline Based on Internal Rankings', 'Spread Based on Sportsbook Odds', 'Spread Based on Internal Rankings', 'Fair Odds Based on Sportsbook Odds', 'Fair Odds Based on Internal Rankings', 'Preseason Rank', 'Adjusted Preseason Rank', 'Current Rank', 'Adjusted Current Rank', 'Preseason Difference', 'Adjusted Preseason Difference', 'Current Difference', 'Adjusted Current Difference','Expected Pick Percent', 'Expected Availability', 'Future Value', 'Weekly Rest', 'Weekly Rest Advantage', 'Season-Long Rest Advantage', 'Season-Long Rest Advantage Including This Week', 'Travel Advantage', 'Weekly Timezone Difference', 'Previous Opponent', 'Previous Game Location', 'Previous Game Timezone', 'Next Opponent', 'Next Game Location', 'Back to Back Away Games', 'Thursday Night Game', 'Divisional Matchup?', '3 Games in 10 Days', '4 Games in 17 Days', 'Away Team Short Rest', 'Thanksgiving Favorite', 'Christmas Favorite', 'Thanksgiving Underdog', 'Christmas Underdog','Total Remaining Entries at Start of Week', 'Expected Picks', 'Expected Survival Rate', 'Expected Contest Elimination Percent', 'Expected Eliminations', 'Expected Survivors', 'Adjusted Current Winner', 'Favorite', 'Underdog', 'Opp Preseason Rank', 'Opp Adjusted Preseason Rank', 'Opp Current Rank', 'Opp Adjusted Current Rank', 'Same Internal Ranking + Sportsbook Winner', 'Same Current and Adjusted Current Winner?', 'Same Adjusted Preseason Winner?', 'ID', 'No Live Odds Available - Internal Rankings Used for Moneyline Calculation']
    home_reformatted_df = home_reformatted_df.reindex(columns=new_order_home) # Use reindex to handle missing columns gracefully

    # --- Combine and Sort ---
    combined_df = pd.concat([home_reformatted_df, away_reformatted_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=['ID', 'Team Is Away'], ascending=True)

    # --- Final Step: Use the 'selected_contest' variable ---
#    if selected_contest != 'Circa':
#        combined_df = combined_df.drop(columns=['Thanksgiving Favorite', 'Christmas Favorite', 'Thanksgiving Underdog', 'Christmas Underdog'])
    
    return combined_df


def get_survivor_picks_based_on_ev(df, config: dict, num_solutions: int):
    # --- ADD THIS BLOCK ---
    # Get settings from config
    starting_week = config['starting_week']
    ending_week = config['ending_week']
    picked_teams = config['prohibited_picks']
    required_weeks_dict = config['required_weeks']
    prohibited_weeks_dict = config['prohibited_weeks']
    selected_contest = config['selected_contest']
    subcontest = config['subcontest']
    week_requiring_two_selections = config.get('weeks_two_picks', [])
    week_requiring_three_selections = config.get('weeks_three_picks', [])
    favored_qualifier = config.get('favored_qualifier', 'Live Sportsbook Odds (If Available)')
    
    # Get all constraints
    pick_must_be_favored = config.get('must_be_favored', False)
    avoid_away_teams_in_close_matchups = config.get('avoid_away_close', False)
    min_away_spread = config.get('min_away_spread', 3.0)
    avoid_close_divisional_matchups = config.get('avoid_close_divisional', False)
    min_div_spread = config.get('min_div_spread', 3.0)
    avoid_away_divisional_matchups = config.get('avoid_away_divisional', False)
    avoid_away_teams_on_short_rest = config.get('avoid_away_short_rest', False)
    avoid_4_games_in_17_days = config.get('avoid_4_in_17', False)
    avoid_3_games_in_10_days = config.get('avoid_3_in_10', False)
    avoid_international_game = config.get('avoid_international', False)
    avoid_thursday_night = config.get('avoid_thursday', False)
    avoid_away_thursday_night = config.get('avoid_away_thursday', False)
    avoid_back_to_back_away = config.get('avoid_b2b_away', False)
    avoid_teams_with_weekly_rest_disadvantage = config.get('avoid_weekly_rest_disadvantage', False)
    avoid_cumulative_rest_disadvantage = config.get('avoid_cumulative_rest', False)
    avoid_away_teams_with_travel_disadvantage = config.get('avoid_travel_disadvantage', False)
    # --- END BLOCK ---
    for iteration in range(num_solutions):

        #Number of weeks that have already been played
        #weeks_completed = starting_week -1

        # Teams already picked - Team name in quotes and separated by commas

        # Filter out weeks that have already been played and reset index

        df = df[(df['Week_Num'] >= starting_week) & (df['Week_Num'] < ending_week)].reset_index(drop=True)
		# Rename columns from reformatted_df to what the solver logic expects
        df.rename(columns={
            "Team": "Hypothetical Current Winner",
            "Opponent": "Hypothetical Current Loser",
            "Expected EV": "Hypothetical Current Winner EV",
            "Adjusted Current Rank": "Hypothetical Current Winner Adjusted Current Rank",
            "Opp Adjusted Current Rank": "Hypothetical Current Loser Adjusted Current Rank",
            "Spread Based on Sportsbook Odds": "Hypothetical Current Winner Sportsbook Spread",
            "Spread Based on Internal Rankings": "Internal Hypothetical Current Winner Spread",
			"Adjusted Current Difference": "Adjusted Current Difference"
        }, inplace=True)
        # Filter out already picked teams
        df = df[~df['Hypothetical Current Winner'].isin(picked_teams)].reset_index(drop=True)
        #print(df)
        # Create the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Create binary variables to represent the picks, and store them in a dictionary for easy lookup
        picks = {}
        for i in range(len(df)):
            picks[i] = solver.IntVar(0, 1, 'pick_%i' % i)
        
        # Add the constraints
        for i in range(len(df)):
            # Can only pick an away team if 'Adjusted Current Difference' > 10
            if pick_must_be_favored:
                if favored_qualifier == 'Internal Rankings':
                    if df.loc[i, 'Hypothetical Current Winner'] != df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                else:
                    if df.loc[i, 'Hypothetical Current Winner'] != df.loc[i, 'Favorite']:
                        solver.Add(picks[i] == 0)    		
            if avoid_away_teams_in_close_matchups == 1:
                if favored_qualifier == 'Internal Rankings':
                    # Check if team is away AND the absolute spread (based on internal ranks) is <= min
                    if df.loc[i, 'Team Is Away'] == 'True' and df.loc[i, 'Adjusted Current Difference'] <= min_away_spread:
                        solver.Add(picks[i] == 0)
                else: 
                    if df.loc[i, 'Team Is Away'] == 'True' and df.loc[i, 'Hypothetical Current Winner Sportsbook Spread'] >= min_away_spread: # Uses sportsbook spread
                        solver.Add(picks[i] == 0)
            #if df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Divisional Matchup?'] == 'Divisional':
                #solver.Add(picks[i] == 0)
            if avoid_back_to_back_away == 1:
                if df.loc[i, 'Team Is Away'] == 'True' and df.loc[i, 'Back to Back Away Games'] == 'True':
                    solver.Add(picks[i] == 0)

            # If 'Divisional Matchup?' is "Divisional", can only pick if 'Adjusted Current Difference' > 10
            if avoid_close_divisional_matchups == 1:
                if favored_qualifier == 'Internal Rankings':
                    if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Adjusted Current Difference'] <= min_away_spread:
                        solver.Add(picks[i] == 0)
                else: 
                    if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Hypothetical Current Winner Sportsbook Spread'] >= min_away_spread: # Uses sportsbook spread
                        solver.Add(picks[i] == 0)
            if avoid_away_divisional_matchups == 1:
                if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Team Is Away'] == 'True':
                    solver.Add(picks[i] == 0)
            # Constraints for short rest and 4 games in 17 days (only if team is the Adjusted Current Winner)
            if avoid_away_teams_on_short_rest == 1:
                if df.loc[i, 'Away Team Short Rest'] == 'Yes':
                    solver.Add(picks[i] == 0)
            if avoid_4_games_in_17_days == 1:
                if df.loc[i, '4 Games in 17 Days'] == 'Yes':
                    solver.Add(picks[i] == 0)
            if avoid_3_games_in_10_days == 1:
                if df.loc[i, '3 Games in 10 Days'] == 'Yes':
                    solver.Add(picks[i] == 0)
            if avoid_international_game == 1:    
                if df.loc[i, 'Location'] == 'London, UK':
                    solver.Add(picks[i] == 0)
            if avoid_thursday_night == 1:
                if df.loc[i, 'Thursday Night Game'] == 'True':
                    solver.Add(picks[i] == 0)
            if avoid_away_thursday_night == 1:
                if df.loc[i, 'Thursday Night Game'] == 'True' and df.loc[i, 'Team Is Away'] == 'True':
                    solver.Add(picks[i] == 0)
            if avoid_teams_with_weekly_rest_disadvantage == 1:
                if df.loc[i, 'Weekly Rest Advantage'] < 0:
                    solver.Add(picks[i] == 0)
            if avoid_cumulative_rest_disadvantage == 1:
                is_away = df.loc[i, 'Team Is Away'] == 'True'
                rest_adv = df.loc[i, 'Season-Long Rest Advantage Including This Week']
                if is_away and rest_adv < -10:
                    solver.Add(picks[i] == 0)
                elif (not is_away) and rest_adv < -5: # Team is Home
                    solver.Add(picks[i] == 0)    
            if avoid_away_teams_with_travel_disadvantage == 1:
                if df.loc[i, 'Team Is Away'] == 'True' and df.loc[i, 'Travel Advantage'] < -850:
                    solver.Add(picks[i] == 0)


            
            prohibited_weeks_dict = config['prohibited_weeks']
            team_name = df.loc[i, 'Hypothetical Current Winner']
            if team_name in prohibited_weeks_dict:
                if df.loc[i, 'Week_Num'] in prohibited_weeks_dict[team_name]:
                    solver.Add(picks[i] == 0)


            if df.loc[i, 'Hypothetical Current Winner'] in picked_teams:
                solver.Add(picks[i] == 0)


        required_weeks_dict = config['required_weeks']
        for team, req_week in required_weeks_dict.items():
            if req_week > 0: # If a required week is set
                # Find all matching game indices for this team/week
                required_game_indices = df[
                    (df['Hypothetical Current Winner'] == team) & 
                    (df['Week_Num'] == req_week)
                ].index.tolist()
        
                if required_game_indices:
                    # Force the pick for the first match found (should only be one)
                    solver.Add(picks[required_game_indices[0]] == 1)

        
        for week in df['Week_Num'].unique():
            # Filter picks for the current week
            weekly_picks = [picks[i] for i in range(len(df)) if df.loc[i, 'Week_Num'] == week]

            if selected_contest == "Splash Sports" and subcontest != "Week 9 Bloody Survivor ($100 Entry)" and week in week_requiring_two_selections:
                # For Splash Sports and later weeks, two teams must be selected
                solver.Add(solver.Sum(weekly_picks) == 2)
            elif selected_contest == "Splash Sports" and subcontest == "Week 9 Bloody Survivor ($100 Entry)" and week in week_requiring_three_selections:
                # For Splash Sports and later weeks, two teams must be selected
                solver.Add(solver.Sum(weekly_picks) == 3)
            else:
                # For other contests or earlier weeks, one team per week
                solver.Add(solver.Sum(weekly_picks) == 1)

        for team in df['Hypothetical Current Winner'].unique():
            # Can't pick a team more than once
            solver.Add(solver.Sum([picks[i] for i in range(len(df)) if df.loc[i, 'Hypothetical Current Winner'] == team]) <= 1)

        
        def create_simple_ev_dataframe(summarized_picks_df, favored_qualifier):
            """
            Creates a DataFrame with one row per week, summarizing picks and their metrics.
        
            Args:
                summarized_picks_df (pd.DataFrame): DataFrame containing 'Week', 'Pick', 
                                                    'Fair Odds', 'Pick %', and 'EV'.
                favored_qualifier (str): 'Internal Rankings' or 'Sportsbook'.
        
            Returns:
                pd.DataFrame: A new DataFrame with the calculated metrics per week.
            """
            # Create a mapping for the Fair Odds column based on the favored_qualifier
            if favored_qualifier == 'Internal Rankings':
                odds_col = '=Fair Odds Based on Internal Rankings'
            else:
                odds_col = 'Fair Odds Based on Sportsbook Odds'
        
            # Group the DataFrame by week and aggregate the desired metrics
            simple_ev_df = summarized_picks_df.groupby('Week').agg(
                Picks=('Pick', lambda x: x.tolist()),
                Survival_Rate=(odds_col, 'prod'),
                Pick_Percentage=('Expected Pick Percent', 'prod'),
                EV=('EV', 'prod')
            ).reset_index()
        
            return simple_ev_df

        # Dynamically create the forbidden solution list
        forbidden_solutions_1 = []
        if iteration > 0:
            for previous_iteration in range(iteration):
                if selected_contest == 'Circa':
                    previous_picks_df = pd.read_csv(f"circa_picks_ev_subset_{previous_iteration + 1}.csv")
                elif selected_contest == 'Splash Sports':
                    previous_picks_df = pd.read_csv(f"splash_picks_ev_subset_{previous_iteration + 1}.csv")
                else:
                    previous_picks_df = pd.read_csv(f"dk_picks_ev_subset_{previous_iteration + 1}.csv")

                # Group picks by week for the forbidden solution
                forbidden_solution_by_week = previous_picks_df.groupby('Week')['Pick'].apply(list).to_dict()
                forbidden_solutions_1.append(forbidden_solution_by_week) 
        # Add constraints for all forbidden solutions
        for forbidden_solution_dict in forbidden_solutions_1:
            # Create a list of the picked variables from the previous solution
            forbidden_pick_variables = []

            # Iterate through each week and its corresponding forbidden picks
            for week, picks_in_week in forbidden_solution_dict.items():
                # Find all rows in the current DataFrame for this specific week
                weekly_rows = df[df['Week_Num'] == week]

                # Check if any of these rows match a forbidden pick from that week
                for _, row in weekly_rows.iterrows():
                    if row['Hypothetical Current Winner'] in picks_in_week: # The 'Favorite' column is what you're using to identify the pick
                        # Get the index of this row
                        pick_index = row.name
                        forbidden_pick_variables.append(picks[pick_index])

            # The constraint now ensures that at least one of the forbidden picks is not selected
            solver.Add(solver.Sum([1 - v for v in forbidden_pick_variables]) >= 1)


        

        # Objective: maximize the sum of Adjusted Current Difference of each game picked
        solver.Maximize(solver.Sum([picks[i] * df.loc[i, 'Hypothetical Current Winner EV'] for i in range(len(df))]))



        # Solve the problem and print the solution
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            st.write('')
            st.write(f'**Solution Based on EV: {iteration + 1}**')

            st.write('Solution found!')
            st.write('Objective value =', solver.Objective().Value())


            # Initialize sums
            sum_preseason_difference = 0
            sum_adjusted_preseason_difference = 0
            sum_current_difference = 0
            sum_adjusted_current_difference = 0
            sum_ev = 0
            sum_sportsbook_spread = 0
            sum_internal_spread = 0

            # Initialize picks_df
            picks_df = pd.DataFrame(columns=df.columns)
            picks_rows_2 = []
            for i in range(len(df)):
                if picks[i].solution_value() > 0:
                    # Determine if it's a divisional game and if the picked team is the home team

                    week = df.loc[i, 'Week_Num']
                    date = df.loc[i, 'Date']
                    time = df.loc[i, 'Time']
                    location = df.loc[i, 'Location']
                    pick = df.loc[i,'Hypothetical Current Winner']
                    opponent = df.loc[i,'Hypothetical Current Loser']
                    win_odds = round(df.loc[i, 'Fair Odds Based on Sportsbook Odds'], 2)
                    pick_percent = round(df.loc[i, 'Expected Pick Percent'], 2)
                    expected_availability = round(df.loc[i, 'Expected Availability'], 2)
                    divisional_game = 'Divisional' if df.loc[i, 'Divisional Matchup?'] == 1 else ''
                    home_team = 'Home Team' if df.loc[i, 'Team Is Away'] == 'False' else 'Away Team'
                    weekly_rest = df.loc[i, 'Weekly Rest']
                    weekly_rest_advantage = df.loc[i, 'Weekly Rest Advantage']
                    cumulative_rest = df.loc[i, 'Season-Long Rest Advantage']
                    cumulative_rest_advantage = df.loc[i, 'Season-Long Rest Advantage Including This Week']
                    travel_advantage = df.loc[i, 'Travel Advantage']
                    back_to_back_away_games = 'True' if df.loc[i, 'Team Is Away'] == 'True' and df.loc[i, 'Back to Back Away Games'] == 'True' else 'False'
                    thursday_night_game = 'Thursday Night Game' if df.loc[i, "Thursday Night Game"] == 'True' else 'Sunday/Monday Game'
                    international_game = 'International Game' if df.loc[i, 'Location'] == 'London, UK' else 'Domestic Game'
                    previous_opponent = df.loc[i, 'Previous Opponent']
                    previous_game_location = df.loc[i, 'Previous Game Location']
                    next_opponent = df.loc[i, 'Next Opponent']
                    next_game_location = df.loc[i, 'Next Game Location']
					
                    internal_ranking_fair_odds = df.loc[i, 'Fair Odds Based on Internal Rankings']
                    future_value = df.loc[i, 'Future Value']
                    sportbook_moneyline = df.loc[i, 'Moneyline Based on Sportsbook Odds']
                    internal_moneyline = df.loc[i, 'Moneyline Based on Internal Rankings']
                    contest_selections = df.loc[i, 'Expected Picks']
                    survival_rate = df.loc[i, 'Expected Survival Rate']
                    elimination_percent = df.loc[i, 'Expected Contest Elimination Percent']
                    survivors = df.loc[i, 'Expected Survivors']
                    eliminations = df.loc[i, 'Expected Eliminations']
                    preseason_rank = df.loc[i, 'Preseason Rank']
                    adjusted_preseason_rank = df.loc[i, 'Adjusted Preseason Rank']
                    current_rank = df.loc[i, 'Current Rank']
                    adjusted_current_rank = df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank']
                    away_team_short_rest = df.loc[i, 'Away Team Short Rest']
                    three_games_in_10_days = df.loc[i, '3 Games in 10 Days']
                    four_games_in_17_days = df.loc[i, '4 Games in 17 Days']
                    thanksgiving_favorite = df.loc[i, 'Thanksgiving Favorite']
                    christmas_favorite = df.loc[i, 'Christmas Favorite']
                    thanksgiving_underdog = df.loc[i, 'Thanksgiving Underdog']
                    christmas_underdog = df.loc[i, 'Christmas Underdog']
                    live_odds_unavailable = df.loc[i, 'No Live Odds Available - Internal Rankings Used for Moneyline Calculation']
                    live_odds_spread = df.loc[i, 'Hypothetical Current Winner Sportsbook Spread']
                    internal_spread = df.loc[i, 'Internal Hypothetical Current Winner Spread']
                    

                    # Get differences
                    preseason_difference = df.loc[i, 'Preseason Difference']
                    adjusted_preseason_difference = df.loc[i, 'Adjusted Preseason Difference']
                    current_difference = df.loc[i, 'Current Difference']
                    adjusted_current_difference = df.loc[i, 'Adjusted Current Difference']
                    # Calculate EV for this game
                    ev = df.loc[i, 'Hypothetical Current Winner EV']


                    print('Week %i: Pick %s %s %s (%i, %i, %i, %i, %.4f)' % (df.loc[i, 'Week_Num'], df.loc[i, 'Hypothetical Current Winner'], divisional_game, home_team,
                                                                       preseason_difference, adjusted_preseason_difference,
                                                                       current_difference, adjusted_current_difference, ev))
                    if selected_contest == 'Circa':
                        new_row_2 = {
                            'Week': week,
                            'Pick': pick,
                            'Opponent': opponent,
                            'Date': date,
                            'Time': time,
                            'Location': location,
                            'Home or Away': home_team,
                            'EV': ev,
                            'Fair Odds Based on Sportsbook Odds': win_odds,
                            'Fair Odds Based on Internal Rankings': internal_ranking_fair_odds,
                            'Expected Pick Percent': pick_percent,	
                            'Expected Availability': expected_availability,
                            'Future Value': future_value,
                            'Moneyline Based on Sportsbook Odds': sportbook_moneyline,
                            'Moneyline Based on Internal Rankings': internal_moneyline,
                            'Spread Based on Sportsbook Odds': live_odds_spread,
                            'Spread Based on Internal Rankings': internal_spread,
                            'No Live Odds Available - Internal Rankings Used for Moneyline Calculation': live_odds_unavailable,
                            'Expected Contest Selections': contest_selections,
                            'Expected Survival Rate': survival_rate,
                            'Expected Elimination Rate': elimination_percent,
                            'Expected Survivors': survivors,
                            'Expected Eliminations': eliminations,
                            'Preseason Rank': preseason_rank,
                            'Adjusted Preseason Rank': adjusted_preseason_rank,
                            'Current Rank': current_rank,
                            'Adjusted Current Rank': adjusted_current_rank,
                            'Preseason Difference': preseason_difference,
                            'Adjusted Preseason Difference': adjusted_preseason_difference,
                            'Current Difference': current_difference,
                            'Adjusted Current Difference': adjusted_current_difference,
                            'Thursday Night Game': thursday_night_game,
                            'International Game': international_game,
                            'Divisional Game': divisional_game,
                            'Weekly Rest': weekly_rest,
                            'Weekly Rest Advantage': weekly_rest_advantage,
                            'Season Long Rest Advantage': cumulative_rest,
                            'Season Long Rest Including This Week': cumulative_rest_advantage,
                            'Travel Advantage': travel_advantage,
                            'Back to Back Away Games': back_to_back_away_games,
                            'Away Team on Short Rest': away_team_short_rest,
                            'Three Games in 10 Days': three_games_in_10_days,
                            'Four Games in 17 Days': four_games_in_17_days,
                            'Thanksgiving Favorite': thanksgiving_favorite,
                            'Christmas Favorite': christmas_favorite,
                            'Thanksgiving Underdog': thanksgiving_underdog,
                            'Christmas Underdog': christmas_underdog,
                            'Previous Opponent': previous_opponent,
                            'Previous Game Location': previous_game_location,
                            'Next Opponent': next_opponent,
                            'Next Game Location': next_game_location
                        }
                    else:
                        new_row_2 = {
                            'Week': week,
                            'Pick': pick,
                            'Opponent': opponent,
                            'Date': date,
                            'Time': time,
                            'Location': location,
                            'Home or Away': home_team,
                            'EV': ev,
                            'Fair Odds Based on Sportsbook Odds': win_odds,
                            'Fair Odds Based on Internal Rankings': internal_ranking_fair_odds,
                            'Expected Pick Percent': pick_percent,	
                            'Expected Availability': expected_availability,
                            'Future Value': future_value,
                            'Moneyline Based on Sportsbook Odds': sportbook_moneyline,
                            'Moneyline Based on Internal Rankings': internal_moneyline,
                            'Spread Based on Sportsbook Odds': live_odds_spread,
                            'Spread Based on Internal Rankings': internal_spread,
                            'No Live Odds Available - Internal Rankings Used for Moneyline Calculation': live_odds_unavailable,
                            'Expected Contest Selections': contest_selections,
                            'Expected Survival Rate': survival_rate,
                            'Expected Elimination Rate': elimination_percent,
                            'Expected Survivors': survivors,
                            'Expected Eliminations': eliminations,
                            'Preseason Rank': preseason_rank,
                            'Adjusted Preseason Rank': adjusted_preseason_rank,
                            'Current Rank': current_rank,
                            'Adjusted Current Rank': adjusted_current_rank,
                            'Preseason Difference': preseason_difference,
                            'Adjusted Preseason Difference': adjusted_preseason_difference,
                            'Current Difference': current_difference,
                            'Adjusted Current Difference': adjusted_current_difference,
                            'Thursday Night Game': thursday_night_game,
                            'International Game': international_game,
                            'Divisional Game': divisional_game,
                            'Weekly Rest': weekly_rest,
                            'Weekly Rest Advantage': weekly_rest_advantage,
                            'Season Long Rest Advantage': cumulative_rest,
                            'Season Long Rest Including This Week': cumulative_rest_advantage,
                            'Travel Advantage': travel_advantage,
                            'Back to Back Away Games': back_to_back_away_games,
                            'Away Team on Short Rest': away_team_short_rest,
                            'Three Games in 10 Days': three_games_in_10_days,
                            'Four Games in 17 Days': four_games_in_17_days,
                            'Thanksgiving Favorite': thanksgiving_favorite,
                            'Christmas Favorite': christmas_favorite,
                            'Thanksgiving Underdog': thanksgiving_underdog,
                            'Christmas Underdog': christmas_underdog,
                            'Previous Opponent': previous_opponent,
                            'Previous Game Location': previous_game_location,
                            'Next Opponent': next_opponent,
                            'Next Game Location': next_game_location
                        }
                    picks_rows_2.append(new_row_2)


                    # Add differences to sums
                    sum_preseason_difference += preseason_difference
                    sum_adjusted_preseason_difference += adjusted_preseason_difference
                    sum_current_difference += current_difference
                    sum_adjusted_current_difference += adjusted_current_difference
                    sum_ev += ev
                    sum_sportsbook_spread += float(live_odds_spread)
                    sum_internal_spread += internal_spread
                    picks_df = pd.concat([picks_df, df.loc[[i]]], ignore_index=True)
                    picks_df['Divisional Matchup?'] = divisional_game
            summarized_picks_df = pd.DataFrame(picks_rows_2)

            st.write(summarized_picks_df)
            st.write('')
            st.write('\nPreseason Difference:', sum_preseason_difference)
            st.write('Adjusted Preseason Difference:', sum_adjusted_preseason_difference)
            st.write('Current Difference:', sum_current_difference)
            st.write('Adjusted Current Difference:', sum_adjusted_current_difference)
            st.write('Total Sportsbook Spread: ', sum_sportsbook_spread)
            st.write('Total Internal Spread: ', sum_internal_spread)
            st.write(f'Total EV: :blue[{sum_ev}]')
        else:
            st.write('No solution found. Consider using fewer constraints. Or you may just be fucked')
            st.write('No solution found. Consider using fewer constraints. Or you may just be fucked')
        st.write("---------------------------------------------------------------------------------------------------------------")
        st.write("")
        st.write("")	    
        st.write("")

    
            # Save the picks to a CSV file for the current iteration
        if selected_contest == 'Circa':
            picks_df.to_csv(f'circa_picks_ev_{iteration + 1}.csv', index=False)
            summarized_picks_df.to_csv(f'circa_picks_ev_subset_{iteration + 1}.csv', index=False)
        elif selected_contest == 'Splash Sports':
            picks_df.to_csv(f'splash_picks_ev_{iteration + 1}.csv', index=False)
            summarized_picks_df.to_csv(f'splash_picks_ev_subset_{iteration + 1}.csv', index=False)
        else:
            picks_df.to_csv(f'dk_picks_ev_{iteration + 1}.csv', index=False)
            summarized_picks_df.to_csv(f'dk_picks_ev_subset_{iteration + 1}.csv', index=False)
            
        # Group the picks from the current iteration to create the solution dictionary
        current_solution_dict = summarized_picks_df.groupby('Week')['Pick'].apply(list).to_dict()
        # Call the function to create the simple EV dataframe for the current solution
        simple_ev_df = create_simple_ev_dataframe(summarized_picks_df, favored_qualifier)
        # Now, you can use simple_ev_df for your analysis or display
        st.write(simple_ev_df)
            
        
        # Append the new forbidden solution to the list
        forbidden_solutions_1.append(picks_df['Hypothetical Current Winner'].tolist())

def get_survivor_picks_based_on_internal_rankings(df, config: dict, num_solutions: int):
    # --- ADD THIS BLOCK ---
    # Get settings from config
    starting_week = config['starting_week']
    ending_week = config['ending_week']
    picked_teams = config['prohibited_picks']
    required_weeks_dict = config['required_weeks']
    prohibited_weeks_dict = config['prohibited_weeks']
    selected_contest = config['selected_contest']
    subcontest = config['subcontest']
    week_requiring_two_selections = config.get('weeks_two_picks', [])
    week_requiring_three_selections = config.get('weeks_three_picks', [])
    favored_qualifier = config.get('favored_qualifier', 'Live Sportsbook Odds (If Available)')
    
    # Get all constraints
    pick_must_be_favored = config.get('must_be_favored', False)
    avoid_away_teams_in_close_matchups = config.get('avoid_away_close', False)
    min_away_spread = config.get('min_away_spread', 3.0)
    avoid_close_divisional_matchups = config.get('avoid_close_divisional', False)
    min_div_spread = config.get('min_div_spread', 3.0)
    avoid_away_divisional_matchups = config.get('avoid_away_divisional', False)
    avoid_away_teams_on_short_rest = config.get('avoid_away_short_rest', False)
    avoid_4_games_in_17_days = config.get('avoid_4_in_17', False)
    avoid_3_games_in_10_days = config.get('avoid_3_in_10', False)
    avoid_international_game = config.get('avoid_international', False)
    avoid_thursday_night = config.get('avoid_thursday', False)
    avoid_away_thursday_night = config.get('avoid_away_thursday', False)
    avoid_back_to_back_away = config.get('avoid_b2b_away', False)
    avoid_teams_with_weekly_rest_disadvantage = config.get('avoid_weekly_rest_disadvantage', False)
    avoid_cumulative_rest_disadvantage = config.get('avoid_cumulative_rest', False)
    avoid_away_teams_with_travel_disadvantage = config.get('avoid_travel_disadvantage', False)
    # --- END BLOCK ---
    for iteration in range(num_solutions):

        df = df[(df['Week_Num'] >= starting_week) & (df['Week_Num'] < ending_week)].reset_index(drop=True)
		# Rename columns from reformatted_df to what the solver logic expects
        df.rename(columns={
            "Team": "Hypothetical Current Winner",
            "Opponent": "Hypothetical Current Loser",
            "Expected EV": "Hypothetical Current Winner EV",
            "Adjusted Current Rank": "Hypothetical Current Winner Adjusted Current Rank",
            "Opp Adjusted Current Rank": "Hypothetical Current Loser Adjusted Current Rank",
            "Spread Based on Sportsbook Odds": "Hypothetical Current Winner Sportsbook Spread",
            "Spread Based on Internal Rankings": "Internal Hypothetical Current Winner Spread",
			"Adjusted Current Difference": "Adjusted Current Difference"
        }, inplace=True)
        # Filter out already picked teams

        df = df[~df['Hypothetical Current Winner'].isin(picked_teams)].reset_index(drop=True)
        #print(df)
        # Create the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Create binary variables to represent the picks, and store them in a dictionary for easy lookup
        picks = {}
        for i in range(len(df)):
            picks[i] = solver.IntVar(0, 1, 'pick_%i' % i)
        
        # Add the constraints
        for i in range(len(df)):
            # Can only pick an away team if 'Adjusted Current Difference' > 10
            if pick_must_be_favored:
                if favored_qualifier == 'Internal Rankings':
                    if df.loc[i, 'Hypothetical Current Winner'] != df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                else:
                    if df.loc[i, 'Hypothetical Current Winner'] != df.loc[i, 'Favorite']:
                        solver.Add(picks[i] == 0)    		
            if avoid_away_teams_in_close_matchups == 1:
                if favored_qualifier == 'Internal Rankings':
                    # Check if team is away AND the absolute spread (based on internal ranks) is <= min
                    if df.loc[i, 'Team Is Away'] == 'True' and df.loc[i, 'Adjusted Current Difference'] <= min_away_spread:
                        solver.Add(picks[i] == 0)
                else: 
                    if df.loc[i, 'Team Is Away'] == 'True' and df.loc[i, 'Hypothetical Current Winner Sportsbook Spread'] > -min_away_spread: # Uses sportsbook spread
                        solver.Add(picks[i] == 0)
            #if df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Divisional Matchup?'] == 'Divisional':
                #solver.Add(picks[i] == 0)
            if avoid_back_to_back_away == 1:
                if df.loc[i, 'Team Is Away'] == 'True' and df.loc[i, 'Back to Back Away Games'] == 'True':
                    solver.Add(picks[i] == 0)

            # If 'Divisional Matchup?' is "Divisional", can only pick if 'Adjusted Current Difference' > 10
            if avoid_close_divisional_matchups == 1:
                if favored_qualifier == 'Internal Rankings':
                    if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Adjusted Current Difference'] <= min_away_spread:
                        solver.Add(picks[i] == 0)
                else: 
                    if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Hypothetical Current Winner Sportsbook Spread'] > -min_away_spread: # Uses sportsbook spread
                        solver.Add(picks[i] == 0)
            if avoid_away_divisional_matchups == 1:
                if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Team Is Away'] == 'True':
                    solver.Add(picks[i] == 0)
            # Constraints for short rest and 4 games in 17 days (only if team is the Adjusted Current Winner)
            if avoid_away_teams_on_short_rest == 1:
                if df.loc[i, 'Away Team Short Rest'] == 'Yes':
                    solver.Add(picks[i] == 0)
            if avoid_4_games_in_17_days == 1:
                if df.loc[i, '4 Games in 17 Days'] == 'Yes':
                    solver.Add(picks[i] == 0)
            if avoid_3_games_in_10_days == 1:
                if df.loc[i, '3 Games in 10 Days'] == 'Yes':
                    solver.Add(picks[i] == 0)
            if avoid_international_game == 1:    
                if df.loc[i, 'Location'] == 'London, UK':
                    solver.Add(picks[i] == 0)
            if avoid_thursday_night == 1:
                if df.loc[i, 'Thursday Night Game'] == 'True':
                    solver.Add(picks[i] == 0)
            if avoid_away_thursday_night == 1:
                if df.loc[i, 'Thursday Night Game'] == 'True' and df.loc[i, 'Team Is Away'] == 'True':
                    solver.Add(picks[i] == 0)
            if avoid_teams_with_weekly_rest_disadvantage == 1:
                if df.loc[i, 'Weekly Rest Advantage'] < 0:
                    solver.Add(picks[i] == 0)
            if avoid_cumulative_rest_disadvantage == 1:
                is_away = df.loc[i, 'Team Is Away'] == 'True'
                rest_adv = df.loc[i, 'Season-Long Rest Advantage Including This Week']
                if is_away and rest_adv < -10:
                    solver.Add(picks[i] == 0)
                elif (not is_away) and rest_adv < -5: # Team is Home
                    solver.Add(picks[i] == 0)    
            if avoid_away_teams_with_travel_disadvantage == 1:
                if df.loc[i, 'Team Is Away'] == 'True' and df.loc[i, 'Travel Advantage'] < -850:
                    solver.Add(picks[i] == 0)


            
            prohibited_weeks_dict = config['prohibited_weeks']
            team_name = df.loc[i, 'Hypothetical Current Winner']
            if team_name in prohibited_weeks_dict:
                if df.loc[i, 'Week_Num'] in prohibited_weeks_dict[team_name]:
                    solver.Add(picks[i] == 0)


            if df.loc[i, 'Hypothetical Current Winner'] in picked_teams:
                solver.Add(picks[i] == 0)


        required_weeks_dict = config['required_weeks']
        for team, req_week in required_weeks_dict.items():
            if req_week > 0: # If a required week is set
                # Find all matching game indices for this team/week
                required_game_indices = df[
                    (df['Hypothetical Current Winner'] == team) & 
                    (df['Week_Num'] == req_week)
                ].index.tolist()
        
                if required_game_indices:
                    # Force the pick for the first match found (should only be one)
                    solver.Add(picks[required_game_indices[0]] == 1)

        
        for week in df['Week_Num'].unique():
            # Filter picks for the current week
            weekly_picks = [picks[i] for i in range(len(df)) if df.loc[i, 'Week_Num'] == week]

            if selected_contest == "Splash Sports" and subcontest != "Week 9 Bloody Survivor ($100 Entry)" and week in week_requiring_two_selections:
                # For Splash Sports and later weeks, two teams must be selected
                solver.Add(solver.Sum(weekly_picks) == 2)
            elif selected_contest == "Splash Sports" and subcontest == "Week 9 Bloody Survivor ($100 Entry)" and week in week_requiring_three_selections:
                # For Splash Sports and later weeks, two teams must be selected
                solver.Add(solver.Sum(weekly_picks) == 3)
            else:
                # For other contests or earlier weeks, one team per week
                solver.Add(solver.Sum(weekly_picks) == 1)

        for team in df['Hypothetical Current Winner'].unique():
            # Can't pick a team more than once
            solver.Add(solver.Sum([picks[i] for i in range(len(df)) if df.loc[i, 'Hypothetical Current Winner'] == team]) <= 1)

        
        def create_simple_ev_dataframe(summarized_picks_df, favored_qualifier):
            """
            Creates a DataFrame with one row per week, summarizing picks and their metrics.
        
            Args:
                summarized_picks_df (pd.DataFrame): DataFrame containing 'Week', 'Pick', 
                                                    'Fair Odds', 'Pick %', and 'EV'.
                favored_qualifier (str): 'Internal Rankings' or 'Sportsbook'.
        
            Returns:
                pd.DataFrame: A new DataFrame with the calculated metrics per week.
            """
            # Create a mapping for the Fair Odds column based on the favored_qualifier
            if favored_qualifier == 'Internal Rankings':
                odds_col = '=Fair Odds Based on Internal Rankings'
            else:
                odds_col = 'Fair Odds Based on Sportsbook Odds'
        
            # Group the DataFrame by week and aggregate the desired metrics
            simple_ev_df = summarized_picks_df.groupby('Week').agg(
                Picks=('Pick', lambda x: x.tolist()),
                Survival_Rate=(odds_col, 'prod'),
                Pick_Percentage=('Expected Pick Percent', 'prod'),
                EV=('EV', 'prod')
            ).reset_index()
        
            return simple_ev_df

        # Dynamically create the forbidden solution list
        forbidden_solutions_1 = []
        if iteration > 0:
            for previous_iteration in range(iteration):
                if selected_contest == 'Circa':
                    previous_picks_df = pd.read_csv(f"circa_picks_ev_subset_{previous_iteration + 1}.csv")
                elif selected_contest == 'Splash Sports':
                    previous_picks_df = pd.read_csv(f"splash_picks_ev_subset_{previous_iteration + 1}.csv")
                else:
                    previous_picks_df = pd.read_csv(f"dk_picks_ev_subset_{previous_iteration + 1}.csv")

                # Group picks by week for the forbidden solution
                forbidden_solution_by_week = previous_picks_df.groupby('Week')['Pick'].apply(list).to_dict()
                forbidden_solutions_1.append(forbidden_solution_by_week) 
        # Add constraints for all forbidden solutions
        for forbidden_solution_dict in forbidden_solutions_1:
            # Create a list of the picked variables from the previous solution
            forbidden_pick_variables = []

            # Iterate through each week and its corresponding forbidden picks
            for week, picks_in_week in forbidden_solution_dict.items():
                # Find all rows in the current DataFrame for this specific week
                weekly_rows = df[df['Week_Num'] == week]

                # Check if any of these rows match a forbidden pick from that week
                for _, row in weekly_rows.iterrows():
                    if row['Hypothetical Current Winner'] in picks_in_week: # The 'Favorite' column is what you're using to identify the pick
                        # Get the index of this row
                        pick_index = row.name
                        forbidden_pick_variables.append(picks[pick_index])

            # The constraint now ensures that at least one of the forbidden picks is not selected
            solver.Add(solver.Sum([1 - v for v in forbidden_pick_variables]) >= 1)


        
        # Objective: maximize the sum of Adjusted Current Difference of each game picked
        solver.Minimize(solver.Sum([picks[i] * df.loc[i, 'Hypothetical Current Winner Sportsbook Spread'] for i in range(len(df))]))



        # Solve the problem and print the solution
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            st.write('')
            st.write(f'**Solution Based on Internal Rankings: {iteration + 1}**')

            st.write('Solution found!')
            st.write('Objective value =', solver.Objective().Value())


            # Initialize sums
            sum_preseason_difference = 0
            sum_adjusted_preseason_difference = 0
            sum_current_difference = 0
            sum_adjusted_current_difference = 0
            sum_ev = 0
            sum_sportsbook_spread = 0
            sum_internal_spread = 0

            # Initialize picks_df
            picks_df = pd.DataFrame(columns=df.columns)
            picks_rows_2 = []
            for i in range(len(df)):
                if picks[i].solution_value() > 0:
                    # Determine if it's a divisional game and if the picked team is the home team

                    week = df.loc[i, 'Week_Num']
                    date = df.loc[i, 'Date']
                    time = df.loc[i, 'Time']
                    location = df.loc[i, 'Location']
                    pick = df.loc[i,'Hypothetical Current Winner']
                    opponent = df.loc[i,'Hypothetical Current Loser']
                    win_odds = round(df.loc[i, 'Fair Odds Based on Sportsbook Odds'], 2)
                    pick_percent = round(df.loc[i, 'Expected Pick Percent'], 2)
                    expected_availability = round(df.loc[i, 'Expected Availability'], 2)
                    divisional_game = 'Divisional' if df.loc[i, 'Divisional Matchup?'] == 1 else ''
                    home_team = 'Home Team' if df.loc[i, 'Team Is Away'] == 'False' else 'Away Team'
                    weekly_rest = df.loc[i, 'Weekly Rest']
                    weekly_rest_advantage = df.loc[i, 'Weekly Rest Advantage']
                    cumulative_rest = df.loc[i, 'Season-Long Rest Advantage']
                    cumulative_rest_advantage = df.loc[i, 'Season-Long Rest Advantage Including This Week']
                    travel_advantage = df.loc[i, 'Travel Advantage']
                    back_to_back_away_games = 'True' if df.loc[i, 'Team Is Away'] == 'True' and df.loc[i, 'Back to Back Away Games'] == 'True' else 'False'
                    thursday_night_game = 'Thursday Night Game' if df.loc[i, "Thursday Night Game"] == 'True' else 'Sunday/Monday Game'
                    international_game = 'International Game' if df.loc[i, 'Location'] == 'London, UK' else 'Domestic Game'
                    previous_opponent = df.loc[i, 'Previous Opponent']
                    previous_game_location = df.loc[i, 'Previous Game Location']
                    next_opponent = df.loc[i, 'Next Opponent']
                    next_game_location = df.loc[i, 'Next Game Location']
					
                    internal_ranking_fair_odds = df.loc[i, 'Fair Odds Based on Internal Rankings']
                    future_value = df.loc[i, 'Future Value']
                    sportbook_moneyline = df.loc[i, 'Moneyline Based on Sportsbook Odds']
                    internal_moneyline = df.loc[i, 'Moneyline Based on Internal Rankings']
                    contest_selections = df.loc[i, 'Expected Picks']
                    survival_rate = df.loc[i, 'Expected Survival Rate']
                    elimination_percent = df.loc[i, 'Expected Contest Elimination Percent']
                    survivors = df.loc[i, 'Expected Survivors']
                    eliminations = df.loc[i, 'Expected Eliminations']
                    preseason_rank = df.loc[i, 'Preseason Rank']
                    adjusted_preseason_rank = df.loc[i, 'Adjusted Preseason Rank']
                    current_rank = df.loc[i, 'Current Rank']
                    adjusted_current_rank = df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank']
                    away_team_short_rest = df.loc[i, 'Away Team Short Rest']
                    three_games_in_10_days = df.loc[i, '3 Games in 10 Days']
                    four_games_in_17_days = df.loc[i, '4 Games in 17 Days']
                    thanksgiving_favorite = df.loc[i, 'Thanksgiving Favorite']
                    christmas_favorite = df.loc[i, 'Christmas Favorite']
                    thanksgiving_underdog = df.loc[i, 'Thanksgiving Underdog']
                    christmas_underdog = df.loc[i, 'Christmas Underdog']
                    live_odds_unavailable = df.loc[i, 'No Live Odds Available - Internal Rankings Used for Moneyline Calculation']
                    live_odds_spread = df.loc[i, 'Hypothetical Current Winner Sportsbook Spread']
                    internal_spread = df.loc[i, 'Internal Hypothetical Current Winner Spread']
                    

                    # Get differences
                    preseason_difference = df.loc[i, 'Preseason Difference']
                    adjusted_preseason_difference = df.loc[i, 'Adjusted Preseason Difference']
                    current_difference = df.loc[i, 'Current Difference']
                    adjusted_current_difference = df.loc[i, 'Adjusted Current Difference']
                    # Calculate EV for this game
                    ev = df.loc[i, 'Hypothetical Current Winner EV']


                    print('Week %i: Pick %s %s %s (%i, %i, %i, %i, %.4f)' % (df.loc[i, 'Week_Num'], df.loc[i, 'Hypothetical Current Winner'], divisional_game, home_team,
                                                                       preseason_difference, adjusted_preseason_difference,
                                                                       current_difference, adjusted_current_difference, ev))
                    if selected_contest == 'Circa':
                        new_row_2 = {
                            'Week': week,
                            'Pick': pick,
                            'Opponent': opponent,
                            'Date': date,
                            'Time': time,
                            'Location': location,
                            'Home or Away': home_team,
                            'EV': ev,
                            'Fair Odds Based on Sportsbook Odds': win_odds,
                            'Fair Odds Based on Internal Rankings': internal_ranking_fair_odds,
                            'Expected Pick Percent': pick_percent,	
                            'Expected Availability': expected_availability,
                            'Future Value': future_value,
                            'Moneyline Based on Sportsbook Odds': sportbook_moneyline,
                            'Moneyline Based on Internal Rankings': internal_moneyline,
                            'Spread Based on Sportsbook Odds': live_odds_spread,
                            'Spread Based on Internal Rankings': internal_spread,
                            'No Live Odds Available - Internal Rankings Used for Moneyline Calculation': live_odds_unavailable,
                            'Expected Contest Selections': contest_selections,
                            'Expected Survival Rate': survival_rate,
                            'Expected Elimination Rate': elimination_percent,
                            'Expected Survivors': survivors,
                            'Expected Eliminations': eliminations,
                            'Preseason Rank': preseason_rank,
                            'Adjusted Preseason Rank': adjusted_preseason_rank,
                            'Current Rank': current_rank,
                            'Adjusted Current Rank': adjusted_current_rank,
                            'Preseason Difference': preseason_difference,
                            'Adjusted Preseason Difference': adjusted_preseason_difference,
                            'Current Difference': current_difference,
                            'Adjusted Current Difference': adjusted_current_difference,
                            'Thursday Night Game': thursday_night_game,
                            'International Game': international_game,
                            'Divisional Game': divisional_game,
                            'Weekly Rest': weekly_rest,
                            'Weekly Rest Advantage': weekly_rest_advantage,
                            'Season Long Rest Advantage': cumulative_rest,
                            'Season Long Rest Including This Week': cumulative_rest_advantage,
                            'Travel Advantage': travel_advantage,
                            'Back to Back Away Games': back_to_back_away_games,
                            'Away Team on Short Rest': away_team_short_rest,
                            'Three Games in 10 Days': three_games_in_10_days,
                            'Four Games in 17 Days': four_games_in_17_days,
                            'Thanksgiving Favorite': thanksgiving_favorite,
                            'Christmas Favorite': christmas_favorite,
                            'Thanksgiving Underdog': thanksgiving_underdog,
                            'Christmas Underdog': christmas_underdog,
                            'Previous Opponent': previous_opponent,
                            'Previous Game Location': previous_game_location,
                            'Next Opponent': next_opponent,
                            'Next Game Location': next_game_location
                        }
                    else:
                        new_row_2 = {
                            'Week': week,
                            'Pick': pick,
                            'Opponent': opponent,
                            'Date': date,
                            'Time': time,
                            'Location': location,
                            'Home or Away': home_team,
                            'EV': ev,
                            'Fair Odds Based on Sportsbook Odds': win_odds,
                            'Fair Odds Based on Internal Rankings': internal_ranking_fair_odds,
                            'Expected Pick Percent': pick_percent,	
                            'Expected Availability': expected_availability,
                            'Future Value': future_value,
                            'Moneyline Based on Sportsbook Odds': sportbook_moneyline,
                            'Moneyline Based on Internal Rankings': internal_moneyline,
                            'Spread Based on Sportsbook Odds': live_odds_spread,
                            'Spread Based on Internal Rankings': internal_spread,
                            'No Live Odds Available - Internal Rankings Used for Moneyline Calculation': live_odds_unavailable,
                            'Expected Contest Selections': contest_selections,
                            'Expected Survival Rate': survival_rate,
                            'Expected Elimination Rate': elimination_percent,
                            'Expected Survivors': survivors,
                            'Expected Eliminations': eliminations,
                            'Preseason Rank': preseason_rank,
                            'Adjusted Preseason Rank': adjusted_preseason_rank,
                            'Current Rank': current_rank,
                            'Adjusted Current Rank': adjusted_current_rank,
                            'Preseason Difference': preseason_difference,
                            'Adjusted Preseason Difference': adjusted_preseason_difference,
                            'Current Difference': current_difference,
                            'Adjusted Current Difference': adjusted_current_difference,
                            'Thursday Night Game': thursday_night_game,
                            'International Game': international_game,
                            'Divisional Game': divisional_game,
                            'Weekly Rest': weekly_rest,
                            'Weekly Rest Advantage': weekly_rest_advantage,
                            'Season Long Rest Advantage': cumulative_rest,
                            'Season Long Rest Including This Week': cumulative_rest_advantage,
                            'Travel Advantage': travel_advantage,
                            'Back to Back Away Games': back_to_back_away_games,
                            'Away Team on Short Rest': away_team_short_rest,
                            'Three Games in 10 Days': three_games_in_10_days,
                            'Four Games in 17 Days': four_games_in_17_days,
                            'Thanksgiving Favorite': thanksgiving_favorite,
                            'Christmas Favorite': christmas_favorite,
                            'Thanksgiving Underdog': thanksgiving_underdog,
                            'Christmas Underdog': christmas_underdog,
                            'Previous Opponent': previous_opponent,
                            'Previous Game Location': previous_game_location,
                            'Next Opponent': next_opponent,
                            'Next Game Location': next_game_location
                        }
                    picks_rows_2.append(new_row_2)


                    # Add differences to sums
                    sum_preseason_difference += preseason_difference
                    sum_adjusted_preseason_difference += adjusted_preseason_difference
                    sum_current_difference += current_difference
                    sum_ev += ev
                    sum_sportsbook_spread += live_odds_spread
                    sum_internal_spread += internal_spread
                    sum_adjusted_current_difference += adjusted_current_difference
                    picks_df = pd.concat([picks_df, df.loc[[i]]], ignore_index=True)
                    picks_df['Divisional Matchup?'] = divisional_game
            summarized_picks_df = pd.DataFrame(picks_rows_2)

            st.write(summarized_picks_df)
            st.write('')
            st.write(f'Total EV: {sum_ev}')
            st.write('\nPreseason Difference:', sum_preseason_difference)
            st.write('Adjusted Preseason Difference:', sum_adjusted_preseason_difference)
            st.write('Current Difference:', sum_current_difference)
            st.write(f'Adjusted Current Difference: {sum_adjusted_current_difference}')
            st.write('Total Internal Spread: ', sum_internal_spread)
            st.write(f'Total Sportsbook Spread: :blue[{sum_sportsbook_spread}]')
            
        else:
            st.write('No solution found. Consider using fewer constraints. Or you may just be fucked')
            st.write('No solution found. Consider using fewer constraints. Or you may just be fucked')
        st.write("---------------------------------------------------------------------------------------------------------------")
        st.write("")
        st.write("")  
        st.write("")

            # Save the picks to a CSV file for the current iteration
        if selected_contest == 'Circa':
            picks_df.to_csv(f'circa_picks_ir_{iteration + 1}.csv', index=False)
            summarized_picks_df.to_csv(f'circa_picks_ir_subset_{iteration + 1}.csv', index=False)
        elif selected_contest == 'Splash Sports':
            picks_df.to_csv(f'splash_picks_ir_{iteration + 1}.csv', index=False)
            summarized_picks_df.to_csv(f'splash_picks_ir_subset_{iteration + 1}.csv', index=False)
        else:
            picks_df.to_csv(f'dk_picks_ir_{iteration + 1}.csv', index=False)
            summarized_picks_df.to_csv(f'dk_picks_ir_subset_{iteration + 1}.csv', index=False)
        # Group the picks from the current iteration to create the solution dictionary
        current_solution_dict = summarized_picks_df.groupby('Week')['Pick'].apply(list).to_dict()
        # Call the function to create the simple EV dataframe for the current solution
        simple_ev_df = create_simple_ev_dataframe(summarized_picks_df, favored_qualifier)
        # Now, you can use simple_ev_df for your analysis or display
        st.write(simple_ev_df)
            
        
        # Append the new forbidden solution to the list
        forbidden_solutions_1.append(picks_df['Hypothetical Current Winner'].tolist())
        #print(forbidden_solutions)


picked_teams = []

# --- 1. Constants and Configuration ---
LOGO_PATH = "GSF Survivor Logo Clear BG.png"
DB_FILE = 'user_configs.db'

# Define NFL teams list (used multiple times)
nfl_teams = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
    "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
    "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
    "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"
]
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
}
team_abbreviations = list(TEAM_NAME_TO_ABBR.values())

# Contest Options
contest_options = ["Circa", "Splash Sports", "Other"]
subcontest_options = [
    "", "The Big Splash ($150 Entry)", "High Roller ($1000 Entry)", "Free RotoWire (Free Entry)",
    "4 for 4 ($50 Entry)", "For the Fans ($40 Entry)", "Walker's Ultimate Survivor ($25 Entry)",
    "Ship It Nation ($25 Entry)", "Week 9 Bloody Survivor ($100 Entry)"
]

# --- 2. Database Functions (Using JSON) ---

def get_db_connection():
    db_file = 'user_configs.db'
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            config_name TEXT NOT NULL,
            config_data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (user_id, config_name)
        )
    """)
    conn.commit()
    return conn

# Updated save_config function
# Note: You need to pass the config JSON string here, not input_1, input_2
def save_config(user_id, config_name, config_json_string):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM user_configs WHERE user_id = ? AND config_name = ?",
            (user_id, config_name)
        )
        if cursor.fetchone():
            conn.close()
            return f"Configuration '{config_name}' already exists. Please choose a new name."

        conn.execute(
            "INSERT INTO user_configs (user_id, config_name, config_data) VALUES (?, ?, ?)",
            (user_id, config_name, config_json_string)
        )
        conn.commit()
        conn.close()
        return f"Configuration '{config_name}' saved successfully!"
    except Exception as e:
        conn.close()
        return f"An error occurred during save: {e}"


def load_config(user_id, config_name):
    """Retrieves the JSON string for a selected configuration."""
    conn = get_db_connection()

    data = conn.execute(
        "SELECT config_data FROM user_configs WHERE user_id = ? AND config_name = ?",
        (user_id, config_name)
    ).fetchone()
    
    conn.close()

    if data:
        return data[0]
    return None

def get_all_configs(user_id: str):
    """Returns a list of saved config names for the user."""
    conn = get_db_connection()
    configs = conn.execute(
        "SELECT config_name FROM user_configs WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [c[0] for c in configs]


# Assuming this is your general update function
def update_config_value(key):
    # Get the value from the corresponding widget
    value = st.session_state[f'{key}_widget'] 
    # Update the config dict
    st.session_state.current_config[key] = value

    # ADD THIS CONDITIONAL LOGIC
    if key == 'subcontest':
        subcontest_name = value
        if subcontest_name == "The Big Splash ($150 Entry)":
            st.session_state.current_config['weeks_two_picks'] = [11, 12, 13, 14, 15, 16, 17, 18]
        elif subcontest_name == "4 for 4 ($50 Entry)":
            st.session_state.current_config['weeks_two_picks'] = [12, 13, 14, 15, 16, 17, 18]
        elif subcontest_name == "Free RotoWire (Free Entry)":
            st.session_state.current_config['weeks_two_picks'] = []
        elif subcontest_name == "For the Fans ($40 Entry)":
            st.session_state.current_config['weeks_two_picks'] = [14, 15, 16, 17, 18]
        elif subcontest_name == "Walker's Ultimate Survivor ($25 Entry)":
            st.session_state.current_config['weeks_two_picks'] = [6, 12, 13, 14, 15, 16, 17, 18]
        elif subcontest_name == "Ship It Nation ($25 Entry)":
            st.session_state.current_config['weeks_two_picks'] = [12, 13, 14, 15, 16, 17, 18]
        elif subcontest_name == "High Roller ($1000 Entry)":
            st.session_state.current_config['weeks_two_picks'] = []
        elif subcontest_name == "Week 9 Bloody Survivor ($100 Entry)":
            st.session_state.current_config['weeks_three_picks'] = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# --- Replace the existing update_nested_value with this updated version ---
def update_nested_value(outer_key, inner_key):
    """Generic callback for nested dictionaries (e.g., rankings, availabilities)."""
    widget_key = f"{outer_key}_{inner_key}_widget".replace(" ", "_").replace("(", "").replace(")", "").replace("$","")
    if widget_key in st.session_state:
        value = st.session_state[widget_key]
        # Handle sentinel -1 from sliders for percentage-style fields
        if outer_key in ('team_availabilities', 'pick_percentages'):
            # Slider uses -1 to indicate "Auto". Map -1 -> -1.0 sentinel, otherwise divide to get float percent.
            if value == -1:
                value = -1.0
            else:
                value = value / 100.0
        elif outer_key == 'team_rankings' and value == "Default":
            value = DEFAULT_RANKS.get(inner_key, 0)

        if outer_key not in st.session_state.current_config:
            st.session_state.current_config[outer_key] = {}
        st.session_state.current_config[outer_key][inner_key] = value


# --- Replace the existing update_pick_percentage with this improved version ---
def update_pick_percentage(week, team_name):
    """Specific callback for the nested pick percentage dictionary."""
    widget_key = f"pick_perc_week_{week}_{team_name.replace(' ', '_')}_widget"
    if widget_key in st.session_state:
        percentage_int = st.session_state[widget_key]
        # Treat slider -1 as sentinel (Auto). Store -1.0 sentinel to remain consistent with initialization.
        if percentage_int == -1:
            percentage_float = -1.0
        else:
            percentage_float = percentage_int / 100.0
        week_key = f"week_{week}"

        if week_key not in st.session_state.current_config['pick_percentages']:
            st.session_state.current_config['pick_percentages'][week_key] = {}
            
        st.session_state.current_config['pick_percentages'][week_key][team_name] = percentage_float


# --- 4. Main Streamlit App ---

st.set_page_config(layout="wide") # Use wide layout


def calculate_alive_entries(file_path: str, config: dict) -> int:
    # 1. Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    start_w = config['starting_week']

    # 3. Filter the DataFrame and count the entries where the status is 'ALIVE'
    alive_entries_count = (df['Total_Wins'] >= (start_w - 1)).sum()

    return alive_entries_count

def calculate_team_availability(historical_data_path, picks_data_path, config):
    """
    Calculates the availability of each team for the target week (start_w) in the 
    Circa Survivor Contest, based on the picks made up to the week prior (W_max).

    Availability is defined as the percentage of currently 'alive' entries 
    (Total_Wins >= W_max) that have NOT yet used a given team.

    Args:
        historical_data_path: File path to the Circa historical data (used to find all team names).
        picks_data_path: File path to the survivor picks history (used to find usage).
        start_w: The week number for which availability should be calculated (e.g., Week 5).

    Returns:
        A pandas DataFrame with team, next week, availability percentage, and counts,
        or None if an error occurs.
    """
    
    print(f"Loading data from: {historical_data_path} and {picks_data_path}")

    # --- 1. Load Data ---

    df_hist = pd.read_csv(historical_data_path, low_memory=False)
    df_picks = pd.read_csv(picks_data_path, low_memory=False)
    start_w = config['starting_week']

    # --- User Provided Data ---
    correction_map = {
        "JAC": "JAX",
		"WSH": "WAS"
		#"WAS": "WSH"
    }

    end_week = start_w - 1
    week_columns = [f"Week_{i}" for i in range(1, end_week + 1)]
    
    # Apply the replacement to all specified columns in df_picks
    for col in week_columns:
        if col in df_picks.columns:
            df_picks[col] = df_picks[col].replace(correction_map)
        else:
            print(f"Warning: Column '{col}' not found in df_picks.")

    print("df_picks successfully updated to use 'JAX' for all weekly columns.")
    # --- 2. Determine the Target Week (W_next) and Last Completed Week (W_max) ---
    # W_next is the week we are calculating availability FOR (start_w)
    W_next = start_w
    
    # W_max is the last completed week (the last week entries had to survive)
    if W_next <= 1:
        # If calculating for Week 1, W_max is 0
        W_max = 0
    else:
        W_max = W_next - 1
    
    print(f"\n--- Analysis Parameters ---")
    print(f"Target availability week (W_next): {W_next} (Source: config['starting_week'])")
    print(f"Last completed pick week (W_max): {W_max}")

    # Define the columns that represent the picks made up to the last completed week
    # This list will be empty if W_max is 0
    pick_cols_theoretical = [f'Week_{i}' for i in range(1, W_max + 1)]
    
    # Check for missing pick columns
    available_pick_cols = [col for col in pick_cols_theoretical if col in df_picks.columns]

    if not available_pick_cols and W_max > 0:
        print("ERROR: Picks data is missing columns for the necessary past weeks.")
        return None
    
    pick_cols_to_use = available_pick_cols 

    # --- 3. Filter Alive Entries ---
    if 'Total_Wins' not in df_picks.columns:
        print("ERROR: '2025_survivor_picks.csv' must contain a 'Total_Wins' column.")
        return None

    df_picks['Total_Wins_Numeric'] = pd.to_numeric(df_picks['Total_Wins'], errors='coerce').fillna(0)
    # Entries are considered 'alive' if Total_Wins >= W_max 
    df_alive = df_picks[df_picks['Total_Wins_Numeric'] >= W_max].copy()


    N_alive = len(df_alive)
    print(f"Total entries considered alive (Total_Wins >= {W_max}): {N_alive}")

    if N_alive == 0:
        print("\nNo entries are currently alive. Cannot calculate availability.")
        return pd.DataFrame({'Team': [], 'Availability_Percent': []})


    # --- 4 & 5. Calculate Availability for Each Team ---
    
    # Get the unique list of all teams from the historical data
    all_teams_home = df_hist['Team'].dropna().unique()
    all_teams_away = df_hist['Opponent'].dropna().unique()
    all_teams = np.unique(np.concatenate((all_teams_home, all_teams_away)))
    availability_list = []

    for team in all_teams:
        # Initialize mask for entries that have used this team
        used_mask = pd.Series(False, index=df_alive.index)
        
        # Only check usage if W_max > 0 (i.e., we are past Week 1)
        if W_max > 0:
            for col in pick_cols_to_use:
                if col in df_alive.columns:
                    # Use str.strip() to handle potential whitespace in team names/picks
                    used_mask = used_mask | (df_alive[col].astype(str).str.strip() == str(team).strip())

        N_used = used_mask.sum()
        N_available = N_alive - N_used
        availability_percent = N_available / N_alive
        
        availability_list.append({
            'Team': team,
            'Availability_Week': W_next,
            'Entries_Used_Count': N_used,
            'Entries_Available_Count': N_available,
            'Total_Alive_Entries': N_alive,
            'Availability_Percent': f"{availability_percent:.4f}"
        })

    # --- 6. Finalize and Return ---
    df_availability = pd.DataFrame(availability_list)
    df_availability['Availability_Percent_Float'] = pd.to_numeric(df_availability['Availability_Percent'], errors='coerce')
    df_availability = df_availability.sort_values(by='Availability_Percent_Float', ascending=False).drop(columns=['Availability_Percent_Float'])

    # Map team abbreviations to full names
    abbreviations_to_full_name = {
        "ARI": "Arizona Cardinals",
        "ATL": "Atlanta Falcons",
        "BAL": "Baltimore Ravens",
        "BUF": "Buffalo Bills",
        "CAR": "Carolina Panthers",
        "CHI": "Chicago Bears",
        "CIN": "Cincinnati Bengals",
        "CLE": "Cleveland Browns",
        "DAL": "Dallas Cowboys",
        "DEN": "Denver Broncos",
        "DET": "Detroit Lions",
        "GB": "Green Bay Packers",
        "HOU": "Houston Texans",
        "IND": "Indianapolis Colts",
        "JAX": "Jacksonville Jaguars",
        "KC": "Kansas City Chiefs",
        "LV": "Las Vegas Raiders",
        "LAC": "Los Angeles Chargers",
        "LAR": "Los Angeles Rams",
        "MIA": "Miami Dolphins",
        "MIN": "Minnesota Vikings",
        "NE": "New England Patriots",
        "NO": "New Orleans Saints",
        "NYG": "New York Giants",
        "NYJ": "New York Jets",
        "PHI": "Philadelphia Eagles",
        "PIT": "Pittsburgh Steelers",
        "SF": "San Francisco 49ers",
        "SEA": "Seattle Seahawks",
        "TB": "Tampa Bay Buccaneers",
        "TEN": "Tennessee Titans",
        "WAS": "Washington Commanders"
    }
    
    df_availability['Team'] = df_availability['Team'].replace(abbreviations_to_full_name)
    print("\n--- Availability Calculation Complete ---")
    return df_availability




# --- Authentication ---
def login_screen():
    st.image(LOGO_PATH, width=200) # Smaller logo for login
    st.title("Welcome to Generic Sports Fan Survivor Optimization")
    st.subheader("Please log in with Google to access the optimizer.")
    st.write("This tool requires authentication to ensure proper usage and to save your progress.")
    st.write("")
    if st.button("Log in with Google", use_container_width=True, type="primary"):
        st.login()

if not st.user.is_logged_in:
    login_screen()
else:
    # --- User Logged In ---
    CURRENT_USER_ID = st.user.email

    # --- Initialize Master Config Dictionary in Session State ---
    if 'current_config' not in st.session_state:
        # Define ALL default values for a new user session
        st.session_state.current_config = {
            'selected_contest': "Circa",
            'subcontest': "",
            'weeks_two_picks': [],
			'weeks_three_picks': [],
            'has_picked_teams': False,
            'prohibited_picks': [],
            'choose_weeks': False,
            'starting_week': 1,
            'ending_week': 21, # Default to Circa max
            'current_week_entries': -1,
            'use_live_data': False,
            'team_availabilities': {team: -1.0 for team in nfl_teams},
            'provide_availability': False,
            'require_team': False,
            'required_weeks': {team: 0 for team in nfl_teams},
            'prohibit_teams': False,
            'prohibited_weeks': {team: [] for team in nfl_teams},
            'custom_rankings': False,
            'team_rankings': {team: 'Default' for team in nfl_teams}, # Use 'Default' string marker
            'must_be_favored': False,
            'favored_qualifier': 'Live Sportsbook Odds (If Available)',
            'add_constraints': False,
            'avoid_away_short_rest': False,
            'avoid_close_divisional': False,
            'min_div_spread': 3.0,
            'avoid_away_divisional': False,
            'avoid_3_in_10': False,
            'avoid_4_in_17': False,
            'avoid_away_close': False,
            'min_away_spread': 3.0,
            'avoid_cumulative_rest': False,
            'avoid_thursday': False,
            'avoid_away_thursday': False,
            'avoid_b2b_away': False,
            'avoid_international': False,
            'avoid_weekly_rest_disadvantage': False,
            'avoid_travel_disadvantage': False,
            'bayesian_constraint': "No Rest, Bayesian, and Travel Constraints",
            'custom_pick_percentages': False,
            'pick_percentages': {f"week_{w}": {team: -1.0 for team in nfl_teams} for w in range(1, 21)},
            'number_solutions': 10,
            # Add placeholders for other sections if needed
        }

        # Dynamically set ending week based on default contest
        if st.session_state.current_config['selected_contest'] != 'Circa':
            st.session_state.current_config['ending_week'] = 19

    # --- Initialize other session state variables ---
    if 'config_status' not in st.session_state:
        st.session_state.config_status = ""
    if 'selected_config_to_load' not in st.session_state:
        st.session_state.selected_config_to_load = None


    # --- Sidebar ---
    with st.sidebar:
        st.image(LOGO_PATH, use_container_width=True)
        st.write(f"Hello, **{st.user.name}**!")
        if st.button("Logout"):
            st.logout()
        st.write("---")
        # Add Save/Load to Sidebar for better access
        st.header("💾 Configurations")

        # Load
        saved_configs = get_all_configs(CURRENT_USER_ID)
        st.write(saved_configs)
        if saved_configs:
            st.selectbox(
                "Load Configuration:",
                options=[""] + saved_configs, # Add empty option
                key='selected_config_to_load',
                index=0 # Default to empty
            )
            if st.button("🔄 Load Selected", use_container_width=True):
                if st.session_state.selected_config_to_load:
                    loaded_json_string = load_config(CURRENT_USER_ID, st.session_state.selected_config_to_load)
                    if loaded_json_string:
                        try:
                            loaded_data = json.loads(loaded_json_string)
                            st.session_state.current_config = loaded_data # Now it's a dictionary
                            st.session_state.config_status = f"'{st.session_state.selected_config_to_load}' loaded!"
                            st.rerun()
                        except json.JSONDecodeError:
                            st.session_state.config_status = "Error: Failed to parse configuration data."
                    else:
                        st.session_state.config_status = "Error loading configuration."
                    # --- END CHANGE ---
                        
                else:
                    st.session_state.config_status = "Please select a configuration to load."

        # Save
        with st.form("save_form_sidebar", clear_on_submit=True):
            save_name = st.text_input("Save Current Config As:", key='save_name_input')
            if st.form_submit_button("💾 Save", use_container_width=True):
                if save_name:
                    config_json_string = json.dumps(st.session_state.current_config)
                    message = save_config(CURRENT_USER_ID, save_name, config_json_string)
                    # --- END CHANGE ---
                    
                    st.session_state.config_status = message
                    # No rerun needed, just update status
                else:
                    st.session_state.config_status = "Please enter a name to save."

        # Display Status in Sidebar
        if st.session_state.config_status:
            st.info(st.session_state.config_status)


    # --- Main Page Content ---
    st.image(LOGO_PATH, width=300)
    st.title("NFL Survivor Optimization")
    st.subheader("The best NFL Survivor Contest optimizer")

    with st.expander("More Information"):
        st.write("Alright, clowns. This site is built to help you optimize your picks for the Circa Survivor contest (And Draftkings and Splash Sports). :red[This tool is just for informational use. It does not take into account injuries, weather, resting players, or certain other factors. Do not use this tool as your only source of information.] Simply input which week you're in, your team rankings, constraints, etc... and the algorithm will do the rest.")
        st.write("Calculating Expected Value, or EV, will take the longest in this process. For a full season, this step will take roughly 5-10 minutes or more. Do not close your browser. It's worth the wait. Good luck!")

    st.write('---')

    # --- A. Contest Selection ---
    st.subheader('Select Contest')
    help_text_seletced_contest = f"""
    \nThe biggest differences between the three contests:
    \n- DraftKings tends to be a standard contest. One pick per week for 18 weeks. A win or tie from your pick means you advance.
    \n- In Splash Sports, you will be required to make two picks per week usually starting at some point between week 11 and week 14, depending on the contest you entered. 
    \n- Circa has 20 Weeks (Christmas and Thanksgiving/Black Friday act as their own individual weeks)
    \n- Thanksgiving/Black Friday week will be Week 13 on this site (If you select Circa)
    \n- Christmas Week will be week 18 on this site (If you select Circa)
    \n- In Circa and Splash Sports, a tie eliminates you, but in Draftkings, you move on with a tie
    \n- Players in Circa and Splash Sports tend to be sharper, making it more difficult to play contrarian and ultimately win
    
    """
    
    # Widget Key and linking to config dictionary
    st.selectbox(
        'Choose Platform:',
        options=contest_options,
        key='selected_contest_widget', # Temporary widget key
        index=contest_options.index(st.session_state.current_config['selected_contest']),
        on_change=update_config_value,
        args=('selected_contest',), # Key in the main dictionary to update
        help=help_text_seletced_contest
    )

    # Conditional UI based on selected contest (reading from config dict)
    selected_contest = st.session_state.current_config['selected_contest'] # Use value from dict
    
    # Update default ending week if contest changes
    if selected_contest == 'Circa' and st.session_state.current_config['ending_week'] != 21:
         st.session_state.current_config['ending_week'] = 21
    elif selected_contest != 'Circa' and st.session_state.current_config['ending_week'] != 19:
         st.session_state.current_config['ending_week'] = 19
    
    
    if selected_contest == "Splash Sports":
        subcontest_help_text = f"""
    \n- Choose the specific contest you're playing on Splash Sports.
    \n- Differences in contests include which weeks require double picks, entry stake, number of entries, and weekly percentage picks.
    
    """	
        two_team_selections_help_text = f"""
    \nIn Splash Sports, most survivor contests have a unique requirement: You must select two teams per week in select weeks. Sometimes this is weeks where all 32 teams play, or sometimes it's the last few weeks of the season.  
    \n- However, when you start selecting two teams per week varies, depending on which contest you enter. For some, this may begin in week 11, and for others it may begin as late as week 16.
    \n- Because it varies, we want to give you the option to select which week this applies to you. Plus, it gives you more flexibility to play around with the tool. 
    
    """

        three_team_selections_help_text = f"""
    \nIn Splash Sports, some rare survivor contests have a unique requirement: You must select :red[three] teams per week in select weeks.
    \n- Because it varies, we want to give you the option to select which week(s) this applies to you. Plus, it gives you more flexibility to play around with the tool. 
    
    """
        st.write('')
        st.selectbox(
            'Choose Specific Contest from Splash Sports:',
            options=subcontest_options,
            key='subcontest_widget',
            index=subcontest_options.index(st.session_state.current_config['subcontest']),
            on_change=update_config_value,
            args=('subcontest',),
            help=subcontest_help_text
        )
        
        subcontest = st.session_state.current_config['subcontest']
        if subcontest != "Week 9 Bloody Survivor ($100 Entry)":
            st.multiselect(
	            "Which weeks do you need to select two teams?:",
	            options=range(1, 19),
	            key='weeks_two_picks_widget',
	            default=st.session_state.current_config['weeks_two_picks'],
	            on_change=update_config_value,
	            args=('weeks_two_picks',),
	            help=two_team_selections_help_text
	        )
        else:
	        st.multiselect(
	            "Which weeks do you need to select three teams?:",
	            options=range(1, 19),
	            key='weeks_three_picks_widget',
	            default=st.session_state.current_config['weeks_three_picks'],
	            on_change=update_config_value,
	            args=('weeks_three_picks',),
	            help=three_team_selections_help_text
	        )
        # Display helper text based on subcontest
        subcontest = st.session_state.current_config['subcontest']
		
        if subcontest == "The Big Splash ($150 Entry)":
            st.write("Weeks requiring double picks in The Big Splash Survivor Contest: :green[11, 12, 13, 14, 15, 16, 17, 18]")
            st.write(f"System is defaulting to require double picks in the following weeks: {st.session_state.current_config['weeks_two_picks']}")
        elif subcontest == "4 for 4 ($50 Entry)":
            st.write("Weeks requiring double picks in the 4 for 4 Survivor Contest: :green[11, 12, 13, 14, 15, 16, 17, 18]")
            st.write(f"System is defaulting to require double picks in the following weeks: {st.session_state.current_config['weeks_two_picks']}")
        elif subcontest == "Free RotoWire (Free Entry)":
            st.write("Weeks requiring double picks in the Free RotoWire Survivor Contest: :green[None]")
            st.write(f"System is defaulting to require double picks in the following weeks: None")
        elif subcontest == "For the Fans ($40 Entry)":
            st.write("Weeks requiring double picks in the For the Fan Survivor Contest: :green[14, 15, 16, 17, 18]")
            st.write(f"System is defaulting to require double picks in the following weeks: {st.session_state.current_config['weeks_two_picks']}")
        elif subcontest == "Walker's Ultimate Survivor ($25 Entry)":
            st.write("Weeks requiring double picks in Walker's Ultimate Survivor Survivor Contest: :green[6, 12, 13, 14, 15, 16, 17, 18]")
            st.write(f"System is defaulting to require double picks in the following weeks: {st.session_state.current_config['weeks_two_picks']}")
        elif subcontest == "Ship It Nation ($25 Entry)":
            st.write("Weeks requiring double picks in the Ship It Nation Survivor Contest: :green[12, 13, 14, 15, 16, 17, 18]")
            st.write(f"System is defaulting to require double picks in the following weeks: {st.session_state.current_config['weeks_two_picks']}")
        elif subcontest == "High Roller ($1000 Entry)":
            st.write("Weeks requiring double picks in the High Roller Survivor Contest: :green[None]")
            st.write(f"System is defaulting to require double picks in the following weeks: {st.session_state.current_config['weeks_two_picks']}")
        elif subcontest == "Week 9 Bloody Survivor ($100 Entry)":
            st.write("Weeks requiring :red[TRIPLE] picks in the Bloody Survivor Contest: :green[9, 10, 11, 12, 13, 14, 15, 16, 17, 18]")
            st.write(f"System is defaulting to require triple picks in the following weeks: {st.session_state.current_config['weeks_three_picks']}")

    st.write('---')

    # --- B. Picked Teams (Prohibited Picks) ---
    st.subheader('Picked Teams / Season-Long Prohibitions')
    st.checkbox(
        'Have you already used any teams OR want to prohibit teams for the entire season?',
        key='has_picked_teams_widget', # Use key for simple boolean
        value=st.session_state.current_config['has_picked_teams'],
        on_change=update_config_value,
        args=('has_picked_teams',)
    )

    if st.session_state.current_config['has_picked_teams']:
        selected_teams_help_text = 'Select teams already used or teams you NEVER want picked.'
        st.multiselect(
            "Season Prohibited Picks:",
            options=nfl_teams,
            key='prohibited_picks_widget',
            default=st.session_state.current_config['prohibited_picks'],
            on_change=update_config_value,
            args=('prohibited_picks',),
            help=selected_teams_help_text
        )
        # Display selected teams
        picked_teams = st.session_state.current_config['prohibited_picks']
        if picked_teams:
            st.write("Currently Prohibited:", ", ".join(picked_teams))
        else:
            st.write("No teams selected for season-long prohibition.")

    st.write('---')


    # --- C. Remaining Weeks ---
    st.subheader('Remaining Weeks')
    st.checkbox(
        'Choose a specific range of weeks (instead of the entire season)?',
        key='choose_weeks_widget',
        value=st.session_state.current_config['choose_weeks'],
        on_change=update_config_value,
        args=('choose_weeks',)
    )

    # Get potentially updated values from config dict
    choose_weeks = st.session_state.current_config['choose_weeks']
    current_start_week = st.session_state.current_config['starting_week']
    current_end_week = st.session_state.current_config['ending_week'] # This is exclusive end
    current_contest = st.session_state.current_config['selected_contest']

    if choose_weeks:
        max_week = 21 if current_contest == 'Circa' else 19
        week_options = range(1, max_week)
        
        # Ensure start week is valid
        if current_start_week not in week_options:
            current_start_week = week_options[0]
            st.session_state.current_config['starting_week'] = current_start_week

        # Starting Week
        help_text = "..." # Your help text
        st.selectbox(
            "Select Starting Week:",
            options=week_options,
            key='starting_week_widget',
            index=week_options.index(current_start_week),
            on_change=update_config_value,
            args=('starting_week',),
            help=help_text
        )
        
        # Update current_start_week immediately for ending week options
        current_start_week = st.session_state.current_config['starting_week']

        # Ending Week
        ending_week_options = range(current_start_week, max_week)
        current_end_week_display = current_end_week - 1 # Value to display/select

        # Ensure end week is valid
        if current_end_week_display not in ending_week_options:
             current_end_week_display = ending_week_options[0]
             st.session_state.current_config['ending_week'] = current_end_week_display + 1

        st.selectbox(
            "Select Ending Week (Inclusive):",
            options=ending_week_options,
            key='ending_week_display_widget', # Use a different key
            index=ending_week_options.index(current_end_week_display),
            # Use a lambda or dedicated function to update the 'ending_week' (exclusive) value
            on_change=lambda: st.session_state.current_config.update({'ending_week': st.session_state.ending_week_display_widget + 1}),
            help="Select the last week you want the algorithm to run for."
        )

    # Display the selected range
    start_disp = st.session_state.current_config['starting_week']
    end_disp = st.session_state.current_config['ending_week'] - 1
    st.write(f"Algorithm will run from week **{start_disp}** to week **{end_disp}**.")
    st.write('---')


    # --- D. Current Week Entries ---
    st.subheader('Current Week Entries')
    current_week_entries_help_text = 'Input remaining entries. Use -1 to estimate automatically.'
    st.number_input(
        "Number of Remaining Entries (-1 for Auto):",
        min_value=-1,
        step=1,
        key='current_week_entries_widget',
        value=st.session_state.current_config['current_week_entries'],
        on_change=update_config_value,
        args=('current_week_entries',),
        help=current_week_entries_help_text
    )

    config = st.session_state.current_config # Use the latest config after user input
    current_year_pick_data = '2025_survivor_picks.csv'
    current_entries_value = config.get('current_week_entries')
    if current_entries_value != -1:
        st.write(f"Entered: {current_entries_value}")
        st.write('---')
    # Only execute if Contest is 'Circa' AND user has left the value as the sentinel '-1'
    elif config.get('selected_contest') == 'Circa' and current_entries_value == -1:       
        # 1. Run the dynamic calculation
        alive_count = calculate_alive_entries(current_year_pick_data, config)
        # 2. Overwrite the -1 flag in the config with the calculated value
        config['current_week_entries'] = alive_count
        
        # 3. Display confirmation and the calculated value
        st.write(f"Entered: -1")
        st.success(f"Automatically Calculated:! **Total ALIVE Circa Entries: {alive_count}**")
        st.caption("The entered value of -1 has been automatically replaced for calculations.")
    elif config.get('selected_contest') == 'Circa' and current_entries_value > 0:
        # Optional: Confirm the manually entered number when it's Circa
        st.info(f"Using manually entered Circa entries: **{current_entries_value}**.")
    else:
        st.write(f"Using default value for {config.get('selected_contest')}")


    # --- E. Current Week Team Availability ---
    st.subheader('Current Week Team Availability')
    if current_contest in ['Circa', 'Splash Sports']:
        st.checkbox(
            'Use live contest data to estimate availability?',
            key='use_live_data_widget',
            value=st.session_state.current_config['use_live_data'],
            on_change=update_config_value,
            args=('use_live_data',) 
        )

    nfl_teams = list(TEAM_NAME_TO_ABBR.keys())

    live_availability_data = None
    show_live_data = False
    
    # 2. Calculate live data early if needed
    if st.session_state.current_config['use_live_data']:
        # Calculate the team availability (DataFrame)
        team_availability_df = calculate_team_availability("Circa_historical_data.csv", "2025_survivor_picks.csv", config)
        if 'Team' in team_availability_df.columns:
            team_availability_df = team_availability_df.set_index('Team', drop=True)
        # CONVERSION FIX: Use the correct column name 'Availability_Percent'
        live_availability_data = team_availability_df['Availability_Percent'].to_dict() 
        show_live_data = True
        # 3. Inject Live Data into Session State for "Auto" teams
        # This ensures that when the user toggles to 'provide_availability', 
        # the sliders are initialized with the live data unless they were previously overridden.
        for abbr, live_value in live_availability_data.items():
            
            # --- FIX APPLIED HERE ---
            # Retrieve the current value. It might be stored as a string if loaded from JSON.
            raw_config_value = st.session_state.current_config['team_availabilities'].get(abbr, -1.0)
            
            # Attempt to safely convert the value to a float for comparison.
            try:
                current_config_value = float(raw_config_value)
            except (ValueError, TypeError):
                current_config_value = -1.0 # Default to auto if value is corrupted
                
            # Only inject the live value if the user hasn't set a custom value yet (i.e., it's still Auto)
            if current_config_value < 0:
                 # CRITICAL FIX: Ensure the injected value is also a float
                 try:
                     st.session_state.current_config['team_availabilities'][abbr] = float(live_value)
                 except (ValueError, TypeError):
                     st.session_state.current_config['team_availabilities'][abbr] = -1.0 # Fallback on bad data
    
    # Display the calculated live data and the checkbox for override
    if show_live_data:
        st.write(f"**Week {current_start_week} Team Availability (Live)**")
        # We display the *calculated* DataFrame here, which shows the source of the data
        st.dataframe(team_availability_df)
        
        st.checkbox(
            "Provide your own estimates for this week's availability for each team?",
            key='provide_availability_widget',
            value=st.session_state.current_config['provide_availability'],
            on_change=update_config_value,
            args=('provide_availability',)
        )
    else: # use_live_data == False
        st.checkbox(
            "Provide your own estimates for this week's availability for each team?",
            key='provide_availability_widget',
            value=st.session_state.current_config['provide_availability'],
            on_change=update_config_value,
            args=('provide_availability',)
        )
    
    # --- Manual Availability Sliders ---
    if st.session_state.current_config['provide_availability']:
        st.write("Set availability % (0-100). Use -1 to estimate automatically.")
        
        # Use columns for better layout
        num_cols = 2
        cols = st.columns(num_cols)
        col_idx = 0
        
        for team in nfl_teams:
            # 4. Use the abbreviation for the internal key lookup
#            team_abbr = TEAM_NAME_TO_ABBR[team]
            
            with cols[col_idx]:
                outer_key = 'team_availabilities'
                inner_key = team#_abbr # IMPORTANT: Use the abbreviation here
                widget_key = f"{outer_key}_{inner_key}_widget".replace(" ", "_")
                
                # --- FIX APPLIED HERE ---
                # Get current value from the config 
                current_val_raw = st.session_state.current_config[outer_key].get(inner_key, -1.0)
                
                # CRITICAL FIX: Ensure the value is a float before comparison
                try:
                    current_val_float = float(current_val_raw)
                except (ValueError, TypeError):
                    current_val_float = -1.0 
                
                # Convert float value (-1.0 to 1.0) to integer slider value (-1 to 100)
                if current_val_float < 0: # Comparison now safe
                    current_val_int = -1
                else:
                    current_val_int = int(current_val_float * 100) # Convert to integer for slider (0..100)
        
                st.slider(
                    f"{team}:",
                    min_value=-1,
                    max_value=100,
                    key=widget_key,
                    value=current_val_int,
                    on_change=update_nested_value,
                    args=(outer_key, inner_key)
                )
                
                # --- FIX APPLIED HERE ---
                # Display current setting from the dictionary
                display_val_raw = st.session_state.current_config[outer_key].get(inner_key, -1.0)

                # CRITICAL FIX: Ensure the value is a float before comparison/display formatting
                try:
                    display_val = float(display_val_raw)
                except (ValueError, TypeError):
                    display_val = -1.0 

                if display_val < 0: # Comparison now safe
                     st.caption(":red[Auto]")
                else:
                     st.caption(f":green[{display_val*100:.0f}%]")
    
            col_idx = (col_idx + 1) % num_cols

    # --- F. Required Team Picks ---
    st.subheader('Required Weekly Picks')
    st.checkbox(
        'Require a specific team to be used in a specific week?',
        key='require_team_widget',
        value=st.session_state.current_config['require_team'],
        on_change=update_config_value,
        args=('require_team',)
    )

    if st.session_state.current_config['require_team']:
        with st.expander("Set Required Picks"):
            st.write("Select the week (0 = No Requirement).")
            # Determine valid week options based on current start/end week
            start_w = st.session_state.current_config['starting_week']
            end_w = st.session_state.current_config['ending_week'] # exclusive
            required_week_options = [0] + list(range(start_w, end_w))
            
            req_cols = st.columns(3)
            req_col_idx = 0
            for team in nfl_teams:
                with req_cols[req_col_idx]:
                    outer_key = 'required_weeks'
                    inner_key = team
                    widget_key = f"{outer_key}_{inner_key}_widget".replace(" ", "_")
                    current_val = st.session_state.current_config[outer_key].get(inner_key, 0)
                    
                    # Ensure current value is in options, default to 0 if not
                    if current_val not in required_week_options:
                        current_val = 0
                        st.session_state.current_config[outer_key][inner_key] = 0

                    st.selectbox(
                        f"{team} Req Week:",
                        options=required_week_options,
                        key=widget_key,
                        index=required_week_options.index(current_val),
                        on_change=update_nested_value,
                        args=(outer_key, inner_key)
                    )
                req_col_idx = (req_col_idx + 1) % 3
    st.write('---')


    # --- G. Prohibited Weekly Picks ---
    st.subheader('Prohibited Weekly Picks')
    st.checkbox(
        'Prohibit specific teams from being picked in specific weeks?',
        key='prohibit_teams_widget',
        value=st.session_state.current_config['prohibit_teams'],
        on_change=update_config_value,
        args=('prohibit_teams',)
    )

    if st.session_state.current_config['prohibit_teams']:
        with st.expander("Set Prohibited Picks"):
            st.write("Select weeks where the team CANNOT be picked.")
            start_w = st.session_state.current_config['starting_week']
            end_w = st.session_state.current_config['ending_week'] # exclusive
            prohibited_week_options = list(range(start_w, end_w))

            pro_cols = st.columns(2) # Fewer columns for multiselect
            pro_col_idx = 0
            for team in nfl_teams:
                with pro_cols[pro_col_idx]:
                    outer_key = 'prohibited_weeks'
                    inner_key = team
                    widget_key = f"{outer_key}_{inner_key}_widget".replace(" ", "_")
                    current_val = st.session_state.current_config[outer_key].get(inner_key, [])
                    
                    # Filter out invalid weeks from current value
                    current_val = [w for w in current_val if w in prohibited_week_options]
                    st.session_state.current_config[outer_key][inner_key] = current_val

                    st.multiselect(
                        f"{team} Prohibited Wks:",
                        options=prohibited_week_options,
                        key=widget_key,
                        default=current_val,
                        on_change=update_nested_value,
                        args=(outer_key, inner_key)
                    )
                pro_col_idx = (pro_col_idx + 1) % 2
    st.write('---')


    # --- H. Team Rankings ---
    st.subheader('NFL Team Rankings')
    st.checkbox(
        'Use customized rankings instead of default rankings?',
        key='custom_rankings_widget',
        value=st.session_state.current_config['custom_rankings'],
        on_change=update_config_value,
        args=('custom_rankings',)
    )

    if st.session_state.current_config['custom_rankings']:
        with st.expander("Set Custom Team Rankings"):
            st.write("Ranking = Expected points vs. average team on neutral field. Select 'Default' to use default.")
            team_rankings_options = ["Default"] + [i / 2.0 for i in range(-30, 31)] # -15 to 15 in 0.5 steps

            rank_cols = st.columns(3)
            rank_col_idx = 0
            for team in nfl_teams:
                 with rank_cols[rank_col_idx]:
                    outer_key = 'team_rankings'
                    inner_key = team
                    widget_key = f"{outer_key}_{inner_key}_widget".replace(" ", "_")
                    current_val = st.session_state.current_config[outer_key].get(inner_key, 'Default')
                    
                    # Ensure value is valid
                    if current_val not in team_rankings_options:
                        current_val = 'Default'

                    st.selectbox(
                        f"{team} Rank:",
                        options=team_rankings_options,
                        key=widget_key,
                        index=team_rankings_options.index(current_val),
                        on_change=update_nested_value,
                        args=(outer_key, inner_key)
                    )
                    # Display the effective rank
                    effective_rank = current_val if current_val != 'Default' else DEFAULT_RANKS.get(team, 0)
                    st.caption(f"Effective: :green[{effective_rank}]")

                 rank_col_idx = (rank_col_idx + 1) % 3
    st.write('---')

    # --- I. Pick Must Be Favored ---
    st.subheader('Pick Exclusively Favorites?')
    st.checkbox(
        'All teams picked must be favored?',
        key='must_be_favored_widget',
        value=st.session_state.current_config['must_be_favored'],
        on_change=update_config_value,
        args=('must_be_favored',)
    )

    if st.session_state.current_config['must_be_favored']:
        must_be_favored_options = ['Live Sportsbook Odds (If Available)', 'Internal Rankings', 'Both Live Sportsbook Odds and Internal Rankings']
        current_qualifier = st.session_state.current_config['favored_qualifier']
        if current_qualifier not in must_be_favored_options:
            current_qualifier = must_be_favored_options[0]

        st.selectbox(
            'What qualifies a team as favored?',
            options=must_be_favored_options,
            key='favored_qualifier_widget',
            index=must_be_favored_options.index(current_qualifier),
            on_change=update_config_value,
            args=('favored_qualifier',),
            help="How to determine if a team is favored."
        )
    st.write('---')


    # --- J. Constraints ---
    st.subheader('Select Constraints')
    st.checkbox(
        'Add scheduling/travel/situational constraints?',
        key='add_constraints_widget',
        value=st.session_state.current_config['add_constraints'],
        on_change=update_config_value,
        args=('add_constraints',)
    )

    if st.session_state.current_config['add_constraints']:
        with st.expander("Configure Constraints"):
            spread_options = [i / 2.0 for i in range(1, 21)] # 0.5 to 10.0
            
            # Simple Checkbox Constraints (update directly in dictionary)
            constraints_simple = {
                'avoid_away_short_rest': 'Avoid Away Teams on Short Rest',
                'avoid_away_divisional': 'Avoid AWAY Divisional Matchups',
                'avoid_3_in_10': 'Avoid 3 games in 10 days',
                'avoid_4_in_17': 'Avoid 4 games in 17 days',
                'avoid_cumulative_rest': 'Avoid Cumulative Rest Disadvantage',
                'avoid_thursday': 'Avoid ALL TEAMS in Thursday Night Games',
                'avoid_away_thursday': 'Avoid ONLY AWAY TEAMS in Thursday Night Games',
                'avoid_b2b_away': 'Avoid Teams on Back to Back Away Games',
                'avoid_international': 'Avoid International Games',
                'avoid_weekly_rest_disadvantage': 'Avoid Teams with Rest Disadvantage',
                'avoid_travel_disadvantage': 'Avoid Teams with Travel Disadvantage'
            }
            for key, label in constraints_simple.items():
                 st.checkbox(label, key=f"{key}_widget", value=st.session_state.current_config[key],
                             on_change=update_config_value, args=(key,))

            # Constraints needing selectboxes
            st.checkbox(
                'Avoid Close Divisional Matchups', key='avoid_close_divisional_widget',
                value=st.session_state.current_config['avoid_close_divisional'],
                on_change=update_config_value, args=('avoid_close_divisional',)
            )
            if st.session_state.current_config['avoid_close_divisional']:
                current_div_spread = st.session_state.current_config['min_div_spread']
                st.selectbox(
                    'Min spread to NOT consider "close" for divisional:', spread_options,
                    key='min_div_spread_widget', index=spread_options.index(current_div_spread),
                    on_change=update_config_value, args=('min_div_spread',)
                )

            st.checkbox(
                'Avoid Away Teams in Close Games', key='avoid_away_close_widget',
                value=st.session_state.current_config['avoid_away_close'],
                on_change=update_config_value, args=('avoid_away_close',)
            )
            if st.session_state.current_config['avoid_away_close']:
                current_away_spread = st.session_state.current_config['min_away_spread']
                st.selectbox(
                    'Min spread to NOT consider "close" for away games:', spread_options,
                    key='min_away_spread_widget', index=spread_options.index(current_away_spread),
                    on_change=update_config_value, args=('min_away_spread',)
                )

            # Bayesian/Rest Constraint
            bayesian_options = [
                "No Rest, Bayesian, and Travel Constraints",
                "Selected team must have been projected to win based on preseason rankings, current rankings, and with and without travel/rest adjustments",
                "Selected team must be projected to win with and without travel and rest impact based on current rankings",
                "Selected team must have been projected to win based on preseason rankings in addition to current rankings",
            ]
            current_bayes = st.session_state.current_config['bayesian_constraint']
            st.selectbox(
                'Bayesian, Rest, and Travel Impact:', options=bayesian_options,
                key='bayesian_constraint_widget', index=bayesian_options.index(current_bayes),
                on_change=update_config_value, args=('bayesian_constraint',),
                help="Apply constraints based on consistency across ranking/adjustment methods."
            )
    st.write('---')


    # --- K. Estimate Pick Percentages ---
    st.subheader('Estimate Pick Percentages')
    st.checkbox(
        'Add custom estimated pick percentages (Overrides automatic estimation)?',
        key='custom_pick_percentages_widget',
        value=st.session_state.current_config['custom_pick_percentages'],
        on_change=update_config_value,
        args=('custom_pick_percentages',)
    )

    if st.session_state.current_config['custom_pick_percentages']:
        with st.expander("Set Custom Pick Percentages"):
            st.write("Set pick % (0-100). Use -1 to use automatic estimation for that specific team/week.")
            start_w = st.session_state.current_config['starting_week']
            end_w = st.session_state.current_config['ending_week']  # exclusive
    
            for week in range(start_w, end_w):
                week_key = f"week_{week}"
                # Use an expander per week (no 'key' argument; label is unique per week)
                with st.expander(f"Week {week} Custom Pick %", expanded=False):
                    st.markdown(f"**Week {week} Custom Pick %**")
                    perc_cols = st.columns(3)
                    perc_col_idx = 0
    
                    for team in nfl_teams:
                        with perc_cols[perc_col_idx]:
                            outer_key = 'pick_percentages'
                            # Ensure week dict exists before accessing team
                            if week_key not in st.session_state.current_config[outer_key]:
                                st.session_state.current_config[outer_key][week_key] = {}
    
                            inner_key = team
                            widget_key = f"pick_perc_{week_key}_{inner_key}_widget".replace(" ", "_")
    
                            current_val_float = st.session_state.current_config[outer_key].get(week_key, {}).get(inner_key, -1.0)
                            # Map sentinel (<0) to slider -1, otherwise convert to 0..100 integer
                            if current_val_float is None or current_val_float < 0:
                                current_val_int = -1
                            else:
                                current_val_int = int(current_val_float * 100)
    
                            st.slider(
                                f"{team} Wk {week} %:",
                                min_value=-1,
                                max_value=100,
                                key=widget_key,
                                value=current_val_int,
                                on_change=update_pick_percentage,
                                args=(week, inner_key)
                            )
                            # Display current setting
                            display_val = st.session_state.current_config[outer_key].get(week_key, {}).get(inner_key, -1.0)
                            if display_val < 0:
                                st.caption(":red[Auto]")
                            else:
                                st.caption(f":green[{display_val*100:.0f}%]")
    
                        perc_col_idx = (perc_col_idx + 1) % 3
    
                    st.write("---")  # Separator between weeks
    st.write('---')


    # --- L. Get Optimized Picks ---
    st.subheader('Get Optimized Survivor Picks')
    number_of_solutions_options = [1, 5, 10, 25, 50, 100]
    current_num_solutions = st.session_state.current_config['number_solutions']
    st.selectbox(
        'Number of Solutions per Method:',
        options=number_of_solutions_options,
        key='number_solutions_widget',
        index=number_of_solutions_options.index(current_num_solutions),
        on_change=update_config_value,
        args=('number_solutions',),
        help="How many top solutions to generate (e.g., 10 EV-based, 10 Ranking-based)."
    )

    with st.expander("More Information"):
        num_sol_display = st.session_state.current_config['number_solutions']
        st.write(f"""This button will find the best picks for each week. It will pump out :green[{num_sol_display * 2} solutions].
	\n- The first {num_sol_display} solutions will be :red[based purely on EV] that is a complicated formula based on their predicted pick percentage of each team in each week, and each team's chances of winning that week.
 	\n- This will use the rankings defined above (or within our system) to determine win probability and thus pick percentage for each team.
  	\n- The remaining {num_sol_display} solutions will be based on the :red[rankings and constraints you provided]. 
  	\n- This method finds the teams that are just straight up most likely to win. While helpful, thsi does not provide you with much of a competitive advantage.
   	\n- All solutions will abide by the constraints you've provided
   	\n- If you have too many constraints, or the solution is impossible, you will see an error
       	\n- :green[Mathematically, EV is most likely to win. However, using your own rankings has advantages as well, which is why we provide both solutions (Sometimes it's just preposterous to pick the Jets)]
        """)

    st.write('')
    if st.button("🚀 Get Optimized Survivor Picks", type="primary"):
        # --- Trigger Backend Logic ---
        st.write("Starting Optimization Process...")
        
        # Get all config settings once
        config = st.session_state.current_config
        start_w = config['starting_week']
        end_w = config['ending_week']
        selected_c = config['selected_contest']
        use_cache = False 
        num_sol = config['number_solutions']
        
        # --- Run your steps ---
        
        # Step 1: Get Schedule
        st.write("Step 1/6: Fetching Schedule Data...")
        # Pass config for get_schedule as it might need it (though it doesn't currently)
        schedule_table, schedule_rows = get_schedule(config) 
        st.write("Step 1 Completed.")
        
        # Step 2: Collect Data
        st.write("Step 2/6: Collecting Travel, Ranking, Odds, Rest Data...")
        # --- FIX: Pass schedule_rows to this function ---
        collect_schedule_travel_ranking_data_df = collect_schedule_travel_ranking_data(pd, config, schedule_rows)
        st.write("Step 2 Completed.")

        # Step 3: Predict Pick % (Preliminary)
        st.write("Step 3/6: Predicting Pick Percentages & Calculating Availability...")
        # --- Pass the dataframe from Step 2 into this function ---
        nfl_schedule_pick_percentages_df = get_predicted_pick_percentages(config, collect_schedule_travel_ranking_data_df)

        st.write("Step 3a Completed (Availability Calculated).")
#        st.write(nfl_schedule_pick_percentages_df)
        
        # Step 4: Calculate EV
        st.write("Step 4/6: Calculating Live Expected Value...")
        with st.spinner('Calculating EV... (This may take 5-10 minutes)'):
            # Pass the dataframe from Step 3 into this function
            full_df_with_ev = calculate_ev(nfl_schedule_pick_percentages_df, config, use_cache)
        st.write("Step 4 Completed.")
#        st.dataframe(full_df_with_ev) 

        # Step 4b: Reformat
        st.write("Reformatting Data for Solver...")
        reformatted_df = reformat_df(full_df_with_ev, config)
        st.write("Reformatting Complete. Full Dataset:")
        st.dataframe(reformatted_df)

        # Step 5 & 6: Run Solvers
        st.write('Step 5/6: Calculating Best Picks Based on EV...')
        # --- FIX: Pass reformatted_df to the solver ---
        get_survivor_picks_based_on_ev(reformatted_df, config, num_sol)
        st.write('Step 5 Completed.')
        
        st.write("---")
        st.write('Step 6/6: Calculating Best Picks Based on Rankings...')
        # --- FIX: Pass reformatted_df to the solver ---
        get_survivor_picks_based_on_internal_rankings(reformatted_df, config, num_sol)
        st.write('Step 6 Completed.')
        
        st.success("Optimization Complete!")

