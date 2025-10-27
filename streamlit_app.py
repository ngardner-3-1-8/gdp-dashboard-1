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

# --------------------------------------------------------------------------
# --- 1. DEFAULT TEAM RANKS (Baseline) ---
# Used if the user selects 'Default' in the UI for their current rank setting.
# --------------------------------------------------------------------------
DEFAULT_RANKS = {
    'Arizona Cardinals': -1.2,
    'Atlanta Falcons': -0.59,
    'Baltimore Ravens': -2.5,
    'Buffalo Bills': 3.54,
    'Carolina Panthers': -3.37,
    'Chicago Bears': -1.91,
    'Cincinnati Bengals': -3.73,
    'Cleveland Browns': -7.53,
    'Dallas Cowboys': 0.13,
    'Denver Broncos': 2.34,
    'Detroit Lions': 6.34,
    'Green Bay Packers': 5.71,
    'Houston Texans': 1.92,
    'Indianapolis Colts': 2.83,
    'Jacksonville Jaguars': -1.29,
    'Kansas City Chiefs': 4.75,
    'Las Vegas Raiders': -3.93,
    'Los Angeles Chargers': 0.29,
    'Los Angeles Rams': 5.1,
    'Miami Dolphins': -3.77,
    'Minnesota Vikings': 0.03,
    'New England Patriots': -0.31,
    'New Orleans Saints': -6.45,
    'New York Giants': -5.4,
    'New York Jets': -3.17,
    'Philadelphia Eagles': 4.7,
    'Pittsburgh Steelers': 1.48,
    'San Francisco 49ers': 3.32,
    'Seattle Seahawks': 5.25,
    'Tampa Bay Buccaneers': 1.79,
    'Tennessee Titans': -6.88,
    'Washington Commanders': 3.0
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
    start_date_str = 'September 3, 2025'
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
                away_rest_days = 'NA'
            if home_team in last_game:
                home_rest_days = (last_date - last_game[home_team]).days
            else:
                home_rest_days = 'NA'   
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
        df.loc[df['Date'] >= pd.to_datetime('2025-11-29'), 'Week'] += 1
        df.loc[df['Date'] >= pd.to_datetime('2025-12-26'), 'Week'] += 1
        df.loc[df['Date'] >= pd.to_datetime('2025-11-29'), 'Week_Num'] += 1
        df.loc[df['Date'] >= pd.to_datetime('2025-12-26'), 'Week_Num'] += 1

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
    df["Thursday Night Game"] = df.apply(lambda row: 'True' if (row['Date'].weekday() == 3) and (row['Date'] != pd.to_datetime('2024-11-28')) and (row['Date'] != pd.to_datetime('2024-12-26'))  else row["Thursday Night Game"], axis =1)


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
    df['Away Team Weekly Rest'] = df['Away Team Weekly Rest'].replace('NA', pd.NA)
    df['Home Team Weekly Rest'] = df['Home Team Weekly Rest'].replace('NA', pd.NA)

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
    
    def get_api_nfl_odds(api_key):
        # API key and parameters
        SPORT = 'americanfootball_nfl'
        # Fetch odds from US bookmakers
        REGIONS = 'us'
        # Request moneyline odds
        MARKETS = 'h2h,spreads'
        # Use decimal odds format for easier averaging
        ODDS_FORMAT = 'decimal'
        # Request ISO format for dates, which includes UTC timezone info
        DATE_FORMAT = 'iso'
    
        url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={api_key}&regions={REGIONS}&markets={MARKETS}&oddsFormat={ODDS_FORMAT}&dateFormat={DATE_FORMAT}'
    
        # Make the GET request
        response = requests.get(url)
    
        # Handle potential errors
        if response.status_code != 200:
            print(f'Failed to get odds from The Odds API: status_code {response.status_code}, response body {response.text}')
            return pd.DataFrame() # Return an empty DataFrame on error
    
        odds_data = response.json()
    
        # Prepare a list to hold data for the DataFrame
        formatted_games = []
    
        # Define the Eastern Timezone (for formatting like your original code)
        eastern_tz = pytz.timezone('America/New_York')
    
        for event in odds_data:
            game_id = event['id']
            home_team = event['home_team']
            away_team = event['away_team']
            
            # Convert commence_time to datetime object and then to Eastern Time
            # The API returns UTC, so localize it as UTC first, then convert
            utc_commence_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            eastern_commence_time = utc_commence_time.astimezone(eastern_tz)
            
            # Format the time for display like your original code ('8:20 PM ET')
            # %I: Hour (12-hour clock) as a zero-padded decimal number [01, 12]
            # %M: Minute as a zero-padded decimal number [00, 59]
            # %p: Locale's equivalent of either AM or PM
            formatted_time = eastern_commence_time.strftime('%I:%M %p ET').replace('AM ET', 'am').replace('PM ET', 'pm').lstrip('0')
            # Removing leading zero for hours for single-digit hours, e.g., '8:20 pm' instead of '08:20 pm'
            # Your original example showed "8:20 PM ET", which implies a leading zero was present in original text but might be removed in the final format.
            # This formatting is to match your original output as closely as possible.
    
            # Create a temporary dictionary to store odds for this game
            game_odds_avg = {'home': [], 'away': [], 'home_spread': [], 'away_spread': []}
            
            for bookmaker in event['bookmakers']:
                for market in bookmaker['markets']:
                    if market['key'] == 'h2h':  # Moneyline market
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team:
                                game_odds_avg['home'].append(outcome['price'])
                            elif outcome['name'] == away_team:
                                game_odds_avg['away'].append(outcome['price'])
                # 2. ADDITION: New logic for spreads
                    elif market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team:
                                game_odds_avg['home_spread'].append(outcome['point'])
                            elif outcome['name'] == away_team:
                                game_odds_avg['away_spread'].append(outcome['point'])
    
            # Calculate average odds (if available)
            avg_home_odds = sum(game_odds_avg['home']) / len(game_odds_avg['home']) if game_odds_avg['home'] else None
            avg_away_odds = sum(game_odds_avg['away']) / len(game_odds_avg['away']) if game_odds_avg['away'] else None

            avg_home_spread = sum(game_odds_avg['home_spread']) / len(game_odds_avg['home_spread']) if game_odds_avg['home_spread'] else None
            avg_away_spread = sum(game_odds_avg['away_spread']) / len(game_odds_avg['away_spread']) if game_odds_avg['away_spread'] else None
    
            # Convert decimal odds to American odds format
            # If decimal odds are >= 2.0, American odds = (decimal odds - 1) * 100
            # If decimal odds are < 2.0, American odds = -100 / (decimal odds - 1)
            # Use round() to get integer odds like your original output
    
            american_home_odds = None
            if avg_home_odds:
                if avg_home_odds >= 2.0:
                    american_home_odds = round((avg_home_odds - 1) * 100)
                else:
                    american_home_odds = round(-100 / (avg_home_odds - 1))
    
            american_away_odds = None
            if avg_away_odds:
                if avg_away_odds >= 2.0:
                    american_away_odds = round((avg_away_odds - 1) * 100)
                else:
                    american_away_odds = round(-100 / (avg_away_odds - 1))
    
            formatted_games.append({
                'Time': formatted_time,
                'Away Team': away_team,
                'Away Odds': american_away_odds,
                'Home Team': home_team,
                'Home Odds': american_home_odds,
				'Away Spread': avg_away_spread,
				'Home Spread': avg_home_spread
            })
    
        # Create pandas DataFrame from the extracted data
        live_api_odds_nfl_df = pd.DataFrame(formatted_games)
        return live_api_odds_nfl_df
    
    # Example usage (replace 'YOUR_API_KEY' with your actual API key)
    # Make sure to set your API_KEY environment variable or replace directly
    API_KEY = '34671f7aeaa8f4fbee2398163f2f45d3'  # Replace with your actual API key
    
    # Only run if API_KEY is set (for example usage)
    if API_KEY != 'YOUR_API_KEY':
        live_api_odds_df = get_api_nfl_odds(API_KEY)
        print(live_api_odds_df)
        # You can save it to CSV as well
        # live_api_odds_df.to_csv('Live_API_Odds_NFL.csv', index=False)
    else:
        print("Please replace 'YOUR_API_KEY' with your actual API key to fetch data.")
    
    st.write("")
    st.write("")
    st.write("")
    st.subheader("Live Odds Aggregated from Multiple Sportsbooks")
    st.write(live_api_odds_df)
    st.write("")
    st.write("")
    st.write("")
	
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
        st.subheader('Games with Unavailable Live Odds')
        st.write('This dataframe contains the games where live odds from the Live Odds API were unavailable. This will likely happen for lookahead lines and future weeks')
        st.write(overridden_games_df)
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
        week_df["Away Team Star Rating"] = week_df["Away Team Cumulative Win Percentage"].apply(lambda x: calculate_star_rating(x, week))
        week_df["Home Team Star Rating"] = week_df["Home Team Cumulative Win Percentage"].apply(lambda x: calculate_star_rating(x, week))

        # Mark Thanksgiving Favorites
        # Find Week 13 games and winners
        week13_df = df[df["Week"] == "Week 13"]
        week13_winners = week13_df["Favorite"].unique()
	

        # Create new columns and mark Thanksgiving Favorites
        week_df["Away Team Thanksgiving Favorite"] = week_df.apply(
            lambda row: True
            if (1 <= int(row["Week"].replace("Week ", "")) <= 12) and (row["Away Team"] in week13_winners)
            else False,
            axis=1,
        )

        week_df["Home Team Thanksgiving Favorite"] = week_df.apply(
            lambda row: True
            if (1 <= int(row["Week"].replace("Week ", "")) <= 12) and (row["Home Team"] in week13_winners)
            else False,
            axis=1,
        )

        # Mark Christmas Favorites
        # Find Week 18 games and winners
        week18_df = df[df["Week"] == "Week 18"]
        week18_winners = week18_df["Favorite"].unique()

        # Create new columns and mark Thanksgiving Favorites
        week_df["Away Team Christmas Favorite"] = week_df.apply(
            lambda row: True
            if (1 <= int(row["Week"].replace("Week ", "")) <= 17) and (row["Away Team"] in week18_winners)
            else False,
            axis=1,
        )
        week_df["Home Team Christmas Favorite"] = week_df.apply(
            lambda row: True
            if (1 <= int(row["Week"].replace("Week ", "")) <= 17) and (row["Home Team"] in week18_winners)
            else False,
            axis=1,
        )
		
        week13_df = df[df["Week"] == "Week 13"]
        week13_winners = week13_df["Underdog"].unique()
	

        # Create new columns and mark Thanksgiving Favorites
        week_df["Away Team Thanksgiving Underdog"] = week_df.apply(
            lambda row: True
            if (1 <= int(row["Week"].replace("Week ", "")) <= 12) and (row["Away Team"] in week13_winners)
            else False,
            axis=1,
        )

        week_df["Home Team Thanksgiving Underdog"] = week_df.apply(
            lambda row: True
            if (1 <= int(row["Week"].replace("Week ", "")) <= 12) and (row["Home Team"] in week13_winners)
            else False,
            axis=1,
        )

        # Mark Christmas Favorites
        # Find Week 18 games and winners
        week18_df = df[df["Week"] == "Week 18"]
        week18_winners = week18_df["Underdog"].unique()

        # Create new columns and mark Thanksgiving Favorites
        week_df["Away Team Christmas Underdog"] = week_df.apply(
            lambda row: True
            if (1 <= int(row["Week"].replace("Week ", "")) <= 17) and (row["Away Team"] in week18_winners)
            else False,
            axis=1,
        )
        week_df["Home Team Christmas Underdog"] = week_df.apply(
            lambda row: True
            if (1 <= int(row["Week"].replace("Week ", "")) <= 17) and (row["Home Team"] in week18_winners)
            else False,
            axis=1,
        )
        consolidated_df = pd.concat([consolidated_df, week_df])

    # Create the 'Divisional Matchup Boolean' column
    consolidated_df["Divisional Matchup Boolean"] = 0

    # Set values based on 'Divisional Matchup?' column
    consolidated_df.loc[consolidated_df["Divisional Matchup?"] == True, "Divisional Matchup Boolean"] = 1

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
		
def get_predicted_pick_percentages(pd, config: dict, schedule_df: pd.DataFrame):
    # Add these definitions near the top of the function
    circa_total_entries = 20000 # Or your default
    splash_total_entries = 5000 # Or your default
    dk_total_entries = 25000 # Or your default
    selected_contest = config['selected_contest']
    starting_week = config['starting_week']
    current_week_entries = config['current_week_entries']
    week_requiring_two_selections = config.get('weeks_two_picks', [])
    team_availability = config.get('team_availabilities', {})
    custom_pick_percentages = config.get('pick_percentages', {})
    # Load your historical data (replace 'historical_pick_data_FV_circa.csv' with your actual file path)
    if selected_contest == 'Circa':
        df = pd.read_csv('Circa_historical_data.csv')
    elif selected_contest == 'Splash Sports':
        df = pd.read_csv('DK_historical_data.csv')
    else:
        df = pd.read_csv('DK_historical_data.csv')
    df.rename(columns={"Week": "Date"}, inplace=True)
    # Remove percentage sign and convert to float
    #df['Win %'] = df['Win %'].str.rstrip('%').astype(float, -1, 100) / 100
    #df['Pick %'] = df['Pick %'].str.rstrip('%').astype(float, -1, 100) / 100
    # Extract the numeric part (week number)
    #df['Week'] = df['Week'].str.extract(r'(\d+)').astype(int)
    #print(df['Date'])
    df['Pick %'].fillna(0.0, inplace=True)

    #print(df)
    # Split data into input features (X) and target variable (y)
    X = df[['Win %', 'Future Value (Stars)', 'Date', 'Away Team', 'Divisional Matchup?']]
    y = df['Pick %']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=50, random_state=0)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf_model.predict(X_test)

    # Evaluate model performance (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    #print(f"Mean Absolute Error: {mae:.2f}")

    # Read the CSV file into a DataFrame
    
    new_df = schedule_df.copy()

    # Create a new DataFrame with selected columns
    selected_columns = ['Week', 'Away Team', 'Home Team', 'Away Team Fair Odds',
                        'Home Team Fair Odds', 'Away Team Star Rating', 'Home Team Star Rating', 'Divisional Matchup Boolean', 'Away Team Thanksgiving Favorite', 'Home Team Thanksgiving Favorite', 'Away Team Christmas Favorite', 'Home Team Christmas Favorite']
    new_df = new_df[selected_columns]
    new_df['Week_Number'] = new_df['Week'].str.split(' ').str[1].astype(int)
    # Filter the DataFrame
    new_df = new_df[new_df['Week_Number'] >= starting_week]
    # You can drop the auxiliary 'Week_Number' column if you no longer need it
    new_df = new_df.drop(columns=['Week_Number'])

    # Read the original CSV file into a DataFrame
    #csv_path = 'nfl_Schedule_circa.csv'
    #df = pd.read_csv(csv_path)

    # Create the new DataFrame with modified column names
    away_df = new_df.rename(columns={
        'Week': 'Date',
        'Away Team': 'Team',
        'Home Team': 'Opponent',
        'Away Team Fair Odds': 'Win %',
        'Away Team Star Rating': 'Future Value (Stars)',
        'Divisional Matchup Boolean': 'Divisional Matchup?'
    })
    away_df['Year'] = 2025
    away_df['Home/Away'] = 'Away'
    away_df['Away Team'] = 1
    # Add the "Pick %" and "EV" columns (initially empty)
    away_df['Pick %'] = None
    away_df['EV'] = None

    # Drop the unwanted columns
    away_df.drop(columns=['Home Team Fair Odds', 'Home Team Star Rating', 'Home Team Thanksgiving Favorite', 'Home Team Christmas Favorite'], inplace=True)

    # Reorder the columns
    column_order = ['EV', 'Win %', 'Pick %', 'Team', 'Opponent', 'Future Value (Stars)', 'Year', 'Date', 'Home/Away', 'Away Team', 'Divisional Matchup?', 'Away Team Thanksgiving Favorite', 'Away Team Christmas Favorite']
    away_df = away_df[column_order]


    # Create the new DataFrame with modified column names
    home_df = new_df.rename(columns={
        'Week': 'Date',
        'Home Team': 'Team',
        'Away Team': 'Opponent',
        'Home Team Fair Odds': 'Win %',
        'Home Team Star Rating': 'Future Value (Stars)',
        'Divisional Matchup Boolean': 'Divisional Matchup?'
    })
    home_df['Year'] = 2025
    home_df['Home/Away'] = 'Home'
    home_df['Away Team'] = 0
    # Add the "Pick %" and "EV" columns (initially empty)
    home_df['Pick %'] = None
    home_df['EV'] = None

    # Drop the unwanted columns
    home_df.drop(columns=['Away Team Fair Odds', 'Away Team Star Rating', 'Away Team Thanksgiving Favorite', 'Away Team Christmas Favorite'], inplace=True)

    # Reorder the columns
    column_order = ['EV', 'Win %', 'Pick %', 'Team', 'Opponent', 'Future Value (Stars)', 'Year', 'Date', 'Home/Away', 'Away Team', 'Divisional Matchup?', 'Home Team Thanksgiving Favorite', 'Home Team Christmas Favorite']
    home_df = home_df[column_order]

    # Now `away_df` contains the desired columns with modified names
    #print(home_df)
    home_df['Date'] = home_df['Date'].str.extract(r'(\d+)').astype(int)
    away_df['Date'] = away_df['Date'].str.extract(r'(\d+)').astype(int)

    #print(home_df)
    #print(away_df)

    predictions = rf_model.predict(away_df[['Win %', 'Future Value (Stars)', 'Date', 'Away Team', 'Divisional Matchup?']])
    away_df['Pick %'] = predictions
    #away_df.to_csv('predicted_away_data_circa.csv', index=False)

    predictions = rf_model.predict(home_df[['Win %', 'Future Value (Stars)', 'Date', 'Away Team', 'Divisional Matchup?']])
    home_df['Pick %'] = predictions
    #home_df.to_csv('predicted_home_data_circa.csv', index=False)

    # Concatenate your DataFrames
    pick_predictions_df = pd.concat([away_df, home_df], ignore_index=True)
    
    # Function to calculate the adjusted "Pick %"
    def adjust_pick_percentage(row):
        """
        Calculates the adjusted 'Pick %' based on holiday favorite status 
        and then multiplies by 'Availability'.
        """
        original_pick_percent = row["Pick %"]
        pre_thanksgiving = row["Date"] < 13
        pre_christmas = row["Date"] < 18
        
        # Thanksgiving Adjustment
        # The original logic has two separate checks that compound if both teams
        # are checked, but the goal seems to be: if a team is a Thanksgiving favorite
        # AND it's NOT Thanksgiving (Date != 13), then apply the / 4 modification.
        if pre_thanksgiving:
            if (row["Home Team Thanksgiving Favorite"] or row["Away Team Thanksgiving Favorite"]) and (row["Home Team Christmas Favorite"] or row["Away Team Christmas Favorite"]):
                original_pick_percent = original_pick_percent / 6
            elif (not row["Home Team Thanksgiving Favorite"] and not row["Away Team Thanksgiving Favorite"]) and (row["Home Team Christmas Favorite"] or row["Away Team Christmas Favorite"]):
                original_pick_percent = original_pick_percent / 3
            elif (row["Home Team Thanksgiving Favorite"] or row["Away Team Thanksgiving Favorite"]) and (not row["Home Team Christmas Favorite"] and not row["Away Team Christmas Favorite"]):
                original_pick_percent = original_pick_percent / 2
        elif pre_christmas:
            if row["Home Team Christmas Favorite"] == row["Team"] or row["Away Team Christmas Favorite"] == row["Team"]:
                original_pick_percent = original_pick_percent / 4
    
        # Final adjustment: multiply by Availability (applied once)
        return original_pick_percent
    
    # Apply the consolidated function
    if selected_contest == 'Circa':
        pick_predictions_df["Pick %"] = pick_predictions_df.apply(
            adjust_pick_percentage,
            axis=1
        )

    pick_predictions_df.drop(columns=['Away Team Thanksgiving Favorite', 'Away Team Christmas Favorite', 'Home Team Thanksgiving Favorite', 'Home Team Christmas Favorite'], inplace=True)

    # Calculate the sum of "Pick %" for each date
    sum_by_date = pick_predictions_df.groupby('Date')['Pick %'].sum()

    # Update the "Pick %" column by dividing each value by the corresponding sum
    pick_predictions_df['Pick %'] = pick_predictions_df.apply(lambda row: row['Pick %'] / sum_by_date[row['Date']], axis=1)

    pick_predictions_df.to_csv('pick_predictions_df.csv', index = False)

    # Filter the DataFrame based on the "Home/Away" column
    home_df = pick_predictions_df[pick_predictions_df["Home/Away"] == "Home"]
    away_df = pick_predictions_df[pick_predictions_df["Home/Away"] == "Away"]


    home_df = home_df.rename(columns={
        'Date': 'Week',
        'Team': 'Home Team',
        'Opponent': 'Away Team',
        'Win %': 'Home Team Fair Odds',
        'Future Value (Stars)': 'Home Team Star Rating',
        "Pick %": "Home Pick %",
        "Away Team": "Home Away Matchup",
        "Divisional Matchup?": "Home Divisional Matchup Boolean"
    })

    away_df = away_df.rename(columns={
        'Date': 'Week',
        'Team': 'Away Team',
        'Opponent': 'Home Team',
        'Win %': 'Away Team Fair Odds',
        'Future Value (Stars)': 'Away Team Star Rating',
        "Pick %": "Away Pick %",
        "Away Team": "Away Away Matchup",
        "Divisional Matchup?": "Away Divisional Matchup Boolean"
    })


    # Drop the redundant columns from the merged DataFrame
    away_df.drop(columns=['EV', 'Home/Away', 'Away Team Star Rating', 'Away Team Fair Odds', 'Year'], inplace=True)
    home_df.drop(columns=['EV', 'Home/Away', 'Home Team Star Rating', 'Home Team Fair Odds', 'Year'], inplace=True)

    #print(home_df)
    #print(away_df)

    nfl_schedule_df = schedule_df.copy()
    nfl_schedule_df['Week_Number'] = nfl_schedule_df['Week'].str.split(' ').str[1].astype(int)
    # Filter the DataFrame
    nfl_schedule_df = nfl_schedule_df[nfl_schedule_df['Week_Number'] >= starting_week]
    # You can drop the auxiliary 'Week_Number' column if you no longer need it
    nfl_schedule_df = nfl_schedule_df.drop(columns=['Week_Number'])

    #nfl_schedule_df['Week'] = nfl_schedule_df['Week'].str.extract(r'(\d+)').astype(int)
    # Merge the DataFrames based on matching columns
    nfl_schedule_df = pd.merge(nfl_schedule_df, away_df, 
                               left_on=['Week_Num', 'Away Team', 'Home Team'],
                               right_on=['Week', 'Away Team', 'Home Team'],
                               how='left')
    nfl_schedule_df = pd.merge(nfl_schedule_df, home_df, 
                               left_on=['Week_Num', 'Away Team', 'Home Team'],
                               right_on=['Week', 'Away Team', 'Home Team'],
                               how='left')

    #print(nfl_schedule_circa_df)

    # Add 'Home Team EV' and 'Away Team EV' columns to nfl_schedule_circa_df
    nfl_schedule_df['Home Team EV'] = 0.0  # Initialize with 0.0
    nfl_schedule_df['Away Team EV'] = 0.0  # Initialize with 0.0

# Use the 'current_week_entries' variable from the config
    if current_week_entries >= 0:
        nfl_schedule_df.loc[nfl_schedule_df['Week'] == starting_week, 'Total Remaining Entries at Start of Week'] = current_week_entries
    else:
        # Handle the -1 (auto-estimate) case based on contest
        if selected_contest == 'Circa':
             default_entries = 18000 # Example
        elif selected_contest == 'Splash Sports':
             default_entries = 5000 # Example
        else: # DraftKings
             default_entries = 20000 # Example
        nfl_schedule_df.loc[nfl_schedule_df['Week'] == starting_week, 'Total Remaining Entries at Start of Week'] = default_entries

    # Create the boolean mask once, as it's used twice
    multiplier_mask = (selected_contest == 'Splash Sports') & \
                  (nfl_schedule_df['Week'].isin(week_requiring_two_selections))
	
    nfl_schedule_df['Home Expected Survival Rate'] = nfl_schedule_df['Home Team Fair Odds'] * nfl_schedule_df['Home Pick %']
    nfl_schedule_df.loc[multiplier_mask, 'Home Expected Survival Rate'] *= 0.65
    nfl_schedule_df['Home Expected Elimination Percent'] = nfl_schedule_df['Home Pick %'] - nfl_schedule_df['Home Expected Survival Rate']
    nfl_schedule_df['Away Expected Survival Rate'] = nfl_schedule_df['Away Team Fair Odds'] * nfl_schedule_df['Away Pick %']
    nfl_schedule_df.loc[multiplier_mask, 'Away Expected Survival Rate'] *= 0.65
    nfl_schedule_df['Away Expected Elimination Percent'] = nfl_schedule_df['Away Pick %'] - nfl_schedule_df['Away Expected Survival Rate']
    nfl_schedule_df['Expected Eliminated Entry Percent From Game'] = nfl_schedule_df['Home Expected Elimination Percent'] + nfl_schedule_df['Away Expected Elimination Percent']



    #Iterate through weeks starting from week 2
    for week in range(starting_week, nfl_schedule_df['Week'].max() + 1):
        previous_week_df = nfl_schedule_df[nfl_schedule_df['Week'] == week - 1]        
        #Handle potential empty previous week (e.g., if week 1 is missing data for some reason)
        if previous_week_df.empty:
            previous_week_median = nfl_schedule_df['Total Remaining Entries at Start of Week'].median() #Fallback to overall median
        else:    
            previous_week_median = previous_week_df['Total Remaining Entries at Start of Week'].median()
        sum_eliminated = previous_week_df['Expected Eliminated Entry Percent From Game'].sum()
        #Calculate total remaining entries for current week. Handle potential NaN from previous calculations.
        current_week_total = previous_week_median * sum_eliminated if not np.isnan(previous_week_median * sum_eliminated) else 0 
        nfl_schedule_df.loc[nfl_schedule_df['Week'] == week, 'Total Remaining Entries at Start of Week'] = round(previous_week_median - current_week_total)
    if selected_contest == 'Circa':
        nfl_schedule_df['Entry Remaining Percent'] = nfl_schedule_df['Total Remaining Entries at Start of Week'] / circa_total_entries
    elif selected_contest == 'Splash Sports':
        nfl_schedule_df['Entry Remaining Percent'] = nfl_schedule_df['Total Remaining Entries at Start of Week'] / splash_total_entries
    else:
        nfl_schedule_df['Entry Remaining Percent'] = nfl_schedule_df['Total Remaining Entries at Start of Week'] / dk_total_entries
        
    nfl_schedule_df['Expected Eliminated Entries From Game'] = nfl_schedule_df['Total Remaining Entries at Start of Week'] * nfl_schedule_df['Expected Eliminated Entry Percent From Game']
    nfl_schedule_df['Expected Home Team Picks'] = nfl_schedule_df['Home Pick %'] * nfl_schedule_df['Total Remaining Entries at Start of Week']
    nfl_schedule_df['Expected Away Team Picks'] = nfl_schedule_df['Away Pick %'] * nfl_schedule_df['Total Remaining Entries at Start of Week']
    nfl_schedule_df['Expected Home Team Eliminations'] = nfl_schedule_df['Expected Home Team Picks'] * (1 - nfl_schedule_df['Home Team Fair Odds'])
    nfl_schedule_df['Expected Home Team Survivors'] = nfl_schedule_df['Expected Home Team Picks'] * nfl_schedule_df['Home Team Fair Odds']
    nfl_schedule_df['Expected Away Team Eliminations'] = nfl_schedule_df['Expected Away Team Picks'] * (1 - nfl_schedule_df['Away Team Fair Odds'])
    nfl_schedule_df['Expected Away Team Survivors'] = nfl_schedule_df['Expected Away Team Picks'] * nfl_schedule_df['Away Team Fair Odds']



#CALCULATE ESTIMATED REMAINING AVAILABILITY
    
    # 1. Initialization
    all_teams_series = pd.unique(nfl_schedule_df[['Home Team', 'Away Team']].values.ravel('K'))
    all_teams = [team for team in all_teams_series if pd.notna(team)] # Ensure no NaNs if any
    
    # U_prev_week stores U[w][team]: number of entries starting week w that have already used 'team'.
    U_prev_week = {team: 0.0 for team in all_teams} # Using float for expected counts
    
    # Add new columns for availability, initialize
    nfl_schedule_df['Home Team Expected Availability'] = 1.0
    nfl_schedule_df['Away Team Expected Availability'] = 1.0

# Function to get availability
    def get_expected_availability(team_name, availability_dict):
        availability = availability_dict.get(team_name) 
    # 2. Check if the value is -1 (from the Streamlit slider) OR None (if team is missing)
        if availability == -.01 or availability is None:
            return 1.0
        else:
            return availability

# Apply the function to update 'Home Team Expected Availability'
    nfl_schedule_df['Home Team Expected Availability'] = nfl_schedule_df['Home Team'].apply(
        lambda team: get_expected_availability(team, team_availability)
    )

# Apply the function to update 'Away Team Expected Availability'
    nfl_schedule_df['Away Team Expected Availability'] = nfl_schedule_df['Away Team'].apply(
        lambda team: get_expected_availability(team, team_availability)
    )
    
    max_week_num = 0
    if not nfl_schedule_df['Week'].empty:
        max_week_num = nfl_schedule_df['Week'].max()
        if pd.isna(max_week_num): # Handle case where all Week_Num might be NaN after conversion
            max_week_num = 0
    
    # 2. Loop through Weeks
    for week_iter_num in range(1, int(max_week_num) + 1):
        print(f"Calculating availability for Week {week_iter_num}...")
        # --- START: Recalibrate U_prev_week at starting_week ---
        if week_iter_num == starting_week:
            print(f"  Reached starting_week ({starting_week}). Recalibrating U_prev_week based on team_availability.")
    
            # Determine S_at_sw (Total Remaining Entries at Start of starting_week)
            S_at_sw = 0.0
            starting_week_df_rows = nfl_schedule_df[nfl_schedule_df['Week'] == starting_week]
            if not starting_week_df_rows.empty:
                S_at_sw_series = starting_week_df_rows['Total Remaining Entries at Start of Week']
                # Ensure the series is not empty and the first value is not NaN
                if not S_at_sw_series.empty and pd.notna(S_at_sw_series.iloc[0]):
                    S_at_sw = S_at_sw_series.iloc[0]
                else:
                    print(f"  Warning: 'Total Remaining Entries at Start of Week' is missing or NaN for starting_week {starting_week}.")
            else:
                print(f"  Warning: No games found for starting_week {starting_week} to determine S_at_sw.")
    
            if S_at_sw > 0:
                temp_U_for_starting_week = {}
                for team_name_iter in all_teams:
                    # Get the availability percentage for this team from the initial dictionary
                    avail_percent = get_expected_availability(team_name_iter, team_availability)
    
                    # Implied used count = TotalEntries * (1 - AvailabilityPercent)
                    implied_used_count = S_at_sw * (1.0 - avail_percent)
    
                    # Ensure used count is not negative and not more than total entries
                    temp_U_for_starting_week[team_name_iter] = max(0.0, min(implied_used_count, S_at_sw))
    
                U_prev_week = temp_U_for_starting_week # U_prev_week is now set for the start of starting_week
                print(f"  U_prev_week for Week {starting_week} recalibrated. Example for 'Chicago Bears': {U_prev_week.get('Chicago Bears', 'Not Found')}")
            else:
                print(f"  Warning: S_at_sw for starting_week {starting_week} is {S_at_sw}. Cannot use team_availability to set U_prev_week. U_prev_week will be based on prior week ({week_iter_num-1}) calculations (if any).")
        # --- END: Recalibrate U_prev_week at starting_week ---
        current_week_mask = nfl_schedule_df['Week'] == week_iter_num
        
        if not current_week_mask.any():
            print(f"  No games found for Week {week_iter_num}.")
            # If U_prev_week needs to be carried over an empty week, this might need adjustment,
            # but typically U_prev_week would just remain the same for the next actual game week.
            continue
            
        week_df_rows = nfl_schedule_df[current_week_mask]
    
        # S_w: Total Remaining Entries at Start of Week 'week_iter_num'
        # This should be a single scalar value for the entire week.
        S_w_series = week_df_rows['Total Remaining Entries at Start of Week']
        S_w = 0.0
        if not S_w_series.empty:
            S_w = S_w_series.iloc[0]
            if pd.isna(S_w): S_w = 0.0 # Handle potential NaN
        else:
            # This case should ideally not be hit if current_week_mask.any() is true
            # and data is structured with one 'Total Remaining Entries' value per week.
            print(f"  Warning: 'Total Remaining Entries at Start of Week' missing or inconsistent for Week {week_iter_num}.")
    
        # Calculate Availability for current week's games
        for idx, row_data in week_df_rows.iterrows():
            home_team = row_data['Home Team']
            away_team = row_data['Away Team']

            if week_iter_num > starting_week:                
                home_avail = 1.0
                away_avail = 1.0
            
                if S_w > 0:
                    if pd.notna(home_team):
                        unavailable_home = U_prev_week.get(home_team, 0.0)
                        home_avail = (S_w - unavailable_home) / S_w
                    if pd.notna(away_team):
                        unavailable_away = U_prev_week.get(away_team, 0.0)
                        away_avail = (S_w - unavailable_away) / S_w
            
                nfl_schedule_df.loc[idx, 'Home Team Expected Availability'] = max(0.0, min(1.0, home_avail))
                nfl_schedule_df.loc[idx, 'Away Team Expected Availability'] = max(0.0, min(1.0, away_avail))
    
        # Prepare U for the next week (U_next_week will become U_prev_week for the next iteration)
        U_next_week = {team: 0.0 for team in all_teams}
        total_survivors_this_week = 0.0
        survivors_who_picked_team_this_week = {team: 0.0 for team in all_teams}
    
        for idx, row_data in week_df_rows.iterrows():
            home_team = row_data['Home Team']
            away_team = row_data['Away Team']
            
            home_survivors = row_data.get('Expected Home Team Survivors', 0.0)
            if pd.isna(home_survivors): home_survivors = 0.0
            away_survivors = row_data.get('Expected Away Team Survivors', 0.0)
            if pd.isna(away_survivors): away_survivors = 0.0
            if pd.notna(home_team):
                survivors_who_picked_team_this_week[home_team] = survivors_who_picked_team_this_week.get(home_team, 0.0) + home_survivors
            if pd.notna(away_team):
                survivors_who_picked_team_this_week[away_team] = survivors_who_picked_team_this_week.get(away_team, 0.0) + away_survivors
                
            total_survivors_this_week += home_survivors + away_survivors

                
            
        overall_survival_rate_this_week = 0.0
        if S_w > 0:
            overall_survival_rate_this_week = total_survivors_this_week / S_w
        elif total_survivors_this_week > 0 and S_w == 0:
            print(f"  Warning: Week {week_iter_num} started with S_w=0 but has total_survivors_this_week={total_survivors_this_week}. Check data consistency.")
            # If S_w is 0, those in U_prev_week couldn't have existed in that pool to survive.
            # So, effectively, their survival rate from that S_w pool is 0.
            overall_survival_rate_this_week = 0.0
    
    
        for team_name in all_teams:
            val1 = survivors_who_picked_team_this_week.get(team_name, 0.0)
            
            num_already_used_team = U_prev_week.get(team_name, 0.0)
            val2 = num_already_used_team * overall_survival_rate_this_week
            
            current_team_used_next_week = val1 + val2
            
            if total_survivors_this_week > 0:
                U_next_week[team_name] = min(current_team_used_next_week, total_survivors_this_week)
            else: # If no one survived overall, then no one could have used this team and also survived.
                U_next_week[team_name] = 0.0
            
            # Ensure non-negative
            U_next_week[team_name] = max(0.0, U_next_week[team_name])
    
        U_prev_week = U_next_week
    
    print("Expected Availability calculation complete.")
    

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
    
    if selected_contest == 'Circa':
        nfl_schedule_df.to_csv("Circa_Predicted_pick_percent.csv", index=False)
    elif selected_contest == 'Splash Sports':
        nfl_schedule_df.to_csv("Splash_Predicted_pick_percent.csv", index=False)
    else:
        nfl_schedule_df.to_csv("DK_Predicted_pick_percent.csv", index=False)
    st.subheader('Estimated Pick Percentages Without Availability')
    st.write(nfl_schedule_df)
    return nfl_schedule_df

def get_predicted_pick_percentages_with_availability(pd, config: dict, schedule_df: pd.DataFrame):
    # Load your historical data (replace 'historical_pick_data_FV_circa.csv' with your actual file path)
    selected_contest = config['selected_contest']
    starting_week = config['starting_week']
    current_week_entries = config['current_week_entries']
    week_requiring_two_selections = config.get('weeks_two_picks', [])
    team_availability = config.get('team_availabilities', {})
    if selected_contest == 'Circa':
        df = pd.read_csv('Circa_historical_data.csv')

        df.rename(columns={"Week": "Date"}, inplace=True)
        # Remove percentage sign and convert to float
        #df['Win %'] = df['Win %'].str.rstrip('%').astype(float, -1, 100) / 100
        #df['Pick %'] = df['Pick %'].str.rstrip('%').astype(float, -1, 100) / 100
        # Extract the numeric part (week number)
        #df['Week'] = df['Week'].str.extract(r'(\d+)').astype(int)
        #print(df['Date'])
        df['Pick %'].fillna(0.0, inplace=True)
    
        #print(df)
        # Split data into input features (X) and target variable (y)
        X = df[['Win %', 'Future Value (Stars)', 'Date', 'Away Team', 'Divisional Matchup?', 'Availability', 'Entry Remaining Percent']]
        y = df['Pick %']
    
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Initialize and train the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=50, random_state=0)
        rf_model.fit(X_train, y_train)
    
        # Make predictions on the test data
        y_pred = rf_model.predict(X_test)
    
        # Evaluate model performance (MAE)
        mae = mean_absolute_error(y_test, y_pred)
        #print(f"Mean Absolute Error: {mae:.2f}")
    
        # Read the CSV file into a DataFrame
        
        new_df = schedule_df.copy()

    
        # Create a new DataFrame with selected columns
        selected_columns = ['Week', 'Away Team', 'Home Team', 'Away Team Fair Odds',
                            'Home Team Fair Odds', 'Away Team Star Rating', 'Home Team Star Rating', 'Divisional Matchup Boolean', 'Away Team Thanksgiving Favorite', 'Home Team Thanksgiving Favorite', 'Away Team Christmas Favorite', 'Home Team Christmas Favorite', 'Entry Remaining Percent', 'Home Team Expected Availability', 'Away Team Expected Availability']
        new_df = new_df[selected_columns]
        if new_df['Week'].dtype == 'object':
            new_df['Week_Number'] = new_df['Week'].str.split(' ').str[1].astype(int)
        else:
            new_df['Week_Number'] = new_df['Week']
        # Filter the DataFrame
        new_df = new_df[new_df['Week_Number'] >= starting_week]
        # You can drop the auxiliary 'Week_Number' column if you no longer need it
        new_df = new_df.drop(columns=['Week_Number'])
    
        # Read the original CSV file into a DataFrame
        #csv_path = 'nfl_Schedule_circa.csv'
        #df = pd.read_csv(csv_path)
    
        # Create the new DataFrame with modified column names
        away_df = new_df.rename(columns={
            'Week': 'Date',
            'Away Team': 'Team',
            'Home Team': 'Opponent',
            'Away Team Fair Odds': 'Win %',
            'Away Team Star Rating': 'Future Value (Stars)',
            'Divisional Matchup Boolean': 'Divisional Matchup?',
            'Away Team Expected Availability': 'Availability'
        })
        away_df['Year'] = 2025
        away_df['Home/Away'] = 'Away'
        away_df['Away Team'] = 1
        # Add the "Pick %" and "EV" columns (initially empty)
        away_df['Pick %'] = None
        away_df['EV'] = None
    
        # Drop the unwanted columns
        away_df.drop(columns=['Home Team Fair Odds', 'Home Team Star Rating', 'Home Team Thanksgiving Favorite', 'Home Team Christmas Favorite', 'Home Team Expected Availability'], inplace=True)
    
        # Reorder the columns
        column_order = ['EV', 'Win %', 'Pick %', 'Team', 'Opponent', 'Future Value (Stars)', 'Year', 'Date', 'Home/Away', 'Away Team', 'Divisional Matchup?', 'Away Team Thanksgiving Favorite', 'Away Team Christmas Favorite', 'Availability', 'Entry Remaining Percent']
        away_df = away_df[column_order]
    
    
        # Create the new DataFrame with modified column names
        home_df = new_df.rename(columns={
            'Week': 'Date',
            'Home Team': 'Team',
            'Away Team': 'Opponent',
            'Home Team Fair Odds': 'Win %',
            'Home Team Star Rating': 'Future Value (Stars)',
            'Divisional Matchup Boolean': 'Divisional Matchup?',
            'Home Team Expected Availability': 'Availability'
        })
        home_df['Year'] = 2025
        home_df['Home/Away'] = 'Home'
        home_df['Away Team'] = 0
        # Add the "Pick %" and "EV" columns (initially empty)
        home_df['Pick %'] = None
        home_df['EV'] = None
    
        # Drop the unwanted columns
        home_df.drop(columns=['Away Team Fair Odds', 'Away Team Star Rating', 'Away Team Thanksgiving Favorite', 'Away Team Christmas Favorite', 'Away Team Expected Availability'], inplace=True)
    
        # Reorder the columns
        column_order = ['EV', 'Win %', 'Pick %', 'Team', 'Opponent', 'Future Value (Stars)', 'Year', 'Date', 'Home/Away', 'Away Team', 'Divisional Matchup?', 'Home Team Thanksgiving Favorite', 'Home Team Christmas Favorite', 'Availability', 'Entry Remaining Percent']
        home_df = home_df[column_order]
    
    
        predictions = rf_model.predict(away_df[['Win %', 'Future Value (Stars)', 'Date', 'Away Team', 'Divisional Matchup?', 'Availability', 'Entry Remaining Percent']])
        away_df['Pick %'] = predictions
        #away_df.to_csv('predicted_away_data_circa.csv', index=False)
    
        predictions = rf_model.predict(home_df[['Win %', 'Future Value (Stars)', 'Date', 'Away Team', 'Divisional Matchup?', 'Availability', 'Entry Remaining Percent']])
        home_df['Pick %'] = predictions
        #home_df.to_csv('predicted_home_data_circa.csv', index=False)
    
        pick_predictions_df = pd.concat([away_df, home_df], ignore_index=True)
        # Function to calculate the adjusted "Pick %"
        def adjust_pick_percentage(row):
            """
            Calculates the adjusted 'Pick %' based on holiday favorite status 
            and then multiplies by 'Availability'.
            """
            original_pick_percent = row["Pick %"]
            pre_thanksgiving = row["Date"] < 13
            pre_christmas = row["Date"] < 18
            
            # Thanksgiving Adjustment
            # The original logic has two separate checks that compound if both teams
            # are checked, but the goal seems to be: if a team is a Thanksgiving favorite
            # AND it's NOT Thanksgiving (Date != 13), then apply the / 4 modification.
            if pre_thanksgiving:
                if (row["Home Team Thanksgiving Favorite"] or row["Away Team Thanksgiving Favorite"]) and (row["Home Team Christmas Favorite"] or row["Away Team Christmas Favorite"]):
                    original_pick_percent = original_pick_percent / 6
                elif (not row["Home Team Thanksgiving Favorite"] and not row["Away Team Thanksgiving Favorite"]) and (row["Home Team Christmas Favorite"] or row["Away Team Christmas Favorite"]):
                    original_pick_percent = original_pick_percent / 3
                elif (row["Home Team Thanksgiving Favorite"] or row["Away Team Thanksgiving Favorite"]) and (not row["Home Team Christmas Favorite"] and not row["Away Team Christmas Favorite"]):
                    original_pick_percent = original_pick_percent / 2
            elif pre_christmas:
                if row["Home Team Christmas Favorite"] == row["Team"] or row["Away Team Christmas Favorite"] == row["Team"]:
                    original_pick_percent = original_pick_percent / 4
        
            # Final adjustment: multiply by Availability (applied once)
            return original_pick_percent
        
        # Apply the consolidated function
        if selected_contest == 'Circa':
            pick_predictions_df["Pick %"] = pick_predictions_df.apply(
                adjust_pick_percentage,
                axis=1
            )
		
        st.write("TEST PICK PERCENTAGES LINE 1858")
        st.write(pick_predictions_df[["Team", "Pick %", "Availability", "Date", 'Away Team Thanksgiving Favorite', 'Home Team Thanksgiving Favorite', 'Away Team Christmas Favorite', 'Home Team Christmas Favorite']])		
        pick_predictions_df["Pick %"] = pick_predictions_df.apply(
            lambda row: row["Pick %"] * row["Availability"],
            axis=1
        )
    
        pick_predictions_df.drop(columns=['Away Team Thanksgiving Favorite', 'Away Team Christmas Favorite', 'Home Team Thanksgiving Favorite', 'Home Team Christmas Favorite'], inplace=True)
    
        # Calculate the sum of "Pick %" for each date
        sum_by_date = pick_predictions_df.groupby('Date')['Pick %'].sum()
        st.write('Sum By Date')		
        st.write(sum_by_date)

		
        # Update the "Pick %" column by dividing each value by the corresponding sum
        pick_predictions_df['Pick %'] = pick_predictions_df.apply(lambda row: row['Pick %'] / sum_by_date[row['Date']], axis=1)
    
        pick_predictions_df.to_csv('pick_predictions_df.csv', index = False)
    
        # Filter the DataFrame based on the "Home/Away" column
        home_df = pick_predictions_df[pick_predictions_df["Home/Away"] == "Home"]
        away_df = pick_predictions_df[pick_predictions_df["Home/Away"] == "Away"]
    
    
        home_df = home_df.rename(columns={
            'Date': 'Week',
            'Team': 'Home Team',
            'Opponent': 'Away Team',
            'Win %': 'Home Team Fair Odds',
            'Future Value (Stars)': 'Home Team Star Rating',
            "Pick %": "Home Pick %",
            "Away Team": "Home Away Matchup",
            "Divisional Matchup?": "Home Divisional Matchup Boolean",
            'Availability': 'Home Team Expected Availability'
        })
    
        away_df = away_df.rename(columns={
            'Date': 'Week',
            'Team': 'Away Team',
            'Opponent': 'Home Team',
            'Win %': 'Away Team Fair Odds',
            'Future Value (Stars)': 'Away Team Star Rating',
            "Pick %": "Away Pick %",
            "Away Team": "Away Away Matchup",
            "Divisional Matchup?": "Away Divisional Matchup Boolean",
            'Availability': 'Away Team Expected Availability'
        })
    
    
        # Drop the redundant columns from the merged DataFrame
        away_df.drop(columns=['EV', 'Home/Away', 'Away Team Star Rating', 'Away Team Fair Odds', 'Year'], inplace=True)
        home_df.drop(columns=['EV', 'Home/Away', 'Home Team Star Rating', 'Home Team Fair Odds', 'Year'], inplace=True)
    
        #print(home_df)
        #print(away_df)
    
        nfl_schedule_df = schedule_df.copy()
# Drop the preliminary pick percentages from the previous step
        # to avoid column conflicts during the merge.
        if 'Home Pick %' in nfl_schedule_df.columns:
            nfl_schedule_df = nfl_schedule_df.drop(columns=['Home Pick %'])
        if 'Away Pick %' in nfl_schedule_df.columns:
            nfl_schedule_df = nfl_schedule_df.drop(columns=['Away Pick %'])
            
        # Also drop any leftover week columns from the previous merge
        # to prevent conflicts.
        cols_to_drop = ['Week_x', 'Week_y', 'Week']
        existing_cols_to_drop = [col for col in cols_to_drop if col in nfl_schedule_df.columns]
        if existing_cols_to_drop:
            nfl_schedule_df = nfl_schedule_df.drop(columns=existing_cols_to_drop)
        # Assumes format "Week X" - adjust if the format is different
# Filter the DataFrame using the 'Week_Num' column which already exists
        nfl_schedule_df = nfl_schedule_df[nfl_schedule_df['Week_Num'] >= starting_week]

        #nfl_schedule_df['Week'] = nfl_schedule_df['Week'].str.extract(r'(\d+)').astype(int)
        # Merge the DataFrames based on matching columns
        nfl_schedule_df = pd.merge(nfl_schedule_df, away_df, 
                                   left_on=['Week_Num', 'Away Team', 'Home Team'],
                                   right_on=['Week', 'Away Team', 'Home Team'],
                                   how='left')
        nfl_schedule_df = pd.merge(nfl_schedule_df, home_df, 
                                   left_on=['Week_Num', 'Away Team', 'Home Team'],
                                   right_on=['Week', 'Away Team', 'Home Team'],
                                   how='left')
    
        #print(nfl_schedule_circa_df)
    
        # Add 'Home Team EV' and 'Away Team EV' columns to nfl_schedule_circa_df
        nfl_schedule_df['Home Team EV'] = 0.0  # Initialize with 0.0
        nfl_schedule_df['Away Team EV'] = 0.0  # Initialize with 0.0
    
        # Use the 'current_week_entries' variable from the config
        if current_week_entries >= 0:
            nfl_schedule_df.loc[nfl_schedule_df['Week_Num'] == starting_week, 'Total Remaining Entries at Start of Week'] = current_week_entries
        else:
            # Handle the -1 (auto-estimate) case based on contest
            if selected_contest == 'Circa':
                 default_entries = 18000 # Example
            elif selected_contest == 'Splash Sports':
                 default_entries = 5000 # Example
            else: # DraftKings
                 default_entries = 20000 # Example
            nfl_schedule_df.loc[nfl_schedule_df['Week'] == starting_week, 'Total Remaining Entries at Start of Week'] = default_entries
    
        nfl_schedule_df['Home Expected Survival Rate'] = nfl_schedule_df['Home Team Fair Odds'] * nfl_schedule_df['Home Pick %']
        nfl_schedule_df['Home Expected Elimination Percent'] = nfl_schedule_df['Home Pick %'] - nfl_schedule_df['Home Expected Survival Rate']
        nfl_schedule_df['Away Expected Survival Rate'] = nfl_schedule_df['Away Team Fair Odds'] * nfl_schedule_df['Away Pick %']
        nfl_schedule_df['Away Expected Elimination Percent'] = nfl_schedule_df['Away Pick %'] - nfl_schedule_df['Away Expected Survival Rate']
        nfl_schedule_df['Expected Eliminated Entry Percent From Game'] = nfl_schedule_df['Home Expected Elimination Percent'] + nfl_schedule_df['Away Expected Elimination Percent']
    
    
    
        #Iterate through weeks starting from week 2
        for week in range(starting_week, nfl_schedule_df['Week_Num'].max() + 1):
            previous_week_df = nfl_schedule_df[nfl_schedule_df['Week_Num'] == week - 1]
            
            #Handle potential empty previous week (e.g., if week 1 is missing data for some reason)
            if previous_week_df.empty:
                previous_week_median = nfl_schedule_df['Total Remaining Entries at Start of Week'].median() #Fallback to overall median
            else:    
                previous_week_median = previous_week_df['Total Remaining Entries at Start of Week'].median()
    
            sum_eliminated = previous_week_df['Expected Eliminated Entry Percent From Game'].sum()
    
            #Calculate total remaining entries for current week. Handle potential NaN from previous calculations.
            current_week_total = previous_week_median * sum_eliminated if not np.isnan(previous_week_median * sum_eliminated) else 0 
    
            nfl_schedule_df.loc[nfl_schedule_df['Week_Num'] == week, 'Total Remaining Entries at Start of Week'] = round(previous_week_median - current_week_total)
            
        nfl_schedule_df['Expected Eliminated Entries From Game'] = nfl_schedule_df['Total Remaining Entries at Start of Week'] * nfl_schedule_df['Expected Eliminated Entry Percent From Game']
        nfl_schedule_df['Expected Home Team Picks'] = nfl_schedule_df['Home Pick %'] * nfl_schedule_df['Total Remaining Entries at Start of Week']
        nfl_schedule_df['Expected Away Team Picks'] = nfl_schedule_df['Away Pick %'] * nfl_schedule_df['Total Remaining Entries at Start of Week']
        nfl_schedule_df['Expected Home Team Eliminations'] = nfl_schedule_df['Expected Home Team Picks'] * (1 - nfl_schedule_df['Home Team Fair Odds'])
        nfl_schedule_df['Expected Home Team Survivors'] = nfl_schedule_df['Expected Home Team Picks'] * nfl_schedule_df['Home Team Fair Odds']
        nfl_schedule_df['Expected Away Team Eliminations'] = nfl_schedule_df['Expected Away Team Picks'] * (1 - nfl_schedule_df['Away Team Fair Odds'])
        nfl_schedule_df['Expected Away Team Survivors'] = nfl_schedule_df['Expected Away Team Picks'] * nfl_schedule_df['Away Team Fair Odds']
    
    #CALCULATE ESTIMATED REMAINING AVAILABILITY
        
        # 1. Initialization
        all_teams_series = pd.unique(nfl_schedule_df[['Home Team', 'Away Team']].values.ravel('K'))
        all_teams = [team for team in all_teams_series if pd.notna(team)] # Ensure no NaNs if any
        
        # U_prev_week stores U[w][team]: number of entries starting week w that have already used 'team'.
        U_prev_week = {team: 0.0 for team in all_teams} # Using float for expected counts
        
        # Add new columns for availability, initialize
        nfl_schedule_df['Home Team Expected Availability'] = 1.0
        nfl_schedule_df['Away Team Expected Availability'] = 1.0
    # Function to get availability
        def get_expected_availability(team_name, availability_dict):
            availability = availability_dict.get(team_name) # Get availability, default to -1 if team not in dict
            if availability != -.01:
                return availability
            else:
                return 1.0
    
    # Apply the function to update 'Home Team Expected Availability'
        nfl_schedule_df['Home Team Expected Availability'] = nfl_schedule_df['Home Team'].apply(
            lambda team: get_expected_availability(team, team_availability)
        )
    
    # Apply the function to update 'Away Team Expected Availability'
        nfl_schedule_df['Away Team Expected Availability'] = nfl_schedule_df['Away Team'].apply(
            lambda team: get_expected_availability(team, team_availability)
        )
        
        max_week_num = 0
        if not nfl_schedule_df['Week_Num'].empty:
            max_week_num = nfl_schedule_df['Week_Num'].max()
            if pd.isna(max_week_num): # Handle case where all Week_Num might be NaN after conversion
                max_week_num = 0
        
        # 2. Loop through Weeks
        for week_iter_num in range(1, int(max_week_num) + 1):
            print(f"Calculating availability for Week {week_iter_num}...")
            # --- START: Recalibrate U_prev_week at starting_week ---
            if week_iter_num == starting_week:
                print(f"  Reached starting_week ({starting_week}). Recalibrating U_prev_week based on team_availability.")
        
                # Determine S_at_sw (Total Remaining Entries at Start of starting_week)
                S_at_sw = 0.0
                starting_week_df_rows = nfl_schedule_df[nfl_schedule_df['Week_Num'] == starting_week]
                if not starting_week_df_rows.empty:
                    S_at_sw_series = starting_week_df_rows['Total Remaining Entries at Start of Week']
                    # Ensure the series is not empty and the first value is not NaN
                    if not S_at_sw_series.empty and pd.notna(S_at_sw_series.iloc[0]):
                        S_at_sw = S_at_sw_series.iloc[0]
                    else:
                        print(f"  Warning: 'Total Remaining Entries at Start of Week' is missing or NaN for starting_week {starting_week}.")
                else:
                    print(f"  Warning: No games found for starting_week {starting_week} to determine S_at_sw.")
        
                if S_at_sw > 0:
                    temp_U_for_starting_week = {}
                    for team_name_iter in all_teams:
                        # Get the availability percentage for this team from the initial dictionary
                        avail_percent = get_expected_availability(team_name_iter, team_availability)
        
                        # Implied used count = TotalEntries * (1 - AvailabilityPercent)
                        implied_used_count = S_at_sw * (1.0 - avail_percent)
        
                        # Ensure used count is not negative and not more than total entries
                        temp_U_for_starting_week[team_name_iter] = max(0.0, min(implied_used_count, S_at_sw))
        
                    U_prev_week = temp_U_for_starting_week # U_prev_week is now set for the start of starting_week
                    print(f"  U_prev_week for Week {starting_week} recalibrated. Example for 'Chicago Bears': {U_prev_week.get('Chicago Bears', 'Not Found')}")
                else:
                    print(f"  Warning: S_at_sw for starting_week {starting_week} is {S_at_sw}. Cannot use team_availability to set U_prev_week. U_prev_week will be based on prior week ({week_iter_num-1}) calculations (if any).")
            # --- END: Recalibrate U_prev_week at starting_week ---
            current_week_mask = nfl_schedule_df['Week_Num'] == week_iter_num
            
            if not current_week_mask.any():
                print(f"  No games found for Week {week_iter_num}.")
                # If U_prev_week needs to be carried over an empty week, this might need adjustment,
                # but typically U_prev_week would just remain the same for the next actual game week.
                continue
                
            week_df_rows = nfl_schedule_df[current_week_mask]
        
            # S_w: Total Remaining Entries at Start of Week 'week_iter_num'
            # This should be a single scalar value for the entire week.
            S_w_series = week_df_rows['Total Remaining Entries at Start of Week']
            S_w = 0.0
            if not S_w_series.empty:
                S_w = S_w_series.iloc[0]
                if pd.isna(S_w): S_w = 0.0 # Handle potential NaN
            else:
                # This case should ideally not be hit if current_week_mask.any() is true
                # and data is structured with one 'Total Remaining Entries' value per week.
                print(f"  Warning: 'Total Remaining Entries at Start of Week' missing or inconsistent for Week {week_iter_num}.")
        
            # Calculate Availability for current week's games
            for idx, row_data in week_df_rows.iterrows():
                home_team = row_data['Home Team']
                away_team = row_data['Away Team']
                if week_iter_num > starting_week:                
                    home_avail = 1.0
                    away_avail = 1.0
                
                    if S_w > 0:
                        if pd.notna(home_team):
                            unavailable_home = U_prev_week.get(home_team, 0.0)
                            home_avail = (S_w - unavailable_home) / S_w
                        if pd.notna(away_team):
                            unavailable_away = U_prev_week.get(away_team, 0.0)
                            away_avail = (S_w - unavailable_away) / S_w
                
                    nfl_schedule_df.loc[idx, 'Home Team Expected Availability'] = max(0.0, min(1.0, home_avail))
                    nfl_schedule_df.loc[idx, 'Away Team Expected Availability'] = max(0.0, min(1.0, away_avail))
        
            # Prepare U for the next week (U_next_week will become U_prev_week for the next iteration)
            U_next_week = {team: 0.0 for team in all_teams}
            total_survivors_this_week = 0.0
            survivors_who_picked_team_this_week = {team: 0.0 for team in all_teams}
        
            for idx, row_data in week_df_rows.iterrows():
                home_team = row_data['Home Team']
                away_team = row_data['Away Team']
                
                home_survivors = row_data.get('Expected Home Team Survivors', 0.0)
                if pd.isna(home_survivors): home_survivors = 0.0
                away_survivors = row_data.get('Expected Away Team Survivors', 0.0)
                if pd.isna(away_survivors): away_survivors = 0.0
                
                if pd.notna(home_team):
                    survivors_who_picked_team_this_week[home_team] = survivors_who_picked_team_this_week.get(home_team, 0.0) + home_survivors
                if pd.notna(away_team):
                    survivors_who_picked_team_this_week[away_team] = survivors_who_picked_team_this_week.get(away_team, 0.0) + away_survivors
                    
                total_survivors_this_week += home_survivors + away_survivors

                
            overall_survival_rate_this_week = 0.0
            if S_w > 0:
                overall_survival_rate_this_week = total_survivors_this_week / S_w
            elif total_survivors_this_week > 0 and S_w == 0:
                print(f"  Warning: Week {week_iter_num} started with S_w=0 but has total_survivors_this_week={total_survivors_this_week}. Check data consistency.")
                # If S_w is 0, those in U_prev_week couldn't have existed in that pool to survive.
                # So, effectively, their survival rate from that S_w pool is 0.
                overall_survival_rate_this_week = 0.0
        
        
            for team_name in all_teams:
                val1 = survivors_who_picked_team_this_week.get(team_name, 0.0)
                
                num_already_used_team = U_prev_week.get(team_name, 0.0)
                val2 = num_already_used_team * overall_survival_rate_this_week
                
                current_team_used_next_week = val1 + val2
                
                if total_survivors_this_week > 0:
                    U_next_week[team_name] = min(current_team_used_next_week, total_survivors_this_week)
                else: # If no one survived overall, then no one could have used this team and also survived.
                    U_next_week[team_name] = 0.0
                
                # Ensure non-negative
                U_next_week[team_name] = max(0.0, U_next_week[team_name])
        
            U_prev_week = U_next_week
        
        print("Expected Availability calculation complete.")
    

    # This function should be identical to the one in get_predicted_pick_percentages
        def assign_pick_percentages_from_config(row, custom_picks_config):
            home_team = row['Home Team']
            away_team = row['Away Team']
            week = row['Week_Num'] # Assumes week is like "Week 1", "Week 2"
            week_num_str = str(week).replace('Week ', '')
            week_key = f"week_{week_num_str}"
    
            home_pick_percent = row.get('Home Pick %') # Default
            away_pick_percent = row.get('Away Pick %') # Default
    
            if week_key in custom_picks_config:
                week_overrides = custom_picks_config[week_key]
                
                if home_team in week_overrides:
                    user_override_value = week_overrides[home_team]
                    if user_override_value >= 0:
                        home_pick_percent = user_override_value
                        
                if away_team in week_overrides:
                    user_override_value = week_overrides[away_team]
                    if user_override_value >= 0:
                        away_pick_percent = user_override_value
    
            return pd.Series({'Home Pick %': home_pick_percent, 'Away Pick %': away_pick_percent})
                                                      
        # Get the single source of truth for custom picks from the config
        custom_pick_percentages = config.get('pick_percentages', {})
        
        nfl_schedule_df[['Home Pick %', 'Away Pick %']] = nfl_schedule_df.apply(
            assign_pick_percentages_from_config, 
            axis=1, 
            args=(custom_pick_percentages,) # Pass the config dict
        )
        
        if selected_contest == 'Circa':
            nfl_schedule_df.to_csv("Circa_Predicted_pick_percent.csv", index=False)
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
    away_reformatted_df = df[['Week_x', 'Date', 'Time', 'Away Team', 'Home Team', 'Away Team Weekly Rest', 'Weekly Away Rest Advantage', 'Away Cumulative Rest Advantage', 'Away Team Current Week Cumulative Rest Advantage', 'Actual Stadium', 'Back to Back Away Games', 'Away Team Previous Opponent', 'Away Team Previous Location', 'Away Previous Game Actual Stadium TimeZone','Away Weekly Timezone Difference', 'Away Team Next Opponent', 'Away Team Next Location', 'Away Travel Advantage', 'Away Timezone Change', 'Away Team Preseason Rank','Away Team Adjusted Preseason Rank', 'Away Team Current Rank', 'Away Team Adjusted Current Rank', 'Thursday Night Game', 'Divisional Matchup?', 'Away Team 3 games in 10 days', 'Away Team 4 games in 17 days', 'Away Team Short Rest', 'Away Team Moneyline', 'Favorite', 'Underdog', 'Adjusted Away Points', 'Adjusted Home Points', 'Internal Away Team Moneyline', 'Away Team Implied Odds to Win', 'Internal Away Team Implied Odds to Win', 'Away Team Fair Odds', 'Internal Away Team Fair Odds', 'Away Team Star Rating', 'Away Team Thanksgiving Favorite', 'Away Team Christmas Favorite', 'Away Team Thanksgiving Underdog', 'Away Team Christmas Underdog', 'Away Team Expected Availability','Away Pick %', 'Away Team EV', 'Total Remaining Entries at Start of Week', 'Away Expected Survival Rate', 'Away Expected Elimination Percent', 'Expected Away Team Picks', 'Expected Away Team Eliminations', 'Expected Away Team Survivors', 'Same Winner?', 'Same Current and Adjusted Current Winner?', 'Same Adjusted Preseason Winner?', 'Adjusted Current Winner', 'Adjusted Current Difference', 'Home Team Preseason Rank', 'Home Team Adjusted Preseason Rank', 'Home Team Current Rank', 'Home Team Adjusted Current Rank', 'ID']]
    
    new_column_names_away = {
        'Week_x': 'Week',
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
        'Favorite': 'Favorite',
        'Underdog': 'Underdog',
        'Adjusted Away Points': 'Adjusted Away Points',
        'Adjusted Home Points': 'Adjusted Home Points',
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
        'Home Team Adjusted Current Rank': 'Opp Adjusted Current Rank'
    }
    away_reformatted_df = away_reformatted_df.rename(columns=new_column_names_away)

    away_reformatted_df['Date'] = pd.to_datetime(away_reformatted_df['Date']).dt.strftime('%m-%d-%Y')
    
    # ... (Rounding) ...
    away_reformatted_df['Expected EV'] = round(away_reformatted_df['Expected EV'], 4)
    away_reformatted_df['Fair Odds Based on Sportsbook Odds'] = round(away_reformatted_df['Fair Odds Based on Sportsbook Odds'], 2)
    away_reformatted_df['Fair Odds Based on Internal Rankings'] = round(away_reformatted_df['Fair Odds Based on Internal Rankings'], 2)
    away_reformatted_df['Expected Pick Percent'] = round(away_reformatted_df['Expected Pick Percent'], 2)
    away_reformatted_df['Expected Availability'] = round(away_reformatted_df['Expected Availability'], 2)
    away_reformatted_df['Expected Survival Rate'] = round(away_reformatted_df['Expected Survival Rate'], 2)
    away_reformatted_df['Expected Contest Elimination Percent'] = round(away_reformatted_df['Expected Contest Elimination Percent'], 2)
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
    
    new_order_away = ['Week_Num', 'Date', 'Time', 'Team', 'Opponent', 'Team Is Away', 'Location', 'Expected EV', 'Moneyline Based on Sportsbook Odds', 'Moneyline Based on Internal Rankings', 'Fair Odds Based on Sportsbook Odds', 'Fair Odds Based on Internal Rankings', 'Preseason Rank', 'Adjusted Preseason Rank', 'Current Rank', 'Adjusted Current Rank', 'Preseason Difference', 'Adjusted Preseason Difference', 'Current Difference', 'Adjusted Current Difference','Expected Pick Percent', 'Expected Availability', 'Future Value', 'Weekly Rest', 'Weekly Rest Advantage', 'Season-Long Rest Advantage', 'Season-Long Rest Advantage Including This Week', 'Travel Advantage', 'Weekly Timezone Difference', 'Previous Opponent', 'Previous Game Location', 'Previous Game Timezone', 'Next Opponent', 'Next Game Location', 'Back to Back Away Games', 'Thursday Night Game', 'Divisional Matchup?', '3 Games in 10 Days', '4 Games in 17 Days', 'Away Team Short Rest', 'Thanksgiving Favorite', 'Christmas Favorite', 'Thanksgiving Underdog', 'Christmas Underdog', 'Total Remaining Entries at Start of Week', 'Expected Picks', 'Expected Survival Rate', 'Expected Contest Elimination Percent', 'Expected Eliminations', 'Expected Survivors', 'Adjusted Current Winner', 'Favorite', 'Underdog', 'Opp Preseason Rank', 'Opp Adjusted Preseason Rank', 'Opp Current Rank', 'Opp Adjusted Current Rank', 'Same Internal Ranking + Sportsbook Winner', 'Same Current and Adjusted Current Winner?', 'Same Adjusted Preseason Winner?', 'ID']
    away_reformatted_df = away_reformatted_df.reindex(columns=new_order_away) # Use reindex to handle missing columns gracefully

    # --- Home Team Dataframe ---
    home_reformatted_df = df[['Week_x', 'Date', 'Time', 'Home Team', 'Away Team', 'Home Team Weekly Rest', 'Weekly Home Rest Advantage', 'Home Cumulative Rest Advantage', 'Home Team Current Week Cumulative Rest Advantage', 'Actual Stadium', 'Back to Back Away Games', 'Home Team Previous Opponent', 'Home Team Previous Location', 'Home Previous Game Actual Stadium TimeZone','Home Weekly Timezone Difference', 'Home Team Next Opponent', 'Home Team Next Location', 'Home Travel Advantage', 'Home Timezone Change', 'Home Team Preseason Rank', 'Home Team Adjusted Preseason Rank', 'Home Team Current Rank', 'Home Team Adjusted Current Rank', 'Thursday Night Game', 'Divisional Matchup?', 'Home Team 3 games in 10 days', 'Home Team 4 games in 17 days', 'Away Team Short Rest', 'Home Team Moneyline', 'Favorite', 'Underdog', 'Adjusted Home Points', 'Adjusted Away Points', 'Internal Home Team Moneyline', 'Home team Implied Odds to Win', 'Internal Home team Implied Odds to Win', 'Home Team Fair Odds', 'Internal Home Team Fair Odds', 'Home Team Star Rating', 'Home Team Thanksgiving Favorite', 'Home Team Christmas Favorite', 'Home Team Thanksgiving Underdog', 'Home Team Christmas Underdog', 'Home Team Expected Availability','Home Pick %', 'Home Team EV', 'Total Remaining Entries at Start of Week', 'Home Expected Survival Rate', 'Home Expected Elimination Percent', 'Expected Home Team Picks', 'Expected Home Team Eliminations', 'Expected Home Team Survivors', 'Same Winner?', 'Same Current and Adjusted Current Winner?', 'Same Adjusted Preseason Winner?', 'Adjusted Current Winner', 'Adjusted Current Difference', 'Away Team Preseason Rank', 'Away Team Adjusted Preseason Rank', 'Away Team Current Rank', 'Away Team Adjusted Current Rank', 'ID']]

    new_column_names_home = {
        'Week_x': 'Week',
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
        'Adjusted Home Points': 'Adjusted Home Points',
        'Adjusted Away Points': 'Adjusted Away Points',
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
        'Away Team Adjusted Current Rank': 'Opp Adjusted Current Rank'
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
    
    
    new_order_home = ['Week', 'Date', 'Time', 'Team', 'Opponent', 'Team Is Away', 'Location', 'Expected EV', 'Moneyline Based on Sportsbook Odds', 'Moneyline Based on Internal Rankings', 'Fair Odds Based on Sportsbook Odds', 'Fair Odds Based on Internal Rankings', 'Preseason Rank', 'Adjusted Preseason Rank', 'Current Rank', 'Adjusted Current Rank', 'Preseason Difference', 'Adjusted Preseason Difference', 'Current Difference', 'Adjusted Current Difference','Expected Pick Percent', 'Expected Availability', 'Future Value', 'Weekly Rest', 'Weekly Rest Advantage', 'Season-Long Rest Advantage', 'Season-Long Rest Advantage Including This Week', 'Travel Advantage', 'Weekly Timezone Difference', 'Previous Opponent', 'Previous Game Location', 'Previous Game Timezone', 'Next Opponent', 'Next Game Location', 'Back to Back Away Games', 'Thursday Night Game', 'Divisional Matchup?', '3 Games in 10 Days', '4 Games in 17 Days', 'Away Team Short Rest', 'Thanksgiving Favorite', 'Christmas Favorite', 'Thanksgiving Underdog', 'Christmas Underdog','Total Remaining Entries at Start of Week', 'Expected Picks', 'Expected Survival Rate', 'Expected Contest Elimination Percent', 'Expected Eliminations', 'Expected Survivors', 'Adjusted Current Winner', 'Favorite', 'Underdog', 'Opp Preseason Rank', 'Opp Adjusted Preseason Rank', 'Opp Current Rank', 'Opp Adjusted Current Rank', 'Same Internal Ranking + Sportsbook Winner', 'Same Current and Adjusted Current Winner?', 'Same Adjusted Preseason Winner?', 'ID']
    home_reformatted_df = home_reformatted_df.reindex(columns=new_order_home) # Use reindex to handle missing columns gracefully

    # --- Combine and Sort ---
    combined_df = pd.concat([home_reformatted_df, away_reformatted_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=['ID', 'Team Is Away'], ascending=True)

    # --- Final Step: Use the 'selected_contest' variable ---
    if selected_contest != 'Circa':
        combined_df = combined_df.drop(columns=['Thanksgiving Favorite', 'Christmas Favorite', 'Thanksgiving Underdog', 'Christmas Underdog'])
    
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
    week_requiring_two_selections = config.get('weeks_two_picks', [])
    favored_qualifier = config.get('favored_qualifier', 'Live Sportsbook Odds (If Available)')
    
    # Get all constraints
    pick_must_be_favored = config.get('must_be_favored', False)
    avoid_away_teams_in_close_matchups = config.get('avoid_away_close', False)
    min_away_spread = config.get('min_away_spread', 3.0)
    avoid_close_divisional_matchups = config.get('avoid_close_divisional', False)
    min_div_spread = config.get('min_div_spread', 7.0)
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
        # Filter out already picked teams
        # --- Create home_ev_df ---
        home_ev_df = df.copy() # Start with a copy to keep all original columns
        
        # Rename columns for home_ev_df
        home_ev_df.rename(columns={
            "Home Team": "Hypothetical Current Winner",
            "Away Team": "Hypothetical Current Loser",
            "Home Team EV": "Hypothetical Current Winner EV",
            "Away Team EV": "Hypothetical Current Loser EV",
            "Away Team Adjusted Current Rank": "Hypothetical Current Loser Adjusted Current Rank",
            "Home Team Adjusted Current Rank": "Hypothetical Current Winner Adjusted Current Rank",
            "Home Team Sportsbook Spread": "Hypothetical Current Winner Sportsbook Spread",
            "Away Team Sportsbook Spread": "Hypothetical Current Loser Sportsbook Spread",
            "Internal Home Team Spread": "Internal Hypothetical Current Winner Spread",
            "Internal Away Team Spread": "Internal Hypothetical Current Loser Spread"
        }, inplace=True)
        
        # Add "Away Team 1" column
        home_ev_df["Away Team 1"] = home_ev_df["Hypothetical Current Loser"]
        home_ev_df["Home Team 1"] = home_ev_df["Hypothetical Current Winner"]
        
        # --- Create away_ev_df ---
        away_ev_df = df.copy() # Start with a copy to keep all original columns
        
        # Rename columns for away_ev_df
        away_ev_df.rename(columns={
            "Home Team": "Hypothetical Current Loser",
            "Away Team": "Hypothetical Current Winner",
            "Home Team EV": "Hypothetical Current Loser EV",
            "Away Team EV": "Hypothetical Current Winner EV",
            "Away Team Adjusted Current Rank": "Hypothetical Current Winner Adjusted Current Rank",
            "Home Team Adjusted Current Rank": "Hypothetical Current Loser Adjusted Current Rank",
            "Away Team Sportsbook Spread": "Hypothetical Current Winner Sportsbook Spread" ,
            "Home Team Sportsbook Spread": "Hypothetical Current Loser Sportsbook Spread",
            "Internal Away Team Spread": "Internal Hypothetical Current Winner Spread",
            "Internal Home Team Spread": "Internal Hypothetical Current Loser Spread"
        }, inplace=True)
        
        # Add "Away Team 1" column
        away_ev_df["Away Team 1"] = away_ev_df["Hypothetical Current Winner"]
        away_ev_df["Home Team 1"] = away_ev_df["Hypothetical Current Loser"]
        
        # --- Combine the two dataframes ---
        combined_df = pd.concat([home_ev_df, away_ev_df], ignore_index=True)
        combined_df = combined_df.sort_values(by='Week_Num')
        
        # Display the results (optional)
        print("Original DataFrame (df):")
        print(df)
        print("\nHome EV DataFrame (home_ev_df):")
        print(home_ev_df)
        print("\nAway EV DataFrame (away_ev_df):")
        print(away_ev_df)
        print("\nCombined DataFrame (combined_df):")
        print(combined_df['Week_Num'])
        df = combined_df
        df = df[~df['Hypothetical Current Winner'].isin(picked_teams)].reset_index(drop=True)
        #print(df)
        # Create the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Create binary variables to represent the picks, and store them in a dictionary for easy lookup
        picks = {}
        for i in range(len(df)):
            picks[i] = solver.IntVar(0, 1, 'pick_%i' % i)
        
        for team, req_week in required_teams:
            if req_week > 0:
                # Find the index of the game where 'team' plays in 'req_week'
                # The '&' operator here applies the condition for matching both team and week
                required_game_indices = df[
                    (df['Hypothetical Current Winner'] == team) & 
                    (df['Week_Num'] == req_week)
                ].index.tolist()
                
                # Add the DataFrame index numbers (i) to the set
                required_pick_indices.update(required_game_indices)
        # Add the constraints
        for i in range(len(df)):
            if i in required_pick_indices:
                continue # Skip all general constraints below for this game
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
                    if df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Adjusted Current Difference'] <= min_away_spread and df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank'] > df.loc[i, 'Hypothetical Current Loser Adjusted Current Rank']:
                        solver.Add(picks[i] == 0)
                else:
                    if df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Adjusted Spread'] <= min_away_spread and df.loc[i, 'Hypothetical Current Winner Sportsbook Spread'] < df.loc[i, 'Hypothetical Current Loser Sportsbook Spread']:
                        solver.Add(picks[i] == 0) 
            #if df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Divisional Matchup?'] == 'Divisional':
                #solver.Add(picks[i] == 0)
            if avoid_back_to_back_away == 1:
                if df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Back to Back Away Games'] == 'True':
                    solver.Add(picks[i] == 0)

            # If 'Divisional Matchup?' is "Divisional", can only pick if 'Adjusted Current Difference' > 10
            if avoid_close_divisional_matchups == 1:
                if favored_qualifier == 'Internal Rankings':
                    if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Adjusted Current Difference'] <= min_div_spread and df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank'] > df.loc[i, 'Hypothetical Current Loser Adjusted Current Rank']:
                        solver.Add(picks[i] == 0)
                else:
                    if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Adjusted Spread'] <= min_div_spread and df.loc[i, 'Hypothetical Current Winner Sportsbook Spread'] < df.loc[i, 'Hypothetical Current Loser Sportsbook Spread']:
                        solver.Add(picks[i] == 0) 
            if avoid_away_divisional_matchups == 1:
                if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
            # Constraints for short rest and 4 games in 17 days (only if team is the Adjusted Current Winner)
            if avoid_away_teams_on_short_rest == 1:
                if df.loc[i, 'Away Team Short Rest'] == 'Yes' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
            if avoid_4_games_in_17_days == 1:
                if df.loc[i, 'Home Team 4 games in 17 days'] == 'Yes' and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Away Team 4 games in 17 days'] == 'No':
                    solver.Add(picks[i] == 0)
                if df.loc[i, 'Away Team 4 games in 17 days'] == 'Yes' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Home Team 4 games in 17 days'] == 'No':
                    solver.Add(picks[i] == 0)
            if avoid_3_games_in_10_days == 1:
                if df.loc[i, 'Home Team 3 games in 10 days'] == 'Yes' and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Away Team 3 games in 10 days'] == 'No':
                    solver.Add(picks[i] == 0)
                if df.loc[i, 'Away Team 3 games in 10 days'] == 'Yes' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Home Team 3 games in 10 days'] == 'No':
                    solver.Add(picks[i] == 0)
            if avoid_international_game == 1:    
                if df.loc[i, 'Actual Stadium'] == 'London, UK' and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
                if df.loc[i, 'Actual Stadium'] == 'London, UK' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
            if avoid_thursday_night == 1:
                if df.loc[i, 'Thursday Night Game'] == 'True':
                    solver.Add(picks[i] == 0)
            if avoid_away_thursday_night == 1:
                if df.loc[i, 'Thursday Night Game'] == 'True' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
            if avoid_teams_with_weekly_rest_disadvantage == 1:
                if df.loc[i, 'Home Team Weekly Rest'] < df.loc [i, 'Away Team Weekly Rest'] and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
                if df.loc[i, 'Away Team Weekly Rest'] < df.loc [i, 'Home Team Weekly Rest'] and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
            if avoid_cumulative_rest_disadvantage == 1:
                if df.loc[i, 'Away Team Current Week Cumulative Rest Advantage'] < -10 and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
                if df.loc[i, 'Home Team Current Week Cumulative Rest Advantage'] < -5 and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
            if avoid_away_teams_with_travel_disadvantage == 1:
                if df.loc[i, 'Away Travel Advantage'] < -850 and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
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

            if selected_contest == "Splash Sports" and week in week_requiring_two_selections:
                # For Splash Sports and later weeks, two teams must be selected
                solver.Add(solver.Sum(weekly_picks) == 2)
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
                weekly_rows = df[df['Week'] == week]

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

                    week = df.loc[i, 'Week']
                    date = df.loc[i, 'Date']
                    date = pd.to_datetime(date, format='%b %d, %Y')
                    time = df.loc[i, 'Time']
                    location = df.loc[i, 'Actual Stadium']
                    pick = df.loc[i,'Hypothetical Current Winner']
                    opponent = df.loc[i,'Hypothetical Current Loser']
                    win_odds = round(df.loc[i, 'Home Team Fair Odds'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Fair Odds'], 2)
                    pick_percent = round(df.loc[i, 'Home Pick %'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Pick %'], 2)
                    expected_availability = round(df.loc[i, 'Home Team Expected Availability'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Expected Availability'], 2)
                    divisional_game = 'Divisional' if df.loc[i, 'Divisional Matchup Boolean'] else ''
                    home_team = 'Home Team' if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else 'Away Team'
                    weekly_rest = df.loc[i, 'Home Team Weekly Rest'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Weekly Rest']
                    weekly_rest_advantage = df.loc[i, 'Weekly Home Rest Advantage'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Weekly Away Rest Advantage']
                    cumulative_rest = df.loc[i, 'Home Cumulative Rest Advantage'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Cumulative Rest Advantage']
                    cumulative_rest_advantage = df.loc[i, 'Home Team Current Week Cumulative Rest Advantage'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Current Week Cumulative Rest Advantage']
                    travel_advantage = df.loc[i, 'Home Travel Advantage'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Travel Advantage']
                    back_to_back_away_games = 'True' if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Away Team 1'] and df.loc[i, 'Back to Back Away Games'] == 'True' else 'False'
                    thursday_night_game = 'Thursday Night Game' if df.loc[i, "Thursday Night Game"] == 'True' else 'Sunday/Monday Game'
                    international_game = 'International Game' if df.loc[i, 'Actual Stadium'] == 'London, UK' else 'Domestic Game'
                    previous_opponent = df.loc[i, 'Home Team Previous Opponent'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Previous Opponent']
                    previous_game_location = df.loc[i, 'Home Team Previous Location'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Previous Location']
                    next_opponent = df.loc[i, 'Home Team Next Opponent'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Next Opponent']
                    next_game_location = df.loc[i, 'Home Team Next Location'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Next Location']

                    internal_ranking_fair_odds = df.loc[i, 'Internal Home Team Fair Odds'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Internal Away Team Fair Odds']
                    future_value = df.loc[i, 'Home Team Star Rating'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Star Rating']
                    sportbook_moneyline = df.loc[i, 'Home Team Moneyline'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Moneyline']
                    internal_moneyline = df.loc[i, 'Internal Home Team Moneyline'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Internal Away Team Moneyline']
                    contest_selections = df.loc[i, 'Expected Home Team Picks'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Expected Away Team Picks']
                    survival_rate = df.loc[i, 'Home Expected Survival Rate'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Expected Survival Rate']
                    elimination_percent = df.loc[i, 'Home Expected Elimination Percent'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Expected Elimination Percent']
                    survivors = df.loc[i, 'Expected Home Team Survivors'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Expected Away Team Survivors']
                    eliminations = df.loc[i, 'Expected Home Team Eliminations'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Expected Away Team Eliminations']
                    preseason_rank = df.loc[i, 'Home Team Preseason Rank'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Preseason Rank']
                    adjusted_preseason_rank = df.loc[i, 'Home Team Adjusted Preseason Rank'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Adjusted Preseason Rank']
                    current_rank = df.loc[i, 'Home Team Current Rank'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Current Rank']
                    adjusted_current_rank = df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank']
                    away_team_short_rest = 'True' if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Away Team 1'] and df.loc[i, 'Away Team Short Rest'] == 'True' else 'False'
                    three_games_in_10_days = df.loc[i, 'Home Team 3 games in 10 days'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team 3 games in 10 days']
                    four_games_in_17_days = df.loc[i, 'Home Team 4 games in 17 days'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Fair Odds']
                    thanksgiving_favorite = df.loc[i, 'Home Team Thanksgiving Favorite'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Thanksgiving Favorite']
                    christmas_favorite = df.loc[i, 'Home Team Christmas Favorite'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Christmas Favorite']
                    thanksgiving_underdog = df.loc[i, 'Home Team Thanksgiving Underdog'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Thanksgiving Underdog']
                    christmas_underdog = df.loc[i, 'Home Team Christmas Underdog'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Christmas Underdog']
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
    week_requiring_two_selections = config.get('weeks_two_picks', [])
    favored_qualifier = config.get('favored_qualifier', 'Live Sportsbook Odds (If Available)')
    
    # Get all constraints
    pick_must_be_favored = config.get('must_be_favored', False)
    avoid_away_teams_in_close_matchups = config.get('avoid_away_close', False)
    min_away_spread = config.get('min_away_spread', 3.0)
    avoid_close_divisional_matchups = config.get('avoid_close_divisional', False)
    min_div_spread = config.get('min_div_spread', 7.0)
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
        # Filter out already picked teams
        # --- Create home_ev_df ---
        home_ev_df = df.copy() # Start with a copy to keep all original columns
        
        # Rename columns for home_ev_df
        home_ev_df.rename(columns={
            "Home Team": "Hypothetical Current Winner",
            "Away Team": "Hypothetical Current Loser",
            "Home Team EV": "Hypothetical Current Winner EV",
            "Away Team EV": "Hypothetical Current Loser EV",
            "Away Team Adjusted Current Rank": "Hypothetical Current Loser Adjusted Current Rank",
            "Home Team Adjusted Current Rank": "Hypothetical Current Winner Adjusted Current Rank",
            "Home Team Preseason Rank": "Hypothetical Current Winner Preseason Rank",
            "Away Team Preseason Rank": "Hypothetical Current Loser Preseason Rank",
            "Home Team Adjusted Preseason Rank": "Hypothetical Current Winner Adjusted Preseason Rank",
            "Away Team Adjusted Preseason Rank": "Hypothetical Current Loser Adjusted Preseason Rank",
            "Home Team Current Rank": "Hypothetical Current Winner Current Rank",
            "Away Team Current Rank": "Hypothetical Current Loser Current Rank",
            "Home Team Sportsbook Spread": "Hypothetical Current Winner Sportsbook Spread" ,
            "Away Team Sportsbook Spread": "Hypothetical Current Loser Sportsbook Spread",
            "Internal Home Team Spread": "Internal Hypothetical Current Winner Spread",
            "Internal Away Team Spread": "Internal Hypothetical Current Loser Spread"
        }, inplace=True)
        
        # Add "Away Team 1" column
        home_ev_df["Away Team 1"] = home_ev_df["Hypothetical Current Loser"]
        home_ev_df["Home Team 1"] = home_ev_df["Hypothetical Current Winner"]
        
        # --- Create away_ev_df ---
        away_ev_df = df.copy() # Start with a copy to keep all original columns
        
        # Rename columns for away_ev_df
        away_ev_df.rename(columns={
            "Home Team": "Hypothetical Current Loser",
            "Away Team": "Hypothetical Current Winner",
            "Home Team EV": "Hypothetical Current Loser EV",
            "Away Team EV": "Hypothetical Current Winner EV",
            "Away Team Adjusted Current Rank": "Hypothetical Current Winner Adjusted Current Rank",
            "Home Team Adjusted Current Rank": "Hypothetical Current Loser Adjusted Current Rank",
            "Away Team Preseason Rank": "Hypothetical Current Winner Preseason Rank",
            "Home Team Preseason Rank": "Hypothetical Current Loser Preseason Rank",
            "Away Team Adjusted Preseason Rank": "Hypothetical Current Winner Adjusted Preseason Rank",
            "Home Team Adjusted Preseason Rank": "Hypothetical Current Loser Adjusted Preseason Rank",
            "Away Team Current Rank": "Hypothetical Current Winner Current Rank",
            "Home Team Current Rank": "Hypothetical Current Loser Current Rank",
            "Away Team Sportsbook Spread": "Hypothetical Current Winner Sportsbook Spread",
            "Home Team Sportsbook Spread": "Hypothetical Current Loser Sportsbook Spread",
            "Internal Away Team Spread": "Internal Hypothetical Current Winner Spread",
            "Internal Home Team Spread": "Internal Hypothetical Current Loser Spread"
        }, inplace=True)
        
        # Add "Away Team 1" column
        away_ev_df["Away Team 1"] = away_ev_df["Hypothetical Current Winner"]
        away_ev_df["Home Team 1"] = away_ev_df["Hypothetical Current Loser"]
        
        # --- Combine the two dataframes ---
        combined_df = pd.concat([home_ev_df, away_ev_df], ignore_index=True)
        combined_df = combined_df.sort_values(by='Week_Num')
        
        # Display the results (optional)
        print("Original DataFrame (df):")
        print(df)
        print("\nHome EV DataFrame (home_ev_df):")
        print(home_ev_df)
        print("\nAway EV DataFrame (away_ev_df):")
        print(away_ev_df)
        print("\nCombined DataFrame (combined_df):")
        print(combined_df['Week_Num'])
        df = combined_df
        df['Hypothetical Winner Preseason Difference'] = df['Hypothetical Current Winner Preseason Rank'] - df['Hypothetical Current Loser Preseason Rank']	
        df['Hypothetical Winner Adjusted Preseason Difference'] = df['Hypothetical Current Winner Adjusted Preseason Rank'] - df['Hypothetical Current Loser Adjusted Preseason Rank']
        df['Hypothetical Winner Current Difference'] = df['Hypothetical Current Winner Current Rank'] - df['Hypothetical Current Loser Current Rank']	
        df['Hypothetical Winner Adjusted Current Difference'] = df['Hypothetical Current Winner Adjusted Current Rank'] - df['Hypothetical Current Loser Adjusted Current Rank']	

    
        df = df[~df['Hypothetical Current Winner'].isin(picked_teams)].reset_index(drop=True)

        #print(df)
        # Create the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Create binary variables to represent the picks, and store them in a dictionary for easy lookup
        picks = {}
        for i in range(len(df)):
            picks[i] = solver.IntVar(0, 1, 'pick_%i' % i)

        
        for team, req_week in required_teams:
            if req_week > 0:
                # Find the index of the game where 'team' plays in 'req_week'
                # The '&' operator here applies the condition for matching both team and week
                required_game_indices = df[
                    (df['Hypothetical Current Winner'] == team) & 
                    (df['Week_Num'] == req_week)
                ].index.tolist()
                
                # Add the DataFrame index numbers (i) to the set
                required_pick_indices.update(required_game_indices)

        # Add the constraints
        for i in range(len(df)):
            if i in required_pick_indices:
                continue # Skip all general constraints below for this game
            if pick_must_be_favored:
                if favored_qualifier == 'Internal Rankings':
                    if df.loc[i, 'Hypothetical Current Winner'] != df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                else:
                    if df.loc[i, 'Hypothetical Current Winner'] != df.loc[i, 'Favorite']:
                        solver.Add(picks[i] == 0)
            if avoid_away_teams_in_close_matchups == 1:
                if favored_qualifier == 'Internal Rankings':
                    if df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Adjusted Current Difference'] <= min_away_spread and df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank'] > df.loc[i, 'Hypothetical Current Loser Adjusted Current Rank']:
                        solver.Add(picks[i] == 0)
                else:
                    if df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Adjusted Spread'] <= min_away_spread and df.loc[i, 'Hypothetical Current Winner Sportsbook Spread'] < df.loc[i, 'Hypothetical Current Loser Sportsbook Spread']:
                        solver.Add(picks[i] == 0) 

            # If 'Divisional Matchup?' is "Divisional", can only pick if 'Adjusted Current Difference' > 10
            if avoid_close_divisional_matchups == 1:
                if favored_qualifier == 'Internal Rankings':
                    if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Adjusted Current Difference'] <= min_div_spread and df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank'] > df.loc[i, 'Hypothetical Current Loser Adjusted Current Rank']:
                        solver.Add(picks[i] == 0)
                else:
                    if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Adjusted Spread'] <= min_div_spread and df.loc[i, 'Hypothetical Current Winner Sportsbook Spread'] < df.loc[i, 'Hypothetical Current Loser Sportsbook Spread']:
                        solver.Add(picks[i] == 0) 
            # If 'Divisional Matchup?' is "Divisional", can only pick if 'Adjusted Current Difference' > 10
            if avoid_away_divisional_matchups == 1:
                if df.loc[i, 'Divisional Matchup?'] == 1 and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)			
            # Constraints for short rest and 4 games in 17 days (only if team is the Adjusted Current Winner)
            if avoid_away_teams_on_short_rest == 1:
                if df.loc[i, 'Away Team Short Rest'] == 'Yes' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
            if avoid_4_games_in_17_days == 1:
                if df.loc[i, 'Home Team 4 games in 17 days'] == 'Yes' and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Away Team 4 games in 17 days'] == 'No':
                    solver.Add(picks[i] == 0)
                if df.loc[i, 'Away Team 4 games in 17 days'] == 'Yes' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Home Team 4 games in 17 days'] == 'No':
                    solver.Add(picks[i] == 0)
            if avoid_3_games_in_10_days == 1:
                if df.loc[i, 'Home Team 3 games in 10 days'] == 'Yes' and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Away Team 3 games in 10 days'] == 'No':
                    solver.Add(picks[i] == 0)
                if df.loc[i, 'Away Team 3 games in 10 days'] == 'Yes' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Home Team 3 games in 10 days'] == 'No':
                    solver.Add(picks[i] == 0)
            if avoid_international_game == 1:    
                if df.loc[i, 'Actual Stadium'] == 'London, UK' and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
                if df.loc[i, 'Actual Stadium'] == 'London, UK' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
            if avoid_thursday_night == 1:
                if df.loc[i, 'Thursday Night Game'] == 'True':
                    solver.Add(picks[i] == 0)
            if avoid_away_thursday_night == 1:
                if df.loc[i, 'Thursday Night Game'] == 'True' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
            if avoid_teams_with_weekly_rest_disadvantage == 1:
                if df.loc[i, 'Home Team Weekly Rest'] < df.loc [i, 'Away Team Weekly Rest'] and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
                if df.loc[i, 'Away Team Weekly Rest'] < df.loc [i, 'Home Team Weekly Rest'] and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
            if avoid_cumulative_rest_disadvantage == 1:
                if df.loc[i, 'Away Team Current Week Cumulative Rest Advantage'] < -10 and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
                if df.loc[i, 'Home Team Current Week Cumulative Rest Advantage'] < -5 and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                    solver.Add(picks[i] == 0)
            if avoid_away_teams_with_travel_disadvantage == 1:
                if df.loc[i, 'Away Travel Advantage'] < -850 and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
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

        
        # Add the constraints
        for week in df['Week_Num'].unique():
            # Filter picks for the current week
            weekly_picks = [picks[i] for i in range(len(df)) if df.loc[i, 'Week_Num'] == week]

            if selected_contest == "Splash Sports" and week in week_requiring_two_selections:
                # For Splash Sports and later weeks, two teams must be selected
                solver.Add(solver.Sum(weekly_picks) == 2)
            else:
                # For other contests or earlier weeks, one team per week
                solver.Add(solver.Sum(weekly_picks) == 1)

        for team in df['Hypothetical Current Winner'].unique():
            # Can't pick a team more than once
            solver.Add(solver.Sum([picks[i] for i in range(len(df)) if df.loc[i, 'Hypothetical Current Winner'] == team]) <= 1)

        


        # Dynamically create the forbidden solution list
        forbidden_solutions_1 = []
        if iteration > 0:
            for previous_iteration in range(iteration):
                if selected_contest == 'Circa':
                    previous_picks_df = pd.read_csv(f"circa_picks_ir_subset_{previous_iteration + 1}.csv")
                elif selected_contest == 'Splash Sports':
                    previous_picks_df = pd.read_csv(f"splash_picks_ir_subset_{previous_iteration + 1}.csv")
                else:
                    previous_picks_df = pd.read_csv(f"dk_picks_ir_subset_{previous_iteration + 1}.csv")

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
                weekly_rows = df[df['Week'] == week]

                # Check if any of these rows match a forbidden pick from that week
                for _, row in weekly_rows.iterrows():
                    if row['Hypothetical Current Winner'] in picks_in_week: # The 'Favorite' column is what you're using to identify the pick
                        # Get the index of this row
                        pick_index = row.name
                        forbidden_pick_variables.append(picks[pick_index])

            # The constraint now ensures that at least one of the forbidden picks is not selected
            solver.Add(solver.Sum([1 - v for v in forbidden_pick_variables]) >= 1)
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

                    week = df.loc[i, 'Week']
                    date = df.loc[i, 'Date']
                    date = pd.to_datetime(date, format='%b %d, %Y')
                    time = df.loc[i, 'Time']
                    location = df.loc[i, 'Actual Stadium']
                    pick = df.loc[i,'Hypothetical Current Winner']
                    opponent = df.loc[i, 'Hypothetical Current Loser']
                    win_odds = round(df.loc[i, 'Home Team Fair Odds'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Fair Odds'], 2)
                    pick_percent = round(df.loc[i, 'Home Pick %'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Pick %'], 2)
                    expected_availability = round(df.loc[i, 'Home Team Expected Availability'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Expected Availability'], 2)
                    divisional_game = 'Divisional' if df.loc[i, 'Divisional Matchup Boolean'] else ''
                    home_team = 'Home Team' if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else 'Away Team'
                    weekly_rest = df.loc[i, 'Home Team Weekly Rest'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Weekly Rest']
                    weekly_rest_advantage = df.loc[i, 'Weekly Home Rest Advantage'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Weekly Away Rest Advantage']
                    cumulative_rest = df.loc[i, 'Home Cumulative Rest Advantage'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Cumulative Rest Advantage']
                    cumulative_rest_advantage = df.loc[i, 'Home Team Current Week Cumulative Rest Advantage'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Current Week Cumulative Rest Advantage']
                    travel_advantage = df.loc[i, 'Home Travel Advantage'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Travel Advantage']
                    back_to_back_away_games = 'True' if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Away Team 1'] and df.loc[i, 'Back to Back Away Games'] == 'True' else 'False'
                    thursday_night_game = 'Thursday Night Game' if df.loc[i, "Thursday Night Game"] == 'True' else 'Sunday/Monday Game'
                    international_game = 'International Game' if df.loc[i, 'Actual Stadium'] == 'London, UK' else 'Domestic Game'
                    previous_opponent = df.loc[i, 'Home Team Previous Opponent'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Previous Opponent']
                    previous_game_location = df.loc[i, 'Home Team Previous Location'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Previous Location']
                    next_opponent = df.loc[i, 'Home Team Next Opponent'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Next Opponent']
                    next_game_location = df.loc[i, 'Home Team Next Location'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Next Location']

                    internal_ranking_fair_odds = df.loc[i, 'Internal Home Team Fair Odds'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Internal Away Team Fair Odds']
                    future_value = df.loc[i, 'Home Team Star Rating'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Star Rating']
                    sportbook_moneyline = df.loc[i, 'Home Team Moneyline'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Moneyline']
                    internal_moneyline = df.loc[i, 'Internal Home Team Moneyline'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Internal Away Team Moneyline']
                    contest_selections = df.loc[i, 'Expected Home Team Picks'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Expected Away Team Picks']
                    survival_rate = df.loc[i, 'Home Expected Survival Rate'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Expected Survival Rate']
                    elimination_percent = df.loc[i, 'Home Expected Elimination Percent'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Expected Elimination Percent']
                    survivors = df.loc[i, 'Expected Home Team Survivors'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Expected Away Team Survivors']
                    eliminations = df.loc[i, 'Expected Home Team Eliminations'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Expected Away Team Eliminations']
                    preseason_rank = df.loc[i, 'Hypothetical Current Winner Preseason Rank']
                    adjusted_preseason_rank = df.loc[i, 'Hypothetical Current Winner Adjusted Preseason Rank']
                    current_rank = df.loc[i, 'Hypothetical Current Winner Current Rank']
                    adjusted_current_rank = df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank']
                    away_team_short_rest = 'True' if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Away Team 1'] and df.loc[i, 'Away Team Short Rest'] == 'True' else 'False'
                    three_games_in_10_days = df.loc[i, 'Home Team 3 games in 10 days'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team 3 games in 10 days']
                    four_games_in_17_days = df.loc[i, 'Home Team 4 games in 17 days'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Fair Odds']
                    thanksgiving_favorite = df.loc[i, 'Home Team Thanksgiving Favorite'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Thanksgiving Favorite']
                    christmas_favorite = df.loc[i, 'Home Team Christmas Favorite'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Christmas Favorite']
                    thanksgiving_underdog = df.loc[i, 'Home Team Thanksgiving Underdog'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Thanksgiving Underdog']
                    christmas_underdog = df.loc[i, 'Home Team Christmas Underdog'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Christmas Underdog']
                    live_odds_unavailable = df.loc[i, 'No Live Odds Available - Internal Rankings Used for Moneyline Calculation']
                    live_odds_spread = df.loc[i, 'Hypothetical Current Winner Sportsbook Spread']
                    internal_spread = df.loc[i, 'Internal Hypothetical Current Winner Spread']
                    

                    # Get differences
                    preseason_difference = df.loc[i, 'Hypothetical Winner Preseason Difference']
                    adjusted_preseason_difference = df.loc[i, 'Hypothetical Winner Adjusted Preseason Difference']
                    current_difference = df.loc[i, 'Hypothetical Winner Current Difference']
                    adjusted_current_difference = df.loc[i, 'Hypothetical Winner Adjusted Current Difference']
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
            st.write(f'Total Sportsbook Spread: , :blue[{sum_sportsbook_spread}]')
            
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

# Contest Options
contest_options = ["Circa", "DraftKings", "Splash Sports"]
subcontest_options = [
    "The Big Splash ($150 Entry)", "High Roller ($1000 Entry)", "Free RotoWire (Free Entry)",
    "4 for 4 ($50 Entry)", "For the Fans ($40 Entry)", "Walker's Ultimate Survivor ($25 Entry)",
    "Ship It Nation ($25 Entry)"
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



def update_config_value(key):
    """Generic callback to update a top-level key in current_config from a widget's session state key."""
    widget_key = key + "_widget"
    if widget_key in st.session_state:
        st.session_state.current_config[key] = st.session_state[widget_key]

def update_nested_value(outer_key, inner_key):
    """Generic callback for nested dictionaries (e.g., rankings, availabilities)."""
    widget_key = f"{outer_key}_{inner_key}_widget".replace(" ", "_").replace("(", "").replace(")", "").replace("$","")
    if widget_key in st.session_state:
        value = st.session_state[widget_key]
        if outer_key == 'team_availabilities' or outer_key == 'pick_percentages':
             value = value / 100.0
        elif outer_key == 'team_rankings' and value == "Default":
             value = DEFAULT_RANKS.get(inner_key, 0)

        if outer_key not in st.session_state.current_config:
             st.session_state.current_config[outer_key] = {}
        st.session_state.current_config[outer_key][inner_key] = value

def update_pick_percentage(week, team_name):
    """Specific callback for the nested pick percentage dictionary."""
    widget_key = f"pick_perc_week_{week}_{team_name.replace(' ', '_')}_widget"
    if widget_key in st.session_state:
        percentage_int = st.session_state[widget_key]
        percentage_float = percentage_int / 100.0
        week_key = f"week_{week}"

        if week_key not in st.session_state.current_config['pick_percentages']:
            st.session_state.current_config['pick_percentages'][week_key] = {}
            
        st.session_state.current_config['pick_percentages'][week_key][team_name] = percentage_float


# --- 4. Main Streamlit App ---

st.set_page_config(layout="wide") # Use wide layout

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
            'subcontest': "The Big Splash ($150 Entry)",
            'weeks_two_picks': [],
            'has_picked_teams': False,
            'prohibited_picks': [],
            'choose_weeks': False,
            'starting_week': 1,
            'ending_week': 21, # Default to Circa max
            'current_week_entries': -1,
            'use_live_data': False,
            'team_availabilities': {team: -1.0 for team in nfl_teams},
            'require_team': False,
            'required_weeks': {team: 0 for team in nfl_teams},
            'prohibit_teams': False,
            'prohibited_weeks': {team: [] for team in nfl_teams},
            'custom_rankings': False,
            'team_rankings': {team: 'Default' for team in nfl_teams}, # Use 'Default' string marker
            'must_be_favored': False,
            'favored_qualifier': 'Live Sportsbook Odds (If Available)',
            'add_constraints': False,
            # Simple constraint flags (can use 0/1 or False/True)
            'avoid_away_short_rest': False,
            'avoid_close_divisional': False,
            'min_div_spread': 7.0,
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
            if st.form_submit_button("๐ ’พ Save", use_container_width=True):
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
    st.image(LOGO_PATH, width=150)
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
        st.multiselect(
            "Which weeks do you need to select two teams?:",
            options=range(1, 19),
            key='weeks_two_picks_widget',
            default=st.session_state.current_config['weeks_two_picks'],
            on_change=update_config_value,
            args=('weeks_two_picks',),
            help=two_team_selections_help_text
        )
        # Display helper text based on subcontest
        subcontest = st.session_state.current_config['subcontest']
        if subcontest == "The Big Splash ($150 Entry)":
            st.write("Weeks requiring double picks in The Big Splash Survivor Contest: :green[11, 12, 13, 14, 15, 16, 17, 18]")
        elif subcontest == "4 for 4 ($50 Entry)":
            st.write("Weeks requiring double picks in the 4 for 4 Survivor Contest: :green[11, 12, 13, 14, 15, 16, 17, 18]")
        elif subcontest == "Free RotoWire (Free Entry)":
            st.write("Weeks requiring double picks in the Free RotoWire Survivor Contest: :green[None]")
        elif subcontest == "For the Fans ($40 Entry)":
            st.write("Weeks requiring double picks in the For the Fan Survivor Contest: :green[14, 15, 16, 17, 18]")
        elif subcontest == "Walker's Ultimate Survivor ($25 Entry)":
            st.write("Weeks requiring double picks in Walker's Ultimate Survivor Survivor Contest: :green[6, 12, 13, 14, 15, 16, 17, 18]")
        elif subcontest == "Ship It Nation ($25 Entry)":
            st.write("Weeks requiring double picks in the Ship It Nation Survivor Contest: :green[12, 13, 14, 15, 16, 17, 18]")
        elif subcontest == "High Roller ($1000 Entry)":
            st.write("Weeks requiring double picks in the High Roller Survivor Contest: :green[None]")

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
    st.write(f"Entered: {st.session_state.current_config['current_week_entries']}")
    st.write('---')


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

    st.write("Set availability % (0-100). Use -1 to estimate automatically.")
    
    # Use columns for better layout
    num_cols = 3
    cols = st.columns(num_cols)
    col_idx = 0

    for team in nfl_teams:
        with cols[col_idx]:
            outer_key = 'team_availabilities'
            inner_key = team
            widget_key = f"{outer_key}_{inner_key}_widget".replace(" ", "_")
            
            # Get current value, handle potential float issues from JSON load
            current_val_float = st.session_state.current_config[outer_key].get(inner_key, -1.0)
            current_val_int = int(current_val_float * 100) # Convert to integer for slider

            st.slider(
                f"{team}:",
                min_value=-1,
                max_value=100,
                key=widget_key,
                value=current_val_int,
                on_change=update_nested_value,
                args=(outer_key, inner_key)
            )
            # Display current setting from the dictionary
            display_val = st.session_state.current_config[outer_key].get(inner_key, -1.0)
            if display_val < 0:
                 st.caption(":red[Auto]")
            else:
                 st.caption(f":green[{display_val*100:.0f}%]")

        col_idx = (col_idx + 1) % num_cols # Cycle through columns

    st.write('---')

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
            end_w = st.session_state.current_config['ending_week'] # exclusive
            
            for week in range(start_w, end_w):
                # Use a checkbox to optionally show/hide each week's sliders
                week_key = f"week_{week}"
                show_week = st.checkbox(f"Adjust Week {week} Pick %?", key=f"show_week_{week}_perc")
                
                if show_week:
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
                             current_val_int = int(current_val_float * 100)

                             st.slider(
                                 f"{team} Wk {week} %:",
                                 min_value=-1,
                                 max_value=100,
                                 key=widget_key,
                                 value=current_val_int,
                                 on_change=update_pick_percentage, # Use specific updater
                                 args=(week, inner_key)
                             )
                             # Display current setting
                             display_val = st.session_state.current_config[outer_key].get(week_key, {}).get(inner_key, -1.0)
                             if display_val < 0:
                                  st.caption(":red[Auto]")
                             else:
                                  st.caption(f":green[{display_val*100:.0f}%]")

                         perc_col_idx = (perc_col_idx + 1) % 3
                     st.write("---") # Separator between weeks
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
        df_with_availability = get_predicted_pick_percentages(pd, config, collect_schedule_travel_ranking_data_df)
        st.write("Step 3a Completed (Availability Calculated).")
        
        # Step 3b: Predict Pick % (With Availability)
        st.write("Step 3b/6: Refining Pick Percentages using Availability...")
        # --- Pass the dataframe from Step 3a into this function ---
        nfl_schedule_pick_percentages_df = get_predicted_pick_percentages_with_availability(pd, config, df_with_availability)
        st.write("Step 3b Completed (Final Pick % Predicted).")
        
        # Step 4: Calculate EV
        st.write("Step 4/6: Calculating Live Expected Value...")
        with st.spinner('Calculating EV... (This may take 5-10 minutes)'):
            # Pass the dataframe from Step 3 into this function
            full_df_with_ev = calculate_ev(nfl_schedule_pick_percentages_df, config, use_cache)
        st.write("Step 4 Completed.")
        st.dataframe(full_df_with_ev) 

        # Step 4b: Reformat
        st.write("Reformatting Data for Solver...")
        reformatted_df = reformat_df(full_df_with_ev, config)
        st.write("Reformatting Complete.")
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

