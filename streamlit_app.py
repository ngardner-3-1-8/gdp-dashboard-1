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


def get_schedule():
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

def collect_schedule_travel_ranking_data(pd):
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
    stadiums = {
        'Arizona Cardinals': ['State Farm Stadium', 33.5277, -112.262608, 'America/Denver', 'NFC West', preseason_az_rank, az_rank, az_home_adv, az_away_adj],
        'Atlanta Falcons': ['Mercedez-Benz Stadium', 33.757614, -84.400972, 'America/New_York', 'NFC South', preseason_atl_rank, atl_rank, atl_home_adv, atl_away_adj],
        'Baltimore Ravens': ['M&T Stadium', 39.277969, -76.622767, 'America/New_York', 'AFC North', preseason_bal_rank, bal_rank, bal_home_adv, bal_away_adj],
        'Buffalo Bills': ['Highmark Stadium', 42.773739, -78.786978, 'America/New_York', 'AFC East', preseason_buf_rank, buf_rank, buf_home_adv, buf_away_adj],
        'Carolina Panthers': ['Bank of America Stadium', 35.225808, -80.852861, 'America/New_York', 'NFC South', preseason_car_rank, car_rank, car_home_adv, car_away_adj],
        'Chicago Bears': ['Soldier Field', 41.862306, -87.616672, 'America/Chicago', 'NFC North', preseason_chi_rank, chi_rank, chi_home_adv, chi_away_adj],
        'Cincinnati Bengals': ['Paycor Stadium', 39.095442, -84.516039, 'America/New_York', 'AFC North', preseason_cin_rank, cin_rank, cin_home_adv, cin_away_adj],
        'Cleveland Browns': ['Cleveland Browns Stadium', 41.506022, -81.699564, 'America/New_York', 'AFC North', preseason_cle_rank, cle_rank, cle_home_adv, cle_away_adj],
        'Dallas Cowboys': ['AT&T Stadium', 32.747778, -97.092778, 'America/Chicago', 'NFC East', preseason_dal_rank, dal_rank, dal_home_adv, dal_away_adj],
        'Denver Broncos': ['Empower Field at Mile High', 39.743936, -105.020097, 'America/Denver', 'AFC West', preseason_den_rank, den_rank, den_home_adv, den_away_adj],
        'Detroit Lions': ['Ford Field', 42.340156, -83.045808, 'America/New_York', 'NFC North', preseason_det_rank, det_rank, det_home_adv, det_away_adj],
        'Green Bay Packers': ['Lambeau Field', 44.501306, -88.062167, 'America/Chicago', 'NFC North', preseason_gb_rank, gb_rank, gb_home_adv, gb_away_adj],
        'Houston Texans': ['NRG Stadium', 29.684781, -95.410956, 'America/Chicago', 'AFC South', preseason_hou_rank, hou_rank, hou_home_adv, hou_away_adj],
        'Indianapolis Colts': ['Lucas Oil Stadium', 39.760056, -86.163806, 'America/New_York', 'AFC South', preseason_ind_rank, ind_rank, ind_home_adv, ind_away_adj],
        'Jacksonville Jaguars': ['Everbank Stadium', 30.323925, -81.637356, 'America/New_York', 'AFC South', preseason_jax_rank, jax_rank, jax_home_adv, jax_away_adj],
        'Kansas City Chiefs': ['Arrowhead Stadium', 39.048786, -94.484566, 'America/Chicago', 'AFC West', preseason_kc_rank, kc_rank, kc_home_adv, kc_away_adj],
        'Las Vegas Raiders': ['Allegiant Stadium', 36.090794, -115.183952, 'America/Los_Angeles', 'AFC West', preseason_lv_rank, lv_rank, lv_home_adv, lv_away_adj],
        'Los Angeles Chargers': ['SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'AFC West', preseason_lac_rank, lac_rank, lac_home_adv, lac_away_adj],
        'Los Angeles Rams': ['SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'NFC West', preseason_lar_rank, lar_rank, lar_home_adv, lar_away_adj],
        'Miami Dolphins': ['Hard Rock Stadium', 25.957919, -80.238842, 'America/New_York', 'AFC East', preseason_mia_rank, mia_rank, mia_home_adv, mia_away_adj],
        'Minnesota Vikings': ['U.S Bank Stadium', 44.973881, -93.258094, 'America/Chicago', 'NFC North', preseason_min_rank, min_rank, min_home_adv, min_away_adj],
        'New England Patriots': ['Gillette Stadium', 42.090925, -71.26435, 'America/New_York', 'AFC East', preseason_ne_rank, ne_rank, ne_home_adv, ne_away_adj],
        'New Orleans Saints': ['Caesars Superdome', 29.950931, -90.081364, 'America/Chicago', 'NFC South', preseason_no_rank, no_rank, no_home_adv, no_away_adj],
        'New York Giants': ['MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'NFC East', preseason_nyg_rank, nyg_rank, nyg_home_adv, nyg_away_adj],
        'New York Jets': ['MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'AFC East', preseason_nyj_rank, nyj_rank, nyj_home_adv, nyj_away_adj],
        'Philadelphia Eagles': ['Lincoln Financial Field', 39.900775, -75.167453, 'America/New_York', 'NFC East', preseason_phi_rank, phi_rank, phi_home_adv, phi_away_adj],
        'Pittsburgh Steelers': ['Acrisure Stadium', 40.446786, -80.015761, 'America/New_York', 'AFC North', preseason_pit_rank, pit_rank, pit_home_adv, pit_away_adj],
        'San Francisco 49ers': ['Levi\'s Stadium', 37.713486, -122.386256, 'America/Los_Angeles', 'NFC West', preseason_sf_rank, sf_rank, sf_home_adv, sf_away_adj],
        'Seattle Seahawks': ['Lumen Field', 47.595153, -122.331625, 'America/Los_Angeles', 'NFC West', preseason_sea_rank, sea_rank, sea_home_adv, sea_away_adj],
        'Tampa Bay Buccaneers': ['Raymomd James Stadium', 27.975967, -82.50335, 'America/New_York', 'NFC South', preseason_tb_rank, tb_rank, tb_home_adv, tb_away_adj],
        'Tennessee Titans': ['Nissan Stadium', 36.166461, -86.771289, 'America/Chicago', 'AFC South', preseason_ten_rank, ten_rank, ten_home_adv, ten_away_adj],
        'Washington Commanders': ['FedExField', 38.907697, -76.864517, 'America/New_York', 'NFC East', preseason_was_rank, was_rank, was_home_adv, was_away_adj]
    }

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

    df['Away Team Adjusted Current Rank'] = df['Away Team'].map(lambda team: stadiums[team][6]) + np.where((df['Away Travel Advantage'] < -100) & (df['Home Stadium'] == df['Actual Stadium']), -.125, 0) - pd.to_numeric(df['Away Timezone Advantage']*.25, errors='coerce').fillna(0)-pd.to_numeric(df['Weekly Away Rest Advantage'], errors='coerce').fillna(0)-.125*df['Away Team Current Week Cumulative Rest Advantage'] - np.where((df['Away Team'].map(lambda team: stadiums[team][0])) != df['Home Team'].map(lambda team: stadiums[team][0]), df['Away Team'].map(lambda team: stadiums[team][8]), 0)
    df['Home Team Adjusted Current Rank'] = df['Home Team'].map(lambda team: stadiums[team][6]) - np.where((df['Away Travel Advantage'] < -100) & (df['Home Stadium'] == df['Actual Stadium']), .125, 0) - pd.to_numeric(df['Home Timezone Advantage']*.25, errors='coerce').fillna(0)-pd.to_numeric(df['Weekly Home Rest Advantage'], errors='coerce').fillna(0)-.125*df['Home Team Current Week Cumulative Rest Advantage'] + np.where((df['Away Team'].map(lambda team: stadiums[team][0])) != df['Home Team'].map(lambda team: stadiums[team][0]), df['Home Team'].map(lambda team: stadiums[team][7]), 0)

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

    
    @st.cache_resource
    def get_webdriver():
        """Initializes and caches the Selenium WebDriver."""
        options = Options()
        options.add_argument("--headless")  # Run in headless mode (no browser UI)
        options.add_argument("--no-sandbox")  # Required for running in many Linux environments (like Docker/Streamlit Cloud)
        options.add_argument("--disable-dev-shm-usage")  # Overcomes limited /dev/shm size in some environments
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36")
        options.add_argument("--window-size=1920,1080") # Set a consistent window size
        
        try:
            # This will work if chromedriver is in the system's PATH, or if 'chromium-driver'
            # from packages.txt installs it to a discoverable location like /usr/bin/chromedriver
            driver = webdriver.Chrome(options=options)
            print("Web Driver Initialized Successfully")
        except Exception as e:
            st.error(f"Failed to initialize WebDriver directly: {e}. Attempting specific path.")
            try:
                # Common path for chromedriver installed via apt in Debian-based systems (like Streamlit Cloud's base)
                service = webdriver.chrome.service.Service('/usr/bin/chromedriver')
                driver = webdriver.Chrome(service=service, options=options)
            except Exception as e_path:
                st.error(f"Failed to initialize WebDriver even with explicit path: {e_path}. "
                         "Ensure Chromium and Chromedriver are correctly installed and accessible.")
                raise e_path # Re-raise to stop execution if essential component fails
        return driver

    def get_preseason_odds():
        st.write ("Getting Oreseason Odds")
        url = "https://sportsbook.draftkings.com/leagues/football/nfl?category=game-lines&subcategory=game"
    
        team_name_mapping = {
            "ARI Cardinals" : "Arizona Cardinals", "ATL Falcons" : "Atlanta Falcons",
            "BAL Ravens" : "Baltimore Ravens", "BUF Bills" : "Buffalo Bills",
            "CAR Panthers" : "Carolina Panthers", "CHI Bears" : "Chicago Bears",
            "CIN Bengals" : 'Cincinnati Bengals', "CLE Browns" : 'Cleveland Browns',
            "DAL Cowboys" : 'Dallas Cowboys', "DEN Broncos" : 'Denver Broncos',
            "DET Lions" : 'Detroit Lions', "GB Packers" : 'Green Bay Packers',
            "HOU Texans" : 'Houston Texans', "IND Colts" : 'Indianapolis Colts',
            "JAX Jaguars" : 'Jacksonville Jaguars', "KC Chiefs" : 'Kansas City Chiefs',
            "LV Raiders" : 'Las Vegas Raiders', "LA Chargers" : 'Los Angeles Chargers',
            "LA Rams" : 'Los Angeles Rams', "MIA Dolphins" : 'Miami Dolphins',
            "MIN Vikings" : 'Minnesota Vikings', "NE Patriots" : 'New England Patriots',
            "NO Saints" : 'New Orleans Saints', "NY Giants" : 'New York Giants',
            "NY Jets" : 'New York Jets', "PHI Eagles" : 'Philadelphia Eagles',
            "PIT Steelers" : 'Pittsburgh Steelers', "SF 49ers" : 'San Francisco 49ers',
            "SEA Seahawks" : 'Seattle Seahawks', "TB Buccaneers" : 'Tampa Bay Buccaneers',
            "TEN Titans" : 'Tennessee Titans', "WAS Commanders" : 'Washington Commanders'
        }
    
        games = []
        
        # Define the Eastern and UTC Timezones
        eastern_tz = pytz.timezone('America/New_York')
        utc_tz = pytz.utc
    
        try:
            driver = get_webdriver() # Get the cached WebDriver instance
            st.write(f"Navigating to URL with Selenium: {url}")
            driver.get(url)
    
            # --- Selenium Wait Condition ---
            # This selector needs to be something present when the main dynamic content loads.
            # Given your original code was looking for `cms-market-selector-section-wrapper bottom-margin`,
            # we'll use that as the target for the wait, assuming it *does* appear after JS loads.
            # If this still fails, THIS is the first place to re-inspect in the browser's Elements tab.
            main_content_load_selector = 'div.cms-market-selector-section-wrapper.bottom-margin'
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, main_content_load_selector))
                )
                st.success(f"Selenium: Dynamic content appears loaded (found: '{main_content_load_selector}').")
            except Exception as e:
                st.error(f"Selenium: Timeout waiting for dynamic content on '{main_content_load_selector}'. "
                         f"Page might not have loaded correctly or selector is wrong: {e}")
                st.info("Attempting to get page source anyway for debugging.")
            
            # Get the page source *after* JavaScript has had a chance to render
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
    
            st.markdown("### Rendered HTML (All characters for debugging):")
            st.code(html_content) # Use st.code for better display in Streamlit
    
            # --- Your Original BeautifulSoup Parsing Logic (Adapted for Selenium's HTML) ---
            # Assuming these classes are now present in the `html_content` from Selenium
            
            game_dates = soup.find_all('div', class_='cms-market-selector-section-wrapper bottom-margin')
            if not game_dates:
                st.error("Error: No game dates (div.cms-market-selector-section-wrapper bottom-margin) found on the rendered page.")
                return pd.DataFrame()
            
            st.info(f"Found {len(game_dates)} game date cards/sections.")
            
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            for card_index, card in enumerate(game_dates):
                game_date_header = card.find('div', class_='cms-market-selector-static-header')
                if not game_date_header:
                    st.warning(f"Warning: No date header found for game date section {card_index + 1}.")
                    continue
                
                current_date_str_raw = "Unknown Date"
                date_head = game_date_header.find('div', class_='cms-market-selector-title__wrapper')
                # st.write("Line [approx 622] - date head found:") # Removed line numbers
                # st.write(date_head)
                if date_head:
                    date_header_title = date_head.find('p', class_='cms-market-selector-event-title')
                    if date_header_title:
                        current_date_str_raw = date_header_title.text.strip()
                        # st.write("Line [approx 628]: current date found:")
                        # st.write(current_date_str_raw)
                    else:
                        st.warning("Line [approx 628]: ERROR: current date title not found.")
                
                # --- Date Parsing Logic ---
                parsed_date_obj = None
                clean_date_str = re.sub(r'(\d+)(TH|ST|ND|RD)', r'\1', current_date_str_raw.upper())
                clean_date_str = clean_date_str.replace(',', '').strip()
                
                # st.write("Line [approx 642]: Cleaned Date String:")
                # st.write(clean_date_str)
                
                date_formats = [
                    "%a %b %d",     # "FRI SEP 5"
                    "%A %B %d",     # "FRIDAY SEPTEMBER 5"
                    "%B %d",        # "SEPTEMBER 5"
                    "%a, %b %d"     # "FRI, SEP 5"
                ]
                
                temp_parsed_date = None
                format_used = None
                for fmt in date_formats:
                    try:
                        # First, try parsing with current year
                        temp_parsed_date = datetime.strptime(f"{clean_date_str} {current_year}", f"{fmt} %Y").date()
                        format_used = fmt
                        break
                    except ValueError:
                        continue
                
                if temp_parsed_date:
                    # Heuristic for next year: if month is earlier than current month AND date is in the past
                    if temp_parsed_date.month < current_month and temp_parsed_date < datetime.now().date():
                        try:
                            # Attempt to parse with the next year
                            temp_parsed_date = datetime.strptime(f"{clean_date_str} {current_year + 1}", f"{format_used} %Y").date()
                        except ValueError:
                            pass # Keep the current year if next year also fails or doesn't apply
                    
                    parsed_date_obj = temp_parsed_date
                    current_date_for_game = parsed_date_obj.strftime('%Y-%m-%d')
                else:
                    st.warning(f"Warning: Could not parse date string: '{current_date_str_raw}'. Date will remain as string.")
                    current_date_for_game = current_date_str_raw
                    parsed_date_obj = None # Ensure it's None if parsing failed
                
                st.info(f"Processing table for date: {current_date_for_game}")
                
                game_body = card.find('div', class_='cms-market-selector-static__event-wrapper') # Changed from find_all to find as per original logic
                if not game_body:
                    st.warning(f"Warning: No games found under date header for {current_date_for_game}.")
                    continue
                
                # This class `cb-market__template cb-market__template--4-columns` is very specific.
                # If `game_body` exists, then `find_all` on it for this class should yield rows.
                team_data_rows = game_body.find_all('div', class_='cb-market__template cb-market__template--4-columns', recursive=False)
                
                actual_team_rows = []
                for row in team_data_rows: # Iterating over `team_data_rows` not `team_dta`
                    # Your original code used `row` here, not `team`. Corrected.
                    if row.find('span', class_='cb-market__label-inner cb-market__label-inner--parlay'):
                        actual_team_rows.append(row)
                
                st.info(f"Found {len(actual_team_rows)} actual team rows for {current_date_for_game}.")
                
                game_data = {} # Reset for each new date section
                for i, row in enumerate(actual_team_rows):
                    team_name_element = row.find('span', class_='cb-market__label-inner cb-market__label-inner--parlay')
    
                    all_odds_buttons = row.find_all('button', class_='cb-market__button cb-market__button--regular')
                    
                    odds_button_element = None
                    team_moneyline_element = None # Renamed from team_moneyline to avoid conflict with `odds` later
                    if len(all_odds_buttons) >= 3:
                        odds_button_element = all_odds_buttons[2] # Select the third button (0-indexed)
                        team_moneyline_element = odds_button_element.find('span', class_='cb-market__button-odds')
                    else:
                        st.warning(f"Warning: Not enough odds buttons found for row {i} in date {current_date_for_game}. Expected 3, got {len(all_odds_buttons)}.")
                    
                    time_element = row.find('span', class_='cb-event-cell__start-time') # This might be the time for each game
    
                    if not team_name_element:
                        st.warning("Warning: Skipping a row that was expected to be a team row but missing name.")
                        continue
                    
                    team_raw = team_name_element.text.strip()
                    team = team_name_mapping.get(team_raw, team_raw)
                    
                    odds_val = None
                    if team_moneyline_element: # Check the specific moneyline span
                        odds_text = team_moneyline_element.text.strip().replace('−', '-')
                        if odds_text:
                            try:
                                odds_val = int(odds_text) # Odds are usually ints for moneyline
                            except ValueError:
                                st.warning(f"Warning: Could not parse odds for {team} on {current_date_for_game}: '{odds_text}'")
                                odds_val = None
                    
                    # Logic for Home/Away teams and collecting game data
                    # This assumes alternating Away/Home rows within actual_team_rows
                    if i % 2 == 0:  # Away Team
                        # Initialize game_data for a new game
                        game_data = {'Date': current_date_for_game} 
                        
                        if time_element:
                            raw_time_str = time_element.text.strip()
                            game_data['Time_Raw'] = raw_time_str # Keep raw time for debugging
                            
                            converted_to_eastern = None
                            if parsed_date_obj: # Ensure we have a valid date object to combine with
                                # This time parsing logic needs to be robust. 
                                # If DK uses a simple 'HH:MM AM/PM' then this is fine.
                                time_to_parse = re.sub(r'\s*(ET|PT|CT|MT)\s*', '', raw_time_str, flags=re.IGNORECASE).strip()
                                time_to_parse = re.sub(r'(\d)([AP]M)', r'\1 \2', time_to_parse) # Ensure space before AM/PM
                                
                                time_formats_for_dt = [
                                    "%I:%M %p", # "12:20 AM", "7:00 PM"
                                    "%I:%M%p",  # "12:20AM", "7:00PM" (no space, regex should add one)
                                    "%H:%M"     # "19:00" (24-hour format)
                                ]
                                
                                time_parsed_success = False
                                for fmt in time_formats_for_dt:
                                    try:
                                        naive_dt_from_site = datetime.strptime(f"{parsed_date_obj.strftime('%Y-%m-%d')} {time_to_parse}", f'%Y-%m-%d {fmt}')
                                        
                                        # Assuming the time on DraftKings is in Eastern Time (America/New_York)
                                        localized_dt_eastern = eastern_tz.localize(naive_dt_from_site)
                                        converted_to_eastern = localized_dt_eastern.isoformat() # Store as ISO string with TZ info
                                        
                                        time_parsed_success = True
                                        break
                                    except ValueError as ve:
                                        continue # Try the next format
                                
                                if time_parsed_success and converted_to_eastern:
                                    game_data['Event DateTime (ET)'] = converted_to_eastern
                                    game_data['Time'] = datetime.fromisoformat(converted_to_eastern).strftime('%I:%M %p')
                                else:
                                    st.warning(f"Warning: Could not parse or convert datetime for {team} on {current_date_for_game} at '{raw_time_str}'. Time will remain raw.")
                                    game_data['Time'] = raw_time_str
                                    game_data['Event DateTime (ET)'] = None
                            else:
                                game_data['Time'] = raw_time_str
                                game_data['Event DateTime (ET)'] = None
                        else:
                            game_data['Time'] = 'N/A'
                            game_data['Event DateTime (ET)'] = None
                            st.warning(f"Warning: Time element not found for Away Team {team} on {current_date_for_game}")
                        
                        game_data['Away Team'] = team
                        game_data['Away Odds'] = odds_val # Renamed variable
    
                        # If this is the last row and it's an Away Team, it means the Home team is missing.
                        if i == len(actual_team_rows) - 1:
                            st.warning(f"Warning: Game for {team} (Away) on {current_date_for_game} is incomplete (missing Home team).")
                            # You might choose to append game_data anyway with N/A for home, or discard.
                            # For now, we'll clear it so it doesn't get processed further.
                            game_data = {} 
                    else:  # Home Team
                        if 'Away Team' not in game_data:
                            st.warning(f"Warning: Home team {team} on {current_date_for_game} found without a preceding Away team. Skipping this pair.")
                            game_data = {} # Clear potentially orphaned data
                            continue
                        
                        game_data['Home Team'] = team
                        game_data['Home Odds'] = odds_val # Renamed variable
                        
                        games.append(game_data)
                        game_data = {} # Reset for the next pair
                
        except Exception as e:
            st.error(f"An overarching error occurred during the overall scraping process: {e}")
            return pd.DataFrame()
        finally:
            # No driver.quit() here as the driver is managed by st.cache_resource
            pass
    
        df = pd.DataFrame(games)
        
        # --- Post-processing the DataFrame ---
        # Ensure 'Event DateTime (ET)' is a proper datetime object and correctly localized
        df['Event DateTime (ET)'] = pd.to_datetime(df['Event DateTime (ET)'], errors='coerce', utc=True)
        df['Event DateTime (ET)'] = df['Event DateTime (ET)'].dt.tz_convert(eastern_tz)
        
        # Update the 'Date' column to reflect the date in Eastern Time from the datetime object
        df['Date'] = df['Event DateTime (ET)'].dt.date.astype(str) # Format as 'YYYY-MM-DD'
        
        # Remove the 'Time_Raw' column as it's for debugging
        if 'Time_Raw' in df.columns:
            df = df.drop(columns=['Time_Raw'])
        
        # Reorder columns to match your desired output format
        desired_columns = [
            'Date', 'Time', 'Event DateTime (ET)',
            'Away Team', 'Away Odds',
            'Home Team', 'Home Odds'
        ]
        # Handle cases where some columns might be missing if data wasn't found
        df = df[[col for col in desired_columns if col in df.columns]]
    
        st.dataframe(df) # Display the final DataFrame in Streamlit
        return df
    
    live_scraped_odds_df = get_preseason_odds()
    
    st.write("")
    st.write("")
    st.write("")
    st.subheader("Live Odds from Draftkings")
    st.write(live_scraped_odds_df)
    st.write("")
    st.write("")
    st.write("")
	
    def add_odds_to_main_csv():
        """
        Adds odds data to the main DataFrame, prioritizing DraftKings data if available and complete.
        If DraftKings data is missing or incomplete for a game, it overrides with internal calculations.
    
        Args:
            df (pd.DataFrame): The main DataFrame to which odds will be added.
            live_scraped_odds_df (pd.DataFrame): DataFrame containing live odds scraped from DraftKings.
            # ... (all preseason_X_rank, X_rank, X_home_adv, X_away_adj parameters for each team)
    
        Returns:
            pd.DataFrame: The updated DataFrame with odds.
        """
    
        # 0: Stadium | 1: Lattitude | 2: Longitude | 3: Timezone | 4: Division | 5: Start of 2023 Season Rank | 6: Current Rank | 7: Average points better than Average Team (Used for Spread and Odds Calculation)
        stadiums = {
            'Arizona Cardinals': ['State Farm Stadium', 33.5277, -112.262608, 'America/Denver', 'NFC West', preseason_az_rank, az_rank, az_home_adv, az_away_adj],
            'Atlanta Falcons': ['Mercedez-Benz Stadium', 33.757614, -84.400972, 'America/New_York', 'NFC South', preseason_atl_rank, atl_rank, atl_home_adv, atl_away_adj],
            'Baltimore Ravens': ['M&T Stadium', 39.277969, -76.622767, 'America/New_York', 'AFC North', preseason_bal_rank, bal_rank, bal_home_adv, bal_away_adj],
            'Buffalo Bills': ['Highmark Stadium', 42.773739, -78.786978, 'America/New_York', 'AFC East', preseason_buf_rank, buf_rank, buf_home_adv, buf_away_adj],
            'Carolina Panthers': ['Bank of America Stadium', 35.225808, -80.852861, 'America/New_York', 'NFC South', preseason_car_rank, car_rank, car_home_adv, car_away_adj],
            'Chicago Bears': ['Soldier Field', 41.862306, -87.616672, 'America/Chicago', 'NFC North', preseason_chi_rank, chi_rank, chi_home_adv, chi_away_adj],
            'Cincinnati Bengals': ['Paycor Stadium', 39.095442, -84.516039, 'America/New_York', 'AFC North', preseason_cin_rank, cin_rank, cin_home_adv, cin_away_adj],
            'Cleveland Browns': ['Cleveland Browns Stadium', 41.506022, -81.699564, 'America/New_York', 'AFC North', preseason_cle_rank, cle_rank, cle_home_adv, cle_away_adj],
            'Dallas Cowboys': ['AT&T Stadium', 32.747778, -97.092778, 'America/Chicago', 'NFC East', preseason_dal_rank, dal_rank, dal_home_adv, dal_away_adj],
            'Denver Broncos': ['Empower Field at Mile High', 39.743936, -105.020097, 'America/Denver', 'AFC West', preseason_den_rank, den_rank, den_home_adv, den_away_adj],
            'Detroit Lions': ['Ford Field', 42.340156, -83.045808, 'America/New_York', 'NFC North', preseason_det_rank, det_rank, det_home_adv, det_away_adj],
            'Green Bay Packers': ['Lambeau Field', 44.501306, -88.062167, 'America/Chicago', 'NFC North', preseason_gb_rank, gb_rank, gb_home_adv, gb_away_adj],
            'Houston Texans': ['NRG Stadium', 29.684781, -95.410956, 'America/Chicago', 'AFC South', preseason_hou_rank, hou_rank, hou_home_adv, hou_away_adj],
            'Indianapolis Colts': ['Lucas Oil Stadium', 39.760056, -86.163806, 'America/New_York', 'AFC South', preseason_ind_rank, ind_rank, ind_home_adv, ind_away_adj],
            'Jacksonville Jaguars': ['Everbank Stadium', 30.323925, -81.637356, 'America/New_York', 'AFC South', preseason_jax_rank, jax_rank, jax_home_adv, jax_away_adj],
            'Kansas City Chiefs': ['Arrowhead Stadium', 39.048786, -94.484566, 'America/Chicago', 'AFC West', preseason_kc_rank, kc_rank, kc_home_adv, kc_away_adj],
            'Las Vegas Raiders': ['Allegiant Stadium', 36.090794, -115.183952, 'America/Los_Angeles', 'AFC West', preseason_lv_rank, lv_rank, lv_home_adv, lv_away_adj],
            'Los Angeles Chargers': ['SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'AFC West', preseason_lac_rank, lac_rank, lac_home_adv, lac_away_adj],
            'Los Angeles Rams': ['SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'NFC West', preseason_lar_rank, lar_rank, lar_home_adv, lar_away_adj],
            'Miami Dolphins': ['Hard Rock Stadium', 25.957919, -80.238842, 'America/New_York', 'AFC East', preseason_mia_rank, mia_rank, mia_home_adv, mia_away_adj],
            'Minnesota Vikings': ['U.S Bank Stadium', 44.973881, -93.258094, 'America/Chicago', 'NFC North', preseason_min_rank, min_rank, min_home_adv, min_away_adj],
            'New England Patriots': ['Gillette Stadium', 42.090925, -71.26435, 'America/New_York', 'AFC East', preseason_ne_rank, ne_rank, ne_home_adv, ne_away_adj],
            'New Orleans Saints': ['Caesars Superdome', 29.950931, -90.081364, 'America/Chicago', 'NFC South', preseason_no_rank, no_rank, no_home_adv, no_away_adj],
            'New York Giants': ['MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'NFC East', preseason_nyg_rank, nyg_rank, nyg_home_adv, nyg_away_adj],
            'New York Jets': ['MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'AFC East', preseason_nyj_rank, nyj_rank, nyj_home_adv, nyj_away_adj],
            'Philadelphia Eagles': ['Lincoln Financial Field', 39.900775, -75.167453, 'America/New_York', 'NFC East', preseason_phi_rank, phi_rank, phi_home_adv, phi_away_adj],
            'Pittsburgh Steelers': ['Acrisure Stadium', 40.446786, -80.015761, 'America/New_York', 'AFC North', preseason_pit_rank, pit_rank, pit_home_adv, pit_away_adj],
            'San Francisco 49ers': ['Levi\'s Stadium', 37.713486, -122.386256, 'America/Los_Angeles', 'NFC West', preseason_sf_rank, sf_rank, sf_home_adv, sf_away_adj],
            'Seattle Seahawks': ['Lumen Field', 47.595153, -122.331625, 'America/Los_Angeles', 'NFC West', preseason_sea_rank, sea_rank, sea_home_adv, sea_away_adj],
            'Tampa Bay Buccaneers': ['Raymomd James Stadium', 27.975967, -82.50335, 'America/New_York', 'NFC South', preseason_tb_rank, tb_rank, tb_home_adv, tb_away_adj],
            'Tennessee Titans': ['Nissan Stadium', 36.166461, -86.771289, 'America/Chicago', 'AFC South', preseason_ten_rank, ten_rank, ten_home_adv, ten_away_adj],
            'Washington Commanders': ['FedExField', 38.907697, -76.864517, 'America/New_York', 'NFC East', preseason_was_rank, was_rank, was_home_adv, was_away_adj]
        }
    
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
        
        # Attempt to update CSV data with scraped odds from DraftKings
        # This block only executes if live_scraped_odds_df is not empty
        if not live_scraped_odds_df.empty:
            for index, row in csv_df.iterrows():
                # Find a matching row in the scraped DraftKings data
                matching_row = live_scraped_odds_df[
                    (live_scraped_odds_df['Away Team'] == row['Away Team']) & 
                    (live_scraped_odds_df['Home Team'] == row['Home Team'])
                ]
                if not matching_row.empty:
                    # If a match is found, apply DraftKings moneyline odds
                    csv_df.loc[index, 'Away Team Moneyline'] = matching_row.iloc[0]['Away Odds']
                    csv_df.loc[index, 'Home Team Moneyline'] = matching_row.iloc[0]['Home Odds']
                    
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
        csv_df['Favorite'] = csv_df.apply(lambda row: row['Home Team'] if row['Home Team Adjusted Current Rank'] >= row['Away Team Adjusted Current Rank'] else row['Away Team'], axis=1)
        csv_df['Underdog'] = csv_df.apply(lambda row: row['Home Team'] if row['Home Team Adjusted Current Rank'] < row['Away Team Adjusted Current Rank'] else row['Away Team'], axis=1)

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
    
        # Iterate through the DataFrame to apply overrides and calculate implied/fair odds
        for index, row in csv_df.iterrows():
            # Override Moneyline if DraftKings data was missing (still NaN)
            if pd.isna(row['Away Team Moneyline']):
                csv_df.loc[index, 'Away Team Moneyline'] = row['Internal Away Team Moneyline']
            if pd.isna(row['Home Team Moneyline']):
                csv_df.loc[index, 'Home Team Moneyline'] = row['Internal Home Team Moneyline']
            
            # Override Favorite/Underdog if not set by DraftKings (i.e., still NaN)
            if pd.isna(row['Favorite']):
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
        week13_winners = week13_df["Adjusted Current Winner"].unique()
	

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
        week18_winners = week18_df["Adjusted Current Winner"].unique()

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
collect_schedule_travel_ranking_data_nfl_schedule_df = collect_schedule_travel_ranking_data

def get_predicted_pick_percentages(pd):
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
    
    new_df = collect_schedule_travel_ranking_data_df

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

    pick_predictions_df = pd.concat([away_df, home_df], ignore_index=True)

    # Adjust pick percentages for Thanksgiving Favorites
    pick_predictions_df["Pick %"] = pick_predictions_df.apply(
        lambda row: row["Pick %"] / 4 if (row["Date"] != 13 and row["Home Team Thanksgiving Favorite"]) else row["Pick %"],
        axis=1
    )

    pick_predictions_df["Pick %"] = pick_predictions_df.apply(
        lambda row: row["Pick %"] / 4 if (row["Date"] != 13 and row["Away Team Thanksgiving Favorite"]) else row["Pick %"],
        axis=1
    )

    # Adjust pick percentages for Thanksgiving Favorites
    pick_predictions_df["Pick %"] = pick_predictions_df.apply(
        lambda row: row["Pick %"] / 4 if row["Home Team Christmas Favorite"] else row["Pick %"],
        axis=1
    )

    pick_predictions_df["Pick %"] = pick_predictions_df.apply(
        lambda row: row["Pick %"] / 4 if row["Away Team Christmas Favorite"] else row["Pick %"],
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

    nfl_schedule_df = collect_schedule_travel_ranking_data_df
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

    #Handle Week 1 as before
    if selected_contest == 'Circa':
        nfl_schedule_df.loc[nfl_schedule_df['Week'] == starting_week, 'Total Remaining Entries at Start of Week'] = current_week_entries_circa
    elif selected_contest == 'DraftKings': # Assuming 'DK' was a shorthand for 'DraftKings' or similar
        nfl_schedule_df.loc[nfl_schedule_df['Week'] == starting_week, 'Total Remaining Entries at Start of Week'] = current_week_entries_dk
    elif selected_contest == 'Splash Sports':
        nfl_schedule_df.loc[nfl_schedule_df['Week'] == starting_week, 'Total Remaining Entries at Start of Week'] = current_week_entries_splash_sports # Assuming you'll have a variable for Splash Sports entries
    else:
        # Handle an unexpected selected_contest value, e.g., raise an error or set a default
        print("Warning: Unknown contest type provided.")
        nfl_schedule_df.loc[nfl_schedule_df['Week'] == starting_week, 'Total Remaining Entries at Start of Week'] = 20000 # Or some default value

    nfl_schedule_df['Home Expected Survival Rate'] = nfl_schedule_df['Home Team Fair Odds'] * nfl_schedule_df['Home Pick %']
    nfl_schedule_df['Home Expected Elimination Percent'] = nfl_schedule_df['Home Pick %'] - nfl_schedule_df['Home Expected Survival Rate']
    nfl_schedule_df['Away Expected Survival Rate'] = nfl_schedule_df['Away Team Fair Odds'] * nfl_schedule_df['Away Pick %']
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
        availability = availability_dict.get(team_name) # Get availability, default to -1 if team not in dict
        if availability != -1:
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
    

    def assign_pick_percentages(row, selected_contest, circa_pick_percentages, dk_pick_percentages, splash_pick_percentages):
    #"""Assigns home and away pick percentages to a row, conditionally overwriting."""

        home_team = row['Home Team']
        away_team = row['Away Team']
        week = row['Week']
        week_index = week - 1

        home_pick_percent = row.get('Home Pick %')  # Get existing value, defaults to None
        away_pick_percent = row.get('Away Pick %')  # Get existing value, defaults to None

        if selected_contest == 'Circa':
            # Check for Home Team overrides
            if home_team in circa_pick_percentages:
                team_pick_percent_list = circa_pick_percentages[home_team]
                if week_index < len(team_pick_percent_list):
                    user_override_value = team_pick_percent_list[week_index]
                    # Apply override ONLY if the user-provided value is NOT -1
                    if user_override_value >= 0:
                        home_pick_percent = user_override_value

            # Check for Away Team overrides
            if away_team in circa_pick_percentages:
                team_pick_percent_list = circa_pick_percentages[away_team]
                if week_index < len(team_pick_percent_list):
                    user_override_value = team_pick_percent_list[week_index]
                    # Apply override ONLY if the user-provided value is NOT -1
                    if user_override_value >= 0:
                        away_pick_percent = user_override_value
        elif selected_contest == 'Splash Sports': 
            if home_team in splash_pick_percentages:
                team_pick_percent_list = splash_pick_percentages[home_team]
                if week_index < len(team_pick_percent_list):
                    user_override_value = team_pick_percent_list[week_index]
                    # Apply override ONLY if the user-provided value is NOT -1
                    if user_override_value >= 0:
                        home_pick_percent = user_override_value

            # Check for Away Team overrides
            if away_team in splash_pick_percentages:
                team_pick_percent_list = splash_pick_percentages[away_team]
                if week_index < len(team_pick_percent_list):
                    user_override_value = team_pick_percent_list[week_index]
                    # Apply override ONLY if the user-provided value is NOT -1
                    if user_override_value >= 0:
                        away_pick_percent = user_override_value
	    
        else: # assuming it's DraftKings
            if home_team in dk_pick_percentages:
                team_pick_percent_list = dk_pick_percentages[home_team]
                if week_index < len(team_pick_percent_list):
                    user_override_value = team_pick_percent_list[week_index]
                    # Apply override ONLY if the user-provided value is NOT -1
                    if user_override_value >= 0:
                        home_pick_percent = user_override_value

            # Check for Away Team overrides
            if away_team in dk_pick_percentages:
                team_pick_percent_list = dk_pick_percentages[away_team]
                if week_index < len(team_pick_percent_list):
                    user_override_value = team_pick_percent_list[week_index]
                    # Apply override ONLY if the user-provided value is NOT -1
                    if user_override_value >= 0:
                        away_pick_percent = user_override_value

        return pd.Series({'Home Pick %': home_pick_percent, 'Away Pick %': away_pick_percent})
                                                                 

    nfl_schedule_df[['Home Pick %', 'Away Pick %']] = nfl_schedule_df.apply(
        assign_pick_percentages, 
        axis=1, 
        args=(selected_contest, circa_pick_percentages, dk_pick_percentages, splash_pick_percentages)
    )
    
    if selected_contest == 'Circa':
        nfl_schedule_df.to_csv("Circa_Predicted_pick_percent.csv", index=False)
    elif selected_contest == 'Splash Sports':
        nfl_schedule_df.to_csv("Splash_Predicted_pick_percent.csv", index=False)
    else:
        nfl_schedule_df.to_csv("DK_Predicted_pick_percent.csv", index=False)
    return nfl_schedule_df

def get_predicted_pick_percentages_with_availability(pd):
    # Load your historical data (replace 'historical_pick_data_FV_circa.csv' with your actual file path)
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
        
        new_df = nfl_schedule_pick_percentages_df

    
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
    
        # Adjust pick percentages for Thanksgiving Favorites
        pick_predictions_df["Pick %"] = pick_predictions_df.apply(
            lambda row: row["Pick %"] / 4 if (row["Date"] != 13 and row["Home Team Thanksgiving Favorite"]) else row["Pick %"],
            axis=1
        )
    
        pick_predictions_df["Pick %"] = pick_predictions_df.apply(
            lambda row: row["Pick %"] / 4 if (row["Date"] != 13 and row["Away Team Thanksgiving Favorite"]) else row["Pick %"],
            axis=1
        )
    
        # Adjust pick percentages for Thanksgiving Favorites
        pick_predictions_df["Pick %"] = pick_predictions_df.apply(
            lambda row: row["Pick %"] / 4 if row["Home Team Christmas Favorite"] else row["Pick %"],
            axis=1
        )
    
        pick_predictions_df["Pick %"] = pick_predictions_df.apply(
            lambda row: row["Pick %"] / 4 if row["Away Team Christmas Favorite"] else row["Pick %"],
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
    
        nfl_schedule_df = collect_schedule_travel_ranking_data_df
        # Assumes format "Week X" - adjust if the format is different
        if nfl_schedule_df['Week'].dtype == 'object':
            nfl_schedule_df['Week_Number'] = nfl_schedule_df['Week'].str.split(' ').str[1].astype(int)
        # Filter the DataFrame
        else:
            nfl_schedule_df['Week_Number'] = nfl_schedule_df['Week']
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
    
        #Handle Week 1 as before
        if selected_contest == 'Circa':
            nfl_schedule_df.loc[nfl_schedule_df['Week'] == starting_week, 'Total Remaining Entries at Start of Week'] = current_week_entries_circa
        elif selected_contest == 'DraftKings': # Assuming 'DK' was a shorthand for 'DraftKings' or similar
            nfl_schedule_df.loc[nfl_schedule_df['Week'] == starting_week, 'Total Remaining Entries at Start of Week'] = current_week_entries_dk
        elif selected_contest == 'Splash Sports':
            nfl_schedule_df.loc[nfl_schedule_df['Week'] == starting_week, 'Total Remaining Entries at Start of Week'] = current_week_entries_splash_sports # Assuming you'll have a variable for Splash Sports entries
        else:
            # Handle an unexpected selected_contest value, e.g., raise an error or set a default
            print("Warning: Unknown contest type provided.")
            nfl_schedule_df.loc[nfl_schedule_df['Week'] == starting_week, 'Total Remaining Entries at Start of Week'] = 20000 # Or some default value
    
        nfl_schedule_df['Home Expected Survival Rate'] = nfl_schedule_df['Home Team Fair Odds'] * nfl_schedule_df['Home Pick %']
        nfl_schedule_df['Home Expected Elimination Percent'] = nfl_schedule_df['Home Pick %'] - nfl_schedule_df['Home Expected Survival Rate']
        nfl_schedule_df['Away Expected Survival Rate'] = nfl_schedule_df['Away Team Fair Odds'] * nfl_schedule_df['Away Pick %']
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
            if availability != -1:
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
    

        def assign_pick_percentages_with_availability(row, selected_contest, circa_pick_percentages, dk_pick_percentages, splash_pick_percentages):
        #"""Assigns home and away pick percentages to a row, conditionally overwriting."""
            if selected_contest == 'Circa':
                home_team = row['Home Team']
                away_team = row['Away Team']
                week = row['Week']
                week_index = week - 1
        
                home_pick_percent = row.get('Home Pick %')  # Get existing value, defaults to None
                away_pick_percent = row.get('Away Pick %')  # Get existing value, defaults to None
        
                # Check for Home Team overrides
                if home_team in circa_pick_percentages:
                    team_pick_percent_list = circa_pick_percentages[home_team]
                    if week_index < len(team_pick_percent_list):
                        user_override_value = team_pick_percent_list[week_index]
                        # Apply override ONLY if the user-provided value is NOT -1
                        if user_override_value >= 0:
                            home_pick_percent = user_override_value
        
                # Check for Away Team overrides
                if away_team in circa_pick_percentages:
                    team_pick_percent_list = circa_pick_percentages[away_team]
                    if week_index < len(team_pick_percent_list):
                        user_override_value = team_pick_percent_list[week_index]
                        # Apply override ONLY if the user-provided value is NOT -1
                        if user_override_value >= 0:
                            away_pick_percent = user_override_value
        
                return pd.Series({'Home Pick %': home_pick_percent, 'Away Pick %': away_pick_percent})
                                                                         
        
        nfl_schedule_df[['Home Pick %', 'Away Pick %']] = nfl_schedule_df.apply(
            assign_pick_percentages_with_availability, 
            axis=1, 
            args=(selected_contest, circa_pick_percentages, dk_pick_percentages, splash_pick_percentages)
        )
        
        if selected_contest == 'Circa':
            nfl_schedule_df.to_csv("Circa_Predicted_pick_percent.csv", index=False)
    return nfl_schedule_df


def calculate_ev(nfl_schedule_pick_percentages_df, starting_week, ending_week, selected_contest, use_cached_expected_value):
    
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


    st.write("Current Week Progress")  # Streamlit progress bar
    progress_bar = st.progress(0)

    all_weeks_ev = {} #Store the EV values for each week

    for week in tqdm(range(starting_week, ending_week), desc="Processing Weeks", leave=False):
        week_df = nfl_schedule_pick_percentages_df[nfl_schedule_pick_percentages_df['Week_Num'] == week].copy() # Create a copy to avoid SettingWithCopyWarning
        weighted_avg_ev, all_outcomes, scenario_weights = calculate_all_scenarios(week_df)

        #Store the EV values for the current week
        all_weeks_ev[week] = weighted_avg_ev

        #More efficient way to set the EV values for the current week
        for team in week_df['Home Team'].unique():
            nfl_schedule_pick_percentages_df.loc[(nfl_schedule_pick_percentages_df['Week_Num'] == week) & (nfl_schedule_pick_percentages_df['Home Team'] == team), 'Home Team EV'] = weighted_avg_ev[team]
        for team in week_df['Away Team'].unique():
            nfl_schedule_pick_percentages_df.loc[(nfl_schedule_pick_percentages_df['Week_Num'] == week) & (nfl_schedule_pick_percentages_df['Away Team'] == team), 'Away Team EV'] = weighted_avg_ev[team]

        progress_percent = int((week / ending_week) * 100)
        progress_bar.progress(progress_percent)

    if selected_contest == 'Circa':
        nfl_schedule_pick_percentages_df.to_csv("NFL Schedule with full ev_circa.csv", index=False)
    elif selected_contest == 'Splash Sports':
        nfl_schedule_pick_percentages_df.to_csv("NFL Schedule with full ev_splash.csv", index=False)
    else:
        nfl_schedule_pick_percentages_df.to_csv("NFL Schedule with full ev_dk.csv", index=False)
    return nfl_schedule_pick_percentages_df


def get_survivor_picks_based_on_ev():
    if pick_must_be_favored:
        for iteration in range(number_solutions):
            df = full_df_with_ev
    
            #Number of weeks that have already been played
            #weeks_completed = starting_week -1
    
            # Teams already picked - Team name in quotes and separated by commas
    
            # Filter out weeks that have already been played and reset index
    
            df = df[(df['Week_Num'] >= starting_week) & (df['Week_Num'] < ending_week)].reset_index(drop=True)
            # Filter out already picked teams
            df = df[~df['Adjusted Current Winner'].isin(picked_teams)].reset_index(drop=True)
            #print(df)
            # Create the solver
            solver = pywraplp.Solver.CreateSolver('SCIP')
    
            # Create binary variables to represent the picks, and store them in a dictionary for easy lookup
            picks = {}
            for i in range(len(df)):
                picks[i] = solver.IntVar(0, 1, 'pick_%i' % i)
    
            # Add the constraints
            for i in range(len(df)):
                # Must pick from 'Adjusted Current Winner'
                #if df.loc[i, 'Adjusted Current Winner'] != df.loc[i, 'Home Team']:
                    #solver.Add(picks[i] == 0)
                # Must pick from 'Same Winner?'
                if bayesian_rest_travel_constraint == "Selected team must have been projected to win based on preseason rankings, current rankings, and with and without travel/rest adjustments":
                    if df.loc[i, 'Same Winner?'] != 'Same':
                        solver.Add(picks[i] == 0)
                elif  bayesian_rest_travel_constraint == "Selected team must be projected to win with and without travel and rest impact based on current rankings":
                    if df.loc[i, 'Same Current and Adjusted Current Winner?'] != 'Same':
                        solver.Add(picks[i] == 0)
                elif  bayesian_rest_travel_constraint == "Selected team must have been projected to win based on preseason rankings in addition to current rankings":   
                    if df.loc[i, 'Same Adjusted Preseason Winner?'] != 'Same':
                        solver.Add(picks[i] == 0)
                # Can only pick an away team if 'Adjusted Current Difference' > 10
                if avoid_away_teams_in_close_matchups == 1:
                    if df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Adjusted Current Difference'] < 10:
                        solver.Add(picks[i] == 0)
                #if df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Divisional Matchup?'] == 'Divisional':
                    #solver.Add(picks[i] == 0)
                if avoid_back_to_back_away == 1:
                    if df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Back to Back Away Games'] == 'True':
                        solver.Add(picks[i] == 0)
    
                # If 'Divisional Matchup?' is "Divisional", can only pick if 'Adjusted Current Difference' > 10
                if avoid_close_divisional_matchups == 1:
                    if df.loc[i, 'Divisional Matchup?'] == 'Divisional' and df.loc[i, 'Adjusted Current Difference'] < 10:
                        solver.Add(picks[i] == 0)
                # Constraints for short rest and 4 games in 17 days (only if team is the Adjusted Current Winner)
                if avoid_away_teams_on_short_rest == 1:
                    if df.loc[i, 'Away Team Short Rest'] == 'Yes' and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                if avoid_4_games_in_17_days == 1:
                    if df.loc[i, 'Home Team 4 games in 17 days'] == 'Yes' and df.loc[i, 'Home Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Away Team 4 games in 17 days'] == 'No':
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'Away Team 4 games in 17 days'] == 'Yes' and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Home Team 4 games in 17 days'] == 'No':
                        solver.Add(picks[i] == 0)
                if avoid_3_games_in_10_days == 1:
                    if df.loc[i, 'Home Team 3 games in 10 days'] == 'Yes' and df.loc[i, 'Home Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Away Team 3 games in 10 days'] == 'No':
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'Away Team 3 games in 10 days'] == 'Yes' and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Home Team 3 games in 10 days'] == 'No':
                        solver.Add(picks[i] == 0)
                if avoid_international_game == 1:    
                    if df.loc[i, 'City'] == 'London, UK' and df.loc[i, 'Home Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'City'] == 'London, UK' and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                if avoid_thursday_night == 1:
                    if df.loc[i, 'Thursday Night Game'] == 'True':
                        solver.Add(picks[i] == 0)
                if avoid_away_thursday_night == 1:
                    if df.loc[i, 'Thursday Night Game'] == 'True' and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                if avoid_teams_with_weekly_rest_disadvantage == 1:
                    if df.loc[i, 'Home Team Weekly Rest'] < df.loc [i, 'Away Team Weekly Rest'] and df.loc[i, 'Home Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'Away Team Weekly Rest'] < df.loc [i, 'Home Team Weekly Rest'] and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                if avoid_cumulative_rest_disadvantage == 1:
                    if df.loc[i, 'Away Team Current Week Cumulative Rest Advantage'] < -10 and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'Home Team Current Week Cumulative Rest Advantage'] < -5 and df.loc[i, 'Home Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                if avoid_away_teams_with_travel_disadvantage == 1:
                    if df.loc[i, 'Away Travel Advantage'] < -850 and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
    
    
                
                if df.loc[i, 'Adjusted Current Winner'] == 'Arizona Cardinals' and df.loc[i, 'Week_Num'] in az_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Atlanta Falcons' and df.loc[i, 'Week_Num'] in atl_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Baltimore Ravens' and df.loc[i, 'Week_Num'] in bal_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Buffalo Bills' and df.loc[i, 'Week_Num'] in buf_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Carolina Panthers' and df.loc[i, 'Week_Num'] in car_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Chicago Bears' and df.loc[i, 'Week_Num'] in chi_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Cincinnati Bengals' and df.loc[i, 'Week_Num'] in cin_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Cleveland Browns' and df.loc[i, 'Week_Num'] in cle_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Dallas Cowboys' and df.loc[i, 'Week_Num'] in dal_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Denver Broncos' and df.loc[i, 'Week_Num'] in den_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Detroit Lions' and df.loc[i, 'Week_Num'] in det_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Green Bay Packers' and df.loc[i, 'Week_Num'] in gb_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Houston Texans' and df.loc[i, 'Week_Num'] in hou_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Indianapolis Colts' and df.loc[i, 'Week_Num'] in ind_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Jacksonville Jaguars' and df.loc[i, 'Week_Num'] in jax_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Kansas City Chiefs' and df.loc[i, 'Week_Num'] in kc_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Las vegas Raiders' and df.loc[i, 'Week_Num'] in lv_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Chargers' and df.loc[i, 'Week_Num'] in lac_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Rams' and df.loc[i, 'Week_Num'] in lar_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Miami Dolphins' and df.loc[i, 'Week_Num'] in mia_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Minnesota Vikings' and df.loc[i, 'Week_Num'] in min_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'New England Patriots' and df.loc[i, 'Week_Num'] in ne_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'New Orleans Saints' and df.loc[i, 'Week_Num'] in no_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'New York Giants' and df.loc[i, 'Week_Num'] in nyg_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'New York Jets' and df.loc[i, 'Week_Num'] in nyj_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Philadelphia Eagles' and df.loc[i, 'Week_Num'] in phi_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Pittsburgh Steelers' and df.loc[i, 'Week_Num'] in pit_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Seattle Seahawks' and df.loc[i, 'Week_Num'] in sea_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Tampa Bay Buccaneers' and df.loc[i, 'Week_Num'] in tb_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Tennessee Titans' and df.loc[i, 'Week_Num'] in ten_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Washington Commanders' and df.loc[i, 'Week_Num'] in was_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'San Francisco 49ers' and df.loc[i, 'Week_Num'] in sf_excluded_weeks:
                        solver.Add(picks[i] == 0)
    
    
                if df.loc[i, 'Adjusted Current Winner'] in picked_teams:
                    solver.Add(picks[i] == 0)
    
    # Add the constraint for San Francisco 49ers in week 11
            if sf_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'San Francisco 49ers' or df.loc[i, 'Away Team'] == 'San Francisco 49ers') and df.loc[i, 'Week_Num'] == sf_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'San Francisco 49ers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'San Francisco 49ers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'San Francisco 49ers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if az_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Arizona Cardinals' or df.loc[i, 'Away Team'] == 'Arizona Cardinals') and df.loc[i, 'Week_Num'] == az_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Arizona Cardinals':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Arizona Cardinals':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Arizona Cardinals':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if atl_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Atlanta Falcons' or df.loc[i, 'Away Team'] == 'Atlanta Falcons') and df.loc[i, 'Week_Num'] == atl_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Atlanta Falcons':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Atlanta Falcons':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Atlanta Falcons':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if bal_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Baltimore Ravens' or df.loc[i, 'Away Team'] == 'Baltimore Ravens') and df.loc[i, 'Week_Num'] == bal_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Baltimore Ravens':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Baltimore Ravens':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Baltimore Ravens':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if buf_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Buffalo Bills' or df.loc[i, 'Away Team'] == 'Buffalo Bills') and df.loc[i, 'Week_Num'] == buf_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Buffalo Bills':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Buffalo Bills':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Buffalo Bills':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if car_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Carolina Panthers' or df.loc[i, 'Away Team'] == 'Carolina Panthers') and df.loc[i, 'Week_Num'] == car_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Carolina Panthers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Carolina Panthers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Carolina Panthers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if chi_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Chicago Bears' or df.loc[i, 'Away Team'] == 'Chicago Bears') and df.loc[i, 'Week_Num'] == chi_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Chicago Bears':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Chicago Bears':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Chicago Bears':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if cin_req_week > 0:
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Cincinnati Bengals' or df.loc[i, 'Away Team'] == 'Cincinnati Bengals') and df.loc[i, 'Week_Num'] == cin_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Cincinnati Bengals':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Cincinnati Bengals':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Cincinnati Bengals':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if cle_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Cleveland Browns' or df.loc[i, 'Away Team'] == 'Cleveland Browns') and df.loc[i, 'Week_Num'] == cle_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Cleveland Browns':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Cleveland Browns':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Cleveland Browns':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if dal_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Dallas Cowboys' or df.loc[i, 'Away Team'] == 'Dallas Cowboys') and df.loc[i, 'Week_Num'] == dal_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Dallas Cowboys':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Dallas Cowboys':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Dallas Cowboys':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if den_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Denver Broncos' or df.loc[i, 'Away Team'] == 'Denver Broncos') and df.loc[i, 'Week_Num'] == den_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Denver Broncos':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Denver Broncos':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Denver Broncos':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if det_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Detroit Lions' or df.loc[i, 'Away Team'] == 'Detroit Lions') and df.loc[i, 'Week_Num'] == det_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Detroit Lions':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Detroit Lions':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Detroit Lions':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if gb_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Green Bay Packers' or df.loc[i, 'Away Team'] == 'Green Bay Packers') and df.loc[i, 'Week_Num'] == gb_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Green Bay Packers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Green Bay Packers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Green Bay Packers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if hou_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Houston Texans' or df.loc[i, 'Away Team'] == 'Houston Texans') and df.loc[i, 'Week_Num'] == hou_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Houston Texans':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Houston Texans':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Houston Texans':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if ind_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Indianapolis Colts' or df.loc[i, 'Away Team'] == 'Indianapolis Colts') and df.loc[i, 'Week_Num'] == ind_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Indianapolis Colts':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Indianapolis Colts':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Indianapolis Colts':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if jax_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Jacksonville Jaguars' or df.loc[i, 'Away Team'] == 'Jacksonville Jaguars') and df.loc[i, 'Week_Num'] == jax_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Jacksonville Jaguars':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Jacksonville Jaguars':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Jacksonville Jaguars':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if kc_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Kansas City Chiefs' or df.loc[i, 'Away Team'] == 'Kansas City Chiefs') and df.loc[i, 'Week_Num'] == kc_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Kansas City Chiefs':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Kansas City Chiefs':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Kansas City Chiefs':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if lv_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Las Vegas Raiders' or df.loc[i, 'Away Team'] == 'Las Vegas Raiders') and df.loc[i, 'Week_Num'] == lv_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Las Vegas Raiders':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Las Vegas Raiders':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Las Vegas Raiders':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if lac_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Los Angeles Chargers' or df.loc[i, 'Away Team'] == 'Los Angeles Chargers') and df.loc[i, 'Week_Num'] == lac_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Chargers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Los Angeles Chargers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Los Angeles Chargers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if lar_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Los Angeles Rams' or df.loc[i, 'Away Team'] == 'Los Angeles Rams') and df.loc[i, 'Week_Num'] == lar_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Rams':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Los Angeles Rams':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Los Angeles Rams':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if mia_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Miami Dolphins' or df.loc[i, 'Away Team'] == 'Miami Dolphins') and df.loc[i, 'Week_Num'] == mia_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Miami Dolphins':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Miami Dolphins':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Miami Dolphins':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if min_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Minnesota Vikings' or df.loc[i, 'Away Team'] == 'Minnesota Vikings') and df.loc[i, 'Week_Num'] == min_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Minnesota Vikings':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Minnesota Vikings':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Minnesota Vikings':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if ne_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'New England Patriots' or df.loc[i, 'Away Team'] == 'New England Patriots') and df.loc[i, 'Week_Num'] == ne_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'New England Patriots':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'New England Patriots':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'New England Patriots':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if no_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'New Orleans Saints' or df.loc[i, 'Away Team'] == 'New Orleans Saints') and df.loc[i, 'Week_Num'] == no_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'New Orleans Saints':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'New Orleans Saints':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'New Orleans Saints':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if nyg_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'New York Giants' or df.loc[i, 'Away Team'] == 'New York Giants') and df.loc[i, 'Week_Num'] == nyg_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'New York Giants':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'New York Giants':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'New York Giants':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if nyj_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'New York Jets' or df.loc[i, 'Away Team'] == 'New York Jets') and df.loc[i, 'Week_Num'] == nyj_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'New York Jets':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'New York Jets':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'New York Jets':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if phi_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Philadelphia Eagles' or df.loc[i, 'Away Team'] == 'Philadelphia Eagles') and df.loc[i, 'Week_Num'] == phi_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Philadelphia Eagles':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Philadelphia Eagles':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Philadelphia Eagles':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if pit_req_week > 0:        
                    if (df.loc[i, 'Home Team'] == 'Pittsburgh Steelers' or df.loc[i, 'Away Team'] == 'Pittsburgh Steelers') and df.loc[i, 'Week_Num'] == pit_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Pittsburgh Steelers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Pittsburgh Steelers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Pittsburgh Steelers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if sea_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Seattle Seahawks' or df.loc[i, 'Away Team'] == 'Seattle Seahawks') and df.loc[i, 'Week_Num'] == sea_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Seattle Seahawks':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Seattle Seahawks':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Seattle Seahawks':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if tb_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Tampa Bay Buccaneers' or df.loc[i, 'Away Team'] == 'Tampa Bay Buccaneers') and df.loc[i, 'Week_Num'] == tb_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Tampa Bay Buccaneers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Tampa Bay Buccaneers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Tampa Bay Buccaneers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if ten_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Tennessee Titans' or df.loc[i, 'Away Team'] == 'Tennessee Titans') and df.loc[i, 'Week_Num'] == ten_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Tennessee Titans':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Tennessee Titans':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Tennessee Titans':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if was_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Washington Commanders' or df.loc[i, 'Away Team'] == 'Washington Commanders') and df.loc[i, 'Week_Num'] == was_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Washington Commanders':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Washington Commanders':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Washington Commanders':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
    
    	    
            for week in df['Week_Num'].unique():
                # Filter picks for the current week
                weekly_picks = [picks[i] for i in range(len(df)) if df.loc[i, 'Week_Num'] == week]

                if selected_contest == "Splash Sports" and week >= week_requiring_two_selections:
                    # For Splash Sports and later weeks, two teams must be selected
                    solver.Add(solver.Sum(weekly_picks) == 2)
                else:
                    # For other contests or earlier weeks, one team per week
                    solver.Add(solver.Sum(weekly_picks) == 1)
    
            for team in df['Adjusted Current Winner'].unique():
                # Can't pick a team more than once
                solver.Add(solver.Sum([picks[i] for i in range(len(df)) if df.loc[i, 'Adjusted Current Winner'] == team]) <= 1)
    
    
            # Dynamically create the forbidden solution list
            forbidden_solutions_1 = []
            if iteration > 0: 
                for previous_iteration in range(iteration):
                    # Load the picks from the previous iteration
                    if selected_contest == 'Circa':
                        previous_picks_df = pd.read_csv(f"circa_picks_ev_{previous_iteration + 1}.csv")
                    elif selected_contest == 'Splash Sports':
                        previous_picks_df = pd.read_csv(f"splash_picks_ev_{previous_iteration + 1}.csv")
                    else:
                        previous_picks_df = pd.read_csv(f"dk_picks_ev_{previous_iteration + 1}.csv")
    
                    # Extract the forbidden solution for this iteration
                    forbidden_solution_1 = previous_picks_df['Adjusted Current Winner'].tolist()
                    forbidden_solutions_1.append(forbidden_solution_1)
    
            # Add constraints for all forbidden solutions
            for forbidden_solution_1 in forbidden_solutions_1:
                # Get the indices of the forbidden solution in the DataFrame
                forbidden_indices_1 = []
                for i in range(len(df)):
                    # Calculate the relative week number within the forbidden solution
                    df_week = df.loc[i, 'Week_Num']
                    relative_week = df_week - starting_week  # Adjust week to be relative to starting week
    
                    #Check if the week is within the range and the solution is forbidden
                    if 0 <= relative_week < len(forbidden_solution_1) and df_week >= starting_week and df_week < ending_week: #Added this to make sure we are only looking at the range
                        if (df.loc[i, 'Adjusted Current Winner'] == forbidden_solution_1[relative_week]):
                            forbidden_indices_1.append(i)
    
                # Add the constraint
                solver.Add(solver.Sum([1 - picks[i] for i in forbidden_indices_1]) >= 1)
    
    
            
    
            # Objective: maximize the sum of Adjusted Current Difference of each game picked
            solver.Maximize(solver.Sum([picks[i] * (df.loc[i, 'Home Team EV'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team EV']) for i in range(len(df))]))
    
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
    
                # Initialize picks_df
                picks_df = pd.DataFrame(columns=df.columns)
                picks_rows_2 = []
                for i in range(len(df)):
                    if picks[i].solution_value() > 0:
                        # Determine if it's a divisional game and if the picked team is the home team
    
                        week = df.loc[i, 'Week']
                        pick = df.loc[i,'Adjusted Current Winner']
                        opponent = df.loc[i, 'Home Team'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Away Team'] else df.loc[i, 'Away Team']
                        win_odds = df.loc[i, 'Home Team Fair Odds'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Fair Odds']
                        pick_percent = df.loc[i, 'Home Pick %'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Pick %']
                        expected_availability = df.loc[i, 'Home Team Expected Availability'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Expected Availability']
                        divisional_game = 'Divisional' if df.loc[i, 'Divisional Matchup Boolean'] else ''
                        home_team = 'Home Team' if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else 'Away Team'
                        weekly_rest = df.loc[i, 'Home Team Weekly Rest'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Weekly Rest']
                        weekly_rest_advantage = df.loc[i, 'Weekly Home Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Weekly Away Rest Advantage']
                        cumulative_rest = df.loc[i, 'Home Cumulative Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Cumulative Rest Advantage']
                        cumulative_rest_advantage = df.loc[i, 'Home Team Current Week Cumulative Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Current Week Cumulative Rest Advantage']
                        travel_advantage = df.loc[i, 'Home Travel Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Travel Advantage']
                        back_to_back_away_games = 'True' if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Away Team'] and df.loc[i, 'Back to Back Away Games'] == 'True' else 'False'
                        thursday_night_game = 'Thursday Night Game' if df.loc[i, "Thursday Night Game"] == 'True' else 'Sunday/Monday Game'
                        international_game = 'International Game' if df.loc[i, 'Actual Stadium'] == 'London, UK' else 'Domestic Game'
                        previous_opponent = df.loc[i, 'Home Team Previous Opponent'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Previous Opponent']
                        previous_game_location = df.loc[i, 'Home Team Previous Location'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Previous Location']
                        next_opponent = df.loc[i, 'Home Team Next Opponent'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Next Opponent']
                        next_game_location = df.loc[i, 'Home Team Next Location'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Next Location']

                        
    
                        # Get differences
                        preseason_difference = df.loc[i, 'Preseason Difference']
                        adjusted_preseason_difference = df.loc[i, 'Adjusted Preseason Difference']
                        current_difference = df.loc[i, 'Current Difference']
                        adjusted_current_difference = df.loc[i, 'Adjusted Current Difference']
                        # Calculate EV for this game
                        ev = (df.loc[i, 'Home Team EV'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team EV'])
    
    
                        print('Week %i: Pick %s %s %s (%i, %i, %i, %i, %.4f)' % (df.loc[i, 'Week_Num'], df.loc[i, 'Adjusted Current Winner'], divisional_game, home_team,
                                                                           preseason_difference, adjusted_preseason_difference,
                                                                           current_difference, adjusted_current_difference, ev))
                        new_row_2 = {
                            'Week': week,
                            'Pick': pick,
                            'Opponent': opponent,
                            'EV': ev,
                            'Win Odds': win_odds,
                            'Pick %': pick_percent,	
                            'Expected Availability': expected_availability,
                            'Divisional Game': divisional_game,
                            'Home Team Status': home_team,
                            'Weekly Rest': weekly_rest,
                            'Weekly Rest Advantage': weekly_rest_advantage,
                            'Cumulative Rest': cumulative_rest,
                            'Cumulative Rest Advantage': cumulative_rest_advantage,
                            'Travel Advantage': travel_advantage,
                            'Back to Back Away Games': back_to_back_away_games,
                            'Thursday Night Game': thursday_night_game,
                            'International Game': international_game,
                            'Previous Opponent': previous_opponent,
                            'Previous Game Location': previous_game_location,
                            'Next Opponent': next_opponent,
                            'Next Game Location': next_game_location,
                            'Preseason Difference': preseason_difference,
                            'Adjusted Preseason Difference': adjusted_preseason_difference,
                            'Current Difference': current_difference,
                            'Adjusted Current Difference': adjusted_current_difference
                        }
                        picks_rows_2.append(new_row_2)
    
    
                        # Add differences to sums
                        sum_preseason_difference += preseason_difference
                        sum_adjusted_preseason_difference += adjusted_preseason_difference
                        sum_current_difference += current_difference
                        sum_adjusted_current_difference += adjusted_current_difference
                        sum_ev += ev
                        picks_df = pd.concat([picks_df, df.loc[[i]]], ignore_index=True)
                        picks_df['Divisional Matchup?'] = divisional_game
                summarized_picks_df = pd.DataFrame(picks_rows_2)
    
                st.write(summarized_picks_df)
                st.write('')
                st.write('\nPreseason Difference:', sum_preseason_difference)
                st.write('Adjusted Preseason Difference:', sum_adjusted_preseason_difference)
                st.write('Current Difference:', sum_current_difference)
                st.write('Adjusted Current Difference:', sum_adjusted_current_difference)
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
            
            # Append the new forbidden solution to the list
            forbidden_solutions_1.append(picks_df['Adjusted Current Winner'].tolist())
            #print(forbidden_solutions)

    else:
        for iteration in range(number_solutions):
            df = full_df_with_ev
    
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
                "Home Team Adjusted Current Rank": "Hypothetical Current Winner Adjusted Current Rank"
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
                "Home Team Adjusted Current Rank": "Hypothetical Current Loser Adjusted Current Rank"
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
    
            # Add the constraints
            for i in range(len(df)):
                # Can only pick an away team if 'Adjusted Current Difference' > 10
                if avoid_away_teams_in_close_matchups == 1:
                    if df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Adjusted Current Difference'] < 10 and df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank'] > df.loc[i, 'Hypothetical Current Loser Adjusted Current Rank']:
                        solver.Add(picks[i] == 0)
                #if df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Divisional Matchup?'] == 'Divisional':
                    #solver.Add(picks[i] == 0)
                if avoid_back_to_back_away == 1:
                    if df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Back to Back Away Games'] == 'True':
                        solver.Add(picks[i] == 0)
    
                # If 'Divisional Matchup?' is "Divisional", can only pick if 'Adjusted Current Difference' > 10
                if avoid_close_divisional_matchups == 1:
                    if df.loc[i, 'Divisional Matchup?'] == 'Divisional' and df.loc[i, 'Adjusted Current Difference'] < 10 and df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank'] > df.loc[i, 'Hypothetical Current Loser Adjusted Current Rank']:
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
                    if df.loc[i, 'City'] == 'London, UK' and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'City'] == 'London, UK' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
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
    
    
                
                if df.loc[i, 'Hypothetical Current Winner'] == 'Arizona Cardinals' and df.loc[i, 'Week_Num'] in az_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Atlanta Falcons' and df.loc[i, 'Week_Num'] in atl_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Baltimore Ravens' and df.loc[i, 'Week_Num'] in bal_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Buffalo Bills' and df.loc[i, 'Week_Num'] in buf_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Carolina Panthers' and df.loc[i, 'Week_Num'] in car_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Chicago Bears' and df.loc[i, 'Week_Num'] in chi_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Cincinnati Bengals' and df.loc[i, 'Week_Num'] in cin_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Cleveland Browns' and df.loc[i, 'Week_Num'] in cle_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Dallas Cowboys' and df.loc[i, 'Week_Num'] in dal_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Denver Broncos' and df.loc[i, 'Week_Num'] in den_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Detroit Lions' and df.loc[i, 'Week_Num'] in det_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Green Bay Packers' and df.loc[i, 'Week_Num'] in gb_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Houston Texans' and df.loc[i, 'Week_Num'] in hou_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Indianapolis Colts' and df.loc[i, 'Week_Num'] in ind_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Jacksonville Jaguars' and df.loc[i, 'Week_Num'] in jax_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Kansas City Chiefs' and df.loc[i, 'Week_Num'] in kc_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Las vegas Raiders' and df.loc[i, 'Week_Num'] in lv_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Los Angeles Chargers' and df.loc[i, 'Week_Num'] in lac_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Los Angeles Rams' and df.loc[i, 'Week_Num'] in lar_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Miami Dolphins' and df.loc[i, 'Week_Num'] in mia_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Minnesota Vikings' and df.loc[i, 'Week_Num'] in min_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'New England Patriots' and df.loc[i, 'Week_Num'] in ne_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'New Orleans Saints' and df.loc[i, 'Week_Num'] in no_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'New York Giants' and df.loc[i, 'Week_Num'] in nyg_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'New York Jets' and df.loc[i, 'Week_Num'] in nyj_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Philadelphia Eagles' and df.loc[i, 'Week_Num'] in phi_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Pittsburgh Steelers' and df.loc[i, 'Week_Num'] in pit_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Seattle Seahawks' and df.loc[i, 'Week_Num'] in sea_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Tampa Bay Buccaneers' and df.loc[i, 'Week_Num'] in tb_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Tennessee Titans' and df.loc[i, 'Week_Num'] in ten_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Washington Commanders' and df.loc[i, 'Week_Num'] in was_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'San Francisco 49ers' and df.loc[i, 'Week_Num'] in sf_excluded_weeks:
                        solver.Add(picks[i] == 0)
    
    
                if df.loc[i, 'Hypothetical Current Winner'] in picked_teams:
                    solver.Add(picks[i] == 0)
    
    # Add the constraint for San Francisco 49ers in week 11
            if sf_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'San Francisco 49ers') and df.loc[i, 'Week_Num'] == sf_req_week:
                        solver.Add(picks[i] == 1)
            if az_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Arizona Cardinals') and df.loc[i, 'Week_Num'] == az_req_week:
                        solver.Add(picks[i] == 1)
            if atl_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Atlanta Falcons') and df.loc[i, 'Week_Num'] == atl_req_week:
                        solver.Add(picks[i] == 1)
            if bal_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Baltimore Ravens') and df.loc[i, 'Week_Num'] == bal_req_week:
                        solver.Add(picks[i] == 1)
            if buf_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Buffalo Bills') and df.loc[i, 'Week_Num'] == buf_req_week:
                        solver.Add(picks[i] == 1)
            if car_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Carolina Panthers') and df.loc[i, 'Week_Num'] == car_req_week:
                        solver.Add(picks[i] == 1)
            if chi_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Chicago Bears') and df.loc[i, 'Week_Num'] == chi_req_week:
                        solver.Add(picks[i] == 1)
            if cin_req_week > 0:
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Cincinnati Bengals') and df.loc[i, 'Week_Num'] == cin_req_week:
                        solver.Add(picks[i] == 1)
            if cle_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Cleveland Browns') and df.loc[i, 'Week_Num'] == cle_req_week:
                        solver.Add(picks[i] == 1)
            if dal_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Dallas Cowboys') and df.loc[i, 'Week_Num'] == dal_req_week:
                        solver.Add(picks[i] == 1)
            if den_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Denver Broncos') and df.loc[i, 'Week_Num'] == den_req_week:
                        solver.Add(picks[i] == 1)
            if det_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Detroit Lions') and df.loc[i, 'Week_Num'] == det_req_week:
                        solver.Add(picks[i] == 1)
            if gb_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Green Bay Packers') and df.loc[i, 'Week_Num'] == gb_req_week:
                        solver.Add(picks[i] == 1)
            if hou_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Houston Texans') and df.loc[i, 'Week_Num'] == hou_req_week:
                        solver.Add(picks[i] == 1)
            if ind_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Indianapolis Colts') and df.loc[i, 'Week_Num'] == ind_req_week:
                        solver.Add(picks[i] == 1)
            if jax_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Jacksonville Jaguars') and df.loc[i, 'Week_Num'] == jax_req_week:
                        solver.Add(picks[i] == 1)
            if kc_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Kansas City Chiefs') and df.loc[i, 'Week_Num'] == kc_req_week:
                        solver.Add(picks[i] == 1)
            if lv_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Las Vegas Raiders') and df.loc[i, 'Week_Num'] == lv_req_week:
                        solver.Add(picks[i] == 1)
            if lac_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Los Angeles Chargers') and df.loc[i, 'Week_Num'] == lac_req_week:
                        solver.Add(picks[i] == 1)
            if lar_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Los Angeles Rams') and df.loc[i, 'Week_Num'] == lar_req_week:
                        solver.Add(picks[i] == 1)
            if mia_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Miami Dolphins') and df.loc[i, 'Week_Num'] == mia_req_week:
                        solver.Add(picks[i] == 1)
            if min_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Minnesota Vikings') and df.loc[i, 'Week_Num'] == min_req_week:
                        solver.Add(picks[i] == 1)
            if ne_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'New England Patriots') and df.loc[i, 'Week_Num'] == ne_req_week:
                        solver.Add(picks[i] == 1)
            if no_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'New Orleans Saints') and df.loc[i, 'Week_Num'] == no_req_week:
                        solver.Add(picks[i] == 1)
            if nyg_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'New York Giants') and df.loc[i, 'Week_Num'] == nyg_req_week:
                        solver.Add(picks[i] == 1)
            if nyj_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'New York Jets') and df.loc[i, 'Week_Num'] == nyj_req_week:
                        solver.Add(picks[i] == 1)
            if phi_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Philadelphia Eagles') and df.loc[i, 'Week_Num'] == phi_req_week:
                        solver.Add(picks[i] == 1)
            if pit_req_week > 0:        
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Pittsburgh Steelers') and df.loc[i, 'Week_Num'] == pit_req_week:
                        solver.Add(picks[i] == 1)
            if sea_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Seattle Seahawks') and df.loc[i, 'Week_Num'] == sea_req_week:
                        solver.Add(picks[i] == 1)
            if tb_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Tampa Bay Buccaneers') and df.loc[i, 'Week_Num'] == tb_req_week:
                        solver.Add(picks[i] == 1)
            if ten_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Tennessee Titans') and df.loc[i, 'Week_Num'] == ten_req_week:
                        solver.Add(picks[i] == 1)
            if was_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Washington Commanders') and df.loc[i, 'Week_Num'] == was_req_week:
                        solver.Add(picks[i] == 1)
    
    	    
            for week in df['Week_Num'].unique():
                # Filter picks for the current week
                weekly_picks = [picks[i] for i in range(len(df)) if df.loc[i, 'Week_Num'] == week]

                if selected_contest == "Splash Sports" and week >= week_requiring_two_selections:
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
                    # Load the picks from the previous iteration
                    if selected_contest == 'Circa':
                        previous_picks_df = pd.read_csv(f"circa_picks_ev_{previous_iteration + 1}.csv")
                    elif selected_contest == 'Splash Sports':
                        previous_picks_df = pd.read_csv(f"splash_picks_ev_{previous_iteration + 1}.csv")
                    else:
                        previous_picks_df = pd.read_csv(f"dk_picks_ev_{previous_iteration + 1}.csv")
    
                    # Extract the forbidden solution for this iteration
                    forbidden_solution_1 = previous_picks_df['Hypothetical Current Winner'].tolist()
                    forbidden_solutions_1.append(forbidden_solution_1)
    
            # Add constraints for all forbidden solutions
            for forbidden_solution_1 in forbidden_solutions_1:
                # Get the indices of the forbidden solution in the DataFrame
                forbidden_indices_1 = []
                for i in range(len(df)):
                    # Calculate the relative week number within the forbidden solution
                    df_week = df.loc[i, 'Week_Num']
                    relative_week = df_week - starting_week  # Adjust week to be relative to starting week
    
                    #Check if the week is within the range and the solution is forbidden
                    if 0 <= relative_week < len(forbidden_solution_1) and df_week >= starting_week and df_week < ending_week: #Added this to make sure we are only looking at the range
                        if (df.loc[i, 'Hypothetical Current Winner'] == forbidden_solution_1[relative_week]):
                            forbidden_indices_1.append(i)
                solver.Add(solver.Sum([1 - picks[i] for i in forbidden_indices_1]) >= 1)
    
            
    
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
    
                # Initialize picks_df
                picks_df = pd.DataFrame(columns=df.columns)
                picks_rows_2 = []
                for i in range(len(df)):
                    if picks[i].solution_value() > 0:
                        # Determine if it's a divisional game and if the picked team is the home team
    
                        week = df.loc[i, 'Week']
                        pick = df.loc[i,'Hypothetical Current Winner']
                        opponent = df.loc[i, 'Hypothetical Current Loser']
                        divisional_game = 'Divisional' if df.loc[i, 'Divisional Matchup Boolean'] else ''
                        pick_percent = df.loc[i, 'Home Pick %'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Pick %']
                        expected_availability = df.loc[i, 'Home Team Expected Availability'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Expected Availability']
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
                        win_odds = df.loc[i, 'Home Team Fair Odds'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Fair Odds']
                        
    
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
                        new_row_2 = {
                            'Week': week,
                            'Pick': pick,
                            'Opponent': opponent,
                            'EV': ev,
                            'Win Odds': win_odds,
                            'Expected Pick %': pick_percent,
                            'Expected Availability': expected_availability,
                            'Divisional Game': divisional_game,
                            'Home Team Status': home_team,
                            'Weekly Rest': weekly_rest,
                            'Weekly Rest Advantage': weekly_rest_advantage,
                            'Cumulative Rest': cumulative_rest,
                            'Cumulative Rest Advantage': cumulative_rest_advantage,
                            'Travel Advantage': travel_advantage,
                            'Back to Back Away Games': back_to_back_away_games,
                            'Thursday Night Game': thursday_night_game,
                            'International Game': international_game,
                            'Previous Opponent': previous_opponent,
                            'Previous Game Location': previous_game_location,
                            'Next Opponent': next_opponent,
                            'Next Game Location': next_game_location,
                            'Preseason Difference': preseason_difference,
                            'Adjusted Preseason Difference': adjusted_preseason_difference,
                            'Current Difference': current_difference,
                            'Adjusted Current Difference': adjusted_current_difference
                        }
                        picks_rows_2.append(new_row_2)
    
    
                        # Add differences to sums
                        sum_preseason_difference += preseason_difference
                        sum_adjusted_preseason_difference += adjusted_preseason_difference
                        sum_current_difference += current_difference
                        sum_adjusted_current_difference += adjusted_current_difference
                        sum_ev += ev
                        picks_df = pd.concat([picks_df, df.loc[[i]]], ignore_index=True)
                        picks_df['Divisional Matchup?'] = divisional_game
                summarized_picks_df = pd.DataFrame(picks_rows_2)
    
                st.write(summarized_picks_df)
                st.write('')
                st.write('\nPreseason Difference:', sum_preseason_difference)
                st.write('Adjusted Preseason Difference:', sum_adjusted_preseason_difference)
                st.write('Current Difference:', sum_current_difference)
                st.write('Adjusted Current Difference:', sum_adjusted_current_difference)
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
            
            # Append the new forbidden solution to the list
            forbidden_solutions_1.append(picks_df['Hypothetical Current Winner'].tolist())

def get_survivor_picks_based_on_internal_rankings():
    if pick_must_be_favored:
        for iteration in range(number_solutions):
            df = nfl_schedule_pick_percentages_df
    
    
            #Number of weeks that have already been played
            #weeks_completed = starting_week -1
    
            # Teams already picked - Team name in quotes and separated by commas
    
            # Filter out weeks that have already been played and reset index
    
            df = df[(df['Week_Num'] >= starting_week) & (df['Week_Num'] < ending_week)].reset_index(drop=True)
            # Filter out already picked teams
            df = df[~df['Adjusted Current Winner'].isin(picked_teams)].reset_index(drop=True)
            #print(df)
            # Create the solver
            solver = pywraplp.Solver.CreateSolver('SCIP')
    
            # Create binary variables to represent the picks, and store them in a dictionary for easy lookup
            picks = {}
            for i in range(len(df)):
                picks[i] = solver.IntVar(0, 1, 'pick_%i' % i)
    
            # Add the constraints
            for week in df['Week_Num'].unique():
                # Filter picks for the current week
                weekly_picks = [picks[i] for i in range(len(df)) if df.loc[i, 'Week_Num'] == week]

                if selected_contest == "Splash Sports" and week >= week_requiring_two_selections:
                    # For Splash Sports and later weeks, two teams must be selected
                    solver.Add(solver.Sum(weekly_picks) == 2)
                else:
                    # For other contests or earlier weeks, one team per week
                    solver.Add(solver.Sum(weekly_picks) == 1)
    
            for team in df['Adjusted Current Winner'].unique():
                # Can't pick a team more than once
                solver.Add(solver.Sum([picks[i] for i in range(len(df)) if df.loc[i, 'Adjusted Current Winner'] == team]) <= 1)
    
            for i in range(len(df)):
                # Must pick from 'Adjusted Current Winner'
                #if df.loc[i, 'Adjusted Current Winner'] != df.loc[i, 'Home Team']:
                    #solver.Add(picks[i] == 0)
                # Must pick from 'Same Winner?'
                if bayesian_rest_travel_constraint == "Selected team must have been projected to win based on preseason rankings, current rankings, and with and without travel/rest adjustments":
                    if df.loc[i, 'Same Winner?'] != 'Same':
                        solver.Add(picks[i] == 0)
                elif  bayesian_rest_travel_constraint == "Selected team must be projected to win with and without travel and rest impact based on current rankings":
                    if df.loc[i, 'Same Current and Adjusted Current Winner?'] != 'Same':
                        solver.Add(picks[i] == 0)
                elif  bayesian_rest_travel_constraint == "Selected team must have been projected to win based on preseason rankings in addition to current rankings":   
                    if df.loc[i, 'Same Adjusted Preseason Winner?'] != 'Same':
                        solver.Add(picks[i] == 0)
                # Can only pick an away team if 'Adjusted Current Difference' > 10
                if avoid_away_teams_in_close_matchups == 1:
                    if df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Adjusted Current Difference'] < 10:
                        solver.Add(picks[i] == 0)
                #if df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Divisional Matchup?'] == 'Divisional':
                    #solver.Add(picks[i] == 0)
                if avoid_back_to_back_away == 1:
                    if df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Back to Back Away Games'] == 'True':
                        solver.Add(picks[i] == 0)
    
                # If 'Divisional Matchup?' is "Divisional", can only pick if 'Adjusted Current Difference' > 10
                if avoid_close_divisional_matchups == 1:
                    if df.loc[i, 'Divisional Matchup?'] == 'Divisional' and df.loc[i, 'Adjusted Current Difference'] < 10:
                        solver.Add(picks[i] == 0)
                # Constraints for short rest and 4 games in 17 days (only if team is the Adjusted Current Winner)
                if avoid_away_teams_on_short_rest == 1:
                    if df.loc[i, 'Away Team Short Rest'] == 'Yes' and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                if avoid_4_games_in_17_days == 1:
                    if df.loc[i, 'Home Team 4 games in 17 days'] == 'Yes' and df.loc[i, 'Home Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Away Team 4 games in 17 days'] == 'No':
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'Away Team 4 games in 17 days'] == 'Yes' and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Home Team 4 games in 17 days'] == 'No':
                        solver.Add(picks[i] == 0)
                if avoid_3_games_in_10_days == 1:
                    if df.loc[i, 'Home Team 3 games in 10 days'] == 'Yes' and df.loc[i, 'Home Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Away Team 3 games in 10 days'] == 'No':
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'Away Team 3 games in 10 days'] == 'Yes' and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Home Team 3 games in 10 days'] == 'No':
                        solver.Add(picks[i] == 0)
                if avoid_international_game == 1:    
                    if df.loc[i, 'City'] == 'London, UK' and df.loc[i, 'Home Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'City'] == 'London, UK' and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                if avoid_thursday_night == 1:
                    if df.loc[i, 'Thursday Night Game'] == 'True':
                        solver.Add(picks[i] == 0)
                if avoid_away_thursday_night == 1:
                    if df.loc[i, 'Thursday Night Game'] == 'True' and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                if avoid_teams_with_weekly_rest_disadvantage == 1:
                    if df.loc[i, 'Home Team Weekly Rest'] < df.loc [i, 'Away Team Weekly Rest'] and df.loc[i, 'Home Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'Away Team Weekly Rest'] < df.loc [i, 'Home Team Weekly Rest'] and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                if avoid_cumulative_rest_disadvantage == 1:
                    if df.loc[i, 'Away Team Current Week Cumulative Rest Advantage'] < -10 and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'Home Team Current Week Cumulative Rest Advantage'] < -5 and df.loc[i, 'Home Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
                if avoid_away_teams_with_travel_disadvantage == 1:
                    if df.loc[i, 'Away Travel Advantage'] < -850 and df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner']:
                        solver.Add(picks[i] == 0)
    
    
                
                if df.loc[i, 'Adjusted Current Winner'] == 'Arizona Cardinals' and df.loc[i, 'Week_Num'] in az_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Atlanta Falcons' and df.loc[i, 'Week_Num'] in atl_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Baltimore Ravens' and df.loc[i, 'Week_Num'] in bal_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Buffalo Bills' and df.loc[i, 'Week_Num'] in buf_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Carolina Panthers' and df.loc[i, 'Week_Num'] in car_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Chicago Bears' and df.loc[i, 'Week_Num'] in chi_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Cincinnati Bengals' and df.loc[i, 'Week_Num'] in cin_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Cleveland Browns' and df.loc[i, 'Week_Num'] in cle_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Dallas Cowboys' and df.loc[i, 'Week_Num'] in dal_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Denver Broncos' and df.loc[i, 'Week_Num'] in den_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Detroit Lions' and df.loc[i, 'Week_Num'] in det_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'AGreen Bay Packers' and df.loc[i, 'Week_Num'] in gb_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Houston Texans' and df.loc[i, 'Week_Num'] in hou_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Indianapolis Colts' and df.loc[i, 'Week_Num'] in ind_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Jacksonville Jaguars' and df.loc[i, 'Week_Num'] in jax_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Kansas City Chiefs' and df.loc[i, 'Week_Num'] in kc_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Las vegas Raiders' and df.loc[i, 'Week_Num'] in lv_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Chargers' and df.loc[i, 'Week_Num'] in lac_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Rams' and df.loc[i, 'Week_Num'] in lar_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Miami Dolphins' and df.loc[i, 'Week_Num'] in mia_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Minnesota Vikings' and df.loc[i, 'Week_Num'] in min_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'New England Patriots' and df.loc[i, 'Week_Num'] in ne_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'New Orleans Saints' and df.loc[i, 'Week_Num'] in no_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'New York Giants' and df.loc[i, 'Week_Num'] in nyg_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'New York Jets' and df.loc[i, 'Week_Num'] in nyj_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Philadelphia Eagles' and df.loc[i, 'Week_Num'] in phi_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Pittsburgh Steelers' and df.loc[i, 'Week_Num'] in pit_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Seattle Seahawks' and df.loc[i, 'Week_Num'] in sea_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Tampa Bay Buccaneers' and df.loc[i, 'Week_Num'] in tb_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Tennessee Titans' and df.loc[i, 'Week_Num'] in ten_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'Washington Commanders' and df.loc[i, 'Week_Num'] in was_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Adjusted Current Winner'] == 'San Francisco 49ers' and df.loc[i, 'Week_Num'] in sf_excluded_weeks:
                        solver.Add(picks[i] == 0)
    
            # Add the constraint for San Francisco 49ers in week 11
            if sf_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'San Francisco 49ers' or df.loc[i, 'Away Team'] == 'San Francisco 49ers') and df.loc[i, 'Week_Num'] == sf_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'San Francisco 49ers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'San Francisco 49ers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'San Francisco 49ers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if az_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Arizona Cardinals' or df.loc[i, 'Away Team'] == 'Arizona Cardinals') and df.loc[i, 'Week_Num'] == az_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Arizona Cardinals':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Arizona Cardinals':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Arizona Cardinals':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if atl_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Atlanta Falcons' or df.loc[i, 'Away Team'] == 'Atlanta Falcons') and df.loc[i, 'Week_Num'] == atl_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Atlanta Falcons':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Atlanta Falcons':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Atlanta Falcons':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if bal_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Baltimore Ravens' or df.loc[i, 'Away Team'] == 'Baltimore Ravens') and df.loc[i, 'Week_Num'] == bal_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Baltimore Ravens':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Baltimore Ravens':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Baltimore Ravens':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if buf_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Buffalo Bills' or df.loc[i, 'Away Team'] == 'Buffalo Bills') and df.loc[i, 'Week_Num'] == buf_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Buffalo Bills':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Buffalo Bills':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Buffalo Bills':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if car_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Carolina Panthers' or df.loc[i, 'Away Team'] == 'Carolina Panthers') and df.loc[i, 'Week_Num'] == car_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Carolina Panthers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Carolina Panthers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Carolina Panthers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if chi_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Chicago Bears' or df.loc[i, 'Away Team'] == 'Chicago Bears') and df.loc[i, 'Week_Num'] == chi_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Chicago Bears':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Chicago Bears':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Chicago Bears':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if cin_req_week > 0:        
                    if (df.loc[i, 'Home Team'] == 'Cincinnati Bengals' or df.loc[i, 'Away Team'] == 'Cincinnati Bengals') and df.loc[i, 'Week_Num'] == cin_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Cincinnati Bengals':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Cincinnati Bengals':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Cincinnati Bengals':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if cle_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Cleveland Browns' or df.loc[i, 'Away Team'] == 'Cleveland Browns') and df.loc[i, 'Week_Num'] == cle_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Cleveland Browns':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Cleveland Browns':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Cleveland Browns':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if dal_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Dallas Cowboys' or df.loc[i, 'Away Team'] == 'Dallas Cowboys') and df.loc[i, 'Week_Num'] == dal_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Dallas Cowboys':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Dallas Cowboys':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Dallas Cowboys':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if den_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Denver Broncos' or df.loc[i, 'Away Team'] == 'Denver Broncos') and df.loc[i, 'Week_Num'] == den_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Denver Broncos':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Denver Broncos':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Denver Broncos':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if det_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Detroit Lions' or df.loc[i, 'Away Team'] == 'Detroit Lions') and df.loc[i, 'Week_Num'] == det_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Detroit Lions':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Detroit Lions':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Detroit Lions':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if gb_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Green Bay Packers' or df.loc[i, 'Away Team'] == 'Green Bay Packers') and df.loc[i, 'Week_Num'] == gb_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Green Bay Packers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Green Bay Packers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Green Bay Packers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if hou_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Houston Texans' or df.loc[i, 'Away Team'] == 'Houston Texans') and df.loc[i, 'Week_Num'] == hou_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Houston Texans':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Houston Texans':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Houston Texans':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if ind_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Indianapolis Colts' or df.loc[i, 'Away Team'] == 'Indianapolis Colts') and df.loc[i, 'Week_Num'] == ind_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Indianapolis Colts':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Indianapolis Colts':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Indianapolis Colts':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if jax_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Jacksonville Jaguars' or df.loc[i, 'Away Team'] == 'Jacksonville Jaguars') and df.loc[i, 'Week_Num'] == jax_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Jacksonville Jaguars':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Jacksonville Jaguars':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Jacksonville Jaguars':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if kc_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Kansas City Chiefs' or df.loc[i, 'Away Team'] == 'Kansas City Chiefs') and df.loc[i, 'Week_Num'] == kc_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Kansas City Chiefs':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Kansas City Chiefs':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Kansas City Chiefs':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if lv_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Las Vegas Raiders' or df.loc[i, 'Away Team'] == 'Las Vegas Raiders') and df.loc[i, 'Week_Num'] == lv_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Las Vegas Raiders':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Las Vegas Raiders':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Las Vegas Raiders':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if lac_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Los Angeles Chargers' or df.loc[i, 'Away Team'] == 'Los Angeles Chargers') and df.loc[i, 'Week_Num'] == lac_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Chargers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Los Angeles Chargers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Los Angeles Chargers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if lar_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Los Angeles Rams' or df.loc[i, 'Away Team'] == 'Los Angeles Rams') and df.loc[i, 'Week_Num'] == lar_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Rams':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Los Angeles Rams':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Los Angeles Rams':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if mia_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Miami Dolphins' or df.loc[i, 'Away Team'] == 'Miami Dolphins') and df.loc[i, 'Week_Num'] == mia_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Miami Dolphins':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Miami Dolphins':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Miami Dolphins':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if min_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Minnesota Vikings' or df.loc[i, 'Away Team'] == 'Minnesota Vikings') and df.loc[i, 'Week_Numv'] == min_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Minnesota Vikings':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Minnesota Vikings':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Minnesota Vikings':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if ne_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'New England Patriots' or df.loc[i, 'Away Team'] == 'New England Patriots') and df.loc[i, 'Week_Num'] == ne_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'New England Patriots':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'New England Patriots':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'New England Patriots':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if no_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'New Orleans Saints' or df.loc[i, 'Away Team'] == 'New Orleans Saints') and df.loc[i, 'Week_Num'] == no_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'New Orleans Saints':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'New Orleans Saints':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'New Orleans Saints':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if nyg_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'New York Giants' or df.loc[i, 'Away Team'] == 'New York Giants') and df.loc[i, 'Week_Num'] == nyg_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'New York Giants':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'New York Giants':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'New York Giants':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if nyj_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'New York Jets' or df.loc[i, 'Away Team'] == 'New York Jets') and df.loc[i, 'Week_Num'] == nyj_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'New York Jets':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'New York Jets':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'New York Jets':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if phi_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Philadelphia Eagles' or df.loc[i, 'Away Team'] == 'Philadelphia Eagles') and df.loc[i, 'Week_Num'] == phi_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Philadelphia Eagles':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Philadelphia Eagles':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Philadelphia Eagles':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if pit_req_week > 0:        
                    if (df.loc[i, 'Home Team'] == 'Pittsburgh Steelers' or df.loc[i, 'Away Team'] == 'Pittsburgh Steelers') and df.loc[i, 'Week_Num'] == pit_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Pittsburgh Steelers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Pittsburgh Steelers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Pittsburgh Steelers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if sea_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Seattle Seahawks' or df.loc[i, 'Away Team'] == 'Seattle Seahawks') and df.loc[i, 'Week_Num'] == sea_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Seattle Seahawks':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Seattle Seahawks':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Seattle Seahawks':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if tb_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Tampa Bay Buccaneers' or df.loc[i, 'Away Team'] == 'Tampa Bay Buccaneers') and df.loc[i, 'Week_Num'] == tb_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Tampa Bay Buccaneers':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Tampa Bay Buccaneers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Tampa Bay Buccaneers':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if ten_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Tennessee Titans' or df.loc[i, 'Away Team'] == 'Tennessee Titans') and df.loc[i, 'Week_Num'] == ten_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Tennessee Titans':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Tennessee Titans':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Tennessee Titans':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
            if was_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Home Team'] == 'Washington Commanders' or df.loc[i, 'Away Team'] == 'Washington Commanders') and df.loc[i, 'Week_Num'] == was_req_week:
                        if df.loc[i, 'Adjusted Current Winner'] == 'Washington Commanders':
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Home Team'] == 'Washington Commanders':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                            solver.Add(picks[i] == 1)
                        elif df.loc[i, 'Away Team'] == 'Washington Commanders':
                            df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                            solver.Add(picks[i] == 1)
    
    
                if df.loc[i, 'Adjusted Current Winner'] in picked_teams:
                    solver.Add(picks[i] == 0)
            # Dynamically create the forbidden solution list
            forbidden_solutions_1 = []
            if iteration > 0: 
                for previous_iteration in range(iteration):
                    # Load the picks from the previous iteration
                    if selected_contest == 'Circa':
                        previous_picks_df = pd.read_csv(f"circa_picks_ir_{previous_iteration + 1}.csv")
                    elif selected_contest == 'Splash Sports':
                        previous_picks_df = pd.read_csv(f"splash_picks_ir_{previous_iteration + 1}.csv")
                    else:
                        previous_picks_df = pd.read_csv(f"dk_picks_ir_{previous_iteration + 1}.csv")
    
                    # Extract the forbidden solution for this iteration
                    forbidden_solution_1 = previous_picks_df['Adjusted Current Winner'].tolist()
                    forbidden_solutions_1.append(forbidden_solution_1)
    
            # Add constraints for all forbidden solutions
            for forbidden_solution_1 in forbidden_solutions_1:
                # Get the indices of the forbidden solution in the DataFrame
                forbidden_indices_1 = []
                for i in range(len(df)):
                    # Calculate the relative week number within the forbidden solution
                    df_week = df.loc[i, 'Week_Num']
                    relative_week = df_week - starting_week  # Adjust week to be relative to starting week
    
                    #Check if the week is within the range and the solution is forbidden
                    if 0 <= relative_week < len(forbidden_solution_1) and df_week >= starting_week and df_week < ending_week: #Added this to make sure we are only looking at the range
                        if (df.loc[i, 'Adjusted Current Winner'] == forbidden_solution_1[relative_week]):
                            forbidden_indices_1.append(i)
    
                # Add the constraint
                solver.Add(solver.Sum([1 - picks[i] for i in forbidden_indices_1]) >= 1)
     
    
            # Objective: maximize the sum of Adjusted Current Difference of each game picked
            solver.Maximize(solver.Sum([picks[i] * df.loc[i, 'Adjusted Current Difference'] for i in range(len(df))]))
    
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
    
                # Initialize picks_df
                picks_df = pd.DataFrame(columns=df.columns)
                picks_rows_2 = []
                for i in range(len(df)):
                    if picks[i].solution_value() > 0:
                        # Determine if it's a divisional game and if the picked team is the home team
    
                        week = df.loc[i, 'Week']
                        pick = df.loc[i,'Adjusted Current Winner']
                        opponent = df.loc[i, 'Home Team'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Away Team'] else df.loc[i, 'Away Team']
                        win_odds = df.loc[i, 'Home Team Fair Odds'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Fair Odds']
                        pick_percent = df.loc[i, 'Home Pick %'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Pick %']
                        expected_availability = df.loc[i, 'Home Team Expected Availability'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Expected Availability']
                        divisional_game = 'Divisional' if df.loc[i, 'Divisional Matchup Boolean'] else ''
                        home_team = 'Home Team' if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else 'Away Team'
                        weekly_rest = df.loc[i, 'Home Team Weekly Rest'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Weekly Rest']
                        weekly_rest_advantage = df.loc[i, 'Weekly Home Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Weekly Away Rest Advantage']
                        cumulative_rest = df.loc[i, 'Home Cumulative Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Cumulative Rest Advantage']
                        cumulative_rest_advantage = df.loc[i, 'Home Team Current Week Cumulative Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Current Week Cumulative Rest Advantage']
                        travel_advantage = df.loc[i, 'Home Travel Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Travel Advantage']
                        back_to_back_away_games = 'True' if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Away Team'] and df.loc[i, 'Back to Back Away Games'] == 'True' else 'False'
                        thursday_night_game = 'Thursday Night Game' if df.loc[i, "Thursday Night Game"] == 'True' else 'Sunday/Monday Game'
                        international_game = 'International Game' if df.loc[i, 'Actual Stadium'] == 'London, UK' else 'Domestic Game'
                        previous_opponent = df.loc[i, 'Home Team Previous Opponent'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Previous Opponent']
                        previous_game_location = df.loc[i, 'Home Team Previous Location'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Previous Location']
                        next_opponent = df.loc[i, 'Home Team Next Opponent'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Next Opponent']
                        next_game_location = df.loc[i, 'Home Team Next Location'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Next Location']

                        
    
                        # Get differences
                        preseason_difference = df.loc[i, 'Preseason Difference']
                        adjusted_preseason_difference = df.loc[i, 'Adjusted Preseason Difference']
                        current_difference = df.loc[i, 'Current Difference']
                        adjusted_current_difference = df.loc[i, 'Adjusted Current Difference']
                        # Calculate EV for this game
                        #ev = (df.loc[i, 'Home Team EV'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team EV'])
    
    
                        #print('Week %i: Pick %s %s %s (%i, %i, %i, %i, %.4f)' % (df.loc[i, 'Week_Num'], df.loc[i, 'Adjusted Current Winner'], divisional_game, home_team,
                        #                                                   preseason_difference, adjusted_preseason_difference,
                        #                                                   current_difference, adjusted_current_difference, ev))
                        new_row_2 = {
                            'Week': week,
                            'Pick': pick,
                            'Opponent': opponent,
                            'EV': ev,
                            'Win Odds': win_odds,
                            'Pick %': pick_percent,
                            'Expected Availability': expected_availability,
                            'Divisional Game': divisional_game,
                            'Home Team Status': home_team,
                            'Weekly Rest': weekly_rest,
                            'Weekly Rest Advantage': weekly_rest_advantage,
                            'Cumulative Rest': cumulative_rest,
                            'Cumulative Rest Advantage': cumulative_rest_advantage,
                            'Travel Advantage': travel_advantage,
                            'Back to Back Away Games': back_to_back_away_games,
                            'Thursday Night Game': thursday_night_game,
                            'International Game': international_game,
                            'Previous Opponent': previous_opponent,
                            'Previous Game Location': previous_game_location,
                            'Next Opponent': next_opponent,
                            'Next Game Location': next_game_location,
                            'Preseason Spread': preseason_difference,
                            'Adjusted Preseason Spread (Homefield, Rest, etc...)': adjusted_preseason_difference,
                            'Current Spread': current_difference,
                            'Adjusted Current Spread (Homefield, Rest, etc...)': adjusted_current_difference
    
                        }
                        picks_rows_2.append(new_row_2)
    
    
                        # Add differences to sums
                        sum_preseason_difference += preseason_difference
                        sum_adjusted_preseason_difference += adjusted_preseason_difference
                        sum_current_difference += current_difference
                        sum_adjusted_current_difference += adjusted_current_difference
                        #sum_ev += ev
                        picks_df = pd.concat([picks_df, df.loc[[i]]], ignore_index=True)
                        picks_df['Divisional Matchup?'] = divisional_game
                summarized_picks_df = pd.DataFrame(picks_rows_2)
    
                st.write(summarized_picks_df)
                st.write('')
                #st.write(f'Total EV: {sum_ev}')
                st.write('\nPreseason Difference:', sum_preseason_difference)
                st.write('Adjusted Preseason Difference:', sum_adjusted_preseason_difference)
                st.write('Current Difference:', sum_current_difference)
                st.write(f'Adjusted Current Difference: :blue[{sum_adjusted_current_difference}]')
                
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
            
            # Append the new forbidden solution to the list
            forbidden_solutions_1.append(picks_df['Adjusted Current Winner'].tolist())
            #print(forbidden_solutions)
    else:
        for iteration in range(number_solutions):
            df = full_df_with_ev
    
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
                "Away Team Current Rank": "Hypothetical Current Loser Current Rank"
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
                "Home Team Current Rank": "Hypothetical Current Loser Current Rank"
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
            df['Hypthetical Winner Preseason Difference'] = df['Hypothetical Current Winner Preseason Rank'] - df['Hypothetical Current Loser Preseason Rank']	
            df['Hypthetical Winner Adjusted Preseason Difference'] = df['Hypothetical Current Winner Adjusted Preseason Rank'] - df['Hypothetical Current Loser Adjusted Preseason Rank']
            df['Hypthetical Winner Current Difference'] = df['Hypothetical Current Winner Current Rank'] - df['Hypothetical Current Loser Current Rank']	
            df['Hypthetical Winner Adjusted Current Difference'] = df['Hypothetical Current Winner Adjusted Current Rank'] - df['Hypothetical Current Loser Adjusted Current Rank']	

		
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
                if avoid_away_teams_in_close_matchups == 1:
                    if df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Adjusted Current Difference'] < 10 and df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank'] > df.loc[i, 'Hypothetical Current Loser Adjusted Current Rank']:
                        solver.Add(picks[i] == 0)
                #if df.loc[i, 'Away Team'] == df.loc[i, 'Adjusted Current Winner'] and df.loc[i, 'Divisional Matchup?'] == 'Divisional':
                    #solver.Add(picks[i] == 0)
                if avoid_back_to_back_away == 1:
                    if df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner'] and df.loc[i, 'Back to Back Away Games'] == 'True':
                        solver.Add(picks[i] == 0)
    
                # If 'Divisional Matchup?' is "Divisional", can only pick if 'Adjusted Current Difference' > 10
                if avoid_close_divisional_matchups == 1:
                    if df.loc[i, 'Divisional Matchup?'] == 'Divisional' and df.loc[i, 'Adjusted Current Difference'] < 10 and df.loc[i, 'Hypothetical Current Winner Adjusted Current Rank'] > df.loc[i, 'Hypothetical Current Loser Adjusted Current Rank']:
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
                    if df.loc[i, 'City'] == 'London, UK' and df.loc[i, 'Home Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
                        solver.Add(picks[i] == 0)
                    if df.loc[i, 'City'] == 'London, UK' and df.loc[i, 'Away Team 1'] == df.loc[i, 'Hypothetical Current Winner']:
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
    
    
                
                if df.loc[i, 'Hypothetical Current Winner'] == 'Arizona Cardinals' and df.loc[i, 'Week_Num'] in az_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Atlanta Falcons' and df.loc[i, 'Week_Num'] in atl_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Baltimore Ravens' and df.loc[i, 'Week_Num'] in bal_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Buffalo Bills' and df.loc[i, 'Week_Num'] in buf_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Carolina Panthers' and df.loc[i, 'Week_Num'] in car_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Chicago Bears' and df.loc[i, 'Week_Num'] in chi_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Cincinnati Bengals' and df.loc[i, 'Week_Num'] in cin_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Cleveland Browns' and df.loc[i, 'Week_Num'] in cle_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Dallas Cowboys' and df.loc[i, 'Week_Num'] in dal_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Denver Broncos' and df.loc[i, 'Week_Num'] in den_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Detroit Lions' and df.loc[i, 'Week_Num'] in det_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Green Bay Packers' and df.loc[i, 'Week_Num'] in gb_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Houston Texans' and df.loc[i, 'Week_Num'] in hou_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Indianapolis Colts' and df.loc[i, 'Week_Num'] in ind_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Jacksonville Jaguars' and df.loc[i, 'Week_Num'] in jax_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Kansas City Chiefs' and df.loc[i, 'Week_Num'] in kc_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Las vegas Raiders' and df.loc[i, 'Week_Num'] in lv_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Los Angeles Chargers' and df.loc[i, 'Week_Num'] in lac_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Los Angeles Rams' and df.loc[i, 'Week_Num'] in lar_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Miami Dolphins' and df.loc[i, 'Week_Num'] in mia_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Minnesota Vikings' and df.loc[i, 'Week_Num'] in min_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'New England Patriots' and df.loc[i, 'Week_Num'] in ne_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'New Orleans Saints' and df.loc[i, 'Week_Num'] in no_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'New York Giants' and df.loc[i, 'Week_Num'] in nyg_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'New York Jets' and df.loc[i, 'Week_Num'] in nyj_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Philadelphia Eagles' and df.loc[i, 'Week_Num'] in phi_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Pittsburgh Steelers' and df.loc[i, 'Week_Num'] in pit_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Seattle Seahawks' and df.loc[i, 'Week_Num'] in sea_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Tampa Bay Buccaneers' and df.loc[i, 'Week_Num'] in tb_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Tennessee Titans' and df.loc[i, 'Week_Num'] in ten_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'Washington Commanders' and df.loc[i, 'Week_Num'] in was_excluded_weeks:
                        solver.Add(picks[i] == 0)
                if df.loc[i, 'Hypothetical Current Winner'] == 'San Francisco 49ers' and df.loc[i, 'Week_Num'] in sf_excluded_weeks:
                        solver.Add(picks[i] == 0)
    
    
                if df.loc[i, 'Hypothetical Current Winner'] in picked_teams:
                    solver.Add(picks[i] == 0)
    
    # Add the constraint for San Francisco 49ers in week 11
            if sf_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'San Francisco 49ers') and df.loc[i, 'Week_Num'] == sf_req_week:
                        solver.Add(picks[i] == 1)
            if az_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Arizona Cardinals') and df.loc[i, 'Week_Num'] == az_req_week:
                        solver.Add(picks[i] == 1)
            if atl_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Atlanta Falcons') and df.loc[i, 'Week_Num'] == atl_req_week:
                        solver.Add(picks[i] == 1)
            if bal_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Baltimore Ravens') and df.loc[i, 'Week_Num'] == bal_req_week:
                        solver.Add(picks[i] == 1)
            if buf_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Buffalo Bills') and df.loc[i, 'Week_Num'] == buf_req_week:
                        solver.Add(picks[i] == 1)
            if car_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Carolina Panthers') and df.loc[i, 'Week_Num'] == car_req_week:
                        solver.Add(picks[i] == 1)
            if chi_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Chicago Bears') and df.loc[i, 'Week_Num'] == chi_req_week:
                        solver.Add(picks[i] == 1)
            if cin_req_week > 0:
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Cincinnati Bengals') and df.loc[i, 'Week_Num'] == cin_req_week:
                        solver.Add(picks[i] == 1)
            if cle_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Cleveland Browns') and df.loc[i, 'Week_Num'] == cle_req_week:
                        solver.Add(picks[i] == 1)
            if dal_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Dallas Cowboys') and df.loc[i, 'Week_Num'] == dal_req_week:
                        solver.Add(picks[i] == 1)
            if den_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Denver Broncos') and df.loc[i, 'Week_Num'] == den_req_week:
                        solver.Add(picks[i] == 1)
            if det_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Detroit Lions') and df.loc[i, 'Week_Num'] == det_req_week:
                        solver.Add(picks[i] == 1)
            if gb_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Green Bay Packers') and df.loc[i, 'Week_Num'] == gb_req_week:
                        solver.Add(picks[i] == 1)
            if hou_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Houston Texans') and df.loc[i, 'Week_Num'] == hou_req_week:
                        solver.Add(picks[i] == 1)
            if ind_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Indianapolis Colts') and df.loc[i, 'Week_Num'] == ind_req_week:
                        solver.Add(picks[i] == 1)
            if jax_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Jacksonville Jaguars') and df.loc[i, 'Week_Num'] == jax_req_week:
                        solver.Add(picks[i] == 1)
            if kc_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Kansas City Chiefs') and df.loc[i, 'Week_Num'] == kc_req_week:
                        solver.Add(picks[i] == 1)
            if lv_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Las Vegas Raiders') and df.loc[i, 'Week_Num'] == lv_req_week:
                        solver.Add(picks[i] == 1)
            if lac_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Los Angeles Chargers') and df.loc[i, 'Week_Num'] == lac_req_week:
                        solver.Add(picks[i] == 1)
            if lar_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Los Angeles Rams') and df.loc[i, 'Week_Num'] == lar_req_week:
                        solver.Add(picks[i] == 1)
            if mia_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Miami Dolphins') and df.loc[i, 'Week_Num'] == mia_req_week:
                        solver.Add(picks[i] == 1)
            if min_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Minnesota Vikings') and df.loc[i, 'Week_Num'] == min_req_week:
                        solver.Add(picks[i] == 1)
            if ne_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'New England Patriots') and df.loc[i, 'Week_Num'] == ne_req_week:
                        solver.Add(picks[i] == 1)
            if no_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'New Orleans Saints') and df.loc[i, 'Week_Num'] == no_req_week:
                        solver.Add(picks[i] == 1)
            if nyg_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'New York Giants') and df.loc[i, 'Week_Num'] == nyg_req_week:
                        solver.Add(picks[i] == 1)
            if nyj_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'New York Jets') and df.loc[i, 'Week_Num'] == nyj_req_week:
                        solver.Add(picks[i] == 1)
            if phi_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Philadelphia Eagles') and df.loc[i, 'Week_Num'] == phi_req_week:
                        solver.Add(picks[i] == 1)
            if pit_req_week > 0:        
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Pittsburgh Steelers') and df.loc[i, 'Week_Num'] == pit_req_week:
                        solver.Add(picks[i] == 1)
            if sea_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Seattle Seahawks') and df.loc[i, 'Week_Num'] == sea_req_week:
                        solver.Add(picks[i] == 1)
            if tb_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Tampa Bay Buccaneers') and df.loc[i, 'Week_Num'] == tb_req_week:
                        solver.Add(picks[i] == 1)
            if ten_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Tennessee Titans') and df.loc[i, 'Week_Num'] == ten_req_week:
                        solver.Add(picks[i] == 1)
            if was_req_week > 0:        
                for i in range(len(df)):
                    if (df.loc[i, 'Hypothetical Current Winner'] == 'Washington Commanders') and df.loc[i, 'Week_Num'] == was_req_week:
                        solver.Add(picks[i] == 1)
    
    	    
            # Add the constraints
            for week in df['Week_Num'].unique():
                # Filter picks for the current week
                weekly_picks = [picks[i] for i in range(len(df)) if df.loc[i, 'Week_Num'] == week]

                if selected_contest == "Splash Sports" and week >= week_requiring_two_selections:
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
                    # Load the picks from the previous iteration
                    if selected_contest == 'Circa':
                        previous_picks_df = pd.read_csv(f"circa_picks_ev_{previous_iteration + 1}.csv")
                    elif selected_contest == 'Splash Sports':
                        previous_picks_df = pd.read_csv(f"splash_picks_ev_{previous_iteration + 1}.csv")
                    else:
                        previous_picks_df = pd.read_csv(f"dk_picks_ev_{previous_iteration + 1}.csv")
    
                    # Extract the forbidden solution for this iteration
                    forbidden_solution_1 = previous_picks_df['Hypothetical Current Winner'].tolist()
                    forbidden_solutions_1.append(forbidden_solution_1)
    
            # Add constraints for all forbidden solutions
            for forbidden_solution_1 in forbidden_solutions_1:
                # Get the indices of the forbidden solution in the DataFrame
                forbidden_indices_1 = []
                for i in range(len(df)):
                    # Calculate the relative week number within the forbidden solution
                    df_week = df.loc[i, 'Week_Num']
                    relative_week = df_week - starting_week  # Adjust week to be relative to starting week
    
                    #Check if the week is within the range and the solution is forbidden
                    if 0 <= relative_week < len(forbidden_solution_1) and df_week >= starting_week and df_week < ending_week: #Added this to make sure we are only looking at the range
                        if (df.loc[i, 'Hypothetical Current Winner'] == forbidden_solution_1[relative_week]):
                            forbidden_indices_1.append(i)
                solver.Add(solver.Sum([1 - picks[i] for i in forbidden_indices_1]) >= 1)
    
            
    
            # Objective: maximize the sum of Adjusted Current Difference of each game picked
            solver.Maximize(solver.Sum([picks[i] * df.loc[i, 'Hypthetical Winner Adjusted Current Difference'] for i in range(len(df))]))


    
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
    
                # Initialize picks_df
                picks_df = pd.DataFrame(columns=df.columns)
                picks_rows_2 = []
                for i in range(len(df)):
                    if picks[i].solution_value() > 0:
                        # Determine if it's a divisional game and if the picked team is the home team
    
                        week = df.loc[i, 'Week']
                        pick = df.loc[i,'Hypothetical Current Winner']
                        opponent = df.loc[i, 'Hypothetical Current Loser']
                        pick_percent = df.loc[i, 'Home Pick %'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Pick %']
                        expected_availability = df.loc[i, 'Home Team Expected Availability'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Expected Availability']
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
                        win_odds = df.loc[i, 'Home Team Fair Odds'] if df.loc[i, 'Hypothetical Current Winner'] == df.loc[i, 'Home Team 1'] else df.loc[i, 'Away Team Fair Odds']
                        
    
                        # Get differences
                        preseason_difference = df.loc[i, 'Hypthetical Winner Preseason Difference']
                        adjusted_preseason_difference = df.loc[i, 'Hypthetical Winner Adjusted Preseason Difference']
                        current_difference = df.loc[i, 'Hypthetical Winner Current Difference']
                        adjusted_current_difference = df.loc[i, 'Hypthetical Winner Adjusted Current Difference']
                        # Calculate EV for this game
                        ev = df.loc[i, 'Hypothetical Current Winner EV']
    
    
                        print('Week %i: Pick %s %s %s (%i, %i, %i, %i, %.4f)' % (df.loc[i, 'Week_Num'], df.loc[i, 'Hypothetical Current Winner'], divisional_game, home_team,
                                                                           preseason_difference, adjusted_preseason_difference,
                                                                           current_difference, adjusted_current_difference, ev))
                        new_row_2 = {
                            'Week': week,
                            'Pick': pick,
                            'Opponent': opponent,
                            'EV': ev,
                            'Win Odds': win_odds,
                            'Expected Pick %': pick_percent,
                            'Expected Availability': expected_availability,	
                            'Divisional Game': divisional_game,
                            'Home Team Status': home_team,
                            'Weekly Rest': weekly_rest,
                            'Weekly Rest Advantage': weekly_rest_advantage,
                            'Cumulative Rest': cumulative_rest,
                            'Cumulative Rest Advantage': cumulative_rest_advantage,
                            'Travel Advantage': travel_advantage,
                            'Back to Back Away Games': back_to_back_away_games,
                            'Thursday Night Game': thursday_night_game,
                            'International Game': international_game,
                            'Previous Opponent': previous_opponent,
                            'Previous Game Location': previous_game_location,
                            'Next Opponent': next_opponent,
                            'Next Game Location': next_game_location,
                            'Preseason Difference': preseason_difference,
                            'Adjusted Preseason Difference': adjusted_preseason_difference,
                            'Current Difference': current_difference,
                            'Adjusted Current Difference': adjusted_current_difference
                        }
                        picks_rows_2.append(new_row_2)
    
    
                        # Add differences to sums
                        sum_preseason_difference += preseason_difference
                        sum_adjusted_preseason_difference += adjusted_preseason_difference
                        sum_current_difference += current_difference
                        sum_adjusted_current_difference += adjusted_current_difference
                        picks_df = pd.concat([picks_df, df.loc[[i]]], ignore_index=True)
                        picks_df['Divisional Matchup?'] = divisional_game
                summarized_picks_df = pd.DataFrame(picks_rows_2)
    
                st.write(summarized_picks_df)
                st.write('')
                #st.write(f'Total EV: {sum_ev}')
                st.write('\nPreseason Difference:', sum_preseason_difference)
                st.write('Adjusted Preseason Difference:', sum_adjusted_preseason_difference)
                st.write('Current Difference:', sum_current_difference)
                st.write(f'Adjusted Current Difference: :blue[{sum_adjusted_current_difference}]')
                
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
            
            # Append the new forbidden solution to the list
            forbidden_solutions_1.append(picks_df['Adjusted Current Winner'].tolist())
            #print(forbidden_solutions)


picked_teams = []

default_az_rank = 1.5
default_atl_rank = -1
default_bal_rank = 7.5
default_buf_rank = 6.5
default_car_rank = -3.5
default_chi_rank = -1
default_cin_rank = 2
default_cle_rank = -6.5
default_dal_rank = -1.5
default_den_rank = .5
default_det_rank = 4
default_gb_rank = 4
default_hou_rank = -.5
default_ind_rank = -4
default_jax_rank = -5
default_kc_rank = 5
default_lv_rank = -2.5
default_lac_rank = 2
default_lar_rank = 2.5
default_mia_rank = -1
default_min_rank = 1.5
default_ne_rank = -3
default_no_rank = -6
default_nyg_rank = -4
default_nyj_rank = -2
default_phi_rank = 5.5
default_pit_rank = -.5
default_sf_rank = -1.5
default_sea_rank = -3
default_tb_rank = 3
default_ten_rank = -5
default_was_rank = 3.5

preseason_az_rank = 1.5
preseason_atl_rank = -1
preseason_bal_rank = 7.5
preseason_buf_rank = 6.5
preseason_car_rank = -3.5
preseason_chi_rank = -1
preseason_cin_rank = 2
preseason_cle_rank = -6.5
preseason_dal_rank = -1.5
preseason_den_rank = .5
preseason_det_rank = 4
preseason_gb_rank = 4
preseason_hou_rank = -.5
preseason_ind_rank = -4
preseason_jax_rank = -5
preseason_kc_rank = 5
preseason_lv_rank = -2.5
preseason_lac_rank = 2
preseason_lar_rank = 2.5
preseason_mia_rank = -1
preseason_min_rank = 1.5
preseason_ne_rank = -3
preseason_no_rank = -6
preseason_nyg_rank = -4
preseason_nyj_rank = -2
preseason_phi_rank = 5.5
preseason_pit_rank = -.5
preseason_sf_rank = -1.5
preseason_sea_rank = -3
preseason_tb_rank = 3
preseason_ten_rank = -5
preseason_was_rank = 3.5


az_home_adv = 1.5/2
atl_home_adv = 2.3/2
bal_home_adv = 3.8/2
buf_home_adv = 3.6/2
car_home_adv = 1.9/2
chi_home_adv = 1.5/2
cin_home_adv = 2.1/2
cle_home_adv = 1.3/2
dal_home_adv = 3.7/2
den_home_adv = 2.6/2
det_home_adv = 2.1/2
gb_home_adv = 3.8/2
hou_home_adv = 1.9/2
ind_home_adv = 2.6/2
jax_home_adv = 1.4/2
kc_home_adv = 3.8/2
lv_home_adv = 1.4/2
lac_home_adv = 2.6/2
lar_home_adv = 2.6/2
mia_home_adv = 2.3/2
min_home_adv = 3.1/2
ne_home_adv = 3.9/2
no_home_adv = 3.1/2
nyg_home_adv = 1.1/2
nyj_home_adv = 1.2/2
phi_home_adv = 3.3/2
pit_home_adv = 3.5/2
sf_home_adv = 3.6/2
sea_home_adv = 2.6/2
tb_home_adv = 2.0/2
ten_home_adv = 2.1/2
was_home_adv = 1.3/2


az_away_adj = -.3/2
atl_away_adj = .2/2
bal_away_adj = -1.5/2
buf_away_adj = -1.1/2
car_away_adj = .5/2
chi_away_adj = 1/2
cin_away_adj = -.2/2
cle_away_adj = 1.5/2
dal_away_adj = -1.2/2
den_away_adj = .6/2
det_away_adj = .7/2
gb_away_adj = -.1/2
hou_away_adj = .7/2
ind_away_adj = -.2/2
jax_away_adj = 1.6/2
kc_away_adj = -1.6/2
lv_away_adj = -.3/2
lac_away_adj = -.8/2
lar_away_adj = 1.3/2
mia_away_adj = 1.4/2
min_away_adj = -.5/2
ne_away_adj = -1.8/2
no_away_adj = -1.6/2
nyg_away_adj = .9/2
nyj_away_adj = 1.9/2
phi_away_adj = -.2/2
pit_away_adj = -.2/2
sf_away_adj = -1.1/2
sea_away_adj = -.4/2
tb_away_adj = -.1/2
ten_away_adj = .4/2
was_away_adj = .6/2

az_excluded_weeks = []
atl_excluded_weeks = []
bal_excluded_weeks = []
buf_excluded_weeks = []
car_excluded_weeks = []
chi_excluded_weeks = []
cin_excluded_weeks = []
cle_excluded_weeks = []
dal_excluded_weeks = []
den_excluded_weeks = []
det_excluded_weeks = []
gb_excluded_weeks = []
hou_excluded_weeks = []
ind_excluded_weeks = []
jax_excluded_weeks = []
kc_excluded_weeks = []
lv_excluded_weeks = []
lac_excluded_weeks = []
lar_excluded_weeks = []
mia_excluded_weeks = []
min_excluded_weeks = []
ne_excluded_weeks = []
no_excluded_weeks = []
nyg_excluded_weeks = []
nyj_excluded_weeks = []
phi_excluded_weeks = []
pit_excluded_weeks = []
sf_excluded_weeks = []
sea_excluded_weeks = []
tb_excluded_weeks = []
ten_excluded_weeks = []
was_excluded_weeks = []
az_req_week = 0
atl_req_week = 0
bal_req_week = 0
buf_req_week = 0
car_req_week = 0
chi_req_week = 0
cin_req_week = 0
cle_req_week = 0
dal_req_week = 0
den_req_week = 0
det_req_week = 0
gb_req_week = 0
hou_req_week = 0
ind_req_week = 0
jax_req_week = 0
kc_req_week = 0
lv_req_week = 0
lac_req_week = 0
lar_req_week = 0
mia_req_week = 0
min_req_week = 0
ne_req_week = 0
no_req_week = 0
nyg_req_week = 0
nyj_req_week = 0
phi_req_week = 0
pit_req_week = 0
sf_req_week = 0
sea_req_week = 0
tb_req_week = 0
ten_req_week = 0
was_req_week = 0
starting_week = 1
az_week_1_pick_percent = -1
atl_week_1_pick_percent = -1
bal_week_1_pick_percent = -1
buf_week_1_pick_percent = -1
car_week_1_pick_percent = -1
chi_week_1_pick_percent = -1
cin_week_1_pick_percent = -1
cle_week_1_pick_percent = -1
dal_week_1_pick_percent = -1
den_week_1_pick_percent = -1
det_week_1_pick_percent = -1
gb_week_1_pick_percent = -1
hou_week_1_pick_percent = -1
ind_week_1_pick_percent = -1
jax_week_1_pick_percent = -1
kc_week_1_pick_percent = -1
lv_week_1_pick_percent = -1
lac_week_1_pick_percent = -1
lar_week_1_pick_percent = -1
mia_week_1_pick_percent = -1
min_week_1_pick_percent = -1
ne_week_1_pick_percent = -1
no_week_1_pick_percent = -1
nyg_week_1_pick_percent = -1
nyj_week_1_pick_percent = -1
phi_week_1_pick_percent = -1
pit_week_1_pick_percent = -1
sf_week_1_pick_percent = -1
sea_week_1_pick_percent = -1
tb_week_1_pick_percent = -1
ten_week_1_pick_percent = -1
was_week_1_pick_percent = -1
az_week_2_pick_percent = -1
atl_week_2_pick_percent = -1
bal_week_2_pick_percent = -1
buf_week_2_pick_percent = -1
car_week_2_pick_percent = -1
chi_week_2_pick_percent = -1
cin_week_2_pick_percent = -1
cle_week_2_pick_percent = -1
dal_week_2_pick_percent = -1
den_week_2_pick_percent = -1
det_week_2_pick_percent = -1
gb_week_2_pick_percent = -1
hou_week_2_pick_percent = -1
ind_week_2_pick_percent = -1
jax_week_2_pick_percent = -1
kc_week_2_pick_percent = -1
lv_week_2_pick_percent = -1
lac_week_2_pick_percent = -1
lar_week_2_pick_percent = -1
mia_week_2_pick_percent = -1
min_week_2_pick_percent = -1
ne_week_2_pick_percent = -1
no_week_2_pick_percent = -1
nyg_week_2_pick_percent = -1
nyj_week_2_pick_percent = -1
phi_week_2_pick_percent = -1
pit_week_2_pick_percent = -1
sf_week_2_pick_percent = -1
sea_week_2_pick_percent = -1
tb_week_2_pick_percent = -1
ten_week_2_pick_percent = -1
was_week_2_pick_percent = -1
az_week_3_pick_percent = -1
atl_week_3_pick_percent = -1
bal_week_3_pick_percent = -1
buf_week_3_pick_percent = -1
car_week_3_pick_percent = -1
chi_week_3_pick_percent = -1
cin_week_3_pick_percent = -1
cle_week_3_pick_percent = -1
dal_week_3_pick_percent = -1
den_week_3_pick_percent = -1
det_week_3_pick_percent = -1
gb_week_3_pick_percent = -1
hou_week_3_pick_percent = -1
ind_week_3_pick_percent = -1
jax_week_3_pick_percent = -1
kc_week_3_pick_percent = -1
lv_week_3_pick_percent = -1
lac_week_3_pick_percent = -1
lar_week_3_pick_percent = -1
mia_week_3_pick_percent = -1
min_week_3_pick_percent = -1
ne_week_3_pick_percent = -1
no_week_3_pick_percent = -1
nyg_week_3_pick_percent = -1
nyj_week_3_pick_percent = -1
phi_week_3_pick_percent = -1
pit_week_3_pick_percent = -1
sf_week_3_pick_percent = -1
sea_week_3_pick_percent = -1
tb_week_3_pick_percent = -1
ten_week_3_pick_percent = -1
was_week_3_pick_percent = -1
az_week_4_pick_percent = -1
atl_week_4_pick_percent = -1
bal_week_4_pick_percent = -1
buf_week_4_pick_percent = -1
car_week_4_pick_percent = -1
chi_week_4_pick_percent = -1
cin_week_4_pick_percent = -1
cle_week_4_pick_percent = -1
dal_week_4_pick_percent = -1
den_week_4_pick_percent = -1
det_week_4_pick_percent = -1
gb_week_4_pick_percent = -1
hou_week_4_pick_percent = -1
ind_week_4_pick_percent = -1
jax_week_4_pick_percent = -1
kc_week_4_pick_percent = -1
lv_week_4_pick_percent = -1
lac_week_4_pick_percent = -1
lar_week_4_pick_percent = -1
mia_week_4_pick_percent = -1
min_week_4_pick_percent = -1
ne_week_4_pick_percent = -1
no_week_4_pick_percent = -1
nyg_week_4_pick_percent = -1
nyj_week_4_pick_percent = -1
phi_week_4_pick_percent = -1
pit_week_4_pick_percent = -1
sf_week_4_pick_percent = -1
sea_week_4_pick_percent = -1
tb_week_4_pick_percent = -1
ten_week_4_pick_percent = -1
was_week_4_pick_percent = -1
az_week_5_pick_percent = -1
atl_week_5_pick_percent = -1
bal_week_5_pick_percent = -1
buf_week_5_pick_percent = -1
car_week_5_pick_percent = -1
chi_week_5_pick_percent = -1
cin_week_5_pick_percent = -1
cle_week_5_pick_percent = -1
dal_week_5_pick_percent = -1
den_week_5_pick_percent = -1
det_week_5_pick_percent = -1
gb_week_5_pick_percent = -1
hou_week_5_pick_percent = -1
ind_week_5_pick_percent = -1
jax_week_5_pick_percent = -1
kc_week_5_pick_percent = -1
lv_week_5_pick_percent = -1
lac_week_5_pick_percent = -1
lar_week_5_pick_percent = -1
mia_week_5_pick_percent = -1
min_week_5_pick_percent = -1
ne_week_5_pick_percent = -1
no_week_5_pick_percent = -1
nyg_week_5_pick_percent = -1
nyj_week_5_pick_percent = -1
phi_week_5_pick_percent = -1
pit_week_5_pick_percent = -1
sf_week_5_pick_percent = -1
sea_week_5_pick_percent = -1
tb_week_5_pick_percent = -1
ten_week_5_pick_percent = -1
was_week_5_pick_percent = -1
az_week_6_pick_percent = -1
atl_week_6_pick_percent = -1
bal_week_6_pick_percent = -1
buf_week_6_pick_percent = -1
car_week_6_pick_percent = -1
chi_week_6_pick_percent = -1
cin_week_6_pick_percent = -1
cle_week_6_pick_percent = -1
dal_week_6_pick_percent = -1
den_week_6_pick_percent = -1
det_week_6_pick_percent = -1
gb_week_6_pick_percent = -1
hou_week_6_pick_percent = -1
ind_week_6_pick_percent = -1
jax_week_6_pick_percent = -1
kc_week_6_pick_percent = -1
lv_week_6_pick_percent = -1
lac_week_6_pick_percent = -1
lar_week_6_pick_percent = -1
mia_week_6_pick_percent = -1
min_week_6_pick_percent = -1
ne_week_6_pick_percent = -1
no_week_6_pick_percent = -1
nyg_week_6_pick_percent = -1
nyj_week_6_pick_percent = -1
phi_week_6_pick_percent = -1
pit_week_6_pick_percent = -1
sf_week_6_pick_percent = -1
sea_week_6_pick_percent = -1
tb_week_6_pick_percent = -1
ten_week_6_pick_percent = -1
was_week_6_pick_percent = -1
az_week_7_pick_percent = -1
atl_week_7_pick_percent = -1
bal_week_7_pick_percent = -1
buf_week_7_pick_percent = -1
car_week_7_pick_percent = -1
chi_week_7_pick_percent = -1
cin_week_7_pick_percent = -1
cle_week_7_pick_percent = -1
dal_week_7_pick_percent = -1
den_week_7_pick_percent = -1
det_week_7_pick_percent = -1
gb_week_7_pick_percent = -1
hou_week_7_pick_percent = -1
ind_week_7_pick_percent = -1
jax_week_7_pick_percent = -1
kc_week_7_pick_percent = -1
lv_week_7_pick_percent = -1
lac_week_7_pick_percent = -1
lar_week_7_pick_percent = -1
mia_week_7_pick_percent = -1
min_week_7_pick_percent = -1
ne_week_7_pick_percent = -1
no_week_7_pick_percent = -1
nyg_week_7_pick_percent = -1
nyj_week_7_pick_percent = -1
phi_week_7_pick_percent = -1
pit_week_7_pick_percent = -1
sf_week_7_pick_percent = -1
sea_week_7_pick_percent = -1
tb_week_7_pick_percent = -1
ten_week_7_pick_percent = -1
was_week_7_pick_percent = -1
az_week_8_pick_percent = -1
atl_week_8_pick_percent = -1
bal_week_8_pick_percent = -1
buf_week_8_pick_percent = -1
car_week_8_pick_percent = -1
chi_week_8_pick_percent = -1
cin_week_8_pick_percent = -1
cle_week_8_pick_percent = -1
dal_week_8_pick_percent = -1
den_week_8_pick_percent = -1
det_week_8_pick_percent = -1
gb_week_8_pick_percent = -1
hou_week_8_pick_percent = -1
ind_week_8_pick_percent = -1
jax_week_8_pick_percent = -1
kc_week_8_pick_percent = -1
lv_week_8_pick_percent = -1
lac_week_8_pick_percent = -1
lar_week_8_pick_percent = -1
mia_week_8_pick_percent = -1
min_week_8_pick_percent = -1
ne_week_8_pick_percent = -1
no_week_8_pick_percent = -1
nyg_week_8_pick_percent = -1
nyj_week_8_pick_percent = -1
phi_week_8_pick_percent = -1
pit_week_8_pick_percent = -1
sf_week_8_pick_percent = -1
sea_week_8_pick_percent = -1
tb_week_8_pick_percent = -1
ten_week_8_pick_percent = -1
was_week_8_pick_percent = -1
az_week_9_pick_percent = -1
atl_week_9_pick_percent = -1
bal_week_9_pick_percent = -1
buf_week_9_pick_percent = -1
car_week_9_pick_percent = -1
chi_week_9_pick_percent = -1
cin_week_9_pick_percent = -1
cle_week_9_pick_percent = -1
dal_week_9_pick_percent = -1
den_week_9_pick_percent = -1
det_week_9_pick_percent = -1
gb_week_9_pick_percent = -1
hou_week_9_pick_percent = -1
ind_week_9_pick_percent = -1
jax_week_9_pick_percent = -1
kc_week_9_pick_percent = -1
lv_week_9_pick_percent = -1
lac_week_9_pick_percent = -1
lar_week_9_pick_percent = -1
mia_week_9_pick_percent = -1
min_week_9_pick_percent = -1
ne_week_9_pick_percent = -1
no_week_9_pick_percent = -1
nyg_week_9_pick_percent = -1
nyj_week_9_pick_percent = -1
phi_week_9_pick_percent = -1
pit_week_9_pick_percent = -1
sf_week_9_pick_percent = -1
sea_week_9_pick_percent = -1
tb_week_9_pick_percent = -1
ten_week_9_pick_percent = -1
was_week_9_pick_percent = -1
az_week_10_pick_percent = -1
atl_week_10_pick_percent = -1
bal_week_10_pick_percent = -1
buf_week_10_pick_percent = -1
car_week_10_pick_percent = -1
chi_week_10_pick_percent = -1
cin_week_10_pick_percent = -1
cle_week_10_pick_percent = -1
dal_week_10_pick_percent = -1
den_week_10_pick_percent = -1
det_week_10_pick_percent = -1
gb_week_10_pick_percent = -1
hou_week_10_pick_percent = -1
ind_week_10_pick_percent = -1
jax_week_10_pick_percent = -1
kc_week_10_pick_percent = -1
lv_week_10_pick_percent = -1
lac_week_10_pick_percent = -1
lar_week_10_pick_percent = -1
mia_week_10_pick_percent = -1
min_week_10_pick_percent = -1
ne_week_10_pick_percent = -1
no_week_10_pick_percent = -1
nyg_week_10_pick_percent = -1
nyj_week_10_pick_percent = -1
phi_week_10_pick_percent = -1
pit_week_10_pick_percent = -1
sf_week_10_pick_percent = -1
sea_week_10_pick_percent = -1
tb_week_10_pick_percent = -1
ten_week_10_pick_percent = -1
was_week_10_pick_percent = -1
az_week_11_pick_percent = -1
atl_week_11_pick_percent = -1
bal_week_11_pick_percent = -1
buf_week_11_pick_percent = -1
car_week_11_pick_percent = -1
chi_week_11_pick_percent = -1
cin_week_11_pick_percent = -1
cle_week_11_pick_percent = -1
dal_week_11_pick_percent = -1
den_week_11_pick_percent = -1
det_week_11_pick_percent = -1
gb_week_11_pick_percent = -1
hou_week_11_pick_percent = -1
ind_week_11_pick_percent = -1
jax_week_11_pick_percent = -1
kc_week_11_pick_percent = -1
lv_week_11_pick_percent = -1
lac_week_11_pick_percent = -1
lar_week_11_pick_percent = -1
mia_week_11_pick_percent = -1
min_week_11_pick_percent = -1
ne_week_11_pick_percent = -1
no_week_11_pick_percent = -1
nyg_week_11_pick_percent = -1
nyj_week_11_pick_percent = -1
phi_week_11_pick_percent = -1
pit_week_11_pick_percent = -1
sf_week_11_pick_percent = -1
sea_week_11_pick_percent = -1
tb_week_11_pick_percent = -1
ten_week_11_pick_percent = -1
was_week_11_pick_percent = -1
az_week_12_pick_percent = -1
atl_week_12_pick_percent = -1
bal_week_12_pick_percent = -1
buf_week_12_pick_percent = -1
car_week_12_pick_percent = -1
chi_week_12_pick_percent = -1
cin_week_12_pick_percent = -1
cle_week_12_pick_percent = -1
dal_week_12_pick_percent = -1
den_week_12_pick_percent = -1
det_week_12_pick_percent = -1
gb_week_12_pick_percent = -1
hou_week_12_pick_percent = -1
ind_week_12_pick_percent = -1
jax_week_12_pick_percent = -1
kc_week_12_pick_percent = -1
lv_week_12_pick_percent = -1
lac_week_12_pick_percent = -1
lar_week_12_pick_percent = -1
mia_week_12_pick_percent = -1
min_week_12_pick_percent = -1
ne_week_12_pick_percent = -1
no_week_12_pick_percent = -1
nyg_week_12_pick_percent = -1
nyj_week_12_pick_percent = -1
phi_week_12_pick_percent = -1
pit_week_12_pick_percent = -1
sf_week_12_pick_percent = -1
sea_week_12_pick_percent = -1
tb_week_12_pick_percent = -1
ten_week_12_pick_percent = -1
was_week_12_pick_percent = -1
az_week_13_pick_percent = -1
atl_week_13_pick_percent = -1
bal_week_13_pick_percent = -1
buf_week_13_pick_percent = -1
car_week_13_pick_percent = -1
chi_week_13_pick_percent = -1
cin_week_13_pick_percent = -1
cle_week_13_pick_percent = -1
dal_week_13_pick_percent = -1
den_week_13_pick_percent = -1
det_week_13_pick_percent = -1
gb_week_13_pick_percent = -1
hou_week_13_pick_percent = -1
ind_week_13_pick_percent = -1
jax_week_13_pick_percent = -1
kc_week_13_pick_percent = -1
lv_week_13_pick_percent = -1
lac_week_13_pick_percent = -1
lar_week_13_pick_percent = -1
mia_week_13_pick_percent = -1
min_week_13_pick_percent = -1
ne_week_13_pick_percent = -1
no_week_13_pick_percent = -1
nyg_week_13_pick_percent = -1
nyj_week_13_pick_percent = -1
phi_week_13_pick_percent = -1
pit_week_13_pick_percent = -1
sf_week_13_pick_percent = -1
sea_week_13_pick_percent = -1
tb_week_13_pick_percent = -1
ten_week_13_pick_percent = -1
was_week_13_pick_percent = -1
az_week_14_pick_percent = -1
atl_week_14_pick_percent = -1
bal_week_14_pick_percent = -1
buf_week_14_pick_percent = -1
car_week_14_pick_percent = -1
chi_week_14_pick_percent = -1
cin_week_14_pick_percent = -1
cle_week_14_pick_percent = -1
dal_week_14_pick_percent = -1
den_week_14_pick_percent = -1
det_week_14_pick_percent = -1
gb_week_14_pick_percent = -1
hou_week_14_pick_percent = -1
ind_week_14_pick_percent = -1
jax_week_14_pick_percent = -1
kc_week_14_pick_percent = -1
lv_week_14_pick_percent = -1
lac_week_14_pick_percent = -1
lar_week_14_pick_percent = -1
mia_week_14_pick_percent = -1
min_week_14_pick_percent = -1
ne_week_14_pick_percent = -1
no_week_14_pick_percent = -1
nyg_week_14_pick_percent = -1
nyj_week_14_pick_percent = -1
phi_week_14_pick_percent = -1
pit_week_14_pick_percent = -1
sf_week_14_pick_percent = -1
sea_week_14_pick_percent = -1
tb_week_14_pick_percent = -1
ten_week_14_pick_percent = -1
was_week_14_pick_percent = -1
az_week_15_pick_percent = -1
atl_week_15_pick_percent = -1
bal_week_15_pick_percent = -1
buf_week_15_pick_percent = -1
car_week_15_pick_percent = -1
chi_week_15_pick_percent = -1
cin_week_15_pick_percent = -1
cle_week_15_pick_percent = -1
dal_week_15_pick_percent = -1
den_week_15_pick_percent = -1
det_week_15_pick_percent = -1
gb_week_15_pick_percent = -1
hou_week_15_pick_percent = -1
ind_week_15_pick_percent = -1
jax_week_15_pick_percent = -1
kc_week_15_pick_percent = -1
lv_week_15_pick_percent = -1
lac_week_15_pick_percent = -1
lar_week_15_pick_percent = -1
mia_week_15_pick_percent = -1
min_week_15_pick_percent = -1
ne_week_15_pick_percent = -1
no_week_15_pick_percent = -1
nyg_week_15_pick_percent = -1
nyj_week_15_pick_percent = -1
phi_week_15_pick_percent = -1
pit_week_15_pick_percent = -1
sf_week_15_pick_percent = -1
sea_week_15_pick_percent = -1
tb_week_15_pick_percent = -1
ten_week_15_pick_percent = -1
was_week_15_pick_percent = -1
az_week_16_pick_percent = -1
atl_week_16_pick_percent = -1
bal_week_16_pick_percent = -1
buf_week_16_pick_percent = -1
car_week_16_pick_percent = -1
chi_week_16_pick_percent = -1
cin_week_16_pick_percent = -1
cle_week_16_pick_percent = -1
dal_week_16_pick_percent = -1
den_week_16_pick_percent = -1
det_week_16_pick_percent = -1
gb_week_16_pick_percent = -1
hou_week_16_pick_percent = -1
ind_week_16_pick_percent = -1
jax_week_16_pick_percent = -1
kc_week_16_pick_percent = -1
lv_week_16_pick_percent = -1
lac_week_16_pick_percent = -1
lar_week_16_pick_percent = -1
mia_week_16_pick_percent = -1
min_week_16_pick_percent = -1
ne_week_16_pick_percent = -1
no_week_16_pick_percent = -1
nyg_week_16_pick_percent = -1
nyj_week_16_pick_percent = -1
phi_week_16_pick_percent = -1
pit_week_16_pick_percent = -1
sf_week_16_pick_percent = -1
sea_week_16_pick_percent = -1
tb_week_16_pick_percent = -1
ten_week_16_pick_percent = -1
was_week_16_pick_percent = -1
az_week_17_pick_percent = -1
atl_week_17_pick_percent = -1
bal_week_17_pick_percent = -1
buf_week_17_pick_percent = -1
car_week_17_pick_percent = -1
chi_week_17_pick_percent = -1
cin_week_17_pick_percent = -1
cle_week_17_pick_percent = -1
dal_week_17_pick_percent = -1
den_week_17_pick_percent = -1
det_week_17_pick_percent = -1
gb_week_17_pick_percent = -1
hou_week_17_pick_percent = -1
ind_week_17_pick_percent = -1
jax_week_17_pick_percent = -1
kc_week_17_pick_percent = -1
lv_week_17_pick_percent = -1
lac_week_17_pick_percent = -1
lar_week_17_pick_percent = -1
mia_week_17_pick_percent = -1
min_week_17_pick_percent = -1
ne_week_17_pick_percent = -1
no_week_17_pick_percent = -1
nyg_week_17_pick_percent = -1
nyj_week_17_pick_percent = -1
phi_week_17_pick_percent = -1
pit_week_17_pick_percent = -1
sf_week_17_pick_percent = -1
sea_week_17_pick_percent = -1
tb_week_17_pick_percent = -1
ten_week_17_pick_percent = -1
was_week_17_pick_percent = -1
az_week_18_pick_percent = -1
atl_week_18_pick_percent = -1
bal_week_18_pick_percent = -1
buf_week_18_pick_percent = -1
car_week_18_pick_percent = -1
chi_week_18_pick_percent = -1
cin_week_18_pick_percent = -1
cle_week_18_pick_percent = -1
dal_week_18_pick_percent = -1
den_week_18_pick_percent = -1
det_week_18_pick_percent = -1
gb_week_18_pick_percent = -1
hou_week_18_pick_percent = -1
ind_week_18_pick_percent = -1
jax_week_18_pick_percent = -1
kc_week_18_pick_percent = -1
lv_week_18_pick_percent = -1
lac_week_18_pick_percent = -1
lar_week_18_pick_percent = -1
mia_week_18_pick_percent = -1
min_week_18_pick_percent = -1
ne_week_18_pick_percent = -1
no_week_18_pick_percent = -1
nyg_week_18_pick_percent = -1
nyj_week_18_pick_percent = -1
phi_week_18_pick_percent = -1
pit_week_18_pick_percent = -1
sf_week_18_pick_percent = -1
sea_week_18_pick_percent = -1
tb_week_18_pick_percent = -1
ten_week_18_pick_percent = -1
was_week_18_pick_percent = -1
az_week_19_pick_percent = -1
atl_week_19_pick_percent = -1
bal_week_19_pick_percent = -1
buf_week_19_pick_percent = -1
car_week_19_pick_percent = -1
chi_week_19_pick_percent = -1
cin_week_19_pick_percent = -1
cle_week_19_pick_percent = -1
dal_week_19_pick_percent = -1
den_week_19_pick_percent = -1
det_week_19_pick_percent = -1
gb_week_19_pick_percent = -1
hou_week_19_pick_percent = -1
ind_week_19_pick_percent = -1
jax_week_19_pick_percent = -1
kc_week_19_pick_percent = -1
lv_week_19_pick_percent = -1
lac_week_19_pick_percent = -1
lar_week_19_pick_percent = -1
mia_week_19_pick_percent = -1
min_week_19_pick_percent = -1
ne_week_19_pick_percent = -1
no_week_19_pick_percent = -1
nyg_week_19_pick_percent = -1
nyj_week_19_pick_percent = -1
phi_week_19_pick_percent = -1
pit_week_19_pick_percent = -1
sf_week_19_pick_percent = -1
sea_week_19_pick_percent = -1
tb_week_19_pick_percent = -1
ten_week_19_pick_percent = -1
was_week_19_pick_percent = -1
az_week_20_pick_percent = -1
atl_week_20_pick_percent = -1
bal_week_20_pick_percent = -1
buf_week_20_pick_percent = -1
car_week_20_pick_percent = -1
chi_week_20_pick_percent = -1
cin_week_20_pick_percent = -1
cle_week_20_pick_percent = -1
dal_week_20_pick_percent = -1
den_week_20_pick_percent = -1
det_week_20_pick_percent = -1
gb_week_20_pick_percent = -1
hou_week_20_pick_percent = -1
ind_week_20_pick_percent = -1
jax_week_20_pick_percent = -1
kc_week_20_pick_percent = -1
lv_week_20_pick_percent = -1
lac_week_20_pick_percent = -1
lar_week_20_pick_percent = -1
mia_week_20_pick_percent = -1
min_week_20_pick_percent = -1
ne_week_20_pick_percent = -1
no_week_20_pick_percent = -1
nyg_week_20_pick_percent = -1
nyj_week_20_pick_percent = -1
phi_week_20_pick_percent = -1
pit_week_20_pick_percent = -1
sf_week_20_pick_percent = -1
sea_week_20_pick_percent = -1
tb_week_20_pick_percent = -1
ten_week_20_pick_percent = -1
was_week_20_pick_percent = -1
az_rank = default_az_rank
atl_rank = default_atl_rank
bal_rank = default_bal_rank
buf_rank = default_buf_rank
car_rank = default_car_rank
chi_rank = default_chi_rank
cin_rank = default_cin_rank
cle_rank = default_cle_rank
dal_rank = default_dal_rank
den_rank = default_den_rank
det_rank = default_det_rank
gb_rank = default_gb_rank
hou_rank = default_hou_rank
ind_rank = default_ind_rank
jax_rank = default_jax_rank
kc_rank = default_kc_rank
lv_rank = default_lv_rank
lac_rank = default_lac_rank
lar_rank = default_lar_rank
mia_rank = default_mia_rank
min_rank = default_min_rank
ne_rank = default_ne_rank
no_rank = default_no_rank
nyg_rank = default_nyg_rank
nyj_rank = default_nyj_rank
phi_rank = default_phi_rank
pit_rank = default_pit_rank
sf_rank = default_sf_rank
sea_rank = default_sea_rank
tb_rank = default_tb_rank
ten_rank = default_ten_rank
was_rank = default_was_rank
avoid_away_teams_on_short_rest = 0
avoid_close_divisional_matchups = 0
avoid_3_games_in_10_days = 0
avoid_4_games_in_17_days = 0
avoid_away_teams_in_close_matchups = 0
avoid_cumulative_rest_disadvantage = 0
avoid_thursday_night = 0
avoid_away_thursday_night = 0
avoid_back_to_back_away = 0
avoid_international_game = 0
avoid_teams_with_weekly_rest_disadvantage = 0
avoid_away_teams_with_travel_disadvantage = 0
bayesian_rest_travel_constraint = "No Rest, Bayesian, and Travel Constraints"
circa_total_entries = 14266
dk_total_entries = 20000
splash_total_entries = 18519
number_solutions = 5
selected_contest = 'Circa'
starting_week = 1
if selected_contest == 'Circa':
    ending_week = 21
else:
    ending_week = 19

dk_pick_percentages = {
    'Arizona Cardinals': [az_week_1_pick_percent, az_week_2_pick_percent, az_week_3_pick_percent, az_week_4_pick_percent, az_week_5_pick_percent, az_week_6_pick_percent, az_week_7_pick_percent, az_week_8_pick_percent, az_week_9_pick_percent, az_week_10_pick_percent, az_week_11_pick_percent, az_week_12_pick_percent, az_week_13_pick_percent, az_week_14_pick_percent, az_week_15_pick_percent, az_week_16_pick_percent, az_week_17_pick_percent, az_week_18_pick_percent],
    'Atlanta Falcons': [atl_week_1_pick_percent, atl_week_2_pick_percent, atl_week_3_pick_percent, atl_week_4_pick_percent, atl_week_5_pick_percent, atl_week_6_pick_percent, atl_week_7_pick_percent, atl_week_8_pick_percent, atl_week_9_pick_percent, atl_week_10_pick_percent, atl_week_11_pick_percent, atl_week_12_pick_percent, atl_week_13_pick_percent, atl_week_14_pick_percent, atl_week_15_pick_percent, atl_week_16_pick_percent, atl_week_17_pick_percent, atl_week_18_pick_percent],
    'Baltimore Ravens': [bal_week_1_pick_percent, bal_week_2_pick_percent, bal_week_3_pick_percent, bal_week_4_pick_percent, bal_week_5_pick_percent, bal_week_6_pick_percent, bal_week_7_pick_percent, bal_week_8_pick_percent, bal_week_9_pick_percent, bal_week_10_pick_percent, bal_week_11_pick_percent, bal_week_12_pick_percent, bal_week_13_pick_percent, bal_week_14_pick_percent, bal_week_15_pick_percent, bal_week_16_pick_percent, bal_week_17_pick_percent, bal_week_18_pick_percent],
    'Buffalo Bills': [buf_week_1_pick_percent, buf_week_2_pick_percent, buf_week_3_pick_percent, buf_week_4_pick_percent, buf_week_5_pick_percent, buf_week_6_pick_percent, buf_week_7_pick_percent, buf_week_8_pick_percent, buf_week_9_pick_percent, buf_week_10_pick_percent, buf_week_11_pick_percent, buf_week_12_pick_percent, buf_week_13_pick_percent, buf_week_14_pick_percent, buf_week_15_pick_percent, buf_week_16_pick_percent, buf_week_17_pick_percent, buf_week_18_pick_percent],
    'Carolina Panthers': [car_week_1_pick_percent, car_week_2_pick_percent, car_week_3_pick_percent, car_week_4_pick_percent, car_week_5_pick_percent, car_week_6_pick_percent, car_week_7_pick_percent, car_week_8_pick_percent, car_week_9_pick_percent, car_week_10_pick_percent, car_week_11_pick_percent, car_week_12_pick_percent, car_week_13_pick_percent, car_week_14_pick_percent, car_week_15_pick_percent, car_week_16_pick_percent, car_week_17_pick_percent, car_week_18_pick_percent],
    'Chicago Bears': [chi_week_1_pick_percent, chi_week_2_pick_percent, chi_week_3_pick_percent, chi_week_4_pick_percent, chi_week_5_pick_percent, chi_week_6_pick_percent, chi_week_7_pick_percent, chi_week_8_pick_percent, chi_week_9_pick_percent, chi_week_10_pick_percent, chi_week_11_pick_percent, chi_week_12_pick_percent, chi_week_13_pick_percent, chi_week_14_pick_percent, chi_week_15_pick_percent, chi_week_16_pick_percent, chi_week_17_pick_percent, chi_week_18_pick_percent],
    'Cincinnati Bengals': [cin_week_1_pick_percent, cin_week_2_pick_percent, cin_week_3_pick_percent, cin_week_4_pick_percent, cin_week_5_pick_percent, cin_week_6_pick_percent, cin_week_7_pick_percent, cin_week_8_pick_percent, cin_week_9_pick_percent, cin_week_10_pick_percent, cin_week_11_pick_percent, cin_week_12_pick_percent, cin_week_13_pick_percent, cin_week_14_pick_percent, cin_week_15_pick_percent, cin_week_16_pick_percent, cin_week_17_pick_percent, cin_week_18_pick_percent],
    'Cleveland Browns': [cle_week_1_pick_percent, cle_week_2_pick_percent, cle_week_3_pick_percent, cle_week_4_pick_percent, cle_week_5_pick_percent, cle_week_6_pick_percent, cle_week_7_pick_percent, cle_week_8_pick_percent, cle_week_9_pick_percent, cle_week_10_pick_percent, cle_week_11_pick_percent, cle_week_12_pick_percent, cle_week_13_pick_percent, cle_week_14_pick_percent, cle_week_15_pick_percent, cle_week_16_pick_percent, cle_week_17_pick_percent, cle_week_18_pick_percent],
    'Dallas Cowboys': [dal_week_1_pick_percent, dal_week_2_pick_percent, dal_week_3_pick_percent, dal_week_4_pick_percent, dal_week_5_pick_percent, dal_week_6_pick_percent, dal_week_7_pick_percent, dal_week_8_pick_percent, dal_week_9_pick_percent, dal_week_10_pick_percent, dal_week_11_pick_percent, dal_week_12_pick_percent, dal_week_13_pick_percent, dal_week_14_pick_percent, dal_week_15_pick_percent, dal_week_16_pick_percent, dal_week_17_pick_percent, dal_week_18_pick_percent],
    'Denver Broncos': [den_week_1_pick_percent, den_week_2_pick_percent, den_week_3_pick_percent, den_week_4_pick_percent, den_week_5_pick_percent, den_week_6_pick_percent, den_week_7_pick_percent, den_week_8_pick_percent, den_week_9_pick_percent, den_week_10_pick_percent, den_week_11_pick_percent, den_week_12_pick_percent, den_week_13_pick_percent, den_week_14_pick_percent, den_week_15_pick_percent, den_week_16_pick_percent, den_week_17_pick_percent, den_week_18_pick_percent],
    'Detroit Lions': [det_week_1_pick_percent, det_week_2_pick_percent, det_week_3_pick_percent, det_week_4_pick_percent, det_week_5_pick_percent, det_week_6_pick_percent, det_week_7_pick_percent, det_week_8_pick_percent, det_week_9_pick_percent, det_week_10_pick_percent, det_week_11_pick_percent, det_week_12_pick_percent, det_week_13_pick_percent, det_week_14_pick_percent, det_week_15_pick_percent, det_week_16_pick_percent, det_week_17_pick_percent, det_week_18_pick_percent],
    'Green Bay Packers': [gb_week_1_pick_percent, gb_week_2_pick_percent, gb_week_3_pick_percent, gb_week_4_pick_percent, gb_week_5_pick_percent, gb_week_6_pick_percent, gb_week_7_pick_percent, gb_week_8_pick_percent, gb_week_9_pick_percent, gb_week_10_pick_percent, gb_week_11_pick_percent, gb_week_12_pick_percent, gb_week_13_pick_percent, gb_week_14_pick_percent, gb_week_15_pick_percent, gb_week_16_pick_percent, gb_week_17_pick_percent, gb_week_18_pick_percent],
    'Houston Texans': [hou_week_1_pick_percent, hou_week_2_pick_percent, hou_week_3_pick_percent, hou_week_4_pick_percent, hou_week_5_pick_percent, hou_week_6_pick_percent, hou_week_7_pick_percent, hou_week_8_pick_percent, hou_week_9_pick_percent, hou_week_10_pick_percent, hou_week_11_pick_percent, hou_week_12_pick_percent, hou_week_13_pick_percent, hou_week_14_pick_percent, hou_week_15_pick_percent, hou_week_16_pick_percent, hou_week_17_pick_percent, hou_week_18_pick_percent],
    'Indianapolis Colts': [ind_week_1_pick_percent, ind_week_2_pick_percent, ind_week_3_pick_percent, ind_week_4_pick_percent, ind_week_5_pick_percent, ind_week_6_pick_percent, ind_week_7_pick_percent, ind_week_8_pick_percent, ind_week_9_pick_percent, ind_week_10_pick_percent, ind_week_11_pick_percent, ind_week_12_pick_percent, ind_week_13_pick_percent, ind_week_14_pick_percent, ind_week_15_pick_percent, ind_week_16_pick_percent, ind_week_17_pick_percent, ind_week_18_pick_percent],
    'Jacksonville Jaguars': [jax_week_1_pick_percent, jax_week_2_pick_percent, jax_week_3_pick_percent, jax_week_4_pick_percent, jax_week_5_pick_percent, jax_week_6_pick_percent, jax_week_7_pick_percent, jax_week_8_pick_percent, jax_week_9_pick_percent, jax_week_10_pick_percent, jax_week_11_pick_percent, jax_week_12_pick_percent, jax_week_13_pick_percent, jax_week_14_pick_percent, jax_week_15_pick_percent, jax_week_16_pick_percent, jax_week_17_pick_percent, jax_week_18_pick_percent],
    'Kansas City Chiefs': [kc_week_1_pick_percent, kc_week_2_pick_percent, kc_week_3_pick_percent, kc_week_4_pick_percent, kc_week_5_pick_percent, kc_week_6_pick_percent, kc_week_7_pick_percent, kc_week_8_pick_percent, kc_week_9_pick_percent, kc_week_10_pick_percent, kc_week_11_pick_percent, kc_week_12_pick_percent, kc_week_13_pick_percent, kc_week_14_pick_percent, kc_week_15_pick_percent, kc_week_16_pick_percent, kc_week_17_pick_percent, kc_week_18_pick_percent],
    'Las Vegas Raiders': [lv_week_1_pick_percent, lv_week_2_pick_percent, lv_week_3_pick_percent, lv_week_4_pick_percent, lv_week_5_pick_percent, lv_week_6_pick_percent, lv_week_7_pick_percent, lv_week_8_pick_percent, lv_week_9_pick_percent, lv_week_10_pick_percent, lv_week_11_pick_percent, lv_week_12_pick_percent, lv_week_13_pick_percent, lv_week_14_pick_percent, lv_week_15_pick_percent, lv_week_16_pick_percent, lv_week_17_pick_percent, lv_week_18_pick_percent],
    'Los Angeles Chargers': [lac_week_1_pick_percent, lac_week_2_pick_percent, lac_week_3_pick_percent, lac_week_4_pick_percent, lac_week_5_pick_percent, lac_week_6_pick_percent, lac_week_7_pick_percent, lac_week_8_pick_percent, lac_week_9_pick_percent, lac_week_10_pick_percent, lac_week_11_pick_percent, lac_week_12_pick_percent, lac_week_13_pick_percent, lac_week_14_pick_percent, lac_week_15_pick_percent, lac_week_16_pick_percent, lac_week_17_pick_percent, lac_week_18_pick_percent],
    'Los Angeles Rams': [lar_week_1_pick_percent, lar_week_2_pick_percent, lar_week_3_pick_percent, lar_week_4_pick_percent, lar_week_5_pick_percent, lar_week_6_pick_percent, lar_week_7_pick_percent, lar_week_8_pick_percent, lar_week_9_pick_percent, lar_week_10_pick_percent, lar_week_11_pick_percent, lar_week_12_pick_percent, lar_week_13_pick_percent, lar_week_14_pick_percent, lar_week_15_pick_percent, lar_week_16_pick_percent, lar_week_17_pick_percent, lar_week_18_pick_percent],
    'Miami Dolphins': [mia_week_1_pick_percent, mia_week_2_pick_percent, mia_week_3_pick_percent, mia_week_4_pick_percent, mia_week_5_pick_percent, mia_week_6_pick_percent, mia_week_7_pick_percent, mia_week_8_pick_percent, mia_week_9_pick_percent, mia_week_10_pick_percent, mia_week_11_pick_percent, mia_week_12_pick_percent, mia_week_13_pick_percent, mia_week_14_pick_percent, mia_week_15_pick_percent, mia_week_16_pick_percent, mia_week_17_pick_percent, mia_week_18_pick_percent],
    'Minnesota Vikings': [min_week_1_pick_percent, min_week_2_pick_percent, min_week_3_pick_percent, min_week_4_pick_percent, min_week_5_pick_percent, min_week_6_pick_percent, min_week_7_pick_percent, min_week_8_pick_percent, min_week_9_pick_percent, min_week_10_pick_percent, min_week_11_pick_percent, min_week_12_pick_percent, min_week_13_pick_percent, min_week_14_pick_percent, min_week_15_pick_percent, min_week_16_pick_percent, min_week_17_pick_percent, min_week_18_pick_percent],
    'New England Patriots': [ne_week_1_pick_percent, ne_week_2_pick_percent, ne_week_3_pick_percent, ne_week_4_pick_percent, ne_week_5_pick_percent, ne_week_6_pick_percent, ne_week_7_pick_percent, ne_week_8_pick_percent, ne_week_9_pick_percent, ne_week_10_pick_percent, ne_week_11_pick_percent, ne_week_12_pick_percent, ne_week_13_pick_percent, ne_week_14_pick_percent, ne_week_15_pick_percent, ne_week_16_pick_percent, ne_week_17_pick_percent, ne_week_18_pick_percent],
    'New Orleans Saints': [no_week_1_pick_percent, no_week_2_pick_percent, no_week_3_pick_percent, no_week_4_pick_percent, no_week_5_pick_percent, no_week_6_pick_percent, no_week_7_pick_percent, no_week_8_pick_percent, no_week_9_pick_percent, no_week_10_pick_percent, no_week_11_pick_percent, no_week_12_pick_percent, no_week_13_pick_percent, no_week_14_pick_percent, no_week_15_pick_percent, no_week_16_pick_percent, no_week_17_pick_percent, no_week_18_pick_percent],
    'New York Giants': [nyg_week_1_pick_percent, nyg_week_2_pick_percent, nyg_week_3_pick_percent, nyg_week_4_pick_percent, nyg_week_5_pick_percent, nyg_week_6_pick_percent, nyg_week_7_pick_percent, nyg_week_8_pick_percent, nyg_week_9_pick_percent, nyg_week_10_pick_percent, nyg_week_11_pick_percent, nyg_week_12_pick_percent, nyg_week_13_pick_percent, nyg_week_14_pick_percent, nyg_week_15_pick_percent, nyg_week_16_pick_percent, nyg_week_17_pick_percent, nyg_week_18_pick_percent],
    'New York Jets': [nyj_week_1_pick_percent, nyj_week_2_pick_percent, nyj_week_3_pick_percent, nyj_week_4_pick_percent, nyj_week_5_pick_percent, nyj_week_6_pick_percent, nyj_week_7_pick_percent, nyj_week_8_pick_percent, nyj_week_9_pick_percent, nyj_week_10_pick_percent, nyj_week_11_pick_percent, nyj_week_12_pick_percent, nyj_week_13_pick_percent, nyj_week_14_pick_percent, nyj_week_15_pick_percent, nyj_week_16_pick_percent, nyj_week_17_pick_percent, nyj_week_18_pick_percent],
    'Philadelphia Eagles': [phi_week_1_pick_percent, phi_week_2_pick_percent, phi_week_3_pick_percent, phi_week_4_pick_percent, phi_week_5_pick_percent, phi_week_6_pick_percent, phi_week_7_pick_percent, phi_week_8_pick_percent, phi_week_9_pick_percent, phi_week_10_pick_percent, phi_week_11_pick_percent, phi_week_12_pick_percent, phi_week_13_pick_percent, phi_week_14_pick_percent, phi_week_15_pick_percent, phi_week_16_pick_percent, phi_week_17_pick_percent, phi_week_18_pick_percent],
    'Pittsburgh Steelers': [pit_week_1_pick_percent, pit_week_2_pick_percent, pit_week_3_pick_percent, pit_week_4_pick_percent, pit_week_5_pick_percent, pit_week_6_pick_percent, pit_week_7_pick_percent, pit_week_8_pick_percent, pit_week_9_pick_percent, pit_week_10_pick_percent, pit_week_11_pick_percent, pit_week_12_pick_percent, pit_week_13_pick_percent, pit_week_14_pick_percent, pit_week_15_pick_percent, pit_week_16_pick_percent, pit_week_17_pick_percent, pit_week_18_pick_percent],
    'San Francisco 49ers': [sf_week_1_pick_percent, sf_week_2_pick_percent, sf_week_3_pick_percent, sf_week_4_pick_percent, sf_week_5_pick_percent, sf_week_6_pick_percent, sf_week_7_pick_percent, sf_week_8_pick_percent, sf_week_9_pick_percent, sf_week_10_pick_percent, sf_week_11_pick_percent, sf_week_12_pick_percent, sf_week_13_pick_percent, sf_week_14_pick_percent, sf_week_15_pick_percent, sf_week_16_pick_percent, sf_week_17_pick_percent, sf_week_18_pick_percent],
    'Seattle Seahawks': [sea_week_1_pick_percent, sea_week_2_pick_percent, sea_week_3_pick_percent, sea_week_4_pick_percent, sea_week_5_pick_percent, sea_week_6_pick_percent, sea_week_7_pick_percent, sea_week_8_pick_percent, sea_week_9_pick_percent, sea_week_10_pick_percent, sea_week_11_pick_percent, sea_week_12_pick_percent, sea_week_13_pick_percent, sea_week_14_pick_percent, sea_week_15_pick_percent, sea_week_16_pick_percent, sea_week_17_pick_percent, sea_week_18_pick_percent],
    'Tampa Bay Buccaneers': [tb_week_1_pick_percent, tb_week_2_pick_percent, tb_week_3_pick_percent, tb_week_4_pick_percent, tb_week_5_pick_percent, tb_week_6_pick_percent, tb_week_7_pick_percent, tb_week_8_pick_percent, tb_week_9_pick_percent, tb_week_10_pick_percent, tb_week_11_pick_percent, tb_week_12_pick_percent, tb_week_13_pick_percent, tb_week_14_pick_percent, tb_week_15_pick_percent, tb_week_16_pick_percent, tb_week_17_pick_percent, tb_week_18_pick_percent],
    'Tennessee Titans': [ten_week_1_pick_percent, ten_week_2_pick_percent, ten_week_3_pick_percent, ten_week_4_pick_percent, ten_week_5_pick_percent, ten_week_6_pick_percent, ten_week_7_pick_percent, ten_week_8_pick_percent, ten_week_9_pick_percent, ten_week_10_pick_percent, ten_week_11_pick_percent, ten_week_12_pick_percent, ten_week_13_pick_percent, ten_week_14_pick_percent, ten_week_15_pick_percent, ten_week_16_pick_percent, ten_week_17_pick_percent, ten_week_18_pick_percent],
     'Washington Commanders': [was_week_1_pick_percent, was_week_2_pick_percent, was_week_3_pick_percent, was_week_4_pick_percent, was_week_5_pick_percent, was_week_6_pick_percent, was_week_7_pick_percent, was_week_8_pick_percent, was_week_9_pick_percent, was_week_10_pick_percent, was_week_11_pick_percent, was_week_12_pick_percent, was_week_13_pick_percent, was_week_14_pick_percent, was_week_15_pick_percent, was_week_16_pick_percent, was_week_17_pick_percent, was_week_18_pick_percent]
}

splash_pick_percentages = {
    'Arizona Cardinals': [az_week_1_pick_percent, az_week_2_pick_percent, az_week_3_pick_percent, az_week_4_pick_percent, az_week_5_pick_percent, az_week_6_pick_percent, az_week_7_pick_percent, az_week_8_pick_percent, az_week_9_pick_percent, az_week_10_pick_percent, az_week_11_pick_percent, az_week_12_pick_percent, az_week_13_pick_percent, az_week_14_pick_percent, az_week_15_pick_percent, az_week_16_pick_percent, az_week_17_pick_percent, az_week_18_pick_percent],
    'Atlanta Falcons': [atl_week_1_pick_percent, atl_week_2_pick_percent, atl_week_3_pick_percent, atl_week_4_pick_percent, atl_week_5_pick_percent, atl_week_6_pick_percent, atl_week_7_pick_percent, atl_week_8_pick_percent, atl_week_9_pick_percent, atl_week_10_pick_percent, atl_week_11_pick_percent, atl_week_12_pick_percent, atl_week_13_pick_percent, atl_week_14_pick_percent, atl_week_15_pick_percent, atl_week_16_pick_percent, atl_week_17_pick_percent, atl_week_18_pick_percent],
    'Baltimore Ravens': [bal_week_1_pick_percent, bal_week_2_pick_percent, bal_week_3_pick_percent, bal_week_4_pick_percent, bal_week_5_pick_percent, bal_week_6_pick_percent, bal_week_7_pick_percent, bal_week_8_pick_percent, bal_week_9_pick_percent, bal_week_10_pick_percent, bal_week_11_pick_percent, bal_week_12_pick_percent, bal_week_13_pick_percent, bal_week_14_pick_percent, bal_week_15_pick_percent, bal_week_16_pick_percent, bal_week_17_pick_percent, bal_week_18_pick_percent],
    'Buffalo Bills': [buf_week_1_pick_percent, buf_week_2_pick_percent, buf_week_3_pick_percent, buf_week_4_pick_percent, buf_week_5_pick_percent, buf_week_6_pick_percent, buf_week_7_pick_percent, buf_week_8_pick_percent, buf_week_9_pick_percent, buf_week_10_pick_percent, buf_week_11_pick_percent, buf_week_12_pick_percent, buf_week_13_pick_percent, buf_week_14_pick_percent, buf_week_15_pick_percent, buf_week_16_pick_percent, buf_week_17_pick_percent, buf_week_18_pick_percent],
    'Carolina Panthers': [car_week_1_pick_percent, car_week_2_pick_percent, car_week_3_pick_percent, car_week_4_pick_percent, car_week_5_pick_percent, car_week_6_pick_percent, car_week_7_pick_percent, car_week_8_pick_percent, car_week_9_pick_percent, car_week_10_pick_percent, car_week_11_pick_percent, car_week_12_pick_percent, car_week_13_pick_percent, car_week_14_pick_percent, car_week_15_pick_percent, car_week_16_pick_percent, car_week_17_pick_percent, car_week_18_pick_percent],
    'Chicago Bears': [chi_week_1_pick_percent, chi_week_2_pick_percent, chi_week_3_pick_percent, chi_week_4_pick_percent, chi_week_5_pick_percent, chi_week_6_pick_percent, chi_week_7_pick_percent, chi_week_8_pick_percent, chi_week_9_pick_percent, chi_week_10_pick_percent, chi_week_11_pick_percent, chi_week_12_pick_percent, chi_week_13_pick_percent, chi_week_14_pick_percent, chi_week_15_pick_percent, chi_week_16_pick_percent, chi_week_17_pick_percent, chi_week_18_pick_percent],
    'Cincinnati Bengals': [cin_week_1_pick_percent, cin_week_2_pick_percent, cin_week_3_pick_percent, cin_week_4_pick_percent, cin_week_5_pick_percent, cin_week_6_pick_percent, cin_week_7_pick_percent, cin_week_8_pick_percent, cin_week_9_pick_percent, cin_week_10_pick_percent, cin_week_11_pick_percent, cin_week_12_pick_percent, cin_week_13_pick_percent, cin_week_14_pick_percent, cin_week_15_pick_percent, cin_week_16_pick_percent, cin_week_17_pick_percent, cin_week_18_pick_percent],
    'Cleveland Browns': [cle_week_1_pick_percent, cle_week_2_pick_percent, cle_week_3_pick_percent, cle_week_4_pick_percent, cle_week_5_pick_percent, cle_week_6_pick_percent, cle_week_7_pick_percent, cle_week_8_pick_percent, cle_week_9_pick_percent, cle_week_10_pick_percent, cle_week_11_pick_percent, cle_week_12_pick_percent, cle_week_13_pick_percent, cle_week_14_pick_percent, cle_week_15_pick_percent, cle_week_16_pick_percent, cle_week_17_pick_percent, cle_week_18_pick_percent],
    'Dallas Cowboys': [dal_week_1_pick_percent, dal_week_2_pick_percent, dal_week_3_pick_percent, dal_week_4_pick_percent, dal_week_5_pick_percent, dal_week_6_pick_percent, dal_week_7_pick_percent, dal_week_8_pick_percent, dal_week_9_pick_percent, dal_week_10_pick_percent, dal_week_11_pick_percent, dal_week_12_pick_percent, dal_week_13_pick_percent, dal_week_14_pick_percent, dal_week_15_pick_percent, dal_week_16_pick_percent, dal_week_17_pick_percent, dal_week_18_pick_percent],
    'Denver Broncos': [den_week_1_pick_percent, den_week_2_pick_percent, den_week_3_pick_percent, den_week_4_pick_percent, den_week_5_pick_percent, den_week_6_pick_percent, den_week_7_pick_percent, den_week_8_pick_percent, den_week_9_pick_percent, den_week_10_pick_percent, den_week_11_pick_percent, den_week_12_pick_percent, den_week_13_pick_percent, den_week_14_pick_percent, den_week_15_pick_percent, den_week_16_pick_percent, den_week_17_pick_percent, den_week_18_pick_percent],
    'Detroit Lions': [det_week_1_pick_percent, det_week_2_pick_percent, det_week_3_pick_percent, det_week_4_pick_percent, det_week_5_pick_percent, det_week_6_pick_percent, det_week_7_pick_percent, det_week_8_pick_percent, det_week_9_pick_percent, det_week_10_pick_percent, det_week_11_pick_percent, det_week_12_pick_percent, det_week_13_pick_percent, det_week_14_pick_percent, det_week_15_pick_percent, det_week_16_pick_percent, det_week_17_pick_percent, det_week_18_pick_percent],
    'Green Bay Packers': [gb_week_1_pick_percent, gb_week_2_pick_percent, gb_week_3_pick_percent, gb_week_4_pick_percent, gb_week_5_pick_percent, gb_week_6_pick_percent, gb_week_7_pick_percent, gb_week_8_pick_percent, gb_week_9_pick_percent, gb_week_10_pick_percent, gb_week_11_pick_percent, gb_week_12_pick_percent, gb_week_13_pick_percent, gb_week_14_pick_percent, gb_week_15_pick_percent, gb_week_16_pick_percent, gb_week_17_pick_percent, gb_week_18_pick_percent],
    'Houston Texans': [hou_week_1_pick_percent, hou_week_2_pick_percent, hou_week_3_pick_percent, hou_week_4_pick_percent, hou_week_5_pick_percent, hou_week_6_pick_percent, hou_week_7_pick_percent, hou_week_8_pick_percent, hou_week_9_pick_percent, hou_week_10_pick_percent, hou_week_11_pick_percent, hou_week_12_pick_percent, hou_week_13_pick_percent, hou_week_14_pick_percent, hou_week_15_pick_percent, hou_week_16_pick_percent, hou_week_17_pick_percent, hou_week_18_pick_percent],
    'Indianapolis Colts': [ind_week_1_pick_percent, ind_week_2_pick_percent, ind_week_3_pick_percent, ind_week_4_pick_percent, ind_week_5_pick_percent, ind_week_6_pick_percent, ind_week_7_pick_percent, ind_week_8_pick_percent, ind_week_9_pick_percent, ind_week_10_pick_percent, ind_week_11_pick_percent, ind_week_12_pick_percent, ind_week_13_pick_percent, ind_week_14_pick_percent, ind_week_15_pick_percent, ind_week_16_pick_percent, ind_week_17_pick_percent, ind_week_18_pick_percent],
    'Jacksonville Jaguars': [jax_week_1_pick_percent, jax_week_2_pick_percent, jax_week_3_pick_percent, jax_week_4_pick_percent, jax_week_5_pick_percent, jax_week_6_pick_percent, jax_week_7_pick_percent, jax_week_8_pick_percent, jax_week_9_pick_percent, jax_week_10_pick_percent, jax_week_11_pick_percent, jax_week_12_pick_percent, jax_week_13_pick_percent, jax_week_14_pick_percent, jax_week_15_pick_percent, jax_week_16_pick_percent, jax_week_17_pick_percent, jax_week_18_pick_percent],
    'Kansas City Chiefs': [kc_week_1_pick_percent, kc_week_2_pick_percent, kc_week_3_pick_percent, kc_week_4_pick_percent, kc_week_5_pick_percent, kc_week_6_pick_percent, kc_week_7_pick_percent, kc_week_8_pick_percent, kc_week_9_pick_percent, kc_week_10_pick_percent, kc_week_11_pick_percent, kc_week_12_pick_percent, kc_week_13_pick_percent, kc_week_14_pick_percent, kc_week_15_pick_percent, kc_week_16_pick_percent, kc_week_17_pick_percent, kc_week_18_pick_percent],
    'Las Vegas Raiders': [lv_week_1_pick_percent, lv_week_2_pick_percent, lv_week_3_pick_percent, lv_week_4_pick_percent, lv_week_5_pick_percent, lv_week_6_pick_percent, lv_week_7_pick_percent, lv_week_8_pick_percent, lv_week_9_pick_percent, lv_week_10_pick_percent, lv_week_11_pick_percent, lv_week_12_pick_percent, lv_week_13_pick_percent, lv_week_14_pick_percent, lv_week_15_pick_percent, lv_week_16_pick_percent, lv_week_17_pick_percent, lv_week_18_pick_percent],
    'Los Angeles Chargers': [lac_week_1_pick_percent, lac_week_2_pick_percent, lac_week_3_pick_percent, lac_week_4_pick_percent, lac_week_5_pick_percent, lac_week_6_pick_percent, lac_week_7_pick_percent, lac_week_8_pick_percent, lac_week_9_pick_percent, lac_week_10_pick_percent, lac_week_11_pick_percent, lac_week_12_pick_percent, lac_week_13_pick_percent, lac_week_14_pick_percent, lac_week_15_pick_percent, lac_week_16_pick_percent, lac_week_17_pick_percent, lac_week_18_pick_percent],
    'Los Angeles Rams': [lar_week_1_pick_percent, lar_week_2_pick_percent, lar_week_3_pick_percent, lar_week_4_pick_percent, lar_week_5_pick_percent, lar_week_6_pick_percent, lar_week_7_pick_percent, lar_week_8_pick_percent, lar_week_9_pick_percent, lar_week_10_pick_percent, lar_week_11_pick_percent, lar_week_12_pick_percent, lar_week_13_pick_percent, lar_week_14_pick_percent, lar_week_15_pick_percent, lar_week_16_pick_percent, lar_week_17_pick_percent, lar_week_18_pick_percent],
    'Miami Dolphins': [mia_week_1_pick_percent, mia_week_2_pick_percent, mia_week_3_pick_percent, mia_week_4_pick_percent, mia_week_5_pick_percent, mia_week_6_pick_percent, mia_week_7_pick_percent, mia_week_8_pick_percent, mia_week_9_pick_percent, mia_week_10_pick_percent, mia_week_11_pick_percent, mia_week_12_pick_percent, mia_week_13_pick_percent, mia_week_14_pick_percent, mia_week_15_pick_percent, mia_week_16_pick_percent, mia_week_17_pick_percent, mia_week_18_pick_percent],
    'Minnesota Vikings': [min_week_1_pick_percent, min_week_2_pick_percent, min_week_3_pick_percent, min_week_4_pick_percent, min_week_5_pick_percent, min_week_6_pick_percent, min_week_7_pick_percent, min_week_8_pick_percent, min_week_9_pick_percent, min_week_10_pick_percent, min_week_11_pick_percent, min_week_12_pick_percent, min_week_13_pick_percent, min_week_14_pick_percent, min_week_15_pick_percent, min_week_16_pick_percent, min_week_17_pick_percent, min_week_18_pick_percent],
    'New England Patriots': [ne_week_1_pick_percent, ne_week_2_pick_percent, ne_week_3_pick_percent, ne_week_4_pick_percent, ne_week_5_pick_percent, ne_week_6_pick_percent, ne_week_7_pick_percent, ne_week_8_pick_percent, ne_week_9_pick_percent, ne_week_10_pick_percent, ne_week_11_pick_percent, ne_week_12_pick_percent, ne_week_13_pick_percent, ne_week_14_pick_percent, ne_week_15_pick_percent, ne_week_16_pick_percent, ne_week_17_pick_percent, ne_week_18_pick_percent],
    'New Orleans Saints': [no_week_1_pick_percent, no_week_2_pick_percent, no_week_3_pick_percent, no_week_4_pick_percent, no_week_5_pick_percent, no_week_6_pick_percent, no_week_7_pick_percent, no_week_8_pick_percent, no_week_9_pick_percent, no_week_10_pick_percent, no_week_11_pick_percent, no_week_12_pick_percent, no_week_13_pick_percent, no_week_14_pick_percent, no_week_15_pick_percent, no_week_16_pick_percent, no_week_17_pick_percent, no_week_18_pick_percent],
    'New York Giants': [nyg_week_1_pick_percent, nyg_week_2_pick_percent, nyg_week_3_pick_percent, nyg_week_4_pick_percent, nyg_week_5_pick_percent, nyg_week_6_pick_percent, nyg_week_7_pick_percent, nyg_week_8_pick_percent, nyg_week_9_pick_percent, nyg_week_10_pick_percent, nyg_week_11_pick_percent, nyg_week_12_pick_percent, nyg_week_13_pick_percent, nyg_week_14_pick_percent, nyg_week_15_pick_percent, nyg_week_16_pick_percent, nyg_week_17_pick_percent, nyg_week_18_pick_percent],
    'New York Jets': [nyj_week_1_pick_percent, nyj_week_2_pick_percent, nyj_week_3_pick_percent, nyj_week_4_pick_percent, nyj_week_5_pick_percent, nyj_week_6_pick_percent, nyj_week_7_pick_percent, nyj_week_8_pick_percent, nyj_week_9_pick_percent, nyj_week_10_pick_percent, nyj_week_11_pick_percent, nyj_week_12_pick_percent, nyj_week_13_pick_percent, nyj_week_14_pick_percent, nyj_week_15_pick_percent, nyj_week_16_pick_percent, nyj_week_17_pick_percent, nyj_week_18_pick_percent],
    'Philadelphia Eagles': [phi_week_1_pick_percent, phi_week_2_pick_percent, phi_week_3_pick_percent, phi_week_4_pick_percent, phi_week_5_pick_percent, phi_week_6_pick_percent, phi_week_7_pick_percent, phi_week_8_pick_percent, phi_week_9_pick_percent, phi_week_10_pick_percent, phi_week_11_pick_percent, phi_week_12_pick_percent, phi_week_13_pick_percent, phi_week_14_pick_percent, phi_week_15_pick_percent, phi_week_16_pick_percent, phi_week_17_pick_percent, phi_week_18_pick_percent],
    'Pittsburgh Steelers': [pit_week_1_pick_percent, pit_week_2_pick_percent, pit_week_3_pick_percent, pit_week_4_pick_percent, pit_week_5_pick_percent, pit_week_6_pick_percent, pit_week_7_pick_percent, pit_week_8_pick_percent, pit_week_9_pick_percent, pit_week_10_pick_percent, pit_week_11_pick_percent, pit_week_12_pick_percent, pit_week_13_pick_percent, pit_week_14_pick_percent, pit_week_15_pick_percent, pit_week_16_pick_percent, pit_week_17_pick_percent, pit_week_18_pick_percent],
    'San Francisco 49ers': [sf_week_1_pick_percent, sf_week_2_pick_percent, sf_week_3_pick_percent, sf_week_4_pick_percent, sf_week_5_pick_percent, sf_week_6_pick_percent, sf_week_7_pick_percent, sf_week_8_pick_percent, sf_week_9_pick_percent, sf_week_10_pick_percent, sf_week_11_pick_percent, sf_week_12_pick_percent, sf_week_13_pick_percent, sf_week_14_pick_percent, sf_week_15_pick_percent, sf_week_16_pick_percent, sf_week_17_pick_percent, sf_week_18_pick_percent],
    'Seattle Seahawks': [sea_week_1_pick_percent, sea_week_2_pick_percent, sea_week_3_pick_percent, sea_week_4_pick_percent, sea_week_5_pick_percent, sea_week_6_pick_percent, sea_week_7_pick_percent, sea_week_8_pick_percent, sea_week_9_pick_percent, sea_week_10_pick_percent, sea_week_11_pick_percent, sea_week_12_pick_percent, sea_week_13_pick_percent, sea_week_14_pick_percent, sea_week_15_pick_percent, sea_week_16_pick_percent, sea_week_17_pick_percent, sea_week_18_pick_percent],
    'Tampa Bay Buccaneers': [tb_week_1_pick_percent, tb_week_2_pick_percent, tb_week_3_pick_percent, tb_week_4_pick_percent, tb_week_5_pick_percent, tb_week_6_pick_percent, tb_week_7_pick_percent, tb_week_8_pick_percent, tb_week_9_pick_percent, tb_week_10_pick_percent, tb_week_11_pick_percent, tb_week_12_pick_percent, tb_week_13_pick_percent, tb_week_14_pick_percent, tb_week_15_pick_percent, tb_week_16_pick_percent, tb_week_17_pick_percent, tb_week_18_pick_percent],
    'Tennessee Titans': [ten_week_1_pick_percent, ten_week_2_pick_percent, ten_week_3_pick_percent, ten_week_4_pick_percent, ten_week_5_pick_percent, ten_week_6_pick_percent, ten_week_7_pick_percent, ten_week_8_pick_percent, ten_week_9_pick_percent, ten_week_10_pick_percent, ten_week_11_pick_percent, ten_week_12_pick_percent, ten_week_13_pick_percent, ten_week_14_pick_percent, ten_week_15_pick_percent, ten_week_16_pick_percent, ten_week_17_pick_percent, ten_week_18_pick_percent],
     'Washington Commanders': [was_week_1_pick_percent, was_week_2_pick_percent, was_week_3_pick_percent, was_week_4_pick_percent, was_week_5_pick_percent, was_week_6_pick_percent, was_week_7_pick_percent, was_week_8_pick_percent, was_week_9_pick_percent, was_week_10_pick_percent, was_week_11_pick_percent, was_week_12_pick_percent, was_week_13_pick_percent, was_week_14_pick_percent, was_week_15_pick_percent, was_week_16_pick_percent, was_week_17_pick_percent, was_week_18_pick_percent]
}

circa_pick_percentages = {
    'Arizona Cardinals': [az_week_1_pick_percent, az_week_2_pick_percent, az_week_3_pick_percent, az_week_4_pick_percent, az_week_5_pick_percent, az_week_6_pick_percent, az_week_7_pick_percent, az_week_8_pick_percent, az_week_9_pick_percent, az_week_10_pick_percent, az_week_11_pick_percent, az_week_12_pick_percent, az_week_13_pick_percent, az_week_14_pick_percent, az_week_15_pick_percent, az_week_16_pick_percent, az_week_17_pick_percent, az_week_18_pick_percent, az_week_19_pick_percent, az_week_20_pick_percent],
    'Atlanta Falcons': [atl_week_1_pick_percent, atl_week_2_pick_percent, atl_week_3_pick_percent, atl_week_4_pick_percent, atl_week_5_pick_percent, atl_week_6_pick_percent, atl_week_7_pick_percent, atl_week_8_pick_percent, atl_week_9_pick_percent, atl_week_10_pick_percent, atl_week_11_pick_percent, atl_week_12_pick_percent, atl_week_13_pick_percent, atl_week_14_pick_percent, atl_week_15_pick_percent, atl_week_16_pick_percent, atl_week_17_pick_percent, atl_week_18_pick_percent, atl_week_19_pick_percent, atl_week_20_pick_percent],
    'Baltimore Ravens': [bal_week_1_pick_percent, bal_week_2_pick_percent, bal_week_3_pick_percent, bal_week_4_pick_percent, bal_week_5_pick_percent, bal_week_6_pick_percent, bal_week_7_pick_percent, bal_week_8_pick_percent, bal_week_9_pick_percent, bal_week_10_pick_percent, bal_week_11_pick_percent, bal_week_12_pick_percent, bal_week_13_pick_percent, bal_week_14_pick_percent, bal_week_15_pick_percent, bal_week_16_pick_percent, bal_week_17_pick_percent, bal_week_18_pick_percent, bal_week_19_pick_percent, bal_week_20_pick_percent],
    'Buffalo Bills': [buf_week_1_pick_percent, buf_week_2_pick_percent, buf_week_3_pick_percent, buf_week_4_pick_percent, buf_week_5_pick_percent, buf_week_6_pick_percent, buf_week_7_pick_percent, buf_week_8_pick_percent, buf_week_9_pick_percent, buf_week_10_pick_percent, buf_week_11_pick_percent, buf_week_12_pick_percent, buf_week_13_pick_percent, buf_week_14_pick_percent, buf_week_15_pick_percent, buf_week_16_pick_percent, buf_week_17_pick_percent, buf_week_18_pick_percent, buf_week_19_pick_percent, buf_week_20_pick_percent],
    'Carolina Panthers': [car_week_1_pick_percent, car_week_2_pick_percent, car_week_3_pick_percent, car_week_4_pick_percent, car_week_5_pick_percent, car_week_6_pick_percent, car_week_7_pick_percent, car_week_8_pick_percent, car_week_9_pick_percent, car_week_10_pick_percent, car_week_11_pick_percent, car_week_12_pick_percent, car_week_13_pick_percent, car_week_14_pick_percent, car_week_15_pick_percent, car_week_16_pick_percent, car_week_17_pick_percent, car_week_18_pick_percent, car_week_19_pick_percent, car_week_20_pick_percent],
    'Chicago Bears': [chi_week_1_pick_percent, chi_week_2_pick_percent, chi_week_3_pick_percent, chi_week_4_pick_percent, chi_week_5_pick_percent, chi_week_6_pick_percent, chi_week_7_pick_percent, chi_week_8_pick_percent, chi_week_9_pick_percent, chi_week_10_pick_percent, chi_week_11_pick_percent, chi_week_12_pick_percent, chi_week_13_pick_percent, chi_week_14_pick_percent, chi_week_15_pick_percent, chi_week_16_pick_percent, chi_week_17_pick_percent, chi_week_18_pick_percent, chi_week_19_pick_percent, chi_week_20_pick_percent],
    'Cincinnati Bengals': [cin_week_1_pick_percent, cin_week_2_pick_percent, cin_week_3_pick_percent, cin_week_4_pick_percent, cin_week_5_pick_percent, cin_week_6_pick_percent, cin_week_7_pick_percent, cin_week_8_pick_percent, cin_week_9_pick_percent, cin_week_10_pick_percent, cin_week_11_pick_percent, cin_week_12_pick_percent, cin_week_13_pick_percent, cin_week_14_pick_percent, cin_week_15_pick_percent, cin_week_16_pick_percent, cin_week_17_pick_percent, cin_week_18_pick_percent, cin_week_19_pick_percent, cin_week_20_pick_percent],
    'Cleveland Browns': [cle_week_1_pick_percent, cle_week_2_pick_percent, cle_week_3_pick_percent, cle_week_4_pick_percent, cle_week_5_pick_percent, cle_week_6_pick_percent, cle_week_7_pick_percent, cle_week_8_pick_percent, cle_week_9_pick_percent, cle_week_10_pick_percent, cle_week_11_pick_percent, cle_week_12_pick_percent, cle_week_13_pick_percent, cle_week_14_pick_percent, cle_week_15_pick_percent, cle_week_16_pick_percent, cle_week_17_pick_percent, cle_week_18_pick_percent, cle_week_19_pick_percent, cle_week_20_pick_percent],
    'Dallas Cowboys': [dal_week_1_pick_percent, dal_week_2_pick_percent, dal_week_3_pick_percent, dal_week_4_pick_percent, dal_week_5_pick_percent, dal_week_6_pick_percent, dal_week_7_pick_percent, dal_week_8_pick_percent, dal_week_9_pick_percent, dal_week_10_pick_percent, dal_week_11_pick_percent, dal_week_12_pick_percent, dal_week_13_pick_percent, dal_week_14_pick_percent, dal_week_15_pick_percent, dal_week_16_pick_percent, dal_week_17_pick_percent, dal_week_18_pick_percent, dal_week_19_pick_percent, dal_week_20_pick_percent],
    'Denver Broncos': [den_week_1_pick_percent, den_week_2_pick_percent, den_week_3_pick_percent, den_week_4_pick_percent, den_week_5_pick_percent, den_week_6_pick_percent, den_week_7_pick_percent, den_week_8_pick_percent, den_week_9_pick_percent, den_week_10_pick_percent, den_week_11_pick_percent, den_week_12_pick_percent, den_week_13_pick_percent, den_week_14_pick_percent, den_week_15_pick_percent, den_week_16_pick_percent, den_week_17_pick_percent, den_week_18_pick_percent, den_week_19_pick_percent, den_week_20_pick_percent],
    'Detroit Lions': [det_week_1_pick_percent, det_week_2_pick_percent, det_week_3_pick_percent, det_week_4_pick_percent, det_week_5_pick_percent, det_week_6_pick_percent, det_week_7_pick_percent, det_week_8_pick_percent, det_week_9_pick_percent, det_week_10_pick_percent, det_week_11_pick_percent, det_week_12_pick_percent, det_week_13_pick_percent, det_week_14_pick_percent, det_week_15_pick_percent, det_week_16_pick_percent, det_week_17_pick_percent, det_week_18_pick_percent, det_week_19_pick_percent, det_week_20_pick_percent],
    'Green Bay Packers': [gb_week_1_pick_percent, gb_week_2_pick_percent, gb_week_3_pick_percent, gb_week_4_pick_percent, gb_week_5_pick_percent, gb_week_6_pick_percent, gb_week_7_pick_percent, gb_week_8_pick_percent, gb_week_9_pick_percent, gb_week_10_pick_percent, gb_week_11_pick_percent, gb_week_12_pick_percent, gb_week_13_pick_percent, gb_week_14_pick_percent, gb_week_15_pick_percent, gb_week_16_pick_percent, gb_week_17_pick_percent, gb_week_18_pick_percent, gb_week_19_pick_percent, gb_week_20_pick_percent],
    'Houston Texans': [hou_week_1_pick_percent, hou_week_2_pick_percent, hou_week_3_pick_percent, hou_week_4_pick_percent, hou_week_5_pick_percent, hou_week_6_pick_percent, hou_week_7_pick_percent, hou_week_8_pick_percent, hou_week_9_pick_percent, hou_week_10_pick_percent, hou_week_11_pick_percent, hou_week_12_pick_percent, hou_week_13_pick_percent, hou_week_14_pick_percent, hou_week_15_pick_percent, hou_week_16_pick_percent, hou_week_17_pick_percent, hou_week_18_pick_percent, hou_week_19_pick_percent, hou_week_20_pick_percent],
    'Indianapolis Colts': [ind_week_1_pick_percent, ind_week_2_pick_percent, ind_week_3_pick_percent, ind_week_4_pick_percent, ind_week_5_pick_percent, ind_week_6_pick_percent, ind_week_7_pick_percent, ind_week_8_pick_percent, ind_week_9_pick_percent, ind_week_10_pick_percent, ind_week_11_pick_percent, ind_week_12_pick_percent, ind_week_13_pick_percent, ind_week_14_pick_percent, ind_week_15_pick_percent, ind_week_16_pick_percent, ind_week_17_pick_percent, ind_week_18_pick_percent, ind_week_19_pick_percent, ind_week_20_pick_percent],
    'Jacksonville Jaguars': [jax_week_1_pick_percent, jax_week_2_pick_percent, jax_week_3_pick_percent, jax_week_4_pick_percent, jax_week_5_pick_percent, jax_week_6_pick_percent, jax_week_7_pick_percent, jax_week_8_pick_percent, jax_week_9_pick_percent, jax_week_10_pick_percent, jax_week_11_pick_percent, jax_week_12_pick_percent, jax_week_13_pick_percent, jax_week_14_pick_percent, jax_week_15_pick_percent, jax_week_16_pick_percent, jax_week_17_pick_percent, jax_week_18_pick_percent, jax_week_19_pick_percent, jax_week_20_pick_percent],
    'Kansas City Chiefs': [kc_week_1_pick_percent, kc_week_2_pick_percent, kc_week_3_pick_percent, kc_week_4_pick_percent, kc_week_5_pick_percent, kc_week_6_pick_percent, kc_week_7_pick_percent, kc_week_8_pick_percent, kc_week_9_pick_percent, kc_week_10_pick_percent, kc_week_11_pick_percent, kc_week_12_pick_percent, kc_week_13_pick_percent, kc_week_14_pick_percent, kc_week_15_pick_percent, kc_week_16_pick_percent, kc_week_17_pick_percent, kc_week_18_pick_percent, kc_week_19_pick_percent, kc_week_20_pick_percent],
    'Las Vegas Raiders': [lv_week_1_pick_percent, lv_week_2_pick_percent, lv_week_3_pick_percent, lv_week_4_pick_percent, lv_week_5_pick_percent, lv_week_6_pick_percent, lv_week_7_pick_percent, lv_week_8_pick_percent, lv_week_9_pick_percent, lv_week_10_pick_percent, lv_week_11_pick_percent, lv_week_12_pick_percent, lv_week_13_pick_percent, lv_week_14_pick_percent, lv_week_15_pick_percent, lv_week_16_pick_percent, lv_week_17_pick_percent, lv_week_18_pick_percent, lv_week_19_pick_percent, lv_week_20_pick_percent],
    'Los Angeles Chargers': [lac_week_1_pick_percent, lac_week_2_pick_percent, lac_week_3_pick_percent, lac_week_4_pick_percent, lac_week_5_pick_percent, lac_week_6_pick_percent, lac_week_7_pick_percent, lac_week_8_pick_percent, lac_week_9_pick_percent, lac_week_10_pick_percent, lac_week_11_pick_percent, lac_week_12_pick_percent, lac_week_13_pick_percent, lac_week_14_pick_percent, lac_week_15_pick_percent, lac_week_16_pick_percent, lac_week_17_pick_percent, lac_week_18_pick_percent, lac_week_19_pick_percent, lac_week_20_pick_percent],
    'Los Angeles Rams': [lar_week_1_pick_percent, lar_week_2_pick_percent, lar_week_3_pick_percent, lar_week_4_pick_percent, lar_week_5_pick_percent, lar_week_6_pick_percent, lar_week_7_pick_percent, lar_week_8_pick_percent, lar_week_9_pick_percent, lar_week_10_pick_percent, lar_week_11_pick_percent, lar_week_12_pick_percent, lar_week_13_pick_percent, lar_week_14_pick_percent, lar_week_15_pick_percent, lar_week_16_pick_percent, lar_week_17_pick_percent, lar_week_18_pick_percent, lar_week_19_pick_percent, lar_week_20_pick_percent],
    'Miami Dolphins': [mia_week_1_pick_percent, mia_week_2_pick_percent, mia_week_3_pick_percent, mia_week_4_pick_percent, mia_week_5_pick_percent, mia_week_6_pick_percent, mia_week_7_pick_percent, mia_week_8_pick_percent, mia_week_9_pick_percent, mia_week_10_pick_percent, mia_week_11_pick_percent, mia_week_12_pick_percent, mia_week_13_pick_percent, mia_week_14_pick_percent, mia_week_15_pick_percent, mia_week_16_pick_percent, mia_week_17_pick_percent, mia_week_18_pick_percent, mia_week_19_pick_percent, mia_week_20_pick_percent],
    'Minnesota Vikings': [min_week_1_pick_percent, min_week_2_pick_percent, min_week_3_pick_percent, min_week_4_pick_percent, min_week_5_pick_percent, min_week_6_pick_percent, min_week_7_pick_percent, min_week_8_pick_percent, min_week_9_pick_percent, min_week_10_pick_percent, min_week_11_pick_percent, min_week_12_pick_percent, min_week_13_pick_percent, min_week_14_pick_percent, min_week_15_pick_percent, min_week_16_pick_percent, min_week_17_pick_percent, min_week_18_pick_percent, min_week_19_pick_percent, min_week_20_pick_percent],
    'New England Patriots': [ne_week_1_pick_percent, ne_week_2_pick_percent, ne_week_3_pick_percent, ne_week_4_pick_percent, ne_week_5_pick_percent, ne_week_6_pick_percent, ne_week_7_pick_percent, ne_week_8_pick_percent, ne_week_9_pick_percent, ne_week_10_pick_percent, ne_week_11_pick_percent, ne_week_12_pick_percent, ne_week_13_pick_percent, ne_week_14_pick_percent, ne_week_15_pick_percent, ne_week_16_pick_percent, ne_week_17_pick_percent, ne_week_18_pick_percent, ne_week_19_pick_percent, ne_week_20_pick_percent],
    'New Orleans Saints': [no_week_1_pick_percent, no_week_2_pick_percent, no_week_3_pick_percent, no_week_4_pick_percent, no_week_5_pick_percent, no_week_6_pick_percent, no_week_7_pick_percent, no_week_8_pick_percent, no_week_9_pick_percent, no_week_10_pick_percent, no_week_11_pick_percent, no_week_12_pick_percent, no_week_13_pick_percent, no_week_14_pick_percent, no_week_15_pick_percent, no_week_16_pick_percent, no_week_17_pick_percent, no_week_18_pick_percent, no_week_19_pick_percent, no_week_20_pick_percent],
    'New York Giants': [nyg_week_1_pick_percent, nyg_week_2_pick_percent, nyg_week_3_pick_percent, nyg_week_4_pick_percent, nyg_week_5_pick_percent, nyg_week_6_pick_percent, nyg_week_7_pick_percent, nyg_week_8_pick_percent, nyg_week_9_pick_percent, nyg_week_10_pick_percent, nyg_week_11_pick_percent, nyg_week_12_pick_percent, nyg_week_13_pick_percent, nyg_week_14_pick_percent, nyg_week_15_pick_percent, nyg_week_16_pick_percent, nyg_week_17_pick_percent, nyg_week_18_pick_percent, nyg_week_19_pick_percent, nyg_week_20_pick_percent],
    'New York Jets': [nyj_week_1_pick_percent, nyj_week_2_pick_percent, nyj_week_3_pick_percent, nyj_week_4_pick_percent, nyj_week_5_pick_percent, nyj_week_6_pick_percent, nyj_week_7_pick_percent, nyj_week_8_pick_percent, nyj_week_9_pick_percent, nyj_week_10_pick_percent, nyj_week_11_pick_percent, nyj_week_12_pick_percent, nyj_week_13_pick_percent, nyj_week_14_pick_percent, nyj_week_15_pick_percent, nyj_week_16_pick_percent, nyj_week_17_pick_percent, nyj_week_18_pick_percent, nyj_week_19_pick_percent, nyj_week_20_pick_percent],
    'Philadelphia Eagles': [phi_week_1_pick_percent, phi_week_2_pick_percent, phi_week_3_pick_percent, phi_week_4_pick_percent, phi_week_5_pick_percent, phi_week_6_pick_percent, phi_week_7_pick_percent, phi_week_8_pick_percent, phi_week_9_pick_percent, phi_week_10_pick_percent, phi_week_11_pick_percent, phi_week_12_pick_percent, phi_week_13_pick_percent, phi_week_14_pick_percent, phi_week_15_pick_percent, phi_week_16_pick_percent, phi_week_17_pick_percent, phi_week_18_pick_percent, phi_week_19_pick_percent, phi_week_20_pick_percent],
    'Pittsburgh Steelers': [pit_week_1_pick_percent, pit_week_2_pick_percent, pit_week_3_pick_percent, pit_week_4_pick_percent, pit_week_5_pick_percent, pit_week_6_pick_percent, pit_week_7_pick_percent, pit_week_8_pick_percent, pit_week_9_pick_percent, pit_week_10_pick_percent, pit_week_11_pick_percent, pit_week_12_pick_percent, pit_week_13_pick_percent, pit_week_14_pick_percent, pit_week_15_pick_percent, pit_week_16_pick_percent, pit_week_17_pick_percent, pit_week_18_pick_percent, pit_week_19_pick_percent, pit_week_20_pick_percent],
    'San Francisco 49ers': [sf_week_1_pick_percent, sf_week_2_pick_percent, sf_week_3_pick_percent, sf_week_4_pick_percent, sf_week_5_pick_percent, sf_week_6_pick_percent, sf_week_7_pick_percent, sf_week_8_pick_percent, sf_week_9_pick_percent, sf_week_10_pick_percent, sf_week_11_pick_percent, sf_week_12_pick_percent, sf_week_13_pick_percent, sf_week_14_pick_percent, sf_week_15_pick_percent, sf_week_16_pick_percent, sf_week_17_pick_percent, sf_week_18_pick_percent, sf_week_19_pick_percent, sf_week_20_pick_percent],
    'Seattle Seahawks': [sea_week_1_pick_percent, sea_week_2_pick_percent, sea_week_3_pick_percent, sea_week_4_pick_percent, sea_week_5_pick_percent, sea_week_6_pick_percent, sea_week_7_pick_percent, sea_week_8_pick_percent, sea_week_9_pick_percent, sea_week_10_pick_percent, sea_week_11_pick_percent, sea_week_12_pick_percent, sea_week_13_pick_percent, sea_week_14_pick_percent, sea_week_15_pick_percent, sea_week_16_pick_percent, sea_week_17_pick_percent, sea_week_18_pick_percent, sea_week_19_pick_percent, sea_week_20_pick_percent],
    'Tampa Bay Buccaneers': [tb_week_1_pick_percent, tb_week_2_pick_percent, tb_week_3_pick_percent, tb_week_4_pick_percent, tb_week_5_pick_percent, tb_week_6_pick_percent, tb_week_7_pick_percent, tb_week_8_pick_percent, tb_week_9_pick_percent, tb_week_10_pick_percent, tb_week_11_pick_percent, tb_week_12_pick_percent, tb_week_13_pick_percent, tb_week_14_pick_percent, tb_week_15_pick_percent, tb_week_16_pick_percent, tb_week_17_pick_percent, tb_week_18_pick_percent, tb_week_19_pick_percent, tb_week_20_pick_percent],
    'Tennessee Titans': [ten_week_1_pick_percent, ten_week_2_pick_percent, ten_week_3_pick_percent, ten_week_4_pick_percent, ten_week_5_pick_percent, ten_week_6_pick_percent, ten_week_7_pick_percent, ten_week_8_pick_percent, ten_week_9_pick_percent, ten_week_10_pick_percent, ten_week_11_pick_percent, ten_week_12_pick_percent, ten_week_13_pick_percent, ten_week_14_pick_percent, ten_week_15_pick_percent, ten_week_16_pick_percent, ten_week_17_pick_percent, ten_week_18_pick_percent, ten_week_19_pick_percent, ten_week_20_pick_percent],
    'Washington Commanders': [was_week_1_pick_percent, was_week_2_pick_percent, was_week_3_pick_percent, was_week_4_pick_percent, was_week_5_pick_percent, was_week_6_pick_percent, was_week_7_pick_percent, was_week_8_pick_percent, was_week_9_pick_percent, was_week_10_pick_percent, was_week_11_pick_percent, was_week_12_pick_percent, was_week_13_pick_percent, was_week_14_pick_percent, was_week_15_pick_percent, was_week_16_pick_percent, was_week_17_pick_percent, was_week_18_pick_percent, was_week_19_pick_percent, was_week_20_pick_percent]
}

az_current_week_availability = -1
atl_current_week_availability = -1
bal_current_week_availability = -1
buf_current_week_availability = -1
car_current_week_availability = -1
chi_current_week_availability = -1
cin_current_week_availability = -1
cle_current_week_availability = -1
dal_current_week_availability = -1
den_current_week_availability = -1
det_current_week_availability = -1
gb_current_week_availability = -1
hou_current_week_availability = -1
ind_current_week_availability = -1
jax_current_week_availability = -1
kc_current_week_availability = -1
lv_current_week_availability = -1
lac_current_week_availability = -1
lar_current_week_availability = -1
mia_current_week_availability = -1
min_current_week_availability = -1
ne_current_week_availability = -1
no_current_week_availability = -1
nyg_current_week_availability = -1
nyj_current_week_availability = -1
phi_current_week_availability = -1
pit_current_week_availability = -1
sf_current_week_availability = -1
sea_current_week_availability = -1
tb_current_week_availability = -1
ten_current_week_availability = -1
was_current_week_availability = -1


default_current_week_entries_circa = 17500
default_current_week_entries_dk = 20000
default_current_week_entries_splash_sports = 18519

current_week_entries_circa = 17500
current_week_entries_dk = 20000
current_week_entries_splash_sports = 18519

LOGO_PATH = "GSF Survivor Logo Clear BG.png"
def login_screen():
    st.title("Welcome to Generic Sports Fan Survivor Optimization")
    st.subheader("Please log in with Google to access the optimizer.")
    st.write("This tool requires authentication to ensure proper usage and to save your progress.")
    st.write("")
    if st.button("Log in with Google", use_container_width=True):
        st.login() # This triggers the Google OAuth flow

if not st.user.is_logged_in:
    # User is not logged in, show the login screen
    login_screen()
else:
        # --- Add Logo in Sidebar when logged in ---
    with st.sidebar:
        st.image(LOGO_PATH, use_container_width=True) # Fills the sidebar width
        if st.button("Logout"):
            st.logout()# Logout button inside sidebar
        st.write(f"Hello, {st.user.name}!")
    # User is logged in, display the main application content
    # --- End Logo in Sidebar ---
    st.image(LOGO_PATH)
    st.title("NFL Survivor Optimization")
    st.subheader("The best NFL Survivor Contest optimizer")
    contest_options = [
        "Circa", "DraftKings", "Splash Sports"
    ]
    with st.expander("More Information"):
        st.write("Alright, clowns. This site is built to help you optimize your picks for the Circa Survivor contest (And Draftkings and Splash Sports). :red[This tool is just for informational use. It does not take into account injuries, weather, resting players, or certain other factors. Do not use this tool as your only source of information.] Simply input which week you're in, your team rankings, constraints, etc... and the algorithm will do the rest.")
        st.write("Calculating Expected Value, or EV, will take the longest in this process. For a full season, this step will take roughly 5-10 minutes or more. Do not close your browser. It's worth the wait. Good luck!")
    st.write('')
    st.write('')
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

    two_team_selections_help_text = f"""
    \nIn Splash Sports, most survivor contests have a unique requirement: You must select two teams per week starting in the second half of the season.  
    \n- However, when you start selecting two teams per week varies, depending on which contest you enter. For some, this may begin in week 11, and for others it may begin as late as week 16.
    \n- Because it varies, we want to give you the option to select which week this applies to you. Plus, it gives you more flexibility to play around with the tool. 
    
    """
	
    selected_contest = st.selectbox('Choose Contest:', options = contest_options, help = 'Choose the contest you are using this algorithm for: Circa (Advanced), Draftkings (Traditional and Pathetic), or Splash Sports (Somewhere between the other two)' + help_text_seletced_contest)
    if selected_contest == "DraftKings":
    	ending_week = 19
    elif selected_contest == "Splash Sports":
        ending_week = 19
        st.write('')
        st.write('')
        week_requiring_two_selections = st.selectbox("Which week do you need to start selecting two teams?:", options=range(1, 19), help = two_team_selections_help_text)
	    
	
    else:
    	ending_week = 21
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.subheader('Picked Teams:')
    yes_i_have_picked_teams = st.checkbox('Have you already used any teams in the contest, or want to prevent the algorithm from using any specific teams at any point in the season?')
    
    #def create_nfl_app():
    
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
    if yes_i_have_picked_teams:
        selected_teams_help_text = 'Select the teams that you have already used in the survivor contest, or teams that you just do not want to pick in the enirety of the contest'
        selected_teams = st.multiselect("Prohibited Picks:", options=nfl_teams, help = selected_teams_help_text)
        picked_teams = selected_teams if selected_teams else []
        if picked_teams:
            st.write("You selected:")
            for team in picked_teams:
                st.write(f"- {team}")
        else:
            st.write("No teams selected")
    st.write('')
    st.write('')
    st.write('')
    st.subheader('Remaining Weeks:')
    yes_i_would_like_to_choose_weeks = st.checkbox('Would you like to choose a range of weeks, instead of the entire season?')
    if yes_i_would_like_to_choose_weeks:
        remaining_weeks_help_text = f"""Select the upcoming week for the starting week and select the week you want the algorithm to stop at. \n- If you select one week, Calculating EV can take up to 30 seconds. \n- All 18 weeks will take 5-10 minutes.\n- Ending Week must be greater than or equal to Starting Week."""
        if selected_contest == "DraftKings":
            starting_week = st.selectbox("Select Starting Week:", options=range(1, 19), help = remaining_weeks_help_text)
        elif selected_contest == "Splash Sports":
            splash_sports_remaining_weeks_help_text = f"""Select the upcoming week for the starting week and select the week you want the algorithm to stop at. \n- If you select one week, Calculating EV can take up to 30 seconds. \n- All 18 weeks will take 5-10 minutes.\n- Ending Week must be greater than or equal to Starting Week. In the later part of the season, you will be required to select two teams per week."""
            starting_week = st.selectbox("Select Starting Week:", options=range(1, 19), help = remaining_weeks_help_text)
        else:
            circa_remaining_weeks_warning = f"""Select the upcoming week for the starting week and select the week you want the algorithm to stop at. \n- If you select one week, Calculating EV can take up to 30 seconds. \n- All 20 weeks will take 5-10 minutes.\n- Ending Week must be greater than or equal to Starting Week. \n- :red[Week 13 is Thanksgiving/Black Friday Week and Week 18 is Christmas Week]"""
            starting_week = st.selectbox("Select Starting Week:", options=range(1, 21), help = circa_remaining_weeks_warning)
        #if starting_week:
            #st.write(f"Selected Starting Week: {starting_week}")
        # Create a dynamic range for ending week based on starting week
        st.write('')
        if selected_contest == "DraftKings":
            ending_week_options = range(starting_week, 19)
        elif selected_contest == "Splash Sports":
            ending_week_options = range(starting_week, 19)
        else:
            ending_week_options = range(starting_week, 21)
        ending_week = st.selectbox("Select Ending Week:", options=ending_week_options)
        ending_week = ending_week + 1
    
        st.write('')
        st.write('')
        st.write('')
        st.subheader('Current Week Entries')
        current_week_entries_help_text = 'For accurate predictions, you must input the number of remaining entries in your contest'
        # Get an integer input with no maximum
        if selected_contest == "DraftKings":
            current_week_entries_dk = st.number_input("Number of Remaining Entries:", value=-1, max_value=None, help = current_week_entries_help_text)
            st.write(f"You entered: {current_week_entries_dk}")
        if selected_contest == "Splash Sports":
            current_week_entries_splash_sports = st.number_input("Number of Remaining Entries:", value=-1, max_value=None, help = current_week_entries_help_text)
            st.write(f"You entered: {current_week_entries_splash_sports}")
        else:
            current_week_entries_circa = st.number_input("Number of Remaining Entries:", value=-1, max_value=None, help = current_week_entries_help_text)
            st.write(f"You entered: {current_week_entries_circa}")
        st.write('')
        st.write('')
        st.write('')
        st.subheader('Current Week Team Availability')
        st.write('')
	    
        az_current_week_availability = st.slider("Arizona Cardinals Current Week Availability:", -1, 100) / 100
        az_current_week_availability_percent = az_current_week_availability*100
        if az_current_week_availability < 0:
            st.write(f"Arizona Cardinals Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Arizona Cardinals Current Week Availability: :green[{az_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        atl_current_week_availability = st.slider("Atlanta Falcons Current Week Availability:", -1, 100) / 100
        atl_current_week_availability_percent = atl_current_week_availability*100
        if atl_current_week_availability < 0:
            st.write(f"Atlanta Falcons Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Atlanta Falcons Current Week Availability: :green[{atl_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        bal_current_week_availability = st.slider("Baltimore Ravens Current Week Availability:", -1, 100) / 100
        bal_current_week_availability_percent = bal_current_week_availability*100
        if bal_current_week_availability < 0:
            st.write(f"Baltimore Ravens Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Baltimore Ravenss Current Week Availability: :green[{bal_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        buf_current_week_availability = st.slider("Buffalo Bills Current Week Availability:", -1, 100) / 100
        buf_current_week_availability_percent = buf_current_week_availability*100
        if buf_current_week_availability < 0:
            st.write(f"Buffalo Bills Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Buffalo Bills Current Week Availability: :green[{buf_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        car_current_week_availability = st.slider("Carolina Panthers Current Week Availability:", -1, 100) / 100
        car_current_week_availability_percent = car_current_week_availability*100
        if car_current_week_availability < 0:
            st.write(f"Carolina Panthers Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Carolina Panthers Current Week Availability: :green[{car_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        chi_current_week_availability = st.slider("Chicago Bears Current Week Availability:", -1, 100) / 100
        chi_current_week_availability_percent = chi_current_week_availability*100
        if chi_current_week_availability < 0:
            st.write(f"Chicago Bears Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Chicago Bears Current Week Availability: :green[{chi_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        cin_current_week_availability = st.slider("Cincinnati Bengals Current Week Availability:", -1, 100) / 100
        cin_current_week_availability_percent = cin_current_week_availability*100
        if cin_current_week_availability < 0:
            st.write(f"Cincinnati Bengals Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Cincinnati Bengals Current Week Availability: :green[{cin_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        cle_current_week_availability = st.slider("Cleveland Browns Current Week Availability:", -1, 100) / 100
        cle_current_week_availability_percent = cle_current_week_availability*100
        if cle_current_week_availability < 0:
            st.write(f"Cleveland Browns Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Cleveland Browns Current Week Availability: :green[{cle_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        dal_current_week_availability = st.slider("Dallas Cowboys Current Week Availability:", -1, 100) / 100
        dal_current_week_availability_percent = dal_current_week_availability*100
        if dal_current_week_availability < 0:
            st.write(f"Dallas Cowboys Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Dallas Cowboys Current Week Availability: :green[{dal_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        den_current_week_availability = st.slider("Denver Broncos Current Week Availability:", -1, 100) / 100
        den_current_week_availability_percent = den_current_week_availability*100
        if den_current_week_availability < 0:
            st.write(f"Denver Broncos Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Denver Broncos Current Week Availability: :green[{den_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        det_current_week_availability = st.slider("Detroit Lions Current Week Availability:", -1, 100) / 100
        det_current_week_availability_percent = det_current_week_availability*100
        if det_current_week_availability < 0:
            st.write(f"Detroit Lions Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Detroit Lions Current Week Availability: :green[{det_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        gb_current_week_availability = st.slider("Green Bay Packers Current Week Availability:", -1, 100) / 100
        gb_current_week_availability_percent = gb_current_week_availability*100
        if gb_current_week_availability < 0:
            st.write(f"Green Bay Packers Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Green Bay Packers Current Week Availability: :green[{gb_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        hou_current_week_availability = st.slider("Houston Texans Current Week Availability:", -1, 100) / 100
        hou_current_week_availability_percent = hou_current_week_availability*100
        if hou_current_week_availability < 0:
            st.write(f"Houston Texans Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Houston Texans Current Week Availability: :green[{hou_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        ind_current_week_availability = st.slider("Indianapoils Colts Current Week Availability:", -1, 100) / 100
        ind_current_week_availability_percent = ind_current_week_availability*100
        if ind_current_week_availability < 0:
            st.write(f"Indianapoils Colts Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Indianapoils Colts Current Week Availability: :green[{ind_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        jax_current_week_availability = st.slider("Jacksonville Jaguars Current Week Availability:", -1, 100) / 100
        jax_current_week_availability_percent = jax_current_week_availability*100
        if jax_current_week_availability < 0:
            st.write(f"Jacksonville Jaguars Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Jacksonville Jaguars Current Week Availability: :green[{jax_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        kc_current_week_availability = st.slider("Kansas City Chiefs Current Week Availability:", -1, 100) / 100
        kc_current_week_availability_percent = kc_current_week_availability*100
        if kc_current_week_availability < 0:
            st.write(f"Kansas City Chiefs Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Kansas City Chiefs Current Week Availability: :green[{kc_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        lv_current_week_availability = st.slider("Las Vegas Raiders Current Week Availability:", -1, 100) / 100
        lv_current_week_availability_percent = lv_current_week_availability*100
        if lv_current_week_availability < 0:
            st.write(f"Las Vegas Raiders Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Las vegas Raiders Current Week Availability: :green[{lv_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        lac_current_week_availability = st.slider("Los Angeles Chargers Current Week Availability:", -1, 100) / 100
        lac_current_week_availability_percent = lac_current_week_availability*100
        if lac_current_week_availability < 0:
            st.write(f"Los Angeles Chargers Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Los angeles Chargers Current Week Availability: :green[{lac_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        lar_current_week_availability = st.slider("Los Angeles Rams Current Week Availability:", -1, 100) / 100
        lar_current_week_availability_percent = lar_current_week_availability*100
        if lar_current_week_availability < 0:
            st.write(f"Los Angeles Rams Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Los Angeles Rams Current Week Availability: :green[{lar_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        mia_current_week_availability = st.slider("Miami Dolphins Current Week Availability:", -1, 100) / 100
        mia_current_week_availability_percent = mia_current_week_availability*100
        if mia_current_week_availability < 0:
            st.write(f"Miami Dolphins Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Miami Dolphins Current Week Availability: :green[{mia_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        min_current_week_availability = st.slider("Minnesota Vikings Current Week Availability:", -1, 100) / 100
        min_current_week_availability_percent = min_current_week_availability*100
        if min_current_week_availability < 0:
            st.write(f"Minnesota Vikings Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Minnesota Vikings Current Week Availability: :green[{min_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        ne_current_week_availability = st.slider("New England Patriots Current Week Availability:", -1, 100) / 100
        ne_current_week_availability_percent = ne_current_week_availability*100
        if ne_current_week_availability < 0:
            st.write(f"New England Patriots Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"New England Patriots Current Week Availability: :green[{ne_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        no_current_week_availability = st.slider("New Orleans Saints Current Week Availability:", -1, 100) / 100
        no_current_week_availability_percent = no_current_week_availability*100
        if no_current_week_availability < 0:
            st.write(f"New Orleans Saints Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"New Orleans Saints Current Week Availability: :green[{no_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        nyg_current_week_availability = st.slider("New York Giants Current Week Availability:", -1, 100) / 100
        nyg_current_week_availability_percent = nyg_current_week_availability*100
        if nyg_current_week_availability < 0:
            st.write(f"New York Giants Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"New York Giants Current Week Availability: :green[{nyg_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        nyj_current_week_availability = st.slider("New York Jets Current Week Availability:", -1, 100) / 100
        nyj_current_week_availability_percent = nyj_current_week_availability*100
        if nyj_current_week_availability < 0:
            st.write(f"New York Jets Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"New York Jets Current Week Availability: :green[{nyj_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        phi_current_week_availability = st.slider("Philadelphia Eagles Current Week Availability:", -1, 100) / 100
        phi_current_week_availability_percent = phi_current_week_availability*100
        if phi_current_week_availability < 0:
            st.write(f"Philadelphia Eagles Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Philadelphia Eagles Current Week Availability: :green[{phi_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        pit_current_week_availability = st.slider("Pittsburgh Steelers Current Week Availability:", -1, 100) / 100
        pit_current_week_availability_percent = pit_current_week_availability*100
        if pit_current_week_availability < 0:
            st.write(f"Pittsburgh Steelers Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Pittsburgh Steelers Current Week Availability: :green[{pit_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        sf_current_week_availability = st.slider("San Francisco 49ers Current Week Availability:", -1, 100) / 100
        sf_current_week_availability_percent = sf_current_week_availability*100
        if sf_current_week_availability < 0:
            st.write(f"San Francisco 49ers Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"San Francisco 49ers Current Week Availability: :green[{sf_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        sea_current_week_availability = st.slider("Seattle Seahawks Current Week Availability:", -1, 100) / 100
        sea_current_week_availability_percent = sea_current_week_availability*100
        if sea_current_week_availability < 0:
            st.write(f"Seattle Seahawks Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Seattle Seahawks Current Week Availability: :green[{sea_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        tb_current_week_availability = st.slider("Tampa Bay Buccaneers Current Week Availability:", -1, 100) / 100
        tb_current_week_availability_percent = tb_current_week_availability*100
        if tb_current_week_availability < 0:
            st.write(f"Tampa Bay Buccaneers Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Tampa Bay Buccaneers Current Week Availability: :green[{tb_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        ten_current_week_availability = st.slider("Tennessee Titans Current Week Availability:", -1, 100) / 100
        ten_current_week_availability_percent = ten_current_week_availability*100
        if ten_current_week_availability < 0:
            st.write(f"Tennessee Titans Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Tennessee Titans Current Week Availability: :green[{ten_current_week_availability_percent}%]")
        st.write("")
        st.write("")
	    
        was_current_week_availability = st.slider("Washington Commanders Current Week Availability:", -1, 100) / 100
        was_current_week_availability_percent = was_current_week_availability*100
        if was_current_week_availability < 0:
            st.write(f"Washington Commanders Current Week Availability: :red[Automatically Calculated]")
        else:
            st.write(f"Washington Commanders Current Week Availability: :green[{was_current_week_availability_percent}%]")
        st.write("")
        st.write("")
    
    team_availability = {
        'Arizona Cardinals': az_current_week_availability,
        'Atlanta Falcons': atl_current_week_availability,
        'Baltimore Ravens': bal_current_week_availability,
        'Buffalo Bills': buf_current_week_availability,
        'Carolina Panthers': car_current_week_availability,
        'Chicago Bears': chi_current_week_availability,
        'Cincinnati Bengals': cin_current_week_availability,
        'Cleveland Browns': cle_current_week_availability,
        'Dallas Cowboys': dal_current_week_availability,
        'Denver Broncos': den_current_week_availability,
        'Detroit Lions': det_current_week_availability,
        'Green Bay Packers': gb_current_week_availability,
        'Houston Texans': hou_current_week_availability,
        'Indianapolis Colts': ind_current_week_availability,
        'Jacksonville Jaguars': jax_current_week_availability,
        'Kansas City Chiefs': kc_current_week_availability,
        'Las Vegas Raiders': lv_current_week_availability,
        'Los Angeles Chargers': lac_current_week_availability,
        'Los Angeles Rams': lar_current_week_availability,
        'Miami Dolphins': mia_current_week_availability,
        'Minnesota Vikings': min_current_week_availability,
        'New England Patriots': ne_current_week_availability,
        'New Orleans Saints': no_current_week_availability,
        'New York Giants': nyg_current_week_availability,
        'New York Jets': nyj_current_week_availability,
        'Philadelphia Eagles': phi_current_week_availability,
        'Pittsburgh Steelers': pit_current_week_availability,
        'San Francisco 49ers': sf_current_week_availability,
        'Seattle Seahawks': sea_current_week_availability,
        'Tampa Bay Buccaneers': tb_current_week_availability,
        'Tennessee Titans': ten_current_week_availability,
        'Washington Commanders': was_current_week_availability
        }
    	
    #return starting_week, ending_week, picked_teams
        #if ending_week:
            #st.write(f"Selected Ending Week: {ending_week}")
    #if __name__ == "__main__":
    #    starting_week, ending_week, picked_teams = create_nfl_app()
    st.write('')
    st.write('')
    st.write('')
    st.subheader('Teams That Have to Be Picked')
    yes_i_have_a_required_team = st.checkbox('Do you have a team that you require to be used in a specific week?')
    required_week_options = [0] + list(range(starting_week, ending_week))
    if yes_i_have_a_required_team:
        with st.expander("More Information"):
            st.write("\n- Select the teams that you would like to be used in a specific week. \n- For example, if you would like to select the Philadelphia Eagles in week 4, just select '4' Under the Eagles' dropdown.")
		
        az_req_week = st.selectbox("Arizona Cardinals Week Required to Be Picked:", options=required_week_options)
        atl_req_week = st.selectbox("Atlanta Falcons Week Required to Be Picked:", options=required_week_options)
        bal_req_week = st.selectbox("Baltimore Ravens Week Required to Be Picked:", options=required_week_options)
        buf_req_week = st.selectbox("Buffalo Bills Week Required to Be Picked:", options=required_week_options)
        car_req_week = st.selectbox("Carolina Panthers Week Required to Be Picked:", options=required_week_options)
        chi_req_week = st.selectbox("Chicago Bears Week Required to Be Picked:", options=required_week_options)
        cin_req_week = st.selectbox("Cincinnati Bengals Week Required to Be Picked:", options=required_week_options)
        cle_req_week = st.selectbox("Cleveland Browns Week Required to Be Picked:", options=required_week_options)
        dal_req_week = st.selectbox("Dallas Cowboys Week Required to Be Picked:", options=required_week_options)
        den_req_week = st.selectbox("Denver Broncos Week Required to Be Picked:", options=required_week_options)
        det_req_week = st.selectbox("Detroit Lions Week Required to Be Picked:", options=required_week_options)
        gb_req_week = st.selectbox("Green Bay Packers Week Required to Be Picked:", options=required_week_options)
        hou_req_week = st.selectbox("Houston Texans Week Required to Be Picked:", options=required_week_options)
        ind_req_week = st.selectbox("Indianapoils Colts Week Required to Be Picked:", options=required_week_options)
        jax_req_week = st.selectbox("Jacksonville Jaguars Week Required to Be Picked:", options=required_week_options)
        kc_req_week = st.selectbox("Kansas City Chiefs Week Required to Be Picked:", options=required_week_options)
        lv_req_week = st.selectbox("Las Vegas Raiders Week Required to Be Picked:", options=required_week_options)
        lac_req_week = st.selectbox("Los Angeles Chargers Week Required to Be Picked:", options=required_week_options)
        lar_req_week = st.selectbox("Los Angeles Rams Week Required to Be Picked:", options=required_week_options)
        mia_req_week = st.selectbox("Miami Dolphins Week Required to Be Picked:", options=required_week_options)
        min_req_week = st.selectbox("Minnesota Vikings Week Required to Be Picked:", options=required_week_options)
        ne_req_week = st.selectbox("New England Patriots Week Required to Be Picked:", options=required_week_options)
        no_req_week = st.selectbox("New Orleans Saints Week Required to Be Picked:", options=required_week_options)
        nyg_req_week = st.selectbox("New York Giants Week Required to Be Picked:", options=required_week_options)
        nyj_req_week = st.selectbox("New York Jets Week Required to Be Picked:", options=required_week_options)
        phi_req_week = st.selectbox("Philadelphia Eagles Week Required to Be Picked:", options=required_week_options)
        pit_req_week = st.selectbox("Pittsburgh Steelers Week Required to Be Picked:", options=required_week_options)
        sf_req_week = st.selectbox("San Francisco 49ers Week Required to Be Picked:", options=required_week_options)
        sea_req_week = st.selectbox("Seattle Seahawks Week Required to Be Picked:", options=required_week_options)
        tb_req_week = st.selectbox("Tampa Bay Buccaneers Week Required to Be Picked:", options=required_week_options)
        ten_req_week = st.selectbox("Tennessee Titans Week Required to Be Picked:", options=required_week_options)
        was_req_week = st.selectbox("Washington Commanders Week Required to Be Picked:", options=required_week_options)
    
    st.write('')
    st.write('')
    st.write('')
    st.subheader('Prohibited Teams')
    yes_i_have_prohibited_teams = st.checkbox('Do you have teams that you want to prohibit the alogrithm from choosing in a specifc week?')
    if yes_i_have_prohibited_teams:
        with st.expander("More Information"):
            st.write('\n-Choose which week you do :red[NOT] want a team to be picked. \n-If, for example, you think the 49ers have a bad matchup in Week 15, and you do not want them to be used then, select "15" for the San Francisco 49ers. \n- You may select multiple weeks for each team.')
    
        az_prohibited_weeks = st.multiselect("Arizona Cardinals Week to Be Excluded:", options=required_week_options)
        atl_prohibited_weeks = st.multiselect("Atlanta Falcons Week to Be Excluded:", options=required_week_options)
        bal_prohibited_weeks = st.multiselect("Baltimore Ravens Week to Be Excluded:", options=required_week_options)
        buf_prohibited_weeks = st.multiselect("Buffalo Bills Week to Be Excluded:", options=required_week_options)
        car_prohibited_weeks = st.multiselect("Carolina Panthers Week to Be Excluded:", options=required_week_options)
        chi_prohibited_weeks = st.multiselect("Chicago Bears Week to Be Excluded:", options=required_week_options)
        cin_prohibited_weeks = st.multiselect("Cincinnati Bengals Week to Be Excluded:", options=required_week_options)
        cle_prohibited_weeks = st.multiselect("Cleveland Browns Week to Be Excluded:", options=required_week_options)
        dal_prohibited_weeks = st.multiselect("Dallas Cowboys Week to Be Excluded:", options=required_week_options)
        den_prohibited_weeks = st.multiselect("Denver Broncos Week to Be Excluded:", options=required_week_options)
        det_prohibited_weeks = st.multiselect("Detroit Lions Week to Be Excluded:", options=required_week_options)
        gb_prohibited_weeks = st.multiselect("Green Bay Packers Week to Be Excluded:", options=required_week_options)
        hou_prohibited_weeks = st.multiselect("Houston Texans Week to Be Excluded:", options=required_week_options)
        ind_prohibited_weeks = st.multiselect("Indianapoils Colts Week to Be Excluded:", options=required_week_options)
        jax_prohibited_weeks = st.multiselect("Jacksonville Jaguars Week to Be Excluded:", options=required_week_options)
        kc_prohibited_weeks = st.multiselect("Kansas City Chiefs Week to Be Excluded:", options=required_week_options)
        lv_prohibited_weeks = st.multiselect("Las Vegas Raiders Week to Be Excluded:", options=required_week_options)
        lac_prohibited_weeks = st.multiselect("Los Angeles Chargers Week to Be Excluded:", options=required_week_options)
        lar_prohibited_weeks = st.multiselect("Los Angeles Rams Week to Be Excluded:", options=required_week_options)
        mia_prohibited_weeks = st.multiselect("Miami Dolphins Week to Be Excluded:", options=required_week_options)
        min_prohibited_weeks = st.multiselect("Minnesota Vikings Week to Be Excluded:", options=required_week_options)
        ne_prohibited_weeks = st.multiselect("New England Patriots Week to Be Excluded:", options=required_week_options)
        no_prohibited_weeks = st.multiselect("New Orleans Saints Week to Be Excluded:", options=required_week_options)
        nyg_prohibited_weeks = st.multiselect("New York Giants Week to Be Excluded:", options=required_week_options)
        nyj_prohibited_weeks = st.multiselect("New York Jets Week to Be Excluded:", options=required_week_options)
        phi_prohibited_weeks = st.multiselect("Philadelphia Eagles Week to Be Excluded:", options=required_week_options)
        pit_prohibited_weeks = st.multiselect("Pittsburgh Steelers Week to Be Excluded:", options=required_week_options)
        sf_prohibited_weeks = st.multiselect("San Francisco 49ers Week to Be Excluded:", options=required_week_options)
        sea_prohibited_weeks = st.multiselect("Seattle Seahawks Week to Be Excluded:", options=required_week_options)
        tb_prohibited_weeks = st.multiselect("Tampa Bay Buccaneers Week to Be Excluded:", options=required_week_options)
        ten_prohibited_weeks = st.multiselect("Tennessee Titans Week to Be Excluded:", options=required_week_options)
        was_prohibited_weeks = st.multiselect("Washington Commanders Week to Be Excluded:", options=required_week_options)
        az_excluded_weeks = az_prohibited_weeks if az_prohibited_weeks else []
        atl_excluded_weeks = atl_prohibited_weeks if atl_prohibited_weeks else []
        bal_excluded_weeks = bal_prohibited_weeks if bal_prohibited_weeks else []
        buf_excluded_weeks = buf_prohibited_weeks if buf_prohibited_weeks else []
        car_excluded_weeks = car_prohibited_weeks if car_prohibited_weeks else []
        chi_excluded_weeks = chi_prohibited_weeks if chi_prohibited_weeks else []
        cin_excluded_weeks = cin_prohibited_weeks if cin_prohibited_weeks else []
        cle_excluded_weeks = cle_prohibited_weeks if cle_prohibited_weeks else []
        dal_excluded_weeks = dal_prohibited_weeks if dal_prohibited_weeks else []
        den_excluded_weeks = den_prohibited_weeks if den_prohibited_weeks else []
        det_excluded_weeks = det_prohibited_weeks if det_prohibited_weeks else []
        gb_excluded_weeks = gb_prohibited_weeks if gb_prohibited_weeks else []
        hou_excluded_weeks = hou_prohibited_weeks if hou_prohibited_weeks else []
        ind_excluded_weeks = ind_prohibited_weeks if ind_prohibited_weeks else []
        jax_excluded_weeks = jax_prohibited_weeks if jax_prohibited_weeks else []
        kc_excluded_weeks = kc_prohibited_weeks if kc_prohibited_weeks else []
        lv_excluded_weeks = lv_prohibited_weeks if lv_prohibited_weeks else []
        lac_excluded_weeks = lac_prohibited_weeks if lac_prohibited_weeks else []
        lar_excluded_weeks = lar_prohibited_weeks if lar_prohibited_weeks else []
        mia_excluded_weeks = mia_prohibited_weeks if mia_prohibited_weeks else []
        min_excluded_weeks = min_prohibited_weeks if min_prohibited_weeks else []
        ne_excluded_weeks = ne_prohibited_weeks if ne_prohibited_weeks else []
        no_excluded_weeks = no_prohibited_weeks if no_prohibited_weeks else []
        nyg_excluded_weeks = nyg_prohibited_weeks if nyg_prohibited_weeks else []
        nyj_excluded_weeks = nyj_prohibited_weeks if nyj_prohibited_weeks else []
        phi_excluded_weeks = phi_prohibited_weeks if phi_prohibited_weeks else []
        pit_excluded_weeks = pit_prohibited_weeks if pit_prohibited_weeks else []
        sf_excluded_weeks = sf_prohibited_weeks if sf_prohibited_weeks else []
        sea_excluded_weeks = sea_prohibited_weeks if sea_prohibited_weeks else []
        tb_excluded_weeks = tb_prohibited_weeks if tb_prohibited_weeks else []
        ten_excluded_weeks = ten_prohibited_weeks if ten_prohibited_weeks else []
        was_excluded_weeks = was_prohibited_weeks if was_prohibited_weeks else []
    
    
    st.write('')
    st.write('')
    st.write('')
    
    st.subheader('NFL Team Rankings')
    yes_i_have_customized_rankings = st.checkbox('Would you like to use customized rankings instead of our default rankings?')
    if yes_i_have_customized_rankings:
        with st.expander("More Information"):
            st.write('\n- The Ranking represents :red[how much a team would either win (positive number) or lose (negative number) by to an average NFL team] on a neutral field. 0 means the team is perfectly average. If you leave the "Default" value, the default rankings will be used. \n- If you use your own rankings, we will use them in the calculation based on internal rankings. \n- If you use your own rankings, We will use public Draftkings ML odds to predict pick percentages for the EV calculation (The public largely relies on public lines to make their decisions), but then we will use your internal rankinsg to predict win percentage and help you find an EV edge based on your internal rankings. \n- If Draftkings moneyline odds are unavailable, we will use your internal rankings throughout.')
		
        
        
        team_rankings = [
            "Default", 0,-15,-14.5,-14,-13.5,-13,-12.5,-12,-11.5,-11,-10.5,-10,-9.5,-9,-8.5,-8,-7.5,
            -7,-6.5,-6,-5.5,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-.5,.5,1,1.5,2,2.5,3,3.5,4,
            4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15
        ]
        az_rank = st.selectbox("Arizona Cardinals Ranking:", options=team_rankings)
        if az_rank == "Default":
            az_rank = default_atl_rank
        st.write(f'Current Arizona Cardinals Ranking: :green[{az_rank}]')
        st.write('')
        atl_rank = st.selectbox("Atlanta Falcons Ranking:", options=team_rankings)
        if atl_rank == "Default":
            atl_rank = default_atl_rank
        st.write(f'Current Atlanta Falcons Ranking: :green[{atl_rank}]')
        st.write('')
        bal_rank = st.selectbox("Baltimore Ravens Ranking:", options=team_rankings)
        if bal_rank == "Default":
            bal_rank = default_bal_rank
        st.write(f'Current Baltimore Ravens Ranking: :green[{bal_rank}]')
        st.write('')
        buf_rank = st.selectbox("Buffalo Bills Ranking:", options=team_rankings)
        if buf_rank == "Default":
            buf_rank = default_buf_rank
        st.write(f'Current Buffalo Bills Ranking: :green[{buf_rank}]')
        st.write('')
        car_rank = st.selectbox("Carolina Panthers Ranking:", options=team_rankings)
        if car_rank == "Default":
            car_rank = default_car_rank
        st.write(f'Current Carolina Panthers Ranking: :green[{car_rank}]')
        st.write('')
        chi_rank = st.selectbox("Chicago Bears Ranking:", options=team_rankings)
        if chi_rank == "Default":
            chi_rank = default_chi_rank
        st.write(f'Current Chicago Bears Ranking: :green[{chi_rank}]')
        st.write('')
        cin_rank = st.selectbox("Cincinnati Bengals Ranking:", options=team_rankings)
        if cin_rank == "Default":
            cin_rank = default_cin_rank
        st.write(f'Current Cincinnati Bengals Ranking: :green[{cin_rank}]')
        st.write('')
        cle_rank = st.selectbox("Cleveland Browns Ranking:", options=team_rankings)
        if cle_rank == "Default":
            cle_rank = default_cle_rank
        st.write(f'Current Cleveland Browns Ranking: :green[{cle_rank}]')
        st.write('')
        dal_rank = st.selectbox("Dallas Cowboys Ranking:", options=team_rankings)
        if dal_rank == "Default":
            dal_rank = default_dal_rank
        st.write(f'Current Dallas Cowboys Ranking: :green[{dal_rank}]')
        st.write('')
        den_rank = st.selectbox("Denver Broncos Ranking:", options=team_rankings)
        if den_rank == "Default":
            den_rank = default_den_rank
        st.write(f'Current Denver Broncos Ranking: :green[{den_rank}]')
        st.write('')
        det_rank = st.selectbox("Detroit Lions Ranking:", options=team_rankings)
        if det_rank == "Default":
            det_rank = default_det_rank
        st.write(f'Current Detroit Lions Ranking: :green[{det_rank}]')
        st.write('')
        gb_rank = st.selectbox("Green Bay Packers Ranking:", options=team_rankings)
        if gb_rank == "Default":
            gb_rank = default_gb_rank
        st.write(f'Current Green Bay Packers Ranking: :green[{gb_rank}]')
        st.write('')
        hou_rank = st.selectbox("Houston Texans Ranking:", options=team_rankings)
        if hou_rank == "Default":
            hou_rank = default_hou_rank
        st.write(f'Current Houston Texans Ranking: :green[{hou_rank}]')
        st.write('')
        ind_rank = st.selectbox("Indianapoils Colts Ranking:", options=team_rankings)
        if ind_rank == "Default":
            ind_rank = default_ind_rank
        st.write(f'Current Indianapoils Colts Ranking: :green[{ind_rank}]')
        st.write('')
        jax_rank = st.selectbox("Jacksonville Jaguars Ranking:", options=team_rankings)
        if jax_rank == "Default":
            jax_rank = default_jax_rank
        st.write(f'Current Jacksonville Jaguars Ranking: :green[{jax_rank}]')
        st.write('')
        kc_rank = st.selectbox("Kansas City Chiefs Ranking:", options=team_rankings)
        if kc_rank == "Default":
            kc_rank = default_kc_rank
        st.write(f'Current Kansas City Chiefs Ranking: :green[{kc_rank}]')
        st.write('')
        lv_rank = st.selectbox("Las Vegas Raiders Ranking:", options=team_rankings)
        if lv_rank == "Default":
            lv_rank = default_lv_rank
        st.write(f'Current Las Vegas Raiders Ranking: :green[{lv_rank}]')
        st.write('')
        lac_rank = st.selectbox("Los Angeles Chargers Ranking:", options=team_rankings)
        if lac_rank == "Default":
            lac_rank = default_lac_rank
        st.write(f'Current Los Angeles Chargers Ranking: :green[{lac_rank}]')
        st.write('')
        lar_rank = st.selectbox("Los Angeles Rams Ranking:", options=team_rankings)
        if lar_rank == "Default":
            lar_rank = default_lar_rank
        st.write(f'Current Los Angeles Rams Ranking: :green[{lar_rank}]')
        st.write('')
        mia_rank = st.selectbox("Miami Dolphins Ranking:", options=team_rankings)
        if mia_rank == "Default":
            mia_rank = default_mia_rank
        st.write(f'Current Miami Dolphins Ranking: :green[{mia_rank}]')
        st.write('')
        min_rank = st.selectbox("Minnesota Vikings Ranking:", options=team_rankings)
        if min_rank == "Default":
            min_rank = default_min_rank
        st.write(f'Current Minnesota Vikings Ranking: :green[{min_rank}]')
        st.write('')
        ne_rank = st.selectbox("New England Patriots Ranking:", options=team_rankings)
        if ne_rank == "Default":
            ne_rank = default_ne_rank
        st.write(f'Current New England Patriots Ranking: :green[{ne_rank}]')
        st.write('')
        no_rank = st.selectbox("New Orleans Saints Ranking:", options=team_rankings)
        if no_rank == "Default":
            no_rank = default_no_rank
        st.write(f'Current New Orleans Saints Ranking: :green[{no_rank}]')
        st.write('')
        nyg_rank = st.selectbox("New York Giants Ranking:", options=team_rankings)
        if nyg_rank == "Default":
            nyg_rank = default_nyg_rank
        st.write(f'Current New York Giants Ranking: :green[{nyg_rank}]')
        st.write('')
        nyj_rank = st.selectbox("New York Jets Ranking:", options=team_rankings)
        if nyj_rank == "Default":
            nyj_rank = default_nyj_rank
        st.write(f'Current New York Jets Ranking: :green[{nyj_rank}]')
        st.write('')
        phi_rank = st.selectbox("Philadelphia Eagles Ranking:", options=team_rankings)
        if phi_rank == "Default":
            phi_rank = default_phi_rank
        st.write(f'Current Philadelphia Eagles Ranking: :green[{phi_rank}]')
        st.write('')
        pit_rank = st.selectbox("Pittsburgh Steelers Ranking:", options=team_rankings)
        if pit_rank == "Default":
            pit_rank = default_pit_rank
        st.write(f'Current Pittsburgh Steelers Ranking: :green[{pit_rank}]')
        st.write('')
        sf_rank = st.selectbox("San Francisco 49ers Ranking:", options=team_rankings)
        if sf_rank == "Default":
            sf_rank = default_sf_rank
        st.write(f'Current San Francisco 49ers Ranking: :green[{sf_rank}]')
        st.write('')
        sea_rank = st.selectbox("Seattle Seahawks Ranking:", options=team_rankings)
        if sea_rank == "Default":
            sea_rank = default_sea_rank
        st.write(f'Current Seattle Seahawks Ranking: :green[{sea_rank}]')
        st.write('')
        tb_rank = st.selectbox("Tampa Bay Buccaneers Ranking:", options=team_rankings)
        if tb_rank == "Default":
            tb_rank = default_tb_rank
        st.write(f'Current Tampa Bay Buccaneers Ranking: :green[{tb_rank}]')
        st.write('')
        ten_rank = st.selectbox("Tennessee Titans Ranking:", options=team_rankings)
        if ten_rank == "Default":
            ten_rank = default_ten_rank
        st.write(f'Current Tennessee Titans Ranking: :green[{ten_rank}]')
        st.write('')
        was_rank = st.selectbox("Washington Commanders Ranking:", options=team_rankings)
        if was_rank == "Default":
            was_rank = default_was_rank
        st.write(f'Current Washington Commanders Ranking: :green[{was_rank}]')
    
    st.write('')
    st.write('')
    st.write('')
    st.subheader('Pick Exclusively Favorites?')
    pick_must_be_favored = st.checkbox('All teams picked must be favored at the time of running this script', help = "When this is selected. All teams that are picked must be favored to win the game in order to be selected for this week. \n- There are many cases where the system may select an underdog to create a larger advantage in a different week. \n- We recommend turning this on for smaller pools, but for Circa we recommend turning this off to give you th best EV advantage.")
    
    st.write('')
    st.write('')
    st.write('')
    st.subheader('Select Constraints')
    yes_i_have_constraints = st.checkbox('Would you like to add constraints? For example, "Avoid Teams on Short Rest"')
    
    if yes_i_have_constraints:
        with st.expander("More Information"):
            st.write("""These constraints will not work 100% of the time (For example in week 18, all Games are divisional matchups). However, it will require a team to be so heavily favored that the impact of the constrained factor should be minimal.'
	    \n- We recommend using only a few constraints at a time, to avoid limiting the  system to failure.
            \n- Most of the time, sprotsbooks' odds take into account these kinds of concerns, however, if you are adamanetly against picking away teams or teams in a divisional matchup, we give you the option to limit them as much as possible.""")
        avoid_away_teams_on_short_rest = 1 if st.checkbox('Avoid Away Teams on Short Rest') else 0
        avoid_close_divisional_matchups = 1 if st.checkbox('Avoid Close Divisional Matchups') else 0
        avoid_3_games_in_10_days = 1 if st.checkbox('Avoid 3 games in 10 days') else 0
        avoid_4_games_in_17_days = 1 if st.checkbox('Avoid 4 games in 17 days') else 0
        avoid_away_teams_in_close_matchups = 1 if st.checkbox('Avoid Away Teams in Close Games') else 0
        avoid_cumulative_rest_disadvantage = 1 if st.checkbox('Avoid Cumulative Rest Disadvantage') else 0
        avoid_thursday_night = 1 if st.checkbox('Avoid :red[ALL TEAMS] in Thursday Night Games') else 0
        avoid_away_thursday_night = 1 if st.checkbox('Avoid :red[ONLY AWAY TEAMS] in Thursday Night Games') else 0
        avoid_back_to_back_away = 1 if st.checkbox('Avoid Teams on Back to Back Away Games') else 0
        avoid_international_game = 1 if st.checkbox('Avoid International Games') else 0
        avoid_teams_with_weekly_rest_disadvantage = 1 if st.checkbox('Avoid Teams with Rest Disadvantage') else 0
        avoid_away_teams_with_travel_disadvantage = 1 if st.checkbox('Avoid Teams with Travel Disadvatage') else 0
        bayesian_and_travel_options = [
            "No Rest, Bayesian, and Travel Constraints",
            "Selected team must have been projected to win based on preseason rankings, current rankings, and with and without travel/rest adjustments",
            "Selected team must be projected to win with and without travel and rest impact based on current rankings",
            "Selected team must have been projected to win based on preseason rankings in addition to current rankings",
        ]
            
        #use_same_winners_across_all_4_metrics = 1 if st.selectbox('Bayesian, Rest, and Travel Impact:', options = bayesian_and_travel_options) == "Selected team must have been projected to win based on preseason rankings, current rankings, and with and without travel/rest adjustments" else 0
        #use_same_current_and_adjusted_current_winners = 1 if st.selectbox('Bayesian, Rest, and Travel Impact:', options = bayesian_and_travel_options) == "Selected team must be projected to win with and without travel and rest impact based on current rankings" else 0
        #use_same_adj_preseason_and_adj_current_winner = 1 if st.selectbox('Bayesian, Rest, and Travel Impact:', options = bayesian_and_travel_options) == "Selected team must have been projected to win based on preseason rankings in addition to current rankings" else 0
        if pick_must_be_favored:
        	bayesian_rest_travel_constraint = st.selectbox('Bayesian, Rest, and Travel Impact:', options = bayesian_and_travel_options, help = """Rest, and Travel Impact allows you to choose teams based on the factors working against them or for them. For example: 
	                                                                                                                                              \n- Distance traveled.
																	              \n- Timezones traveled through
																	              \n- Rest advantage
																	              \n- Season-long cumulative rest advantage 
																	              \n- Many more factors
																	              \n\n Bayesian Impact is the weight/importance of preseason rankings. When checked:
																	              \n- This takes into account both preseason rankings and current rankings
																	              \n- Team must have been projected to win this game before the season started and based on currrent rankings
																	              \n- This is important, because in theory the general public's intitial opinion on a team has validity. For example, if a team has great talent, but hasn't put it together so far this season, they could easily upset a team you picked.
																	              \n- Use with caution, this can be too large of a constraint and lead to no possible solutions.""")
    
    st.write('')
    st.write('')
    st.write('')
    
    st.subheader('Estimate Pick Percentages')
    yes_i_have_pick_percents = st.checkbox('Would you like to add your own estimated pick percentages, instead of using our estimated picks?')
    if yes_i_have_pick_percents:
        with st.expander("More Information"):
            st.write('\n- Select your own estimated pick percentages for each team. \n- If you do not change the default value of -1, then it will automatically calculate the estimated pick percentages for that team. \n- This tool will be especially useful later in the season \n- For teams on bye, you can leave them at -1.')
        st.write('')
        st.write('')
        if starting_week <= 1 and ending_week > 1:
            week_1_pick_percents = st.checkbox('Add Week 1 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 1.")
            if week_1_pick_percents:
                st.write('')
                st.subheader('Week 1 Estimated Pick Percentages')
                st.write('')
                az_week_1_pick_percent = st.slider("Arizona Cardinals Week 1 Pick Percent:", -1, 100) / 100
                az_week_1_pick_percent_percent = az_week_1_pick_percent*100
                if az_week_1_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 1 Pick Percent: :green[{az_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_1_pick_percent = st.slider("Atlanta Falcons Week 1 Pick Percent:", -1, 100) / 100
                atl_week_1_pick_percent_percent = atl_week_1_pick_percent*100
                if atl_week_1_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 1 Pick Percent: :green[{atl_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_1_pick_percent = st.slider("Baltimore Ravens Week 1 Pick Percent:", -1, 100) / 100
                bal_week_1_pick_percent_percent = bal_week_1_pick_percent*100
                if bal_week_1_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 1 Pick Percent: :green[{bal_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_1_pick_percent = st.slider("Buffalo Bills Week 1 Pick Percent:", -1, 100) / 100
                buf_week_1_pick_percent_percent = buf_week_1_pick_percent*100
                if buf_week_1_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 1 Pick Percent: :green[{buf_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_1_pick_percent = st.slider("Carolina Panthers Week 1 Pick Percent:", -1, 100) / 100
                car_week_1_pick_percent_percent = car_week_1_pick_percent*100
                if car_week_1_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 1 Pick Percent: :green[{car_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_1_pick_percent = st.slider("Chicago Bears Week 1 Pick Percent:", -1, 100) / 100
                chi_week_1_pick_percent_percent = chi_week_1_pick_percent*100
                if chi_week_1_pick_percent < 0:
                    st.write(f"Chicago Bears Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 1 Pick Percent: :green[{chi_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_1_pick_percent = st.slider("Cincinnati Bengals Week 1 Pick Percent:", -1, 100) / 100
                cin_week_1_pick_percent_percent = cin_week_1_pick_percent*100
                if cin_week_1_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 1 Pick Percent: :green[{cin_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_1_pick_percent = st.slider("Cleveland Browns Week 1 Pick Percent:", -1, 100) / 100
                cle_week_1_pick_percent_percent = cle_week_1_pick_percent*100
                if cle_week_1_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 1 Pick Percent: :green[{cle_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_1_pick_percent = st.slider("Dallas Cowboys Week 1 Pick Percent:", -1, 100) / 100
                dal_week_1_pick_percent_percent = dal_week_1_pick_percent*100
                if dal_week_1_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 1 Pick Percent: :green[{dal_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_1_pick_percent = st.slider("Denver Broncos Week 1 Pick Percent:", -1, 100) / 100
                den_week_1_pick_percent_percent = den_week_1_pick_percent*100
                if den_week_1_pick_percent < 0:
                    st.write(f"Denver Broncos Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 1 Pick Percent: :green[{den_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_1_pick_percent = st.slider("Detroit Lions Week 1 Pick Percent:", -1, 100) / 100
                det_week_1_pick_percent_percent = det_week_1_pick_percent*100
                if det_week_1_pick_percent < 0:
                    st.write(f"Detroit Lions Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 1 Pick Percent: :green[{det_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_1_pick_percent = st.slider("Green Bay Packers Week 1 Pick Percent:", -1, 100) / 100
                gb_week_1_pick_percent_percent = gb_week_1_pick_percent*100
                if gb_week_1_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 1 Pick Percent: :green[{gb_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_1_pick_percent = st.slider("Houston Texans Week 1 Pick Percent:", -1, 100) / 100
                hou_week_1_pick_percent_percent = hou_week_1_pick_percent*100
                if hou_week_1_pick_percent < 0:
                    st.write(f"Houston Texans Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 1 Pick Percent: :green[{hou_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_1_pick_percent = st.slider("Indianapoils Colts Week 1 Pick Percent:", -1, 100) / 100
                ind_week_1_pick_percent_percent = ind_week_1_pick_percent*100
                if ind_week_1_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 1 Pick Percent: :green[{ind_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_1_pick_percent = st.slider("Jacksonville Jaguars Week 1 Pick Percent:", -1, 100) / 100
                jax_week_1_pick_percent_percent = jax_week_1_pick_percent*100
                if jax_week_1_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 1 Pick Percent: :green[{jax_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_1_pick_percent = st.slider("Kansas City Chiefs Week 1 Pick Percent:", -1, 100) / 100
                kc_week_1_pick_percent_percent = kc_week_1_pick_percent*100
                if kc_week_1_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 1 Pick Percent: :green[{kc_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_1_pick_percent = st.slider("Las Vegas Raiders Week 1 Pick Percent:", -1, 100) / 100
                lv_week_1_pick_percent_percent = lv_week_1_pick_percent*100
                if lv_week_1_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 1 Pick Percent: :green[{lv_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_1_pick_percent = st.slider("Los Angeles Chargers Week 1 Pick Percent:", -1, 100) / 100
                lac_week_1_pick_percent_percent = lac_week_1_pick_percent*100
                if lac_week_1_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 1 Pick Percent: :green[{lac_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_1_pick_percent = st.slider("Los Angeles Rams Week 1 Pick Percent:", -1, 100) / 100
                lar_week_1_pick_percent_percent = lar_week_1_pick_percent*100
                if lar_week_1_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 1 Pick Percent: :green[{lar_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_1_pick_percent = st.slider("Miami Dolphins Week 1 Pick Percent:", -1, 100) / 100
                mia_week_1_pick_percent_percent = mia_week_1_pick_percent*100
                if mia_week_1_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 1 Pick Percent: :green[{mia_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_1_pick_percent = st.slider("Minnesota Vikings Week 1 Pick Percent:", -1, 100) / 100
                min_week_1_pick_percent_percent = min_week_1_pick_percent*100
                if min_week_1_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 1 Pick Percent: :green[{min_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_1_pick_percent = st.slider("New England Patriots Week 1 Pick Percent:", -1, 100) / 100
                ne_week_1_pick_percent_percent = ne_week_1_pick_percent*100
                if ne_week_1_pick_percent < 0:
                    st.write(f"New England Patriots Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 1 Pick Percent: :green[{ne_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_1_pick_percent = st.slider("New Orleans Saints Week 1 Pick Percent:", -1, 100) / 100
                no_week_1_pick_percent_percent = no_week_1_pick_percent*100
                if no_week_1_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 1 Pick Percent: :green[{no_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_1_pick_percent = st.slider("New York Giants Week 1 Pick Percent:", -1, 100) / 100
                nyg_week_1_pick_percent_percent = nyg_week_1_pick_percent*100
                if nyg_week_1_pick_percent < 0:
                    st.write(f"New York Giants Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 1 Pick Percent: :green[{nyg_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_1_pick_percent = st.slider("New York Jets Week 1 Pick Percent:", -1, 100) / 100
                nyj_week_1_pick_percent_percent = nyj_week_1_pick_percent*100
                if nyj_week_1_pick_percent < 0:
                    st.write(f"New York Jets Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 1 Pick Percent: :green[{nyj_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_1_pick_percent = st.slider("Philadelphia Eagles Week 1 Pick Percent:", -1, 100) / 100
                phi_week_1_pick_percent_percent = phi_week_1_pick_percent*100
                if phi_week_1_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 1 Pick Percent: :green[{phi_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_1_pick_percent = st.slider("Pittsburgh Steelers Week 1 Pick Percent:", -1, 100) / 100
                pit_week_1_pick_percent_percent = pit_week_1_pick_percent*100
                if pit_week_1_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 1 Pick Percent: :green[{pit_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_1_pick_percent = st.slider("San Francisco 49ers Week 1 Pick Percent:", -1, 100) / 100
                sf_week_1_pick_percent_percent = sf_week_1_pick_percent*100
                if sf_week_1_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 1 Pick Percent: :green[{sf_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_1_pick_percent = st.slider("Seattle Seahawks Week 1 Pick Percent:", -1, 100) / 100
                sea_week_1_pick_percent_percent = sea_week_1_pick_percent*100
                if sea_week_1_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 1 Pick Percent: :green[{sea_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_1_pick_percent = st.slider("Tampa Bay Buccaneers Week 1 Pick Percent:", -1, 100) / 100
                tb_week_1_pick_percent_percent = tb_week_1_pick_percent*100
                if tb_week_1_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 1 Pick Percent: :green[{tb_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_1_pick_percent = st.slider("Tennessee Titans Week 1 Pick Percent:", -1, 100) / 100
                ten_week_1_pick_percent_percent = ten_week_1_pick_percent*100
                if ten_week_1_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 1 Pick Percent: :green[{ten_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_1_pick_percent = st.slider("Washington Commanders Week 1 Pick Percent:", -1, 100) / 100
                was_week_1_pick_percent_percent = was_week_1_pick_percent*100
                if was_week_1_pick_percent < 0:
                    st.write(f"Washington Commanders Week 1 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 1 Pick Percent: :green[{was_week_1_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 2 and ending_week > 2:
            week_2_pick_percents = st.checkbox('Add Week 2 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 2.")
            if week_2_pick_percents:
                st.write('')
                st.subheader('Week 2 Estimated Pick Percentages')
                st.write('')
                az_week_2_pick_percent = st.slider("Arizona Cardinals Week 2 Pick Percent:", -1, 100) / 100
                az_week_2_pick_percent_percent = az_week_2_pick_percent*100
                if az_week_2_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 2 Pick Percent: :green[{az_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_2_pick_percent = st.slider("Atlanta Falcons Week 2 Pick Percent:", -1, 100) / 100
                atl_week_2_pick_percent_percent = atl_week_2_pick_percent*100
                if atl_week_2_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 2 Pick Percent: :green[{atl_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_2_pick_percent = st.slider("Baltimore Ravens Week 2 Pick Percent:", -1, 100) / 100
                bal_week_2_pick_percent_percent = bal_week_2_pick_percent*100
                if bal_week_2_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 2 Pick Percent: :green[{bal_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_2_pick_percent = st.slider("Buffalo Bills Week 2 Pick Percent:", -1, 100) / 100
                buf_week_2_pick_percent_percent = buf_week_2_pick_percent*100
                if buf_week_2_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 2 Pick Percent: :green[{buf_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_2_pick_percent = st.slider("Carolina Panthers Week 2 Pick Percent:", -1, 100) / 100
                car_week_2_pick_percent_percent = car_week_2_pick_percent*100
                if car_week_2_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 2 Pick Percent: :green[{car_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_2_pick_percent = st.slider("Chicago Bears Week 2 Pick Percent:", -1, 100) / 100
                chi_week_2_pick_percent_percent = chi_week_2_pick_percent*100
                if chi_week_2_pick_percent < 0:
                    st.write(f"Chicago Bears Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 2 Pick Percent: :green[{chi_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_2_pick_percent = st.slider("Cincinnati Bengals Week 2 Pick Percent:", -1, 100) / 100
                cin_week_2_pick_percent_percent = cin_week_2_pick_percent*100
                if cin_week_2_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 2 Pick Percent: :green[{cin_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_2_pick_percent = st.slider("Cleveland Browns Week 2 Pick Percent:", -1, 100) / 100
                cle_week_2_pick_percent_percent = cle_week_2_pick_percent*100
                if cle_week_2_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 2 Pick Percent: :green[{cle_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_2_pick_percent = st.slider("Dallas Cowboys Week 2 Pick Percent:", -1, 100) / 100
                dal_week_2_pick_percent_percent = dal_week_2_pick_percent*100
                if dal_week_2_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 2 Pick Percent: :green[{dal_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_2_pick_percent = st.slider("Denver Broncos Week 2 Pick Percent:", -1, 100) / 100
                den_week_2_pick_percent_percent = den_week_2_pick_percent*100
                if den_week_2_pick_percent < 0:
                    st.write(f"Denver Broncos Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 2 Pick Percent: :green[{den_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_2_pick_percent = st.slider("Detroit Lions Week 2 Pick Percent:", -1, 100) / 100
                det_week_2_pick_percent_percent = det_week_2_pick_percent*100
                if det_week_2_pick_percent < 0:
                    st.write(f"Detroit Lions Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 2 Pick Percent: :green[{det_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_2_pick_percent = st.slider("Green Bay Packers Week 2 Pick Percent:", -1, 100) / 100
                gb_week_2_pick_percent_percent = gb_week_2_pick_percent*100
                if gb_week_2_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 2 Pick Percent: :green[{gb_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_2_pick_percent = st.slider("Houston Texans Week 2 Pick Percent:", -1, 100) / 100
                hou_week_2_pick_percent_percent = hou_week_2_pick_percent*100
                if hou_week_2_pick_percent < 0:
                    st.write(f"Houston Texans Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 2 Pick Percent: :green[{hou_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_2_pick_percent = st.slider("Indianapoils Colts Week 2 Pick Percent:", -1, 100) / 100
                ind_week_2_pick_percent_percent = ind_week_2_pick_percent*100
                if ind_week_2_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 2 Pick Percent: :green[{ind_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_2_pick_percent = st.slider("Jacksonville Jaguars Week 2 Pick Percent:", -1, 100) / 100
                jax_week_2_pick_percent_percent = jax_week_2_pick_percent*100
                if jax_week_2_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 2 Pick Percent: :green[{jax_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_2_pick_percent = st.slider("Kansas City Chiefs Week 2 Pick Percent:", -1, 100) / 100
                kc_week_2_pick_percent_percent = kc_week_2_pick_percent*100
                if kc_week_2_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 2 Pick Percent: :green[{kc_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_2_pick_percent = st.slider("Las Vegas Raiders Week 2 Pick Percent:", -1, 100) / 100
                lv_week_2_pick_percent_percent = lv_week_2_pick_percent*100
                if lv_week_2_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 2 Pick Percent: :green[{lv_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_2_pick_percent = st.slider("Los Angeles Chargers Week 2 Pick Percent:", -1, 100) / 100
                lac_week_2_pick_percent_percent = lac_week_2_pick_percent*100
                if lac_week_2_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 2 Pick Percent: :green[{lac_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_2_pick_percent = st.slider("Los Angeles Rams Week 2 Pick Percent:", -1, 100) / 100
                lar_week_2_pick_percent_percent = lar_week_2_pick_percent*100
                if lar_week_2_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 2 Pick Percent: :green[{lar_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_2_pick_percent = st.slider("Miami Dolphins Week 2 Pick Percent:", -1, 100) / 100
                mia_week_2_pick_percent_percent = mia_week_2_pick_percent*100
                if mia_week_2_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 2 Pick Percent: :green[{mia_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_2_pick_percent = st.slider("Minnesota Vikings Week 2 Pick Percent:", -1, 100) / 100
                min_week_2_pick_percent_percent = min_week_2_pick_percent*100
                if min_week_2_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 2 Pick Percent: :green[{min_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_2_pick_percent = st.slider("New England Patriots Week 2 Pick Percent:", -1, 100) / 100
                ne_week_2_pick_percent_percent = ne_week_2_pick_percent*100
                if ne_week_2_pick_percent < 0:
                    st.write(f"New England Patriots Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 2 Pick Percent: :green[{ne_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_2_pick_percent = st.slider("New Orleans Saints Week 2 Pick Percent:", -1, 100) / 100
                no_week_2_pick_percent_percent = no_week_2_pick_percent*100
                if no_week_2_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 2 Pick Percent: :green[{no_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_2_pick_percent = st.slider("New York Giants Week 2 Pick Percent:", -1, 100) / 100
                nyg_week_2_pick_percent_percent = nyg_week_2_pick_percent*100
                if nyg_week_2_pick_percent < 0:
                    st.write(f"New York Giants Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 2 Pick Percent: :green[{nyg_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_2_pick_percent = st.slider("New York Jets Week 2 Pick Percent:", -1, 100) / 100
                nyj_week_2_pick_percent_percent = nyj_week_2_pick_percent*100
                if nyj_week_2_pick_percent < 0:
                    st.write(f"New York Jets Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 2 Pick Percent: :green[{nyj_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_2_pick_percent = st.slider("Philadelphia Eagles Week 2 Pick Percent:", -1, 100) / 100
                phi_week_2_pick_percent_percent = phi_week_2_pick_percent*100
                if phi_week_2_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 2 Pick Percent: :green[{phi_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_2_pick_percent = st.slider("Pittsburgh Steelers Week 2 Pick Percent:", -1, 100) / 100
                pit_week_2_pick_percent_percent = pit_week_2_pick_percent*100
                if pit_week_2_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 2 Pick Percent: :green[{pit_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_2_pick_percent = st.slider("San Francisco 49ers Week 2 Pick Percent:", -1, 100) / 100
                sf_week_2_pick_percent_percent = sf_week_2_pick_percent*100
                if sf_week_2_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 2 Pick Percent: :green[{sf_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_2_pick_percent = st.slider("Seattle Seahawks Week 2 Pick Percent:", -1, 100) / 100
                sea_week_2_pick_percent_percent = sea_week_2_pick_percent*100
                if sea_week_2_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 2 Pick Percent: :green[{sea_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_2_pick_percent = st.slider("Tampa Bay Buccaneers Week 2 Pick Percent:", -1, 100) / 100
                tb_week_2_pick_percent_percent = tb_week_2_pick_percent*100
                if tb_week_2_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 2 Pick Percent: :green[{tb_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_2_pick_percent = st.slider("Tennessee Titans Week 2 Pick Percent:", -1, 100) / 100
                ten_week_2_pick_percent_percent = ten_week_2_pick_percent*100
                if ten_week_2_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 2 Pick Percent: :green[{ten_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_2_pick_percent = st.slider("Washington Commanders Week 2 Pick Percent:", -1, 100) / 100
                was_week_2_pick_percent_percent = was_week_2_pick_percent*100
                if was_week_2_pick_percent < 0:
                    st.write(f"Washington Commanders Week 2 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 2 Pick Percent: :green[{was_week_2_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 3 and ending_week > 3:
            week_3_pick_percents = st.checkbox('Add Week 3 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 3.")
            if week_3_pick_percents:
                st.write('')
                st.subheader('Week 3 Estimated Pick Percentages')
                st.write('')
                az_week_3_pick_percent = st.slider("Arizona Cardinals Week 3 Pick Percent:", -1, 100) / 100
                az_week_3_pick_percent_percent = az_week_3_pick_percent*100
                if az_week_3_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 3 Pick Percent: :green[{az_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_3_pick_percent = st.slider("Atlanta Falcons Week 3 Pick Percent:", -1, 100) / 100
                atl_week_3_pick_percent_percent = atl_week_3_pick_percent*100
                if atl_week_3_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 3 Pick Percent: :green[{atl_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_3_pick_percent = st.slider("Baltimore Ravens Week 3 Pick Percent:", -1, 100) / 100
                bal_week_3_pick_percent_percent = bal_week_3_pick_percent*100
                if bal_week_3_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 3 Pick Percent: :green[{bal_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_3_pick_percent = st.slider("Buffalo Bills Week 3 Pick Percent:", -1, 100) / 100
                buf_week_3_pick_percent_percent = buf_week_3_pick_percent*100
                if buf_week_3_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 3 Pick Percent: :green[{buf_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_3_pick_percent = st.slider("Carolina Panthers Week 3 Pick Percent:", -1, 100) / 100
                car_week_3_pick_percent_percent = car_week_3_pick_percent*100
                if car_week_3_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 3 Pick Percent: :green[{car_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_3_pick_percent = st.slider("Chicago Bears Week 3 Pick Percent:", -1, 100) / 100
                chi_week_3_pick_percent_percent = chi_week_3_pick_percent*100
                if chi_week_3_pick_percent < 0:
                    st.write(f"Chicago Bears Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 3 Pick Percent: :green[{chi_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_3_pick_percent = st.slider("Cincinnati Bengals Week 3 Pick Percent:", -1, 100) / 100
                cin_week_3_pick_percent_percent = cin_week_3_pick_percent*100
                if cin_week_3_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 3 Pick Percent: :green[{cin_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_3_pick_percent = st.slider("Cleveland Browns Week 3 Pick Percent:", -1, 100) / 100
                cle_week_3_pick_percent_percent = cle_week_3_pick_percent*100
                if cle_week_3_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 3 Pick Percent: :green[{cle_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_3_pick_percent = st.slider("Dallas Cowboys Week 3 Pick Percent:", -1, 100) / 100
                dal_week_3_pick_percent_percent = dal_week_3_pick_percent*100
                if dal_week_3_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 3 Pick Percent: :green[{dal_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_3_pick_percent = st.slider("Denver Broncos Week 3 Pick Percent:", -1, 100) / 100
                den_week_3_pick_percent_percent = den_week_3_pick_percent*100
                if den_week_3_pick_percent < 0:
                    st.write(f"Denver Broncos Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 3 Pick Percent: :green[{den_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_3_pick_percent = st.slider("Detroit Lions Week 3 Pick Percent:", -1, 100) / 100
                det_week_3_pick_percent_percent = det_week_3_pick_percent*100
                if det_week_3_pick_percent < 0:
                    st.write(f"Detroit Lions Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 3 Pick Percent: :green[{det_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_3_pick_percent = st.slider("Green Bay Packers Week 3 Pick Percent:", -1, 100) / 100
                gb_week_3_pick_percent_percent = gb_week_3_pick_percent*100
                if gb_week_3_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 3 Pick Percent: :green[{gb_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_3_pick_percent = st.slider("Houston Texans Week 3 Pick Percent:", -1, 100) / 100
                hou_week_3_pick_percent_percent = hou_week_3_pick_percent*100
                if hou_week_3_pick_percent < 0:
                    st.write(f"Houston Texans Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 3 Pick Percent: :green[{hou_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_3_pick_percent = st.slider("Indianapoils Colts Week 3 Pick Percent:", -1, 100) / 100
                ind_week_3_pick_percent_percent = ind_week_3_pick_percent*100
                if ind_week_3_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 3 Pick Percent: :green[{ind_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_3_pick_percent = st.slider("Jacksonville Jaguars Week 3 Pick Percent:", -1, 100) / 100
                jax_week_3_pick_percent_percent = jax_week_3_pick_percent*100
                if jax_week_3_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 3 Pick Percent: :green[{jax_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_3_pick_percent = st.slider("Kansas City Chiefs Week 3 Pick Percent:", -1, 100) / 100
                kc_week_3_pick_percent_percent = kc_week_3_pick_percent*100
                if kc_week_3_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 3 Pick Percent: :green[{kc_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_3_pick_percent = st.slider("Las Vegas Raiders Week 3 Pick Percent:", -1, 100) / 100
                lv_week_3_pick_percent_percent = lv_week_3_pick_percent*100
                if lv_week_3_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 3 Pick Percent: :green[{lv_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_3_pick_percent = st.slider("Los Angeles Chargers Week 3 Pick Percent:", -1, 100) / 100
                lac_week_3_pick_percent_percent = lac_week_3_pick_percent*100
                if lac_week_3_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 3 Pick Percent: :green[{lac_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_3_pick_percent = st.slider("Los Angeles Rams Week 3 Pick Percent:", -1, 100) / 100
                lar_week_3_pick_percent_percent = lar_week_3_pick_percent*100
                if lar_week_3_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 3 Pick Percent: :green[{lar_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_3_pick_percent = st.slider("Miami Dolphins Week 3 Pick Percent:", -1, 100) / 100
                mia_week_3_pick_percent_percent = mia_week_3_pick_percent*100
                if mia_week_3_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 3 Pick Percent: :green[{mia_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_3_pick_percent = st.slider("Minnesota Vikings Week 3 Pick Percent:", -1, 100) / 100
                min_week_3_pick_percent_percent = min_week_3_pick_percent*100
                if min_week_3_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 3 Pick Percent: :green[{min_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_3_pick_percent = st.slider("New England Patriots Week 3 Pick Percent:", -1, 100) / 100
                ne_week_3_pick_percent_percent = ne_week_3_pick_percent*100
                if ne_week_3_pick_percent < 0:
                    st.write(f"New England Patriots Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 3 Pick Percent: :green[{ne_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_3_pick_percent = st.slider("New Orleans Saints Week 3 Pick Percent:", -1, 100) / 100
                no_week_3_pick_percent_percent = no_week_3_pick_percent*100
                if no_week_3_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 3 Pick Percent: :green[{no_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_3_pick_percent = st.slider("New York Giants Week 3 Pick Percent:", -1, 100) / 100
                nyg_week_3_pick_percent_percent = nyg_week_3_pick_percent*100
                if nyg_week_3_pick_percent < 0:
                    st.write(f"New York Giants Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 3 Pick Percent: :green[{nyg_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_3_pick_percent = st.slider("New York Jets Week 3 Pick Percent:", -1, 100) / 100
                nyj_week_3_pick_percent_percent = nyj_week_3_pick_percent*100
                if nyj_week_3_pick_percent < 0:
                    st.write(f"New York Jets Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 3 Pick Percent: :green[{nyj_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_3_pick_percent = st.slider("Philadelphia Eagles Week 3 Pick Percent:", -1, 100) / 100
                phi_week_3_pick_percent_percent = phi_week_3_pick_percent*100
                if phi_week_3_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 3 Pick Percent: :green[{phi_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_3_pick_percent = st.slider("Pittsburgh Steelers Week 3 Pick Percent:", -1, 100) / 100
                pit_week_3_pick_percent_percent = pit_week_3_pick_percent*100
                if pit_week_3_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 3 Pick Percent: :green[{pit_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_3_pick_percent = st.slider("San Francisco 49ers Week 3 Pick Percent:", -1, 100) / 100
                sf_week_3_pick_percent_percent = sf_week_3_pick_percent*100
                if sf_week_3_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 3 Pick Percent: :green[{sf_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_3_pick_percent = st.slider("Seattle Seahawks Week 3 Pick Percent:", -1, 100) / 100
                sea_week_3_pick_percent_percent = sea_week_3_pick_percent*100
                if sea_week_3_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 3 Pick Percent: :green[{sea_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_3_pick_percent = st.slider("Tampa Bay Buccaneers Week 3 Pick Percent:", -1, 100) / 100
                tb_week_3_pick_percent_percent = tb_week_3_pick_percent*100
                if tb_week_3_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 3 Pick Percent: :green[{tb_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_3_pick_percent = st.slider("Tennessee Titans Week 3 Pick Percent:", -1, 100) / 100
                ten_week_3_pick_percent_percent = ten_week_3_pick_percent*100
                if ten_week_3_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 3 Pick Percent: :green[{ten_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_3_pick_percent = st.slider("Washington Commanders Week 3 Pick Percent:", -1, 100) / 100
                was_week_3_pick_percent_percent = was_week_3_pick_percent*100
                if was_week_3_pick_percent < 0:
                    st.write(f"Washington Commanders Week 3 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 3 Pick Percent: :green[{was_week_3_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 4 and ending_week > 4:
            week_4_pick_percents = st.checkbox('Add Week 4 Pick Percentages?')
            if week_4_pick_percents:
                st.write('')
                st.subheader('Week 4 Estimated Pick Percentages', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 4.")
                st.write('')
                az_week_4_pick_percent = st.slider("Arizona Cardinals Week 4 Pick Percent:", -1, 100) / 100
                az_week_4_pick_percent_percent = az_week_4_pick_percent*100
                if az_week_4_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 4 Pick Percent: :green[{az_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_4_pick_percent = st.slider("Atlanta Falcons Week 4 Pick Percent:", -1, 100) / 100
                atl_week_4_pick_percent_percent = atl_week_4_pick_percent*100
                if atl_week_4_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 4 Pick Percent: :green[{atl_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_4_pick_percent = st.slider("Baltimore Ravens Week 4 Pick Percent:", -1, 100) / 100
                bal_week_4_pick_percent_percent = bal_week_4_pick_percent*100
                if bal_week_4_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 4 Pick Percent: :green[{bal_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_4_pick_percent = st.slider("Buffalo Bills Week 4 Pick Percent:", -1, 100) / 100
                buf_week_4_pick_percent_percent = buf_week_4_pick_percent*100
                if buf_week_4_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 4 Pick Percent: :green[{buf_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_4_pick_percent = st.slider("Carolina Panthers Week 4 Pick Percent:", -1, 100) / 100
                car_week_4_pick_percent_percent = car_week_4_pick_percent*100
                if car_week_4_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 4 Pick Percent: :green[{car_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_4_pick_percent = st.slider("Chicago Bears Week 4 Pick Percent:", -1, 100) / 100
                chi_week_4_pick_percent_percent = chi_week_4_pick_percent*100
                if chi_week_4_pick_percent < 0:
                    st.write(f"Chicago Bears Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 4 Pick Percent: :green[{chi_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_4_pick_percent = st.slider("Cincinnati Bengals Week 4 Pick Percent:", -1, 100) / 100
                cin_week_4_pick_percent_percent = cin_week_4_pick_percent*100
                if cin_week_4_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 4 Pick Percent: :green[{cin_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_4_pick_percent = st.slider("Cleveland Browns Week 4 Pick Percent:", -1, 100) / 100
                cle_week_4_pick_percent_percent = cle_week_4_pick_percent*100
                if cle_week_4_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 4 Pick Percent: :green[{cle_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_4_pick_percent = st.slider("Dallas Cowboys Week 4 Pick Percent:", -1, 100) / 100
                dal_week_4_pick_percent_percent = dal_week_4_pick_percent*100
                if dal_week_4_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 4 Pick Percent: :green[{dal_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_4_pick_percent = st.slider("Denver Broncos Week 4 Pick Percent:", -1, 100) / 100
                den_week_4_pick_percent_percent = den_week_4_pick_percent*100
                if den_week_4_pick_percent < 0:
                    st.write(f"Denver Broncos Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 4 Pick Percent: :green[{den_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_4_pick_percent = st.slider("Detroit Lions Week 4 Pick Percent:", -1, 100) / 100
                det_week_4_pick_percent_percent = det_week_4_pick_percent*100
                if det_week_4_pick_percent < 0:
                    st.write(f"Detroit Lions Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 4 Pick Percent: :green[{det_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_4_pick_percent = st.slider("Green Bay Packers Week 4 Pick Percent:", -1, 100) / 100
                gb_week_4_pick_percent_percent = gb_week_4_pick_percent*100
                if gb_week_4_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 4 Pick Percent: :green[{gb_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_4_pick_percent = st.slider("Houston Texans Week 4 Pick Percent:", -1, 100) / 100
                hou_week_4_pick_percent_percent = hou_week_4_pick_percent*100
                if hou_week_4_pick_percent < 0:
                    st.write(f"Houston Texans Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 4 Pick Percent: :green[{hou_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_4_pick_percent = st.slider("Indianapoils Colts Week 4 Pick Percent:", -1, 100) / 100
                ind_week_4_pick_percent_percent = ind_week_4_pick_percent*100
                if ind_week_4_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 4 Pick Percent: :green[{ind_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_4_pick_percent = st.slider("Jacksonville Jaguars Week 4 Pick Percent:", -1, 100) / 100
                jax_week_4_pick_percent_percent = jax_week_4_pick_percent*100
                if jax_week_4_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 4 Pick Percent: :green[{jax_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_4_pick_percent = st.slider("Kansas City Chiefs Week 4 Pick Percent:", -1, 100) / 100
                kc_week_4_pick_percent_percent = kc_week_4_pick_percent*100
                if kc_week_4_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 4 Pick Percent: :green[{kc_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_4_pick_percent = st.slider("Las Vegas Raiders Week 4 Pick Percent:", -1, 100) / 100
                lv_week_4_pick_percent_percent = lv_week_4_pick_percent*100
                if lv_week_4_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 4 Pick Percent: :green[{lv_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_4_pick_percent = st.slider("Los Angeles Chargers Week 4 Pick Percent:", -1, 100) / 100
                lac_week_4_pick_percent_percent = lac_week_4_pick_percent*100
                if lac_week_4_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 4 Pick Percent: :green[{lac_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_4_pick_percent = st.slider("Los Angeles Rams Week 4 Pick Percent:", -1, 100) / 100
                lar_week_4_pick_percent_percent = lar_week_4_pick_percent*100
                if lar_week_4_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 4 Pick Percent: :green[{lar_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_4_pick_percent = st.slider("Miami Dolphins Week 4 Pick Percent:", -1, 100) / 100
                mia_week_4_pick_percent_percent = mia_week_4_pick_percent*100
                if mia_week_4_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 4 Pick Percent: :green[{mia_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_4_pick_percent = st.slider("Minnesota Vikings Week 4 Pick Percent:", -1, 100) / 100
                min_week_4_pick_percent_percent = min_week_4_pick_percent*100
                if min_week_4_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 4 Pick Percent: :green[{min_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_4_pick_percent = st.slider("New England Patriots Week 4 Pick Percent:", -1, 100) / 100
                ne_week_4_pick_percent_percent = ne_week_4_pick_percent*100
                if ne_week_4_pick_percent < 0:
                    st.write(f"New England Patriots Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 4 Pick Percent: :green[{ne_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_4_pick_percent = st.slider("New Orleans Saints Week 4 Pick Percent:", -1, 100) / 100
                no_week_4_pick_percent_percent = no_week_4_pick_percent*100
                if no_week_4_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 4 Pick Percent: :green[{no_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_4_pick_percent = st.slider("New York Giants Week 4 Pick Percent:", -1, 100) / 100
                nyg_week_4_pick_percent_percent = nyg_week_4_pick_percent*100
                if nyg_week_4_pick_percent < 0:
                    st.write(f"New York Giants Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 4 Pick Percent: :green[{nyg_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_4_pick_percent = st.slider("New York Jets Week 4 Pick Percent:", -1, 100) / 100
                nyj_week_4_pick_percent_percent = nyj_week_4_pick_percent*100
                if nyj_week_4_pick_percent < 0:
                    st.write(f"New York Jets Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 4 Pick Percent: :green[{nyj_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_4_pick_percent = st.slider("Philadelphia Eagles Week 4 Pick Percent:", -1, 100) / 100
                phi_week_4_pick_percent_percent = phi_week_4_pick_percent*100
                if phi_week_4_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 4 Pick Percent: :green[{phi_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_4_pick_percent = st.slider("Pittsburgh Steelers Week 4 Pick Percent:", -1, 100) / 100
                pit_week_4_pick_percent_percent = pit_week_4_pick_percent*100
                if pit_week_4_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 4 Pick Percent: :green[{pit_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_4_pick_percent = st.slider("San Francisco 49ers Week 4 Pick Percent:", -1, 100) / 100
                sf_week_4_pick_percent_percent = sf_week_4_pick_percent*100
                if sf_week_4_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 4 Pick Percent: :green[{sf_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_4_pick_percent = st.slider("Seattle Seahawks Week 4 Pick Percent:", -1, 100) / 100
                sea_week_4_pick_percent_percent = sea_week_4_pick_percent*100
                if sea_week_4_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 4 Pick Percent: :green[{sea_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_4_pick_percent = st.slider("Tampa Bay Buccaneers Week 4 Pick Percent:", -1, 100) / 100
                tb_week_4_pick_percent_percent = tb_week_4_pick_percent*100
                if tb_week_4_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 4 Pick Percent: :green[{tb_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_4_pick_percent = st.slider("Tennessee Titans Week 4 Pick Percent:", -1, 100) / 100
                ten_week_4_pick_percent_percent = ten_week_4_pick_percent*100
                if ten_week_4_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 4 Pick Percent: :green[{ten_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_4_pick_percent = st.slider("Washington Commanders Week 4 Pick Percent:", -1, 100) / 100
                was_week_4_pick_percent_percent = was_week_4_pick_percent*100
                if was_week_4_pick_percent < 0:
                    st.write(f"Washington Commanders Week 4 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 4 Pick Percent: :green[{was_week_4_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 5 and ending_week > 5:
            week_5_pick_percents = st.checkbox('Add Week 5 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 5.")
            if week_5_pick_percents:
                st.write('')
                st.subheader('Week 5 Estimated Pick Percentages')
                st.write('')
                az_week_5_pick_percent = st.slider("Arizona Cardinals Week 5 Pick Percent:", -1, 100) / 100
                az_week_5_pick_percent_percent = az_week_5_pick_percent*100
                if az_week_5_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 5 Pick Percent: :green[{az_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_5_pick_percent = st.slider("Atlanta Falcons Week 5 Pick Percent:", -1, 100) / 100
                atl_week_5_pick_percent_percent = atl_week_5_pick_percent*100
                if atl_week_5_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 5 Pick Percent: :green[{atl_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_5_pick_percent = st.slider("Baltimore Ravens Week 5 Pick Percent:", -1, 100) / 100
                bal_week_5_pick_percent_percent = bal_week_5_pick_percent*100
                if bal_week_5_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 5 Pick Percent: :green[{bal_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_5_pick_percent = st.slider("Buffalo Bills Week 5 Pick Percent:", -1, 100) / 100
                buf_week_5_pick_percent_percent = buf_week_5_pick_percent*100
                if buf_week_5_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 5 Pick Percent: :green[{buf_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_5_pick_percent = st.slider("Carolina Panthers Week 5 Pick Percent:", -1, 100) / 100
                car_week_5_pick_percent_percent = car_week_5_pick_percent*100
                if car_week_5_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 5 Pick Percent: :green[{car_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_5_pick_percent = st.slider("Chicago Bears Week 5 Pick Percent:", -1, 100) / 100
                chi_week_5_pick_percent_percent = chi_week_5_pick_percent*100
                if chi_week_5_pick_percent < 0:
                    st.write(f"Chicago Bears Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 5 Pick Percent: :green[{chi_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_5_pick_percent = st.slider("Cincinnati Bengals Week 5 Pick Percent:", -1, 100) / 100
                cin_week_5_pick_percent_percent = cin_week_5_pick_percent*100
                if cin_week_5_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 5 Pick Percent: :green[{cin_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_5_pick_percent = st.slider("Cleveland Browns Week 5 Pick Percent:", -1, 100) / 100
                cle_week_5_pick_percent_percent = cle_week_5_pick_percent*100
                if cle_week_5_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 5 Pick Percent: :green[{cle_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_5_pick_percent = st.slider("Dallas Cowboys Week 5 Pick Percent:", -1, 100) / 100
                dal_week_5_pick_percent_percent = dal_week_5_pick_percent*100
                if dal_week_5_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 5 Pick Percent: :green[{dal_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_5_pick_percent = st.slider("Denver Broncos Week 5 Pick Percent:", -1, 100) / 100
                den_week_5_pick_percent_percent = den_week_5_pick_percent*100
                if den_week_5_pick_percent < 0:
                    st.write(f"Denver Broncos Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 5 Pick Percent: :green[{den_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_5_pick_percent = st.slider("Detroit Lions Week 5 Pick Percent:", -1, 100) / 100
                det_week_5_pick_percent_percent = det_week_5_pick_percent*100
                if det_week_5_pick_percent < 0:
                    st.write(f"Detroit Lions Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 5 Pick Percent: :green[{det_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_5_pick_percent = st.slider("Green Bay Packers Week 5 Pick Percent:", -1, 100) / 100
                gb_week_5_pick_percent_percent = gb_week_5_pick_percent*100
                if gb_week_5_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 5 Pick Percent: :green[{gb_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_5_pick_percent = st.slider("Houston Texans Week 5 Pick Percent:", -1, 100) / 100
                hou_week_5_pick_percent_percent = hou_week_5_pick_percent*100
                if hou_week_5_pick_percent < 0:
                    st.write(f"Houston Texans Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 5 Pick Percent: :green[{hou_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_5_pick_percent = st.slider("Indianapoils Colts Week 5 Pick Percent:", -1, 100) / 100
                ind_week_5_pick_percent_percent = ind_week_5_pick_percent*100
                if ind_week_5_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 5 Pick Percent: :green[{ind_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_5_pick_percent = st.slider("Jacksonville Jaguars Week 5 Pick Percent:", -1, 100) / 100
                jax_week_5_pick_percent_percent = jax_week_5_pick_percent*100
                if jax_week_5_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 5 Pick Percent: :green[{jax_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_5_pick_percent = st.slider("Kansas City Chiefs Week 5 Pick Percent:", -1, 100) / 100
                kc_week_5_pick_percent_percent = kc_week_5_pick_percent*100
                if kc_week_5_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 5 Pick Percent: :green[{kc_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_5_pick_percent = st.slider("Las Vegas Raiders Week 5 Pick Percent:", -1, 100) / 100
                lv_week_5_pick_percent_percent = lv_week_5_pick_percent*100
                if lv_week_5_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 5 Pick Percent: :green[{lv_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_5_pick_percent = st.slider("Los Angeles Chargers Week 5 Pick Percent:", -1, 100) / 100
                lac_week_5_pick_percent_percent = lac_week_5_pick_percent*100
                if lac_week_5_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 5 Pick Percent: :green[{lac_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_5_pick_percent = st.slider("Los Angeles Rams Week 5 Pick Percent:", -1, 100) / 100
                lar_week_5_pick_percent_percent = lar_week_5_pick_percent*100
                if lar_week_5_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 5 Pick Percent: :green[{lar_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_5_pick_percent = st.slider("Miami Dolphins Week 5 Pick Percent:", -1, 100) / 100
                mia_week_5_pick_percent_percent = mia_week_5_pick_percent*100
                if mia_week_5_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 5 Pick Percent: :green[{mia_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_5_pick_percent = st.slider("Minnesota Vikings Week 5 Pick Percent:", -1, 100) / 100
                min_week_5_pick_percent_percent = min_week_5_pick_percent*100
                if min_week_5_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 5 Pick Percent: :green[{min_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_5_pick_percent = st.slider("New England Patriots Week 5 Pick Percent:", -1, 100) / 100
                ne_week_5_pick_percent_percent = ne_week_5_pick_percent*100
                if ne_week_5_pick_percent < 0:
                    st.write(f"New England Patriots Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 5 Pick Percent: :green[{ne_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_5_pick_percent = st.slider("New Orleans Saints Week 5 Pick Percent:", -1, 100) / 100
                no_week_5_pick_percent_percent = no_week_5_pick_percent*100
                if no_week_5_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 5 Pick Percent: :green[{no_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_5_pick_percent = st.slider("New York Giants Week 5 Pick Percent:", -1, 100) / 100
                nyg_week_5_pick_percent_percent = nyg_week_5_pick_percent*100
                if nyg_week_5_pick_percent < 0:
                    st.write(f"New York Giants Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 5 Pick Percent: :green[{nyg_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_5_pick_percent = st.slider("New York Jets Week 5 Pick Percent:", -1, 100) / 100
                nyj_week_5_pick_percent_percent = nyj_week_5_pick_percent*100
                if nyj_week_5_pick_percent < 0:
                    st.write(f"New York Jets Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 5 Pick Percent: :green[{nyj_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_5_pick_percent = st.slider("Philadelphia Eagles Week 5 Pick Percent:", -1, 100) / 100
                phi_week_5_pick_percent_percent = phi_week_5_pick_percent*100
                if phi_week_5_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 5 Pick Percent: :green[{phi_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_5_pick_percent = st.slider("Pittsburgh Steelers Week 5 Pick Percent:", -1, 100) / 100
                pit_week_5_pick_percent_percent = pit_week_5_pick_percent*100
                if pit_week_5_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 5 Pick Percent: :green[{pit_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_5_pick_percent = st.slider("San Francisco 49ers Week 5 Pick Percent:", -1, 100) / 100
                sf_week_5_pick_percent_percent = sf_week_5_pick_percent*100
                if sf_week_5_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 5 Pick Percent: :green[{sf_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_5_pick_percent = st.slider("Seattle Seahawks Week 5 Pick Percent:", -1, 100) / 100
                sea_week_5_pick_percent_percent = sea_week_5_pick_percent*100
                if sea_week_5_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 5 Pick Percent: :green[{sea_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_5_pick_percent = st.slider("Tampa Bay Buccaneers Week 5 Pick Percent:", -1, 100) / 100
                tb_week_5_pick_percent_percent = tb_week_5_pick_percent*100
                if tb_week_5_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 5 Pick Percent: :green[{tb_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_5_pick_percent = st.slider("Tennessee Titans Week 5 Pick Percent:", -1, 100) / 100
                ten_week_5_pick_percent_percent = ten_week_5_pick_percent*100
                if ten_week_5_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 5 Pick Percent: :green[{ten_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_5_pick_percent = st.slider("Washington Commanders Week 5 Pick Percent:", -1, 100) / 100
                was_week_5_pick_percent_percent = was_week_5_pick_percent*100
                if was_week_5_pick_percent < 0:
                    st.write(f"Washington Commanders Week 5 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 5 Pick Percent: :green[{was_week_5_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 6 and ending_week > 6:
            week_6_pick_percents = st.checkbox('Add Week 6 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 6.")
            if week_6_pick_percents:
                st.write('')
                st.subheader('Week 6 Estimated Pick Percentages')
                st.write('')
                az_week_6_pick_percent = st.slider("Arizona Cardinals Week 6 Pick Percent:", -1, 100) / 100
                az_week_6_pick_percent_percent = az_week_6_pick_percent*100
                if az_week_6_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 6 Pick Percent: :green[{az_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_6_pick_percent = st.slider("Atlanta Falcons Week 6 Pick Percent:", -1, 100) / 100
                atl_week_6_pick_percent_percent = atl_week_6_pick_percent*100
                if atl_week_6_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 6 Pick Percent: :green[{atl_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_6_pick_percent = st.slider("Baltimore Ravens Week 6 Pick Percent:", -1, 100) / 100
                bal_week_6_pick_percent_percent = bal_week_6_pick_percent*100
                if bal_week_6_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 6 Pick Percent: :green[{bal_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_6_pick_percent = st.slider("Buffalo Bills Week 6 Pick Percent:", -1, 100) / 100
                buf_week_6_pick_percent_percent = buf_week_6_pick_percent*100
                if buf_week_6_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 6 Pick Percent: :green[{buf_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_6_pick_percent = st.slider("Carolina Panthers Week 6 Pick Percent:", -1, 100) / 100
                car_week_6_pick_percent_percent = car_week_6_pick_percent*100
                if car_week_6_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 6 Pick Percent: :green[{car_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_6_pick_percent = st.slider("Chicago Bears Week 6 Pick Percent:", -1, 100) / 100
                chi_week_6_pick_percent_percent = chi_week_6_pick_percent*100
                if chi_week_6_pick_percent < 0:
                    st.write(f"Chicago Bears Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 6 Pick Percent: :green[{chi_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_6_pick_percent = st.slider("Cincinnati Bengals Week 6 Pick Percent:", -1, 100) / 100
                cin_week_6_pick_percent_percent = cin_week_6_pick_percent*100
                if cin_week_6_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 6 Pick Percent: :green[{cin_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_6_pick_percent = st.slider("Cleveland Browns Week 6 Pick Percent:", -1, 100) / 100
                cle_week_6_pick_percent_percent = cle_week_6_pick_percent*100
                if cle_week_6_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 6 Pick Percent: :green[{cle_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_6_pick_percent = st.slider("Dallas Cowboys Week 6 Pick Percent:", -1, 100) / 100
                dal_week_6_pick_percent_percent = dal_week_6_pick_percent*100
                if dal_week_6_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 6 Pick Percent: :green[{dal_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_6_pick_percent = st.slider("Denver Broncos Week 6 Pick Percent:", -1, 100) / 100
                den_week_6_pick_percent_percent = den_week_6_pick_percent*100
                if den_week_6_pick_percent < 0:
                    st.write(f"Denver Broncos Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 6 Pick Percent: :green[{den_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_6_pick_percent = st.slider("Detroit Lions Week 6 Pick Percent:", -1, 100) / 100
                det_week_6_pick_percent_percent = det_week_6_pick_percent*100
                if det_week_6_pick_percent < 0:
                    st.write(f"Detroit Lions Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 6 Pick Percent: :green[{det_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_6_pick_percent = st.slider("Green Bay Packers Week 6 Pick Percent:", -1, 100) / 100
                gb_week_6_pick_percent_percent = gb_week_6_pick_percent*100
                if gb_week_6_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 6 Pick Percent: :green[{gb_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_6_pick_percent = st.slider("Houston Texans Week 6 Pick Percent:", -1, 100) / 100
                hou_week_6_pick_percent_percent = hou_week_6_pick_percent*100
                if hou_week_6_pick_percent < 0:
                    st.write(f"Houston Texans Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 6 Pick Percent: :green[{hou_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_6_pick_percent = st.slider("Indianapoils Colts Week 6 Pick Percent:", -1, 100) / 100
                ind_week_6_pick_percent_percent = ind_week_6_pick_percent*100
                if ind_week_6_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 6 Pick Percent: :green[{ind_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_6_pick_percent = st.slider("Jacksonville Jaguars Week 6 Pick Percent:", -1, 100) / 100
                jax_week_6_pick_percent_percent = jax_week_6_pick_percent*100
                if jax_week_6_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 6 Pick Percent: :green[{jax_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_6_pick_percent = st.slider("Kansas City Chiefs Week 6 Pick Percent:", -1, 100) / 100
                kc_week_6_pick_percent_percent = kc_week_6_pick_percent*100
                if kc_week_6_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 6 Pick Percent: :green[{kc_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_6_pick_percent = st.slider("Las Vegas Raiders Week 6 Pick Percent:", -1, 100) / 100
                lv_week_6_pick_percent_percent = lv_week_6_pick_percent*100
                if lv_week_6_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 6 Pick Percent: :green[{lv_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_6_pick_percent = st.slider("Los Angeles Chargers Week 6 Pick Percent:", -1, 100) / 100
                lac_week_6_pick_percent_percent = lac_week_6_pick_percent*100
                if lac_week_6_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 6 Pick Percent: :green[{lac_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_6_pick_percent = st.slider("Los Angeles Rams Week 6 Pick Percent:", -1, 100) / 100
                lar_week_6_pick_percent_percent = lar_week_6_pick_percent*100
                if lar_week_6_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 6 Pick Percent: :green[{lar_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_6_pick_percent = st.slider("Miami Dolphins Week 6 Pick Percent:", -1, 100) / 100
                mia_week_6_pick_percent_percent = mia_week_6_pick_percent*100
                if mia_week_6_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 6 Pick Percent: :green[{mia_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_6_pick_percent = st.slider("Minnesota Vikings Week 6 Pick Percent:", -1, 100) / 100
                min_week_6_pick_percent_percent = min_week_6_pick_percent*100
                if min_week_6_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 6 Pick Percent: :green[{min_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_6_pick_percent = st.slider("New England Patriots Week 6 Pick Percent:", -1, 100) / 100
                ne_week_6_pick_percent_percent = ne_week_6_pick_percent*100
                if ne_week_6_pick_percent < 0:
                    st.write(f"New England Patriots Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 6 Pick Percent: :green[{ne_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_6_pick_percent = st.slider("New Orleans Saints Week 6 Pick Percent:", -1, 100) / 100
                no_week_6_pick_percent_percent = no_week_6_pick_percent*100
                if no_week_6_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 6 Pick Percent: :green[{no_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_6_pick_percent = st.slider("New York Giants Week 6 Pick Percent:", -1, 100) / 100
                nyg_week_6_pick_percent_percent = nyg_week_6_pick_percent*100
                if nyg_week_6_pick_percent < 0:
                    st.write(f"New York Giants Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 6 Pick Percent: :green[{nyg_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_6_pick_percent = st.slider("New York Jets Week 6 Pick Percent:", -1, 100) / 100
                nyj_week_6_pick_percent_percent = nyj_week_6_pick_percent*100
                if nyj_week_6_pick_percent < 0:
                    st.write(f"New York Jets Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 6 Pick Percent: :green[{nyj_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_6_pick_percent = st.slider("Philadelphia Eagles Week 6 Pick Percent:", -1, 100) / 100
                phi_week_6_pick_percent_percent = phi_week_6_pick_percent*100
                if phi_week_6_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 6 Pick Percent: :green[{phi_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_6_pick_percent = st.slider("Pittsburgh Steelers Week 6 Pick Percent:", -1, 100) / 100
                pit_week_6_pick_percent_percent = pit_week_6_pick_percent*100
                if pit_week_6_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 6 Pick Percent: :green[{pit_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_6_pick_percent = st.slider("San Francisco 49ers Week 6 Pick Percent:", -1, 100) / 100
                sf_week_6_pick_percent_percent = sf_week_6_pick_percent*100
                if sf_week_6_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 6 Pick Percent: :green[{sf_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_6_pick_percent = st.slider("Seattle Seahawks Week 6 Pick Percent:", -1, 100) / 100
                sea_week_6_pick_percent_percent = sea_week_6_pick_percent*100
                if sea_week_6_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 6 Pick Percent: :green[{sea_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_6_pick_percent = st.slider("Tampa Bay Buccaneers Week 6 Pick Percent:", -1, 100) / 100
                tb_week_6_pick_percent_percent = tb_week_6_pick_percent*100
                if tb_week_6_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 6 Pick Percent: :green[{tb_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_6_pick_percent = st.slider("Tennessee Titans Week 6 Pick Percent:", -1, 100) / 100
                ten_week_6_pick_percent_percent = ten_week_6_pick_percent*100
                if ten_week_6_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 6 Pick Percent: :green[{ten_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_6_pick_percent = st.slider("Washington Commanders Week 6 Pick Percent:", -1, 100) / 100
                was_week_6_pick_percent_percent = was_week_6_pick_percent*100
                if was_week_6_pick_percent < 0:
                    st.write(f"Washington Commanders Week 6 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 6 Pick Percent: :green[{was_week_6_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 7 and ending_week > 7:
            week_7_pick_percents = st.checkbox('Add Week 7 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 7.")
            if week_7_pick_percents:
                st.write('')
                st.subheader('Week 7 Estimated Pick Percentages')
                st.write('')
                az_week_7_pick_percent = st.slider("Arizona Cardinals Week 7 Pick Percent:", -1, 100) / 100
                az_week_7_pick_percent_percent = az_week_7_pick_percent*100
                if az_week_7_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 7 Pick Percent: :green[{az_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_7_pick_percent = st.slider("Atlanta Falcons Week 7 Pick Percent:", -1, 100) / 100
                atl_week_7_pick_percent_percent = atl_week_7_pick_percent*100
                if atl_week_7_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 7 Pick Percent: :green[{atl_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_7_pick_percent = st.slider("Baltimore Ravens Week 7 Pick Percent:", -1, 100) / 100
                bal_week_7_pick_percent_percent = bal_week_7_pick_percent*100
                if bal_week_7_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 7 Pick Percent: :green[{bal_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_7_pick_percent = st.slider("Buffalo Bills Week 7 Pick Percent:", -1, 100) / 100
                buf_week_7_pick_percent_percent = buf_week_7_pick_percent*100
                if buf_week_7_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 7 Pick Percent: :green[{buf_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_7_pick_percent = st.slider("Carolina Panthers Week 7 Pick Percent:", -1, 100) / 100
                car_week_7_pick_percent_percent = car_week_7_pick_percent*100
                if car_week_7_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 7 Pick Percent: :green[{car_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_7_pick_percent = st.slider("Chicago Bears Week 7 Pick Percent:", -1, 100) / 100
                chi_week_7_pick_percent_percent = chi_week_7_pick_percent*100
                if chi_week_7_pick_percent < 0:
                    st.write(f"Chicago Bears Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 7 Pick Percent: :green[{chi_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_7_pick_percent = st.slider("Cincinnati Bengals Week 7 Pick Percent:", -1, 100) / 100
                cin_week_7_pick_percent_percent = cin_week_7_pick_percent*100
                if cin_week_7_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 7 Pick Percent: :green[{cin_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_7_pick_percent = st.slider("Cleveland Browns Week 7 Pick Percent:", -1, 100) / 100
                cle_week_7_pick_percent_percent = cle_week_7_pick_percent*100
                if cle_week_7_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 7 Pick Percent: :green[{cle_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_7_pick_percent = st.slider("Dallas Cowboys Week 7 Pick Percent:", -1, 100) / 100
                dal_week_7_pick_percent_percent = dal_week_7_pick_percent*100
                if dal_week_7_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 7 Pick Percent: :green[{dal_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_7_pick_percent = st.slider("Denver Broncos Week 7 Pick Percent:", -1, 100) / 100
                den_week_7_pick_percent_percent = den_week_7_pick_percent*100
                if den_week_7_pick_percent < 0:
                    st.write(f"Denver Broncos Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 7 Pick Percent: :green[{den_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_7_pick_percent = st.slider("Detroit Lions Week 7 Pick Percent:", -1, 100) / 100
                det_week_7_pick_percent_percent = det_week_7_pick_percent*100
                if det_week_7_pick_percent < 0:
                    st.write(f"Detroit Lions Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 7 Pick Percent: :green[{det_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_7_pick_percent = st.slider("Green Bay Packers Week 7 Pick Percent:", -1, 100) / 100
                gb_week_7_pick_percent_percent = gb_week_7_pick_percent*100
                if gb_week_7_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 7 Pick Percent: :green[{gb_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_7_pick_percent = st.slider("Houston Texans Week 7 Pick Percent:", -1, 100) / 100
                hou_week_7_pick_percent_percent = hou_week_7_pick_percent*100
                if hou_week_7_pick_percent < 0:
                    st.write(f"Houston Texans Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 7 Pick Percent: :green[{hou_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_7_pick_percent = st.slider("Indianapoils Colts Week 7 Pick Percent:", -1, 100) / 100
                ind_week_7_pick_percent_percent = ind_week_7_pick_percent*100
                if ind_week_7_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 7 Pick Percent: :green[{ind_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_7_pick_percent = st.slider("Jacksonville Jaguars Week 7 Pick Percent:", -1, 100) / 100
                jax_week_7_pick_percent_percent = jax_week_7_pick_percent*100
                if jax_week_7_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 7 Pick Percent: :green[{jax_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_7_pick_percent = st.slider("Kansas City Chiefs Week 7 Pick Percent:", -1, 100) / 100
                kc_week_7_pick_percent_percent = kc_week_7_pick_percent*100
                if kc_week_7_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 7 Pick Percent: :green[{kc_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_7_pick_percent = st.slider("Las Vegas Raiders Week 7 Pick Percent:", -1, 100) / 100
                lv_week_7_pick_percent_percent = lv_week_7_pick_percent*100
                if lv_week_7_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 7 Pick Percent: :green[{lv_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_7_pick_percent = st.slider("Los Angeles Chargers Week 7 Pick Percent:", -1, 100) / 100
                lac_week_7_pick_percent_percent = lac_week_7_pick_percent*100
                if lac_week_7_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 7 Pick Percent: :green[{lac_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_7_pick_percent = st.slider("Los Angeles Rams Week 7 Pick Percent:", -1, 100) / 100
                lar_week_7_pick_percent_percent = lar_week_7_pick_percent*100
                if lar_week_7_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 7 Pick Percent: :green[{lar_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_7_pick_percent = st.slider("Miami Dolphins Week 7 Pick Percent:", -1, 100) / 100
                mia_week_7_pick_percent_percent = mia_week_7_pick_percent*100
                if mia_week_7_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 7 Pick Percent: :green[{mia_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_7_pick_percent = st.slider("Minnesota Vikings Week 7 Pick Percent:", -1, 100) / 100
                min_week_7_pick_percent_percent = min_week_7_pick_percent*100
                if min_week_7_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 7 Pick Percent: :green[{min_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_7_pick_percent = st.slider("New England Patriots Week 7 Pick Percent:", -1, 100) / 100
                ne_week_7_pick_percent_percent = ne_week_7_pick_percent*100
                if ne_week_7_pick_percent < 0:
                    st.write(f"New England Patriots Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 7 Pick Percent: :green[{ne_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_7_pick_percent = st.slider("New Orleans Saints Week 7 Pick Percent:", -1, 100) / 100
                no_week_7_pick_percent_percent = no_week_7_pick_percent*100
                if no_week_7_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 7 Pick Percent: :green[{no_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_7_pick_percent = st.slider("New York Giants Week 7 Pick Percent:", -1, 100) / 100
                nyg_week_7_pick_percent_percent = nyg_week_7_pick_percent*100
                if nyg_week_7_pick_percent < 0:
                    st.write(f"New York Giants Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 7 Pick Percent: :green[{nyg_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_7_pick_percent = st.slider("New York Jets Week 7 Pick Percent:", -1, 100) / 100
                nyj_week_7_pick_percent_percent = nyj_week_7_pick_percent*100
                if nyj_week_7_pick_percent < 0:
                    st.write(f"New York Jets Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 7 Pick Percent: :green[{nyj_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_7_pick_percent = st.slider("Philadelphia Eagles Week 7 Pick Percent:", -1, 100) / 100
                phi_week_7_pick_percent_percent = phi_week_7_pick_percent*100
                if phi_week_7_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 7 Pick Percent: :green[{phi_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_7_pick_percent = st.slider("Pittsburgh Steelers Week 7 Pick Percent:", -1, 100) / 100
                pit_week_7_pick_percent_percent = pit_week_7_pick_percent*100
                if pit_week_7_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 7 Pick Percent: :green[{pit_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_7_pick_percent = st.slider("San Francisco 49ers Week 7 Pick Percent:", -1, 100) / 100
                sf_week_7_pick_percent_percent = sf_week_7_pick_percent*100
                if sf_week_7_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 7 Pick Percent: :green[{sf_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_7_pick_percent = st.slider("Seattle Seahawks Week 7 Pick Percent:", -1, 100) / 100
                sea_week_7_pick_percent_percent = sea_week_7_pick_percent*100
                if sea_week_7_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 7 Pick Percent: :green[{sea_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_7_pick_percent = st.slider("Tampa Bay Buccaneers Week 7 Pick Percent:", -1, 100) / 100
                tb_week_7_pick_percent_percent = tb_week_7_pick_percent*100
                if tb_week_7_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 7 Pick Percent: :green[{tb_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_7_pick_percent = st.slider("Tennessee Titans Week 7 Pick Percent:", -1, 100) / 100
                ten_week_7_pick_percent_percent = ten_week_7_pick_percent*100
                if ten_week_7_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 7 Pick Percent: :green[{ten_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_7_pick_percent = st.slider("Washington Commanders Week 7 Pick Percent:", -1, 100) / 100
                was_week_7_pick_percent_percent = was_week_7_pick_percent*100
                if was_week_7_pick_percent < 0:
                    st.write(f"Washington Commanders Week 7 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 7 Pick Percent: :green[{was_week_7_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 8 and ending_week > 8:
            week_8_pick_percents = st.checkbox('Add Week 8 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 8.")
            if week_8_pick_percents:
                st.write('')
                st.subheader('Week 8 Estimated Pick Percentages')
                st.write('')
                az_week_8_pick_percent = st.slider("Arizona Cardinals Week 8 Pick Percent:", -1, 100) / 100
                az_week_8_pick_percent_percent = az_week_8_pick_percent*100
                if az_week_8_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 8 Pick Percent: :green[{az_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_8_pick_percent = st.slider("Atlanta Falcons Week 8 Pick Percent:", -1, 100) / 100
                atl_week_8_pick_percent_percent = atl_week_8_pick_percent*100
                if atl_week_8_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 8 Pick Percent: :green[{atl_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_8_pick_percent = st.slider("Baltimore Ravens Week 8 Pick Percent:", -1, 100) / 100
                bal_week_8_pick_percent_percent = bal_week_8_pick_percent*100
                if bal_week_8_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 8 Pick Percent: :green[{bal_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_8_pick_percent = st.slider("Buffalo Bills Week 8 Pick Percent:", -1, 100) / 100
                buf_week_8_pick_percent_percent = buf_week_8_pick_percent*100
                if buf_week_8_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 8 Pick Percent: :green[{buf_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_8_pick_percent = st.slider("Carolina Panthers Week 8 Pick Percent:", -1, 100) / 100
                car_week_8_pick_percent_percent = car_week_8_pick_percent*100
                if car_week_8_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 8 Pick Percent: :green[{car_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_8_pick_percent = st.slider("Chicago Bears Week 8 Pick Percent:", -1, 100) / 100
                chi_week_8_pick_percent_percent = chi_week_8_pick_percent*100
                if chi_week_8_pick_percent < 0:
                    st.write(f"Chicago Bears Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 8 Pick Percent: :green[{chi_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_8_pick_percent = st.slider("Cincinnati Bengals Week 8 Pick Percent:", -1, 100) / 100
                cin_week_8_pick_percent_percent = cin_week_8_pick_percent*100
                if cin_week_8_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 8 Pick Percent: :green[{cin_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_8_pick_percent = st.slider("Cleveland Browns Week 8 Pick Percent:", -1, 100) / 100
                cle_week_8_pick_percent_percent = cle_week_8_pick_percent*100
                if cle_week_8_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 8 Pick Percent: :green[{cle_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_8_pick_percent = st.slider("Dallas Cowboys Week 8 Pick Percent:", -1, 100) / 100
                dal_week_8_pick_percent_percent = dal_week_8_pick_percent*100
                if dal_week_8_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 8 Pick Percent: :green[{dal_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_8_pick_percent = st.slider("Denver Broncos Week 8 Pick Percent:", -1, 100) / 100
                den_week_8_pick_percent_percent = den_week_8_pick_percent*100
                if den_week_8_pick_percent < 0:
                    st.write(f"Denver Broncos Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 8 Pick Percent: :green[{den_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_8_pick_percent = st.slider("Detroit Lions Week 8 Pick Percent:", -1, 100) / 100
                det_week_8_pick_percent_percent = det_week_8_pick_percent*100
                if det_week_8_pick_percent < 0:
                    st.write(f"Detroit Lions Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 8 Pick Percent: :green[{det_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_8_pick_percent = st.slider("Green Bay Packers Week 8 Pick Percent:", -1, 100) / 100
                gb_week_8_pick_percent_percent = gb_week_8_pick_percent*100
                if gb_week_8_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 8 Pick Percent: :green[{gb_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_8_pick_percent = st.slider("Houston Texans Week 8 Pick Percent:", -1, 100) / 100
                hou_week_8_pick_percent_percent = hou_week_8_pick_percent*100
                if hou_week_8_pick_percent < 0:
                    st.write(f"Houston Texans Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 8 Pick Percent: :green[{hou_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_8_pick_percent = st.slider("Indianapoils Colts Week 8 Pick Percent:", -1, 100) / 100
                ind_week_8_pick_percent_percent = ind_week_8_pick_percent*100
                if ind_week_8_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 8 Pick Percent: :green[{ind_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_8_pick_percent = st.slider("Jacksonville Jaguars Week 8 Pick Percent:", -1, 100) / 100
                jax_week_8_pick_percent_percent = jax_week_8_pick_percent*100
                if jax_week_8_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 8 Pick Percent: :green[{jax_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_8_pick_percent = st.slider("Kansas City Chiefs Week 8 Pick Percent:", -1, 100) / 100
                kc_week_8_pick_percent_percent = kc_week_8_pick_percent*100
                if kc_week_8_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 8 Pick Percent: :green[{kc_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_8_pick_percent = st.slider("Las Vegas Raiders Week 8 Pick Percent:", -1, 100) / 100
                lv_week_8_pick_percent_percent = lv_week_8_pick_percent*100
                if lv_week_8_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 8 Pick Percent: :green[{lv_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_8_pick_percent = st.slider("Los Angeles Chargers Week 8 Pick Percent:", -1, 100) / 100
                lac_week_8_pick_percent_percent = lac_week_8_pick_percent*100
                if lac_week_8_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 8 Pick Percent: :green[{lac_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_8_pick_percent = st.slider("Los Angeles Rams Week 8 Pick Percent:", -1, 100) / 100
                lar_week_8_pick_percent_percent = lar_week_8_pick_percent*100
                if lar_week_8_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 8 Pick Percent: :green[{lar_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_8_pick_percent = st.slider("Miami Dolphins Week 8 Pick Percent:", -1, 100) / 100
                mia_week_8_pick_percent_percent = mia_week_8_pick_percent*100
                if mia_week_8_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 8 Pick Percent: :green[{mia_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_8_pick_percent = st.slider("Minnesota Vikings Week 8 Pick Percent:", -1, 100) / 100
                min_week_8_pick_percent_percent = min_week_8_pick_percent*100
                if min_week_8_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 8 Pick Percent: :green[{min_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_8_pick_percent = st.slider("New England Patriots Week 8 Pick Percent:", -1, 100) / 100
                ne_week_8_pick_percent_percent = ne_week_8_pick_percent*100
                if ne_week_8_pick_percent < 0:
                    st.write(f"New England Patriots Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 8 Pick Percent: :green[{ne_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_8_pick_percent = st.slider("New Orleans Saints Week 8 Pick Percent:", -1, 100) / 100
                no_week_8_pick_percent_percent = no_week_8_pick_percent*100
                if no_week_8_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 8 Pick Percent: :green[{no_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_8_pick_percent = st.slider("New York Giants Week 8 Pick Percent:", -1, 100) / 100
                nyg_week_8_pick_percent_percent = nyg_week_8_pick_percent*100
                if nyg_week_8_pick_percent < 0:
                    st.write(f"New York Giants Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 8 Pick Percent: :green[{nyg_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_8_pick_percent = st.slider("New York Jets Week 8 Pick Percent:", -1, 100) / 100
                nyj_week_8_pick_percent_percent = nyj_week_8_pick_percent*100
                if nyj_week_8_pick_percent < 0:
                    st.write(f"New York Jets Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 8 Pick Percent: :green[{nyj_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_8_pick_percent = st.slider("Philadelphia Eagles Week 8 Pick Percent:", -1, 100) / 100
                phi_week_8_pick_percent_percent = phi_week_8_pick_percent*100
                if phi_week_8_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 8 Pick Percent: :green[{phi_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_8_pick_percent = st.slider("Pittsburgh Steelers Week 8 Pick Percent:", -1, 100) / 100
                pit_week_8_pick_percent_percent = pit_week_8_pick_percent*100
                if pit_week_8_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 8 Pick Percent: :green[{pit_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_8_pick_percent = st.slider("San Francisco 49ers Week 8 Pick Percent:", -1, 100) / 100
                sf_week_8_pick_percent_percent = sf_week_8_pick_percent*100
                if sf_week_8_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 8 Pick Percent: :green[{sf_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_8_pick_percent = st.slider("Seattle Seahawks Week 8 Pick Percent:", -1, 100) / 100
                sea_week_8_pick_percent_percent = sea_week_8_pick_percent*100
                if sea_week_8_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 8 Pick Percent: :green[{sea_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_8_pick_percent = st.slider("Tampa Bay Buccaneers Week 8 Pick Percent:", -1, 100) / 100
                tb_week_8_pick_percent_percent = tb_week_8_pick_percent*100
                if tb_week_8_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 8 Pick Percent: :green[{tb_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_8_pick_percent = st.slider("Tennessee Titans Week 8 Pick Percent:", -1, 100) / 100
                ten_week_8_pick_percent_percent = ten_week_8_pick_percent*100
                if ten_week_8_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 8 Pick Percent: :green[{ten_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_8_pick_percent = st.slider("Washington Commanders Week 8 Pick Percent:", -1, 100) / 100
                was_week_8_pick_percent_percent = was_week_8_pick_percent*100
                if was_week_8_pick_percent < 0:
                    st.write(f"Washington Commanders Week 8 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 8 Pick Percent: :green[{was_week_8_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 9 and ending_week > 9:
            week_9_pick_percents = st.checkbox('Add Week 9 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 9.")
            if week_9_pick_percents:
                st.write('')
                st.subheader('Week 9 Estimated Pick Percentages')
                st.write('')
                az_week_9_pick_percent = st.slider("Arizona Cardinals Week 9 Pick Percent:", -1, 100) / 100
                az_week_9_pick_percent_percent = az_week_9_pick_percent*100
                if az_week_9_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 9 Pick Percent: :green[{az_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_9_pick_percent = st.slider("Atlanta Falcons Week 9 Pick Percent:", -1, 100) / 100
                atl_week_9_pick_percent_percent = atl_week_9_pick_percent*100
                if atl_week_9_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 9 Pick Percent: :green[{atl_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_9_pick_percent = st.slider("Baltimore Ravens Week 9 Pick Percent:", -1, 100) / 100
                bal_week_9_pick_percent_percent = bal_week_9_pick_percent*100
                if bal_week_9_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 9 Pick Percent: :green[{bal_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_9_pick_percent = st.slider("Buffalo Bills Week 9 Pick Percent:", -1, 100) / 100
                buf_week_9_pick_percent_percent = buf_week_9_pick_percent*100
                if buf_week_9_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 9 Pick Percent: :green[{buf_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_9_pick_percent = st.slider("Carolina Panthers Week 9 Pick Percent:", -1, 100) / 100
                car_week_9_pick_percent_percent = car_week_9_pick_percent*100
                if car_week_9_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 9 Pick Percent: :green[{car_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_9_pick_percent = st.slider("Chicago Bears Week 9 Pick Percent:", -1, 100) / 100
                chi_week_9_pick_percent_percent = chi_week_9_pick_percent*100
                if chi_week_9_pick_percent < 0:
                    st.write(f"Chicago Bears Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 9 Pick Percent: :green[{chi_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_9_pick_percent = st.slider("Cincinnati Bengals Week 9 Pick Percent:", -1, 100) / 100
                cin_week_9_pick_percent_percent = cin_week_9_pick_percent*100
                if cin_week_9_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 9 Pick Percent: :green[{cin_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_9_pick_percent = st.slider("Cleveland Browns Week 9 Pick Percent:", -1, 100) / 100
                cle_week_9_pick_percent_percent = cle_week_9_pick_percent*100
                if cle_week_9_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 9 Pick Percent: :green[{cle_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_9_pick_percent = st.slider("Dallas Cowboys Week 9 Pick Percent:", -1, 100) / 100
                dal_week_9_pick_percent_percent = dal_week_9_pick_percent*100
                if dal_week_9_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 9 Pick Percent: :green[{dal_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_9_pick_percent = st.slider("Denver Broncos Week 9 Pick Percent:", -1, 100) / 100
                den_week_9_pick_percent_percent = den_week_9_pick_percent*100
                if den_week_9_pick_percent < 0:
                    st.write(f"Denver Broncos Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 9 Pick Percent: :green[{den_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_9_pick_percent = st.slider("Detroit Lions Week 9 Pick Percent:", -1, 100) / 100
                det_week_9_pick_percent_percent = det_week_9_pick_percent*100
                if det_week_9_pick_percent < 0:
                    st.write(f"Detroit Lions Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 9 Pick Percent: :green[{det_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_9_pick_percent = st.slider("Green Bay Packers Week 9 Pick Percent:", -1, 100) / 100
                gb_week_9_pick_percent_percent = gb_week_9_pick_percent*100
                if gb_week_9_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 9 Pick Percent: :green[{gb_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_9_pick_percent = st.slider("Houston Texans Week 9 Pick Percent:", -1, 100) / 100
                hou_week_9_pick_percent_percent = hou_week_9_pick_percent*100
                if hou_week_9_pick_percent < 0:
                    st.write(f"Houston Texans Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 9 Pick Percent: :green[{hou_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_9_pick_percent = st.slider("Indianapoils Colts Week 9 Pick Percent:", -1, 100) / 100
                ind_week_9_pick_percent_percent = ind_week_9_pick_percent*100
                if ind_week_9_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 9 Pick Percent: :green[{ind_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_9_pick_percent = st.slider("Jacksonville Jaguars Week 9 Pick Percent:", -1, 100) / 100
                jax_week_9_pick_percent_percent = jax_week_9_pick_percent*100
                if jax_week_9_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 9 Pick Percent: :green[{jax_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_9_pick_percent = st.slider("Kansas City Chiefs Week 9 Pick Percent:", -1, 100) / 100
                kc_week_9_pick_percent_percent = kc_week_9_pick_percent*100
                if kc_week_9_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 9 Pick Percent: :green[{kc_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_9_pick_percent = st.slider("Las Vegas Raiders Week 9 Pick Percent:", -1, 100) / 100
                lv_week_9_pick_percent_percent = lv_week_9_pick_percent*100
                if lv_week_9_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 9 Pick Percent: :green[{lv_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_9_pick_percent = st.slider("Los Angeles Chargers Week 9 Pick Percent:", -1, 100) / 100
                lac_week_9_pick_percent_percent = lac_week_9_pick_percent*100
                if lac_week_9_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 9 Pick Percent: :green[{lac_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_9_pick_percent = st.slider("Los Angeles Rams Week 9 Pick Percent:", -1, 100) / 100
                lar_week_9_pick_percent_percent = lar_week_9_pick_percent*100
                if lar_week_9_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 9 Pick Percent: :green[{lar_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_9_pick_percent = st.slider("Miami Dolphins Week 9 Pick Percent:", -1, 100) / 100
                mia_week_9_pick_percent_percent = mia_week_9_pick_percent*100
                if mia_week_9_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 9 Pick Percent: :green[{mia_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_9_pick_percent = st.slider("Minnesota Vikings Week 9 Pick Percent:", -1, 100) / 100
                min_week_9_pick_percent_percent = min_week_9_pick_percent*100
                if min_week_9_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 9 Pick Percent: :green[{min_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_9_pick_percent = st.slider("New England Patriots Week 9 Pick Percent:", -1, 100) / 100
                ne_week_9_pick_percent_percent = ne_week_9_pick_percent*100
                if ne_week_9_pick_percent < 0:
                    st.write(f"New England Patriots Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 9 Pick Percent: :green[{ne_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_9_pick_percent = st.slider("New Orleans Saints Week 9 Pick Percent:", -1, 100) / 100
                no_week_9_pick_percent_percent = no_week_9_pick_percent*100
                if no_week_9_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 9 Pick Percent: :green[{no_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_9_pick_percent = st.slider("New York Giants Week 9 Pick Percent:", -1, 100) / 100
                nyg_week_9_pick_percent_percent = nyg_week_9_pick_percent*100
                if nyg_week_9_pick_percent < 0:
                    st.write(f"New York Giants Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 9 Pick Percent: :green[{nyg_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_9_pick_percent = st.slider("New York Jets Week 9 Pick Percent:", -1, 100) / 100
                nyj_week_9_pick_percent_percent = nyj_week_9_pick_percent*100
                if nyj_week_9_pick_percent < 0:
                    st.write(f"New York Jets Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 9 Pick Percent: :green[{nyj_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_9_pick_percent = st.slider("Philadelphia Eagles Week 9 Pick Percent:", -1, 100) / 100
                phi_week_9_pick_percent_percent = phi_week_9_pick_percent*100
                if phi_week_9_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 9 Pick Percent: :green[{phi_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_9_pick_percent = st.slider("Pittsburgh Steelers Week 9 Pick Percent:", -1, 100) / 100
                pit_week_9_pick_percent_percent = pit_week_9_pick_percent*100
                if pit_week_9_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 9 Pick Percent: :green[{pit_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_9_pick_percent = st.slider("San Francisco 49ers Week 9 Pick Percent:", -1, 100) / 100
                sf_week_9_pick_percent_percent = sf_week_9_pick_percent*100
                if sf_week_9_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 9 Pick Percent: :green[{sf_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_9_pick_percent = st.slider("Seattle Seahawks Week 9 Pick Percent:", -1, 100) / 100
                sea_week_9_pick_percent_percent = sea_week_9_pick_percent*100
                if sea_week_9_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 9 Pick Percent: :green[{sea_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_9_pick_percent = st.slider("Tampa Bay Buccaneers Week 9 Pick Percent:", -1, 100) / 100
                tb_week_9_pick_percent_percent = tb_week_9_pick_percent*100
                if tb_week_9_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 9 Pick Percent: :green[{tb_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_9_pick_percent = st.slider("Tennessee Titans Week 9 Pick Percent:", -1, 100) / 100
                ten_week_9_pick_percent_percent = ten_week_9_pick_percent*100
                if ten_week_9_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 9 Pick Percent: :green[{ten_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_9_pick_percent = st.slider("Washington Commanders Week 9 Pick Percent:", -1, 100) / 100
                was_week_9_pick_percent_percent = was_week_9_pick_percent*100
                if was_week_9_pick_percent < 0:
                    st.write(f"Washington Commanders Week 9 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 9 Pick Percent: :green[{was_week_9_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 10 and ending_week > 10:
            week_10_pick_percents = st.checkbox('Add Week 10 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 10.")
            if week_10_pick_percents:
                st.write('')
                st.subheader('Week 10 Estimated Pick Percentages')
                st.write('')
                az_week_10_pick_percent = st.slider("Arizona Cardinals Week 10 Pick Percent:", -1, 100) / 100
                az_week_10_pick_percent_percent = az_week_10_pick_percent*100
                if az_week_10_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 10 Pick Percent: :green[{az_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_10_pick_percent = st.slider("Atlanta Falcons Week 10 Pick Percent:", -1, 100) / 100
                atl_week_10_pick_percent_percent = atl_week_10_pick_percent*100
                if atl_week_10_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 10 Pick Percent: :green[{atl_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_10_pick_percent = st.slider("Baltimore Ravens Week 10 Pick Percent:", -1, 100) / 100
                bal_week_10_pick_percent_percent = bal_week_10_pick_percent*100
                if bal_week_10_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 10 Pick Percent: :green[{bal_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_10_pick_percent = st.slider("Buffalo Bills Week 10 Pick Percent:", -1, 100) / 100
                buf_week_10_pick_percent_percent = buf_week_10_pick_percent*100
                if buf_week_10_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 10 Pick Percent: :green[{buf_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_10_pick_percent = st.slider("Carolina Panthers Week 10 Pick Percent:", -1, 100) / 100
                car_week_10_pick_percent_percent = car_week_10_pick_percent*100
                if car_week_10_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 10 Pick Percent: :green[{car_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_10_pick_percent = st.slider("Chicago Bears Week 10 Pick Percent:", -1, 100) / 100
                chi_week_10_pick_percent_percent = chi_week_10_pick_percent*100
                if chi_week_10_pick_percent < 0:
                    st.write(f"Chicago Bears Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 10 Pick Percent: :green[{chi_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_10_pick_percent = st.slider("Cincinnati Bengals Week 10 Pick Percent:", -1, 100) / 100
                cin_week_10_pick_percent_percent = cin_week_10_pick_percent*100
                if cin_week_10_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 10 Pick Percent: :green[{cin_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_10_pick_percent = st.slider("Cleveland Browns Week 10 Pick Percent:", -1, 100) / 100
                cle_week_10_pick_percent_percent = cle_week_10_pick_percent*100
                if cle_week_10_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 10 Pick Percent: :green[{cle_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_10_pick_percent = st.slider("Dallas Cowboys Week 10 Pick Percent:", -1, 100) / 100
                dal_week_10_pick_percent_percent = dal_week_10_pick_percent*100
                if dal_week_10_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 10 Pick Percent: :green[{dal_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_10_pick_percent = st.slider("Denver Broncos Week 10 Pick Percent:", -1, 100) / 100
                den_week_10_pick_percent_percent = den_week_10_pick_percent*100
                if den_week_10_pick_percent < 0:
                    st.write(f"Denver Broncos Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 10 Pick Percent: :green[{den_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_10_pick_percent = st.slider("Detroit Lions Week 10 Pick Percent:", -1, 100) / 100
                det_week_10_pick_percent_percent = det_week_10_pick_percent*100
                if det_week_10_pick_percent < 0:
                    st.write(f"Detroit Lions Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 10 Pick Percent: :green[{det_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_10_pick_percent = st.slider("Green Bay Packers Week 10 Pick Percent:", -1, 100) / 100
                gb_week_10_pick_percent_percent = gb_week_10_pick_percent*100
                if gb_week_10_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 10 Pick Percent: :green[{gb_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_10_pick_percent = st.slider("Houston Texans Week 10 Pick Percent:", -1, 100) / 100
                hou_week_10_pick_percent_percent = hou_week_10_pick_percent*100
                if hou_week_10_pick_percent < 0:
                    st.write(f"Houston Texans Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 10 Pick Percent: :green[{hou_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_10_pick_percent = st.slider("Indianapoils Colts Week 10 Pick Percent:", -1, 100) / 100
                ind_week_10_pick_percent_percent = ind_week_10_pick_percent*100
                if ind_week_10_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 10 Pick Percent: :green[{ind_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_10_pick_percent = st.slider("Jacksonville Jaguars Week 10 Pick Percent:", -1, 100) / 100
                jax_week_10_pick_percent_percent = jax_week_10_pick_percent*100
                if jax_week_10_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 10 Pick Percent: :green[{jax_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_10_pick_percent = st.slider("Kansas City Chiefs Week 10 Pick Percent:", -1, 100) / 100
                kc_week_10_pick_percent_percent = kc_week_10_pick_percent*100
                if kc_week_10_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 10 Pick Percent: :green[{kc_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_10_pick_percent = st.slider("Las Vegas Raiders Week 10 Pick Percent:", -1, 100) / 100
                lv_week_10_pick_percent_percent = lv_week_10_pick_percent*100
                if lv_week_10_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 10 Pick Percent: :green[{lv_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_10_pick_percent = st.slider("Los Angeles Chargers Week 10 Pick Percent:", -1, 100) / 100
                lac_week_10_pick_percent_percent = lac_week_10_pick_percent*100
                if lac_week_10_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 10 Pick Percent: :green[{lac_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_10_pick_percent = st.slider("Los Angeles Rams Week 10 Pick Percent:", -1, 100) / 100
                lar_week_10_pick_percent_percent = lar_week_10_pick_percent*100
                if lar_week_10_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 10 Pick Percent: :green[{lar_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_10_pick_percent = st.slider("Miami Dolphins Week 10 Pick Percent:", -1, 100) / 100
                mia_week_10_pick_percent_percent = mia_week_10_pick_percent*100
                if mia_week_10_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 10 Pick Percent: :green[{mia_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_10_pick_percent = st.slider("Minnesota Vikings Week 10 Pick Percent:", -1, 100) / 100
                min_week_10_pick_percent_percent = min_week_10_pick_percent*100
                if min_week_10_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 10 Pick Percent: :green[{min_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_10_pick_percent = st.slider("New England Patriots Week 10 Pick Percent:", -1, 100) / 100
                ne_week_10_pick_percent_percent = ne_week_10_pick_percent*100
                if ne_week_10_pick_percent < 0:
                    st.write(f"New England Patriots Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 10 Pick Percent: :green[{ne_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_10_pick_percent = st.slider("New Orleans Saints Week 10 Pick Percent:", -1, 100) / 100
                no_week_10_pick_percent_percent = no_week_10_pick_percent*100
                if no_week_10_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 10 Pick Percent: :green[{no_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_10_pick_percent = st.slider("New York Giants Week 10 Pick Percent:", -1, 100) / 100
                nyg_week_10_pick_percent_percent = nyg_week_10_pick_percent*100
                if nyg_week_10_pick_percent < 0:
                    st.write(f"New York Giants Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 10 Pick Percent: :green[{nyg_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_10_pick_percent = st.slider("New York Jets Week 10 Pick Percent:", -1, 100) / 100
                nyj_week_10_pick_percent_percent = nyj_week_10_pick_percent*100
                if nyj_week_10_pick_percent < 0:
                    st.write(f"New York Jets Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 10 Pick Percent: :green[{nyj_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_10_pick_percent = st.slider("Philadelphia Eagles Week 10 Pick Percent:", -1, 100) / 100
                phi_week_10_pick_percent_percent = phi_week_10_pick_percent*100
                if phi_week_10_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 10 Pick Percent: :green[{phi_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_10_pick_percent = st.slider("Pittsburgh Steelers Week 10 Pick Percent:", -1, 100) / 100
                pit_week_10_pick_percent_percent = pit_week_10_pick_percent*100
                if pit_week_10_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 10 Pick Percent: :green[{pit_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_10_pick_percent = st.slider("San Francisco 49ers Week 10 Pick Percent:", -1, 100) / 100
                sf_week_10_pick_percent_percent = sf_week_10_pick_percent*100
                if sf_week_10_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 10 Pick Percent: :green[{sf_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_10_pick_percent = st.slider("Seattle Seahawks Week 10 Pick Percent:", -1, 100) / 100
                sea_week_10_pick_percent_percent = sea_week_10_pick_percent*100
                if sea_week_10_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 10 Pick Percent: :green[{sea_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_10_pick_percent = st.slider("Tampa Bay Buccaneers Week 10 Pick Percent:", -1, 100) / 100
                tb_week_10_pick_percent_percent = tb_week_10_pick_percent*100
                if tb_week_10_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 10 Pick Percent: :green[{tb_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_10_pick_percent = st.slider("Tennessee Titans Week 10 Pick Percent:", -1, 100) / 100
                ten_week_10_pick_percent_percent = ten_week_10_pick_percent*100
                if ten_week_10_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 10 Pick Percent: :green[{ten_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_10_pick_percent = st.slider("Washington Commanders Week 10 Pick Percent:", -1, 100) / 100
                was_week_10_pick_percent_percent = was_week_10_pick_percent*100
                if was_week_10_pick_percent < 0:
                    st.write(f"Washington Commanders Week 10 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 10 Pick Percent: :green[{was_week_10_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 11 and ending_week > 11:
            week_11_pick_percents = st.checkbox('Add Week 11 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 11.")
            if week_11_pick_percents:
                st.write('')
                st.subheader('Week 11 Estimated Pick Percentages')
                st.write('')
                az_week_11_pick_percent = st.slider("Arizona Cardinals Week 11 Pick Percent:", -1, 100) / 100
                az_week_11_pick_percent_percent = az_week_11_pick_percent*100
                if az_week_11_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 11 Pick Percent: :green[{az_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_11_pick_percent = st.slider("Atlanta Falcons Week 11 Pick Percent:", -1, 100) / 100
                atl_week_11_pick_percent_percent = atl_week_11_pick_percent*100
                if atl_week_11_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 11 Pick Percent: :green[{atl_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_11_pick_percent = st.slider("Baltimore Ravens Week 11 Pick Percent:", -1, 100) / 100
                bal_week_11_pick_percent_percent = bal_week_11_pick_percent*100
                if bal_week_11_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 11 Pick Percent: :green[{bal_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_11_pick_percent = st.slider("Buffalo Bills Week 11 Pick Percent:", -1, 100) / 100
                buf_week_11_pick_percent_percent = buf_week_11_pick_percent*100
                if buf_week_11_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 11 Pick Percent: :green[{buf_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_11_pick_percent = st.slider("Carolina Panthers Week 11 Pick Percent:", -1, 100) / 100
                car_week_11_pick_percent_percent = car_week_11_pick_percent*100
                if car_week_11_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 11 Pick Percent: :green[{car_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_11_pick_percent = st.slider("Chicago Bears Week 11 Pick Percent:", -1, 100) / 100
                chi_week_11_pick_percent_percent = chi_week_11_pick_percent*100
                if chi_week_11_pick_percent < 0:
                    st.write(f"Chicago Bears Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 11 Pick Percent: :green[{chi_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_11_pick_percent = st.slider("Cincinnati Bengals Week 11 Pick Percent:", -1, 100) / 100
                cin_week_11_pick_percent_percent = cin_week_11_pick_percent*100
                if cin_week_11_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 11 Pick Percent: :green[{cin_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_11_pick_percent = st.slider("Cleveland Browns Week 11 Pick Percent:", -1, 100) / 100
                cle_week_11_pick_percent_percent = cle_week_11_pick_percent*100
                if cle_week_11_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 11 Pick Percent: :green[{cle_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_11_pick_percent = st.slider("Dallas Cowboys Week 11 Pick Percent:", -1, 100) / 100
                dal_week_11_pick_percent_percent = dal_week_11_pick_percent*100
                if dal_week_11_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 11 Pick Percent: :green[{dal_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_11_pick_percent = st.slider("Denver Broncos Week 11 Pick Percent:", -1, 100) / 100
                den_week_11_pick_percent_percent = den_week_11_pick_percent*100
                if den_week_11_pick_percent < 0:
                    st.write(f"Denver Broncos Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 11 Pick Percent: :green[{den_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_11_pick_percent = st.slider("Detroit Lions Week 11 Pick Percent:", -1, 100) / 100
                det_week_11_pick_percent_percent = det_week_11_pick_percent*100
                if det_week_11_pick_percent < 0:
                    st.write(f"Detroit Lions Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 11 Pick Percent: :green[{det_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_11_pick_percent = st.slider("Green Bay Packers Week 11 Pick Percent:", -1, 100) / 100
                gb_week_11_pick_percent_percent = gb_week_11_pick_percent*100
                if gb_week_11_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 11 Pick Percent: :green[{gb_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_11_pick_percent = st.slider("Houston Texans Week 11 Pick Percent:", -1, 100) / 100
                hou_week_11_pick_percent_percent = hou_week_11_pick_percent*100
                if hou_week_11_pick_percent < 0:
                    st.write(f"Houston Texans Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 11 Pick Percent: :green[{hou_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_11_pick_percent = st.slider("Indianapoils Colts Week 11 Pick Percent:", -1, 100) / 100
                ind_week_11_pick_percent_percent = ind_week_11_pick_percent*100
                if ind_week_11_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 11 Pick Percent: :green[{ind_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_11_pick_percent = st.slider("Jacksonville Jaguars Week 11 Pick Percent:", -1, 100) / 100
                jax_week_11_pick_percent_percent = jax_week_11_pick_percent*100
                if jax_week_11_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 11 Pick Percent: :green[{jax_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_11_pick_percent = st.slider("Kansas City Chiefs Week 11 Pick Percent:", -1, 100) / 100
                kc_week_11_pick_percent_percent = kc_week_11_pick_percent*100
                if kc_week_11_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 11 Pick Percent: :green[{kc_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_11_pick_percent = st.slider("Las Vegas Raiders Week 11 Pick Percent:", -1, 100) / 100
                lv_week_11_pick_percent_percent = lv_week_11_pick_percent*100
                if lv_week_11_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 11 Pick Percent: :green[{lv_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_11_pick_percent = st.slider("Los Angeles Chargers Week 11 Pick Percent:", -1, 100) / 100
                lac_week_11_pick_percent_percent = lac_week_11_pick_percent*100
                if lac_week_11_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 11 Pick Percent: :green[{lac_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_11_pick_percent = st.slider("Los Angeles Rams Week 11 Pick Percent:", -1, 100) / 100
                lar_week_11_pick_percent_percent = lar_week_11_pick_percent*100
                if lar_week_11_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 11 Pick Percent: :green[{lar_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_11_pick_percent = st.slider("Miami Dolphins Week 11 Pick Percent:", -1, 100) / 100
                mia_week_11_pick_percent_percent = mia_week_11_pick_percent*100
                if mia_week_11_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 11 Pick Percent: :green[{mia_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_11_pick_percent = st.slider("Minnesota Vikings Week 11 Pick Percent:", -1, 100) / 100
                min_week_11_pick_percent_percent = min_week_11_pick_percent*100
                if min_week_11_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 11 Pick Percent: :green[{min_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_11_pick_percent = st.slider("New England Patriots Week 11 Pick Percent:", -1, 100) / 100
                ne_week_11_pick_percent_percent = ne_week_11_pick_percent*100
                if ne_week_11_pick_percent < 0:
                    st.write(f"New England Patriots Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 11 Pick Percent: :green[{ne_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_11_pick_percent = st.slider("New Orleans Saints Week 11 Pick Percent:", -1, 100) / 100
                no_week_11_pick_percent_percent = no_week_11_pick_percent*100
                if no_week_11_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 11 Pick Percent: :green[{no_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_11_pick_percent = st.slider("New York Giants Week 11 Pick Percent:", -1, 100) / 100
                nyg_week_11_pick_percent_percent = nyg_week_11_pick_percent*100
                if nyg_week_11_pick_percent < 0:
                    st.write(f"New York Giants Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 11 Pick Percent: :green[{nyg_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_11_pick_percent = st.slider("New York Jets Week 11 Pick Percent:", -1, 100) / 100
                nyj_week_11_pick_percent_percent = nyj_week_11_pick_percent*100
                if nyj_week_11_pick_percent < 0:
                    st.write(f"New York Jets Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 11 Pick Percent: :green[{nyj_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_11_pick_percent = st.slider("Philadelphia Eagles Week 11 Pick Percent:", -1, 100) / 100
                phi_week_11_pick_percent_percent = phi_week_11_pick_percent*100
                if phi_week_11_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 11 Pick Percent: :green[{phi_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_11_pick_percent = st.slider("Pittsburgh Steelers Week 11 Pick Percent:", -1, 100) / 100
                pit_week_11_pick_percent_percent = pit_week_11_pick_percent*100
                if pit_week_11_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 11 Pick Percent: :green[{pit_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_11_pick_percent = st.slider("San Francisco 49ers Week 11 Pick Percent:", -1, 100) / 100
                sf_week_11_pick_percent_percent = sf_week_11_pick_percent*100
                if sf_week_11_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 11 Pick Percent: :green[{sf_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_11_pick_percent = st.slider("Seattle Seahawks Week 11 Pick Percent:", -1, 100) / 100
                sea_week_11_pick_percent_percent = sea_week_11_pick_percent*100
                if sea_week_11_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 11 Pick Percent: :green[{sea_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_11_pick_percent = st.slider("Tampa Bay Buccaneers Week 11 Pick Percent:", -1, 100) / 100
                tb_week_11_pick_percent_percent = tb_week_11_pick_percent*100
                if tb_week_11_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 11 Pick Percent: :green[{tb_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_11_pick_percent = st.slider("Tennessee Titans Week 11 Pick Percent:", -1, 100) / 100
                ten_week_11_pick_percent_percent = ten_week_11_pick_percent*100
                if ten_week_11_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 11 Pick Percent: :green[{ten_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_11_pick_percent = st.slider("Washington Commanders Week 11 Pick Percent:", -1, 100) / 100
                was_week_11_pick_percent_percent = was_week_11_pick_percent*100
                if was_week_11_pick_percent < 0:
                    st.write(f"Washington Commanders Week 11 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 11 Pick Percent: :green[{was_week_11_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 12 and ending_week > 12:
            week_12_pick_percents = st.checkbox('Add Week 12 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 12.")
            if week_12_pick_percents:
                st.write('')
                st.subheader('Week 12 Estimated Pick Percentages')
                st.write('')
                az_week_12_pick_percent = st.slider("Arizona Cardinals Week 12 Pick Percent:", -1, 100) / 100
                az_week_12_pick_percent_percent = az_week_12_pick_percent*100
                if az_week_12_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 12 Pick Percent: :green[{az_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_12_pick_percent = st.slider("Atlanta Falcons Week 12 Pick Percent:", -1, 100) / 100
                atl_week_12_pick_percent_percent = atl_week_12_pick_percent*100
                if atl_week_12_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 12 Pick Percent: :green[{atl_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_12_pick_percent = st.slider("Baltimore Ravens Week 12 Pick Percent:", -1, 100) / 100
                bal_week_12_pick_percent_percent = bal_week_12_pick_percent*100
                if bal_week_12_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 12 Pick Percent: :green[{bal_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_12_pick_percent = st.slider("Buffalo Bills Week 12 Pick Percent:", -1, 100) / 100
                buf_week_12_pick_percent_percent = buf_week_12_pick_percent*100
                if buf_week_12_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 12 Pick Percent: :green[{buf_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_12_pick_percent = st.slider("Carolina Panthers Week 12 Pick Percent:", -1, 100) / 100
                car_week_12_pick_percent_percent = car_week_12_pick_percent*100
                if car_week_12_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 12 Pick Percent: :green[{car_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_12_pick_percent = st.slider("Chicago Bears Week 12 Pick Percent:", -1, 100) / 100
                chi_week_12_pick_percent_percent = chi_week_12_pick_percent*100
                if chi_week_12_pick_percent < 0:
                    st.write(f"Chicago Bears Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 12 Pick Percent: :green[{chi_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_12_pick_percent = st.slider("Cincinnati Bengals Week 12 Pick Percent:", -1, 100) / 100
                cin_week_12_pick_percent_percent = cin_week_12_pick_percent*100
                if cin_week_12_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 12 Pick Percent: :green[{cin_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_12_pick_percent = st.slider("Cleveland Browns Week 12 Pick Percent:", -1, 100) / 100
                cle_week_12_pick_percent_percent = cle_week_12_pick_percent*100
                if cle_week_12_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 12 Pick Percent: :green[{cle_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_12_pick_percent = st.slider("Dallas Cowboys Week 12 Pick Percent:", -1, 100) / 100
                dal_week_12_pick_percent_percent = dal_week_12_pick_percent*100
                if dal_week_12_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 12 Pick Percent: :green[{dal_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_12_pick_percent = st.slider("Denver Broncos Week 12 Pick Percent:", -1, 100) / 100
                den_week_12_pick_percent_percent = den_week_12_pick_percent*100
                if den_week_12_pick_percent < 0:
                    st.write(f"Denver Broncos Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 12 Pick Percent: :green[{den_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_12_pick_percent = st.slider("Detroit Lions Week 12 Pick Percent:", -1, 100) / 100
                det_week_12_pick_percent_percent = det_week_12_pick_percent*100
                if det_week_12_pick_percent < 0:
                    st.write(f"Detroit Lions Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 12 Pick Percent: :green[{det_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_12_pick_percent = st.slider("Green Bay Packers Week 12 Pick Percent:", -1, 100) / 100
                gb_week_12_pick_percent_percent = gb_week_12_pick_percent*100
                if gb_week_12_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 12 Pick Percent: :green[{gb_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_12_pick_percent = st.slider("Houston Texans Week 12 Pick Percent:", -1, 100) / 100
                hou_week_12_pick_percent_percent = hou_week_12_pick_percent*100
                if hou_week_12_pick_percent < 0:
                    st.write(f"Houston Texans Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 12 Pick Percent: :green[{hou_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_12_pick_percent = st.slider("Indianapoils Colts Week 12 Pick Percent:", -1, 100) / 100
                ind_week_12_pick_percent_percent = ind_week_12_pick_percent*100
                if ind_week_12_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 12 Pick Percent: :green[{ind_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_12_pick_percent = st.slider("Jacksonville Jaguars Week 12 Pick Percent:", -1, 100) / 100
                jax_week_12_pick_percent_percent = jax_week_12_pick_percent*100
                if jax_week_12_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 12 Pick Percent: :green[{jax_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_12_pick_percent = st.slider("Kansas City Chiefs Week 12 Pick Percent:", -1, 100) / 100
                kc_week_12_pick_percent_percent = kc_week_12_pick_percent*100
                if kc_week_12_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 12 Pick Percent: :green[{kc_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_12_pick_percent = st.slider("Las Vegas Raiders Week 12 Pick Percent:", -1, 100) / 100
                lv_week_12_pick_percent_percent = lv_week_12_pick_percent*100
                if lv_week_12_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 12 Pick Percent: :green[{lv_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_12_pick_percent = st.slider("Los Angeles Chargers Week 12 Pick Percent:", -1, 100) / 100
                lac_week_12_pick_percent_percent = lac_week_12_pick_percent*100
                if lac_week_12_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 12 Pick Percent: :green[{lac_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_12_pick_percent = st.slider("Los Angeles Rams Week 12 Pick Percent:", -1, 100) / 100
                lar_week_12_pick_percent_percent = lar_week_12_pick_percent*100
                if lar_week_12_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 12 Pick Percent: :green[{lar_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_12_pick_percent = st.slider("Miami Dolphins Week 12 Pick Percent:", -1, 100) / 100
                mia_week_12_pick_percent_percent = mia_week_12_pick_percent*100
                if mia_week_12_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 12 Pick Percent: :green[{mia_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_12_pick_percent = st.slider("Minnesota Vikings Week 12 Pick Percent:", -1, 100) / 100
                min_week_12_pick_percent_percent = min_week_12_pick_percent*100
                if min_week_12_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 12 Pick Percent: :green[{min_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_12_pick_percent = st.slider("New England Patriots Week 12 Pick Percent:", -1, 100) / 100
                ne_week_12_pick_percent_percent = ne_week_12_pick_percent*100
                if ne_week_12_pick_percent < 0:
                    st.write(f"New England Patriots Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 12 Pick Percent: :green[{ne_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_12_pick_percent = st.slider("New Orleans Saints Week 12 Pick Percent:", -1, 100) / 100
                no_week_12_pick_percent_percent = no_week_12_pick_percent*100
                if no_week_12_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 12 Pick Percent: :green[{no_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_12_pick_percent = st.slider("New York Giants Week 12 Pick Percent:", -1, 100) / 100
                nyg_week_12_pick_percent_percent = nyg_week_12_pick_percent*100
                if nyg_week_12_pick_percent < 0:
                    st.write(f"New York Giants Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 12 Pick Percent: :green[{nyg_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_12_pick_percent = st.slider("New York Jets Week 12 Pick Percent:", -1, 100) / 100
                nyj_week_12_pick_percent_percent = nyj_week_12_pick_percent*100
                if nyj_week_12_pick_percent < 0:
                    st.write(f"New York Jets Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 12 Pick Percent: :green[{nyj_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_12_pick_percent = st.slider("Philadelphia Eagles Week 12 Pick Percent:", -1, 100) / 100
                phi_week_12_pick_percent_percent = phi_week_12_pick_percent*100
                if phi_week_12_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 12 Pick Percent: :green[{phi_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_12_pick_percent = st.slider("Pittsburgh Steelers Week 12 Pick Percent:", -1, 100) / 100
                pit_week_12_pick_percent_percent = pit_week_12_pick_percent*100
                if pit_week_12_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 12 Pick Percent: :green[{pit_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_12_pick_percent = st.slider("San Francisco 49ers Week 12 Pick Percent:", -1, 100) / 100
                sf_week_12_pick_percent_percent = sf_week_12_pick_percent*100
                if sf_week_12_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 12 Pick Percent: :green[{sf_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_12_pick_percent = st.slider("Seattle Seahawks Week 12 Pick Percent:", -1, 100) / 100
                sea_week_12_pick_percent_percent = sea_week_12_pick_percent*100
                if sea_week_12_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 12 Pick Percent: :green[{sea_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_12_pick_percent = st.slider("Tampa Bay Buccaneers Week 12 Pick Percent:", -1, 100) / 100
                tb_week_12_pick_percent_percent = tb_week_12_pick_percent*100
                if tb_week_12_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 12 Pick Percent: :green[{tb_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_12_pick_percent = st.slider("Tennessee Titans Week 12 Pick Percent:", -1, 100) / 100
                ten_week_12_pick_percent_percent = ten_week_12_pick_percent*100
                if ten_week_12_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 12 Pick Percent: :green[{ten_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_12_pick_percent = st.slider("Washington Commanders Week 12 Pick Percent:", -1, 100) / 100
                was_week_12_pick_percent_percent = was_week_12_pick_percent*100
                if was_week_12_pick_percent < 0:
                    st.write(f"Washington Commanders Week 12 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 12 Pick Percent: :green[{was_week_12_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 13 and ending_week > 13:
            week_13_pick_percents = st.checkbox('Add Week 13 (Thanksgiving Week if in Circa Survivor Contest) Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 13 (Thanksgiving Week if in Circa Survivor Contest).")
            if week_13_pick_percents:
                st.write('')
                st.subheader('Week 13 Estimated Pick Percentages')
                st.write('')
                az_week_13_pick_percent = st.slider("Arizona Cardinals Week 13 Pick Percent:", -1, 100) / 100
                az_week_13_pick_percent_percent = az_week_13_pick_percent*100
                if az_week_13_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 13 Pick Percent: :green[{az_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_13_pick_percent = st.slider("Atlanta Falcons Week 13 Pick Percent:", -1, 100) / 100
                atl_week_13_pick_percent_percent = atl_week_13_pick_percent*100
                if atl_week_13_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 13 Pick Percent: :green[{atl_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_13_pick_percent = st.slider("Baltimore Ravens Week 13 Pick Percent:", -1, 100) / 100
                bal_week_13_pick_percent_percent = bal_week_13_pick_percent*100
                if bal_week_13_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 13 Pick Percent: :green[{bal_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_13_pick_percent = st.slider("Buffalo Bills Week 13 Pick Percent:", -1, 100) / 100
                buf_week_13_pick_percent_percent = buf_week_13_pick_percent*100
                if buf_week_13_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 13 Pick Percent: :green[{buf_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_13_pick_percent = st.slider("Carolina Panthers Week 13 Pick Percent:", -1, 100) / 100
                car_week_13_pick_percent_percent = car_week_13_pick_percent*100
                if car_week_13_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 13 Pick Percent: :green[{car_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_13_pick_percent = st.slider("Chicago Bears Week 13 Pick Percent:", -1, 100) / 100
                chi_week_13_pick_percent_percent = chi_week_13_pick_percent*100
                if chi_week_13_pick_percent < 0:
                    st.write(f"Chicago Bears Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 13 Pick Percent: :green[{chi_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_13_pick_percent = st.slider("Cincinnati Bengals Week 13 Pick Percent:", -1, 100) / 100
                cin_week_13_pick_percent_percent = cin_week_13_pick_percent*100
                if cin_week_13_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 13 Pick Percent: :green[{cin_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_13_pick_percent = st.slider("Cleveland Browns Week 13 Pick Percent:", -1, 100) / 100
                cle_week_13_pick_percent_percent = cle_week_13_pick_percent*100
                if cle_week_13_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 13 Pick Percent: :green[{cle_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_13_pick_percent = st.slider("Dallas Cowboys Week 13 Pick Percent:", -1, 100) / 100
                dal_week_13_pick_percent_percent = dal_week_13_pick_percent*100
                if dal_week_13_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 13 Pick Percent: :green[{dal_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_13_pick_percent = st.slider("Denver Broncos Week 13 Pick Percent:", -1, 100) / 100
                den_week_13_pick_percent_percent = den_week_13_pick_percent*100
                if den_week_13_pick_percent < 0:
                    st.write(f"Denver Broncos Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 13 Pick Percent: :green[{den_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_13_pick_percent = st.slider("Detroit Lions Week 13 Pick Percent:", -1, 100) / 100
                det_week_13_pick_percent_percent = det_week_13_pick_percent*100
                if det_week_13_pick_percent < 0:
                    st.write(f"Detroit Lions Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 13 Pick Percent: :green[{det_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_13_pick_percent = st.slider("Green Bay Packers Week 13 Pick Percent:", -1, 100) / 100
                gb_week_13_pick_percent_percent = gb_week_13_pick_percent*100
                if gb_week_13_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 13 Pick Percent: :green[{gb_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_13_pick_percent = st.slider("Houston Texans Week 13 Pick Percent:", -1, 100) / 100
                hou_week_13_pick_percent_percent = hou_week_13_pick_percent*100
                if hou_week_13_pick_percent < 0:
                    st.write(f"Houston Texans Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 13 Pick Percent: :green[{hou_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_13_pick_percent = st.slider("Indianapoils Colts Week 13 Pick Percent:", -1, 100) / 100
                ind_week_13_pick_percent_percent = ind_week_13_pick_percent*100
                if ind_week_13_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 13 Pick Percent: :green[{ind_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_13_pick_percent = st.slider("Jacksonville Jaguars Week 13 Pick Percent:", -1, 100) / 100
                jax_week_13_pick_percent_percent = jax_week_13_pick_percent*100
                if jax_week_13_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 13 Pick Percent: :green[{jax_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_13_pick_percent = st.slider("Kansas City Chiefs Week 13 Pick Percent:", -1, 100) / 100
                kc_week_13_pick_percent_percent = kc_week_13_pick_percent*100
                if kc_week_13_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 13 Pick Percent: :green[{kc_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_13_pick_percent = st.slider("Las Vegas Raiders Week 13 Pick Percent:", -1, 100) / 100
                lv_week_13_pick_percent_percent = lv_week_13_pick_percent*100
                if lv_week_13_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 13 Pick Percent: :green[{lv_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_13_pick_percent = st.slider("Los Angeles Chargers Week 13 Pick Percent:", -1, 100) / 100
                lac_week_13_pick_percent_percent = lac_week_13_pick_percent*100
                if lac_week_13_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 13 Pick Percent: :green[{lac_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_13_pick_percent = st.slider("Los Angeles Rams Week 13 Pick Percent:", -1, 100) / 100
                lar_week_13_pick_percent_percent = lar_week_13_pick_percent*100
                if lar_week_13_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 13 Pick Percent: :green[{lar_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_13_pick_percent = st.slider("Miami Dolphins Week 13 Pick Percent:", -1, 100) / 100
                mia_week_13_pick_percent_percent = mia_week_13_pick_percent*100
                if mia_week_13_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 13 Pick Percent: :green[{mia_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_13_pick_percent = st.slider("Minnesota Vikings Week 13 Pick Percent:", -1, 100) / 100
                min_week_13_pick_percent_percent = min_week_13_pick_percent*100
                if min_week_13_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 13 Pick Percent: :green[{min_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_13_pick_percent = st.slider("New England Patriots Week 13 Pick Percent:", -1, 100) / 100
                ne_week_13_pick_percent_percent = ne_week_13_pick_percent*100
                if ne_week_13_pick_percent < 0:
                    st.write(f"New England Patriots Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 13 Pick Percent: :green[{ne_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_13_pick_percent = st.slider("New Orleans Saints Week 13 Pick Percent:", -1, 100) / 100
                no_week_13_pick_percent_percent = no_week_13_pick_percent*100
                if no_week_13_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 13 Pick Percent: :green[{no_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_13_pick_percent = st.slider("New York Giants Week 13 Pick Percent:", -1, 100) / 100
                nyg_week_13_pick_percent_percent = nyg_week_13_pick_percent*100
                if nyg_week_13_pick_percent < 0:
                    st.write(f"New York Giants Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 13 Pick Percent: :green[{nyg_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_13_pick_percent = st.slider("New York Jets Week 13 Pick Percent:", -1, 100) / 100
                nyj_week_13_pick_percent_percent = nyj_week_13_pick_percent*100
                if nyj_week_13_pick_percent < 0:
                    st.write(f"New York Jets Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 13 Pick Percent: :green[{nyj_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_13_pick_percent = st.slider("Philadelphia Eagles Week 13 Pick Percent:", -1, 100) / 100
                phi_week_13_pick_percent_percent = phi_week_13_pick_percent*100
                if phi_week_13_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 13 Pick Percent: :green[{phi_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_13_pick_percent = st.slider("Pittsburgh Steelers Week 13 Pick Percent:", -1, 100) / 100
                pit_week_13_pick_percent_percent = pit_week_13_pick_percent*100
                if pit_week_13_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 13 Pick Percent: :green[{pit_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_13_pick_percent = st.slider("San Francisco 49ers Week 13 Pick Percent:", -1, 100) / 100
                sf_week_13_pick_percent_percent = sf_week_13_pick_percent*100
                if sf_week_13_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 13 Pick Percent: :green[{sf_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_13_pick_percent = st.slider("Seattle Seahawks Week 13 Pick Percent:", -1, 100) / 100
                sea_week_13_pick_percent_percent = sea_week_13_pick_percent*100
                if sea_week_13_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 13 Pick Percent: :green[{sea_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_13_pick_percent = st.slider("Tampa Bay Buccaneers Week 13 Pick Percent:", -1, 100) / 100
                tb_week_13_pick_percent_percent = tb_week_13_pick_percent*100
                if tb_week_13_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 13 Pick Percent: :green[{tb_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_13_pick_percent = st.slider("Tennessee Titans Week 13 Pick Percent:", -1, 100) / 100
                ten_week_13_pick_percent_percent = ten_week_13_pick_percent*100
                if ten_week_13_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 13 Pick Percent: :green[{ten_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_13_pick_percent = st.slider("Washington Commanders Week 13 Pick Percent:", -1, 100) / 100
                was_week_13_pick_percent_percent = was_week_13_pick_percent*100
                if was_week_13_pick_percent < 0:
                    st.write(f"Washington Commanders Week 13 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 13 Pick Percent: :green[{was_week_13_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 14 and ending_week > 14:
            week_14_pick_percents = st.checkbox('Add Week 14 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 14.")
            if week_14_pick_percents:
                st.write('')
                st.subheader('Week 14 Estimated Pick Percentages')
                st.write('')
                az_week_14_pick_percent = st.slider("Arizona Cardinals Week 14 Pick Percent:", -1, 100) / 100
                az_week_14_pick_percent_percent = az_week_14_pick_percent*100
                if az_week_14_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 14 Pick Percent: :green[{az_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_14_pick_percent = st.slider("Atlanta Falcons Week 14 Pick Percent:", -1, 100) / 100
                atl_week_14_pick_percent_percent = atl_week_14_pick_percent*100
                if atl_week_14_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 14 Pick Percent: :green[{atl_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_14_pick_percent = st.slider("Baltimore Ravens Week 14 Pick Percent:", -1, 100) / 100
                bal_week_14_pick_percent_percent = bal_week_14_pick_percent*100
                if bal_week_14_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 14 Pick Percent: :green[{bal_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_14_pick_percent = st.slider("Buffalo Bills Week 14 Pick Percent:", -1, 100) / 100
                buf_week_14_pick_percent_percent = buf_week_14_pick_percent*100
                if buf_week_14_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 14 Pick Percent: :green[{buf_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_14_pick_percent = st.slider("Carolina Panthers Week 14 Pick Percent:", -1, 100) / 100
                car_week_14_pick_percent_percent = car_week_14_pick_percent*100
                if car_week_14_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 14 Pick Percent: :green[{car_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_14_pick_percent = st.slider("Chicago Bears Week 14 Pick Percent:", -1, 100) / 100
                chi_week_14_pick_percent_percent = chi_week_14_pick_percent*100
                if chi_week_14_pick_percent < 0:
                    st.write(f"Chicago Bears Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 14 Pick Percent: :green[{chi_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_14_pick_percent = st.slider("Cincinnati Bengals Week 14 Pick Percent:", -1, 100) / 100
                cin_week_14_pick_percent_percent = cin_week_14_pick_percent*100
                if cin_week_14_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 14 Pick Percent: :green[{cin_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_14_pick_percent = st.slider("Cleveland Browns Week 14 Pick Percent:", -1, 100) / 100
                cle_week_14_pick_percent_percent = cle_week_14_pick_percent*100
                if cle_week_14_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 14 Pick Percent: :green[{cle_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_14_pick_percent = st.slider("Dallas Cowboys Week 14 Pick Percent:", -1, 100) / 100
                dal_week_14_pick_percent_percent = dal_week_14_pick_percent*100
                if dal_week_14_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 14 Pick Percent: :green[{dal_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_14_pick_percent = st.slider("Denver Broncos Week 14 Pick Percent:", -1, 100) / 100
                den_week_14_pick_percent_percent = den_week_14_pick_percent*100
                if den_week_14_pick_percent < 0:
                    st.write(f"Denver Broncos Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 14 Pick Percent: :green[{den_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_14_pick_percent = st.slider("Detroit Lions Week 14 Pick Percent:", -1, 100) / 100
                det_week_14_pick_percent_percent = det_week_14_pick_percent*100
                if det_week_14_pick_percent < 0:
                    st.write(f"Detroit Lions Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 14 Pick Percent: :green[{det_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_14_pick_percent = st.slider("Green Bay Packers Week 14 Pick Percent:", -1, 100) / 100
                gb_week_14_pick_percent_percent = gb_week_14_pick_percent*100
                if gb_week_14_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 14 Pick Percent: :green[{gb_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_14_pick_percent = st.slider("Houston Texans Week 14 Pick Percent:", -1, 100) / 100
                hou_week_14_pick_percent_percent = hou_week_14_pick_percent*100
                if hou_week_14_pick_percent < 0:
                    st.write(f"Houston Texans Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 14 Pick Percent: :green[{hou_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_14_pick_percent = st.slider("Indianapoils Colts Week 14 Pick Percent:", -1, 100) / 100
                ind_week_14_pick_percent_percent = ind_week_14_pick_percent*100
                if ind_week_14_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 14 Pick Percent: :green[{ind_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_14_pick_percent = st.slider("Jacksonville Jaguars Week 14 Pick Percent:", -1, 100) / 100
                jax_week_14_pick_percent_percent = jax_week_14_pick_percent*100
                if jax_week_14_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 14 Pick Percent: :green[{jax_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_14_pick_percent = st.slider("Kansas City Chiefs Week 14 Pick Percent:", -1, 100) / 100
                kc_week_14_pick_percent_percent = kc_week_14_pick_percent*100
                if kc_week_14_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 14 Pick Percent: :green[{kc_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_14_pick_percent = st.slider("Las Vegas Raiders Week 14 Pick Percent:", -1, 100) / 100
                lv_week_14_pick_percent_percent = lv_week_14_pick_percent*100
                if lv_week_14_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 14 Pick Percent: :green[{lv_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_14_pick_percent = st.slider("Los Angeles Chargers Week 14 Pick Percent:", -1, 100) / 100
                lac_week_14_pick_percent_percent = lac_week_14_pick_percent*100
                if lac_week_14_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 14 Pick Percent: :green[{lac_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_14_pick_percent = st.slider("Los Angeles Rams Week 14 Pick Percent:", -1, 100) / 100
                lar_week_14_pick_percent_percent = lar_week_14_pick_percent*100
                if lar_week_14_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 14 Pick Percent: :green[{lar_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_14_pick_percent = st.slider("Miami Dolphins Week 14 Pick Percent:", -1, 100) / 100
                mia_week_14_pick_percent_percent = mia_week_14_pick_percent*100
                if mia_week_14_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 14 Pick Percent: :green[{mia_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_14_pick_percent = st.slider("Minnesota Vikings Week 14 Pick Percent:", -1, 100) / 100
                min_week_14_pick_percent_percent = min_week_14_pick_percent*100
                if min_week_14_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 14 Pick Percent: :green[{min_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_14_pick_percent = st.slider("New England Patriots Week 14 Pick Percent:", -1, 100) / 100
                ne_week_14_pick_percent_percent = ne_week_14_pick_percent*100
                if ne_week_14_pick_percent < 0:
                    st.write(f"New England Patriots Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 14 Pick Percent: :green[{ne_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_14_pick_percent = st.slider("New Orleans Saints Week 14 Pick Percent:", -1, 100) / 100
                no_week_14_pick_percent_percent = no_week_14_pick_percent*100
                if no_week_14_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 14 Pick Percent: :green[{no_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_14_pick_percent = st.slider("New York Giants Week 14 Pick Percent:", -1, 100) / 100
                nyg_week_14_pick_percent_percent = nyg_week_14_pick_percent*100
                if nyg_week_14_pick_percent < 0:
                    st.write(f"New York Giants Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 14 Pick Percent: :green[{nyg_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_14_pick_percent = st.slider("New York Jets Week 14 Pick Percent:", -1, 100) / 100
                nyj_week_14_pick_percent_percent = nyj_week_14_pick_percent*100
                if nyj_week_14_pick_percent < 0:
                    st.write(f"New York Jets Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 14 Pick Percent: :green[{nyj_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_14_pick_percent = st.slider("Philadelphia Eagles Week 14 Pick Percent:", -1, 100) / 100
                phi_week_14_pick_percent_percent = phi_week_14_pick_percent*100
                if phi_week_14_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 14 Pick Percent: :green[{phi_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_14_pick_percent = st.slider("Pittsburgh Steelers Week 14 Pick Percent:", -1, 100) / 100
                pit_week_14_pick_percent_percent = pit_week_14_pick_percent*100
                if pit_week_14_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 14 Pick Percent: :green[{pit_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_14_pick_percent = st.slider("San Francisco 49ers Week 14 Pick Percent:", -1, 100) / 100
                sf_week_14_pick_percent_percent = sf_week_14_pick_percent*100
                if sf_week_14_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 14 Pick Percent: :green[{sf_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_14_pick_percent = st.slider("Seattle Seahawks Week 14 Pick Percent:", -1, 100) / 100
                sea_week_14_pick_percent_percent = sea_week_14_pick_percent*100
                if sea_week_14_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 14 Pick Percent: :green[{sea_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_14_pick_percent = st.slider("Tampa Bay Buccaneers Week 14 Pick Percent:", -1, 100) / 100
                tb_week_14_pick_percent_percent = tb_week_14_pick_percent*100
                if tb_week_14_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 14 Pick Percent: :green[{tb_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_14_pick_percent = st.slider("Tennessee Titans Week 14 Pick Percent:", -1, 100) / 100
                ten_week_14_pick_percent_percent = ten_week_14_pick_percent*100
                if ten_week_14_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 14 Pick Percent: :green[{ten_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_14_pick_percent = st.slider("Washington Commanders Week 14 Pick Percent:", -1, 100) / 100
                was_week_14_pick_percent_percent = was_week_14_pick_percent*100
                if was_week_14_pick_percent < 0:
                    st.write(f"Washington Commanders Week 14 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 14 Pick Percent: :green[{was_week_14_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 15 and ending_week > 15:
            week_15_pick_percents = st.checkbox('Add Week 15 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 15.")
            if week_15_pick_percents:
                st.write('')
                st.subheader('Week 15 Estimated Pick Percentages')
                st.write('')
                az_week_15_pick_percent = st.slider("Arizona Cardinals Week 15 Pick Percent:", -1, 100) / 100
                az_week_15_pick_percent_percent = az_week_15_pick_percent*100
                if az_week_15_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 15 Pick Percent: :green[{az_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_15_pick_percent = st.slider("Atlanta Falcons Week 15 Pick Percent:", -1, 100) / 100
                atl_week_15_pick_percent_percent = atl_week_15_pick_percent*100
                if atl_week_15_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 15 Pick Percent: :green[{atl_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_15_pick_percent = st.slider("Baltimore Ravens Week 15 Pick Percent:", -1, 100) / 100
                bal_week_15_pick_percent_percent = bal_week_15_pick_percent*100
                if bal_week_15_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 15 Pick Percent: :green[{bal_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_15_pick_percent = st.slider("Buffalo Bills Week 15 Pick Percent:", -1, 100) / 100
                buf_week_15_pick_percent_percent = buf_week_15_pick_percent*100
                if buf_week_15_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 15 Pick Percent: :green[{buf_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_15_pick_percent = st.slider("Carolina Panthers Week 15 Pick Percent:", -1, 100) / 100
                car_week_15_pick_percent_percent = car_week_15_pick_percent*100
                if car_week_15_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 15 Pick Percent: :green[{car_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_15_pick_percent = st.slider("Chicago Bears Week 15 Pick Percent:", -1, 100) / 100
                chi_week_15_pick_percent_percent = chi_week_15_pick_percent*100
                if chi_week_15_pick_percent < 0:
                    st.write(f"Chicago Bears Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 15 Pick Percent: :green[{chi_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_15_pick_percent = st.slider("Cincinnati Bengals Week 15 Pick Percent:", -1, 100) / 100
                cin_week_15_pick_percent_percent = cin_week_15_pick_percent*100
                if cin_week_15_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 15 Pick Percent: :green[{cin_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_15_pick_percent = st.slider("Cleveland Browns Week 15 Pick Percent:", -1, 100) / 100
                cle_week_15_pick_percent_percent = cle_week_15_pick_percent*100
                if cle_week_15_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 15 Pick Percent: :green[{cle_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_15_pick_percent = st.slider("Dallas Cowboys Week 15 Pick Percent:", -1, 100) / 100
                dal_week_15_pick_percent_percent = dal_week_15_pick_percent*100
                if dal_week_15_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 15 Pick Percent: :green[{dal_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_15_pick_percent = st.slider("Denver Broncos Week 15 Pick Percent:", -1, 100) / 100
                den_week_15_pick_percent_percent = den_week_15_pick_percent*100
                if den_week_15_pick_percent < 0:
                    st.write(f"Denver Broncos Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 15 Pick Percent: :green[{den_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_15_pick_percent = st.slider("Detroit Lions Week 15 Pick Percent:", -1, 100) / 100
                det_week_15_pick_percent_percent = det_week_15_pick_percent*100
                if det_week_15_pick_percent < 0:
                    st.write(f"Detroit Lions Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 15 Pick Percent: :green[{det_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_15_pick_percent = st.slider("Green Bay Packers Week 15 Pick Percent:", -1, 100) / 100
                gb_week_15_pick_percent_percent = gb_week_15_pick_percent*100
                if gb_week_15_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 15 Pick Percent: :green[{gb_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_15_pick_percent = st.slider("Houston Texans Week 15 Pick Percent:", -1, 100) / 100
                hou_week_15_pick_percent_percent = hou_week_15_pick_percent*100
                if hou_week_15_pick_percent < 0:
                    st.write(f"Houston Texans Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 15 Pick Percent: :green[{hou_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_15_pick_percent = st.slider("Indianapoils Colts Week 15 Pick Percent:", -1, 100) / 100
                ind_week_15_pick_percent_percent = ind_week_15_pick_percent*100
                if ind_week_15_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 15 Pick Percent: :green[{ind_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_15_pick_percent = st.slider("Jacksonville Jaguars Week 15 Pick Percent:", -1, 100) / 100
                jax_week_15_pick_percent_percent = jax_week_15_pick_percent*100
                if jax_week_15_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 15 Pick Percent: :green[{jax_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_15_pick_percent = st.slider("Kansas City Chiefs Week 15 Pick Percent:", -1, 100) / 100
                kc_week_15_pick_percent_percent = kc_week_15_pick_percent*100
                if kc_week_15_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 15 Pick Percent: :green[{kc_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_15_pick_percent = st.slider("Las Vegas Raiders Week 15 Pick Percent:", -1, 100) / 100
                lv_week_15_pick_percent_percent = lv_week_15_pick_percent*100
                if lv_week_15_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 15 Pick Percent: :green[{lv_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_15_pick_percent = st.slider("Los Angeles Chargers Week 15 Pick Percent:", -1, 100) / 100
                lac_week_15_pick_percent_percent = lac_week_15_pick_percent*100
                if lac_week_15_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 15 Pick Percent: :green[{lac_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_15_pick_percent = st.slider("Los Angeles Rams Week 15 Pick Percent:", -1, 100) / 100
                lar_week_15_pick_percent_percent = lar_week_15_pick_percent*100
                if lar_week_15_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 15 Pick Percent: :green[{lar_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_15_pick_percent = st.slider("Miami Dolphins Week 15 Pick Percent:", -1, 100) / 100
                mia_week_15_pick_percent_percent = mia_week_15_pick_percent*100
                if mia_week_15_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 15 Pick Percent: :green[{mia_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_15_pick_percent = st.slider("Minnesota Vikings Week 15 Pick Percent:", -1, 100) / 100
                min_week_15_pick_percent_percent = min_week_15_pick_percent*100
                if min_week_15_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 15 Pick Percent: :green[{min_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_15_pick_percent = st.slider("New England Patriots Week 15 Pick Percent:", -1, 100) / 100
                ne_week_15_pick_percent_percent = ne_week_15_pick_percent*100
                if ne_week_15_pick_percent < 0:
                    st.write(f"New England Patriots Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 15 Pick Percent: :green[{ne_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_15_pick_percent = st.slider("New Orleans Saints Week 15 Pick Percent:", -1, 100) / 100
                no_week_15_pick_percent_percent = no_week_15_pick_percent*100
                if no_week_15_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 15 Pick Percent: :green[{no_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_15_pick_percent = st.slider("New York Giants Week 15 Pick Percent:", -1, 100) / 100
                nyg_week_15_pick_percent_percent = nyg_week_15_pick_percent*100
                if nyg_week_15_pick_percent < 0:
                    st.write(f"New York Giants Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 15 Pick Percent: :green[{nyg_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_15_pick_percent = st.slider("New York Jets Week 15 Pick Percent:", -1, 100) / 100
                nyj_week_15_pick_percent_percent = nyj_week_15_pick_percent*100
                if nyj_week_15_pick_percent < 0:
                    st.write(f"New York Jets Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 15 Pick Percent: :green[{nyj_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_15_pick_percent = st.slider("Philadelphia Eagles Week 15 Pick Percent:", -1, 100) / 100
                phi_week_15_pick_percent_percent = phi_week_15_pick_percent*100
                if phi_week_15_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 15 Pick Percent: :green[{phi_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_15_pick_percent = st.slider("Pittsburgh Steelers Week 15 Pick Percent:", -1, 100) / 100
                pit_week_15_pick_percent_percent = pit_week_15_pick_percent*100
                if pit_week_15_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 15 Pick Percent: :green[{pit_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_15_pick_percent = st.slider("San Francisco 49ers Week 15 Pick Percent:", -1, 100) / 100
                sf_week_15_pick_percent_percent = sf_week_15_pick_percent*100
                if sf_week_15_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 15 Pick Percent: :green[{sf_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_15_pick_percent = st.slider("Seattle Seahawks Week 15 Pick Percent:", -1, 100) / 100
                sea_week_15_pick_percent_percent = sea_week_15_pick_percent*100
                if sea_week_15_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 15 Pick Percent: :green[{sea_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_15_pick_percent = st.slider("Tampa Bay Buccaneers Week 15 Pick Percent:", -1, 100) / 100
                tb_week_15_pick_percent_percent = tb_week_15_pick_percent*100
                if tb_week_15_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 15 Pick Percent: :green[{tb_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_15_pick_percent = st.slider("Tennessee Titans Week 15 Pick Percent:", -1, 100) / 100
                ten_week_15_pick_percent_percent = ten_week_15_pick_percent*100
                if ten_week_15_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 15 Pick Percent: :green[{ten_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_15_pick_percent = st.slider("Washington Commanders Week 15 Pick Percent:", -1, 100) / 100
                was_week_15_pick_percent_percent = was_week_15_pick_percent*100
                if was_week_15_pick_percent < 0:
                    st.write(f"Washington Commanders Week 15 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 15 Pick Percent: :green[{was_week_15_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 16 and ending_week > 16:
            week_16_pick_percents = st.checkbox('Add Week 16 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 16.")
            if week_16_pick_percents:
                st.write('')
                st.subheader('Week 16 Estimated Pick Percentages')
                st.write('')
                az_week_16_pick_percent = st.slider("Arizona Cardinals Week 16 Pick Percent:", -1, 100) / 100
                az_week_16_pick_percent_percent = az_week_16_pick_percent*100
                if az_week_16_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 16 Pick Percent: :green[{az_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_16_pick_percent = st.slider("Atlanta Falcons Week 16 Pick Percent:", -1, 100) / 100
                atl_week_16_pick_percent_percent = atl_week_16_pick_percent*100
                if atl_week_16_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 16 Pick Percent: :green[{atl_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_16_pick_percent = st.slider("Baltimore Ravens Week 16 Pick Percent:", -1, 100) / 100
                bal_week_16_pick_percent_percent = bal_week_16_pick_percent*100
                if bal_week_16_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 16 Pick Percent: :green[{bal_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_16_pick_percent = st.slider("Buffalo Bills Week 16 Pick Percent:", -1, 100) / 100
                buf_week_16_pick_percent_percent = buf_week_16_pick_percent*100
                if buf_week_16_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 16 Pick Percent: :green[{buf_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_16_pick_percent = st.slider("Carolina Panthers Week 16 Pick Percent:", -1, 100) / 100
                car_week_16_pick_percent_percent = car_week_16_pick_percent*100
                if car_week_16_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 16 Pick Percent: :green[{car_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_16_pick_percent = st.slider("Chicago Bears Week 16 Pick Percent:", -1, 100) / 100
                chi_week_16_pick_percent_percent = chi_week_16_pick_percent*100
                if chi_week_16_pick_percent < 0:
                    st.write(f"Chicago Bears Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 16 Pick Percent: :green[{chi_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_16_pick_percent = st.slider("Cincinnati Bengals Week 16 Pick Percent:", -1, 100) / 100
                cin_week_16_pick_percent_percent = cin_week_16_pick_percent*100
                if cin_week_16_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 16 Pick Percent: :green[{cin_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_16_pick_percent = st.slider("Cleveland Browns Week 16 Pick Percent:", -1, 100) / 100
                cle_week_16_pick_percent_percent = cle_week_16_pick_percent*100
                if cle_week_16_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 16 Pick Percent: :green[{cle_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_16_pick_percent = st.slider("Dallas Cowboys Week 16 Pick Percent:", -1, 100) / 100
                dal_week_16_pick_percent_percent = dal_week_16_pick_percent*100
                if dal_week_16_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 16 Pick Percent: :green[{dal_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_16_pick_percent = st.slider("Denver Broncos Week 16 Pick Percent:", -1, 100) / 100
                den_week_16_pick_percent_percent = den_week_16_pick_percent*100
                if den_week_16_pick_percent < 0:
                    st.write(f"Denver Broncos Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 16 Pick Percent: :green[{den_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_16_pick_percent = st.slider("Detroit Lions Week 16 Pick Percent:", -1, 100) / 100
                det_week_16_pick_percent_percent = det_week_16_pick_percent*100
                if det_week_16_pick_percent < 0:
                    st.write(f"Detroit Lions Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 16 Pick Percent: :green[{det_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_16_pick_percent = st.slider("Green Bay Packers Week 16 Pick Percent:", -1, 100) / 100
                gb_week_16_pick_percent_percent = gb_week_16_pick_percent*100
                if gb_week_16_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 16 Pick Percent: :green[{gb_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_16_pick_percent = st.slider("Houston Texans Week 16 Pick Percent:", -1, 100) / 100
                hou_week_16_pick_percent_percent = hou_week_16_pick_percent*100
                if hou_week_16_pick_percent < 0:
                    st.write(f"Houston Texans Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 16 Pick Percent: :green[{hou_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_16_pick_percent = st.slider("Indianapoils Colts Week 16 Pick Percent:", -1, 100) / 100
                ind_week_16_pick_percent_percent = ind_week_16_pick_percent*100
                if ind_week_16_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 16 Pick Percent: :green[{ind_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_16_pick_percent = st.slider("Jacksonville Jaguars Week 16 Pick Percent:", -1, 100) / 100
                jax_week_16_pick_percent_percent = jax_week_16_pick_percent*100
                if jax_week_16_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 16 Pick Percent: :green[{jax_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_16_pick_percent = st.slider("Kansas City Chiefs Week 16 Pick Percent:", -1, 100) / 100
                kc_week_16_pick_percent_percent = kc_week_16_pick_percent*100
                if kc_week_16_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 16 Pick Percent: :green[{kc_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_16_pick_percent = st.slider("Las Vegas Raiders Week 16 Pick Percent:", -1, 100) / 100
                lv_week_16_pick_percent_percent = lv_week_16_pick_percent*100
                if lv_week_16_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 16 Pick Percent: :green[{lv_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_16_pick_percent = st.slider("Los Angeles Chargers Week 16 Pick Percent:", -1, 100) / 100
                lac_week_16_pick_percent_percent = lac_week_16_pick_percent*100
                if lac_week_16_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 16 Pick Percent: :green[{lac_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_16_pick_percent = st.slider("Los Angeles Rams Week 16 Pick Percent:", -1, 100) / 100
                lar_week_16_pick_percent_percent = lar_week_16_pick_percent*100
                if lar_week_16_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 16 Pick Percent: :green[{lar_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_16_pick_percent = st.slider("Miami Dolphins Week 16 Pick Percent:", -1, 100) / 100
                mia_week_16_pick_percent_percent = mia_week_16_pick_percent*100
                if mia_week_16_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 16 Pick Percent: :green[{mia_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_16_pick_percent = st.slider("Minnesota Vikings Week 16 Pick Percent:", -1, 100) / 100
                min_week_16_pick_percent_percent = min_week_16_pick_percent*100
                if min_week_16_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 16 Pick Percent: :green[{min_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_16_pick_percent = st.slider("New England Patriots Week 16 Pick Percent:", -1, 100) / 100
                ne_week_16_pick_percent_percent = ne_week_16_pick_percent*100
                if ne_week_16_pick_percent < 0:
                    st.write(f"New England Patriots Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 16 Pick Percent: :green[{ne_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_16_pick_percent = st.slider("New Orleans Saints Week 16 Pick Percent:", -1, 100) / 100
                no_week_16_pick_percent_percent = no_week_16_pick_percent*100
                if no_week_16_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 16 Pick Percent: :green[{no_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_16_pick_percent = st.slider("New York Giants Week 16 Pick Percent:", -1, 100) / 100
                nyg_week_16_pick_percent_percent = nyg_week_16_pick_percent*100
                if nyg_week_16_pick_percent < 0:
                    st.write(f"New York Giants Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 16 Pick Percent: :green[{nyg_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_16_pick_percent = st.slider("New York Jets Week 16 Pick Percent:", -1, 100) / 100
                nyj_week_16_pick_percent_percent = nyj_week_16_pick_percent*100
                if nyj_week_16_pick_percent < 0:
                    st.write(f"New York Jets Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 16 Pick Percent: :green[{nyj_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_16_pick_percent = st.slider("Philadelphia Eagles Week 16 Pick Percent:", -1, 100) / 100
                phi_week_16_pick_percent_percent = phi_week_16_pick_percent*100
                if phi_week_16_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 16 Pick Percent: :green[{phi_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_16_pick_percent = st.slider("Pittsburgh Steelers Week 16 Pick Percent:", -1, 100) / 100
                pit_week_16_pick_percent_percent = pit_week_16_pick_percent*100
                if pit_week_16_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 16 Pick Percent: :green[{pit_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_16_pick_percent = st.slider("San Francisco 49ers Week 16 Pick Percent:", -1, 100) / 100
                sf_week_16_pick_percent_percent = sf_week_16_pick_percent*100
                if sf_week_16_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 16 Pick Percent: :green[{sf_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_16_pick_percent = st.slider("Seattle Seahawks Week 16 Pick Percent:", -1, 100) / 100
                sea_week_16_pick_percent_percent = sea_week_16_pick_percent*100
                if sea_week_16_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 16 Pick Percent: :green[{sea_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_16_pick_percent = st.slider("Tampa Bay Buccaneers Week 16 Pick Percent:", -1, 100) / 100
                tb_week_16_pick_percent_percent = tb_week_16_pick_percent*100
                if tb_week_16_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 16 Pick Percent: :green[{tb_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_16_pick_percent = st.slider("Tennessee Titans Week 16 Pick Percent:", -1, 100) / 100
                ten_week_16_pick_percent_percent = ten_week_16_pick_percent*100
                if ten_week_16_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 16 Pick Percent: :green[{ten_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_16_pick_percent = st.slider("Washington Commanders Week 16 Pick Percent:", -1, 100) / 100
                was_week_16_pick_percent_percent = was_week_16_pick_percent*100
                if was_week_16_pick_percent < 0:
                    st.write(f"Washington Commanders Week 16 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 16 Pick Percent: :green[{was_week_16_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 17 and ending_week > 17:
            week_17_pick_percents = st.checkbox('Add Week 17 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 17.")
            if week_17_pick_percents:
                st.write('')
                st.subheader('Week 17 Estimated Pick Percentages')
                st.write('')
                az_week_17_pick_percent = st.slider("Arizona Cardinals Week 17 Pick Percent:", -1, 100) / 100
                az_week_17_pick_percent_percent = az_week_17_pick_percent*100
                if az_week_17_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 17 Pick Percent: :green[{az_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_17_pick_percent = st.slider("Atlanta Falcons Week 17 Pick Percent:", -1, 100) / 100
                atl_week_17_pick_percent_percent = atl_week_17_pick_percent*100
                if atl_week_17_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 17 Pick Percent: :green[{atl_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_17_pick_percent = st.slider("Baltimore Ravens Week 17 Pick Percent:", -1, 100) / 100
                bal_week_17_pick_percent_percent = bal_week_17_pick_percent*100
                if bal_week_17_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 17 Pick Percent: :green[{bal_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_17_pick_percent = st.slider("Buffalo Bills Week 17 Pick Percent:", -1, 100) / 100
                buf_week_17_pick_percent_percent = buf_week_17_pick_percent*100
                if buf_week_17_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 17 Pick Percent: :green[{buf_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_17_pick_percent = st.slider("Carolina Panthers Week 17 Pick Percent:", -1, 100) / 100
                car_week_17_pick_percent_percent = car_week_17_pick_percent*100
                if car_week_17_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 17 Pick Percent: :green[{car_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_17_pick_percent = st.slider("Chicago Bears Week 17 Pick Percent:", -1, 100) / 100
                chi_week_17_pick_percent_percent = chi_week_17_pick_percent*100
                if chi_week_17_pick_percent < 0:
                    st.write(f"Chicago Bears Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 17 Pick Percent: :green[{chi_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_17_pick_percent = st.slider("Cincinnati Bengals Week 17 Pick Percent:", -1, 100) / 100
                cin_week_17_pick_percent_percent = cin_week_17_pick_percent*100
                if cin_week_17_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 17 Pick Percent: :green[{cin_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_17_pick_percent = st.slider("Cleveland Browns Week 17 Pick Percent:", -1, 100) / 100
                cle_week_17_pick_percent_percent = cle_week_17_pick_percent*100
                if cle_week_17_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 17 Pick Percent: :green[{cle_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_17_pick_percent = st.slider("Dallas Cowboys Week 17 Pick Percent:", -1, 100) / 100
                dal_week_17_pick_percent_percent = dal_week_17_pick_percent*100
                if dal_week_17_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 17 Pick Percent: :green[{dal_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_17_pick_percent = st.slider("Denver Broncos Week 17 Pick Percent:", -1, 100) / 100
                den_week_17_pick_percent_percent = den_week_17_pick_percent*100
                if den_week_17_pick_percent < 0:
                    st.write(f"Denver Broncos Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 17 Pick Percent: :green[{den_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_17_pick_percent = st.slider("Detroit Lions Week 17 Pick Percent:", -1, 100) / 100
                det_week_17_pick_percent_percent = det_week_17_pick_percent*100
                if det_week_17_pick_percent < 0:
                    st.write(f"Detroit Lions Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 17 Pick Percent: :green[{det_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_17_pick_percent = st.slider("Green Bay Packers Week 17 Pick Percent:", -1, 100) / 100
                gb_week_17_pick_percent_percent = gb_week_17_pick_percent*100
                if gb_week_17_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 17 Pick Percent: :green[{gb_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_17_pick_percent = st.slider("Houston Texans Week 17 Pick Percent:", -1, 100) / 100
                hou_week_17_pick_percent_percent = hou_week_17_pick_percent*100
                if hou_week_17_pick_percent < 0:
                    st.write(f"Houston Texans Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 17 Pick Percent: :green[{hou_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_17_pick_percent = st.slider("Indianapoils Colts Week 17 Pick Percent:", -1, 100) / 100
                ind_week_17_pick_percent_percent = ind_week_17_pick_percent*100
                if ind_week_17_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 17 Pick Percent: :green[{ind_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_17_pick_percent = st.slider("Jacksonville Jaguars Week 17 Pick Percent:", -1, 100) / 100
                jax_week_17_pick_percent_percent = jax_week_17_pick_percent*100
                if jax_week_17_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 17 Pick Percent: :green[{jax_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_17_pick_percent = st.slider("Kansas City Chiefs Week 17 Pick Percent:", -1, 100) / 100
                kc_week_17_pick_percent_percent = kc_week_17_pick_percent*100
                if kc_week_17_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 17 Pick Percent: :green[{kc_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_17_pick_percent = st.slider("Las Vegas Raiders Week 17 Pick Percent:", -1, 100) / 100
                lv_week_17_pick_percent_percent = lv_week_17_pick_percent*100
                if lv_week_17_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 17 Pick Percent: :green[{lv_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_17_pick_percent = st.slider("Los Angeles Chargers Week 17 Pick Percent:", -1, 100) / 100
                lac_week_17_pick_percent_percent = lac_week_17_pick_percent*100
                if lac_week_17_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 17 Pick Percent: :green[{lac_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_17_pick_percent = st.slider("Los Angeles Rams Week 17 Pick Percent:", -1, 100) / 100
                lar_week_17_pick_percent_percent = lar_week_17_pick_percent*100
                if lar_week_17_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 17 Pick Percent: :green[{lar_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_17_pick_percent = st.slider("Miami Dolphins Week 17 Pick Percent:", -1, 100) / 100
                mia_week_17_pick_percent_percent = mia_week_17_pick_percent*100
                if mia_week_17_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 17 Pick Percent: :green[{mia_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_17_pick_percent = st.slider("Minnesota Vikings Week 17 Pick Percent:", -1, 100) / 100
                min_week_17_pick_percent_percent = min_week_17_pick_percent*100
                if min_week_17_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 17 Pick Percent: :green[{min_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_17_pick_percent = st.slider("New England Patriots Week 17 Pick Percent:", -1, 100) / 100
                ne_week_17_pick_percent_percent = ne_week_17_pick_percent*100
                if ne_week_17_pick_percent < 0:
                    st.write(f"New England Patriots Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 17 Pick Percent: :green[{ne_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_17_pick_percent = st.slider("New Orleans Saints Week 17 Pick Percent:", -1, 100) / 100
                no_week_17_pick_percent_percent = no_week_17_pick_percent*100
                if no_week_17_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 17 Pick Percent: :green[{no_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_17_pick_percent = st.slider("New York Giants Week 17 Pick Percent:", -1, 100) / 100
                nyg_week_17_pick_percent_percent = nyg_week_17_pick_percent*100
                if nyg_week_17_pick_percent < 0:
                    st.write(f"New York Giants Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 17 Pick Percent: :green[{nyg_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_17_pick_percent = st.slider("New York Jets Week 17 Pick Percent:", -1, 100) / 100
                nyj_week_17_pick_percent_percent = nyj_week_17_pick_percent*100
                if nyj_week_17_pick_percent < 0:
                    st.write(f"New York Jets Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 17 Pick Percent: :green[{nyj_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_17_pick_percent = st.slider("Philadelphia Eagles Week 17 Pick Percent:", -1, 100) / 100
                phi_week_17_pick_percent_percent = phi_week_17_pick_percent*100
                if phi_week_17_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 17 Pick Percent: :green[{phi_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_17_pick_percent = st.slider("Pittsburgh Steelers Week 17 Pick Percent:", -1, 100) / 100
                pit_week_17_pick_percent_percent = pit_week_17_pick_percent*100
                if pit_week_17_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 17 Pick Percent: :green[{pit_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_17_pick_percent = st.slider("San Francisco 49ers Week 17 Pick Percent:", -1, 100) / 100
                sf_week_17_pick_percent_percent = sf_week_17_pick_percent*100
                if sf_week_17_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 17 Pick Percent: :green[{sf_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_17_pick_percent = st.slider("Seattle Seahawks Week 17 Pick Percent:", -1, 100) / 100
                sea_week_17_pick_percent_percent = sea_week_17_pick_percent*100
                if sea_week_17_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 17 Pick Percent: :green[{sea_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_17_pick_percent = st.slider("Tampa Bay Buccaneers Week 17 Pick Percent:", -1, 100) / 100
                tb_week_17_pick_percent_percent = tb_week_17_pick_percent*100
                if tb_week_17_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 17 Pick Percent: :green[{tb_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_17_pick_percent = st.slider("Tennessee Titans Week 17 Pick Percent:", -1, 100) / 100
                ten_week_17_pick_percent_percent = ten_week_17_pick_percent*100
                if ten_week_17_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 17 Pick Percent: :green[{ten_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_17_pick_percent = st.slider("Washington Commanders Week 17 Pick Percent:", -1, 100) / 100
                was_week_17_pick_percent_percent = was_week_17_pick_percent*100
                if was_week_17_pick_percent < 0:
                    st.write(f"Washington Commanders Week 17 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 17 Pick Percent: :green[{was_week_17_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if starting_week <= 18 and ending_week > 18:
            week_18_pick_percents = st.checkbox('Add Week 18 (Christmas Week if in Circa Survivor Contest) Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 18. (Christmas Week if in Circa Survivor Contest)")
            if week_18_pick_percents:
                st.write('')
                st.subheader('Week 18 Estimated Pick Percentages')
                st.write('')
                az_week_18_pick_percent = st.slider("Arizona Cardinals Week 18 Pick Percent:", -1, 100) / 100
                az_week_18_pick_percent_percent = az_week_18_pick_percent*100
                if az_week_18_pick_percent < 0:
                    st.write(f"Arizona Cardinals Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Arizona Cardinals Week 18 Pick Percent: :green[{az_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                atl_week_18_pick_percent = st.slider("Atlanta Falcons Week 18 Pick Percent:", -1, 100) / 100
                atl_week_18_pick_percent_percent = atl_week_18_pick_percent*100
                if atl_week_18_pick_percent < 0:
                    st.write(f"Atlanta Falcons Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Atlanta Falcons Week 18 Pick Percent: :green[{atl_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                bal_week_18_pick_percent = st.slider("Baltimore Ravens Week 18 Pick Percent:", -1, 100) / 100
                bal_week_18_pick_percent_percent = bal_week_18_pick_percent*100
                if bal_week_18_pick_percent < 0:
                    st.write(f"Baltimore Ravens Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Baltimore Ravenss Week 18 Pick Percent: :green[{bal_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                buf_week_18_pick_percent = st.slider("Buffalo Bills Week 18 Pick Percent:", -1, 100) / 100
                buf_week_18_pick_percent_percent = buf_week_18_pick_percent*100
                if buf_week_18_pick_percent < 0:
                    st.write(f"Buffalo Bills Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Buffalo Bills Week 18 Pick Percent: :green[{buf_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                car_week_18_pick_percent = st.slider("Carolina Panthers Week 18 Pick Percent:", -1, 100) / 100
                car_week_18_pick_percent_percent = car_week_18_pick_percent*100
                if car_week_18_pick_percent < 0:
                    st.write(f"Carolina Panthers Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Carolina Panthers Week 18 Pick Percent: :green[{car_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                chi_week_18_pick_percent = st.slider("Chicago Bears Week 18 Pick Percent:", -1, 100) / 100
                chi_week_18_pick_percent_percent = chi_week_18_pick_percent*100
                if chi_week_18_pick_percent < 0:
                    st.write(f"Chicago Bears Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Chicago Bears Week 18 Pick Percent: :green[{chi_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cin_week_18_pick_percent = st.slider("Cincinnati Bengals Week 18 Pick Percent:", -1, 100) / 100
                cin_week_18_pick_percent_percent = cin_week_18_pick_percent*100
                if cin_week_18_pick_percent < 0:
                    st.write(f"Cincinnati Bengals Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cincinnati Bengals Week 18 Pick Percent: :green[{cin_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                cle_week_18_pick_percent = st.slider("Cleveland Browns Week 18 Pick Percent:", -1, 100) / 100
                cle_week_18_pick_percent_percent = cle_week_18_pick_percent*100
                if cle_week_18_pick_percent < 0:
                    st.write(f"Cleveland Browns Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Cleveland Browns Week 18 Pick Percent: :green[{cle_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                dal_week_18_pick_percent = st.slider("Dallas Cowboys Week 18 Pick Percent:", -1, 100) / 100
                dal_week_18_pick_percent_percent = dal_week_18_pick_percent*100
                if dal_week_18_pick_percent < 0:
                    st.write(f"Dallas Cowboys Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Dallas Cowboys Week 18 Pick Percent: :green[{dal_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                den_week_18_pick_percent = st.slider("Denver Broncos Week 18 Pick Percent:", -1, 100) / 100
                den_week_18_pick_percent_percent = den_week_18_pick_percent*100
                if den_week_18_pick_percent < 0:
                    st.write(f"Denver Broncos Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Denver Broncos Week 18 Pick Percent: :green[{den_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                det_week_18_pick_percent = st.slider("Detroit Lions Week 18 Pick Percent:", -1, 100) / 100
                det_week_18_pick_percent_percent = det_week_18_pick_percent*100
                if det_week_18_pick_percent < 0:
                    st.write(f"Detroit Lions Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Detroit Lions Week 18 Pick Percent: :green[{det_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                gb_week_18_pick_percent = st.slider("Green Bay Packers Week 18 Pick Percent:", -1, 100) / 100
                gb_week_18_pick_percent_percent = gb_week_18_pick_percent*100
                if gb_week_18_pick_percent < 0:
                    st.write(f"Green Bay Packers Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Green Bay Packers Week 18 Pick Percent: :green[{gb_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                hou_week_18_pick_percent = st.slider("Houston Texans Week 18 Pick Percent:", -1, 100) / 100
                hou_week_18_pick_percent_percent = hou_week_18_pick_percent*100
                if hou_week_18_pick_percent < 0:
                    st.write(f"Houston Texans Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Houston Texans Week 18 Pick Percent: :green[{hou_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ind_week_18_pick_percent = st.slider("Indianapoils Colts Week 18 Pick Percent:", -1, 100) / 100
                ind_week_18_pick_percent_percent = ind_week_18_pick_percent*100
                if ind_week_18_pick_percent < 0:
                    st.write(f"Indianapoils Colts Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Indianapoils Colts Week 18 Pick Percent: :green[{ind_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                jax_week_18_pick_percent = st.slider("Jacksonville Jaguars Week 18 Pick Percent:", -1, 100) / 100
                jax_week_18_pick_percent_percent = jax_week_18_pick_percent*100
                if jax_week_18_pick_percent < 0:
                    st.write(f"Jacksonville Jaguars Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Jacksonville Jaguars Week 18 Pick Percent: :green[{jax_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                kc_week_18_pick_percent = st.slider("Kansas City Chiefs Week 18 Pick Percent:", -1, 100) / 100
                kc_week_18_pick_percent_percent = kc_week_18_pick_percent*100
                if kc_week_18_pick_percent < 0:
                    st.write(f"Kansas City Chiefs Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Kansas City Chiefs Week 18 Pick Percent: :green[{kc_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lv_week_18_pick_percent = st.slider("Las Vegas Raiders Week 18 Pick Percent:", -1, 100) / 100
                lv_week_18_pick_percent_percent = lv_week_18_pick_percent*100
                if lv_week_18_pick_percent < 0:
                    st.write(f"Las Vegas Raiders Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Las vegas Raiders Week 18 Pick Percent: :green[{lv_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lac_week_18_pick_percent = st.slider("Los Angeles Chargers Week 18 Pick Percent:", -1, 100) / 100
                lac_week_18_pick_percent_percent = lac_week_18_pick_percent*100
                if lac_week_18_pick_percent < 0:
                    st.write(f"Los Angeles Chargers Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los angeles Chargers Week 18 Pick Percent: :green[{lac_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                lar_week_18_pick_percent = st.slider("Los Angeles Rams Week 18 Pick Percent:", -1, 100) / 100
                lar_week_18_pick_percent_percent = lar_week_18_pick_percent*100
                if lar_week_18_pick_percent < 0:
                    st.write(f"Los Angeles Rams Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Los Angeles Rams Week 18 Pick Percent: :green[{lar_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                mia_week_18_pick_percent = st.slider("Miami Dolphins Week 18 Pick Percent:", -1, 100) / 100
                mia_week_18_pick_percent_percent = mia_week_18_pick_percent*100
                if mia_week_18_pick_percent < 0:
                    st.write(f"Miami Dolphins Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Miami Dolphins Week 18 Pick Percent: :green[{mia_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                min_week_18_pick_percent = st.slider("Minnesota Vikings Week 18 Pick Percent:", -1, 100) / 100
                min_week_18_pick_percent_percent = min_week_18_pick_percent*100
                if min_week_18_pick_percent < 0:
                    st.write(f"Minnesota Vikings Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Minnesota Vikings Week 18 Pick Percent: :green[{min_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ne_week_18_pick_percent = st.slider("New England Patriots Week 18 Pick Percent:", -1, 100) / 100
                ne_week_18_pick_percent_percent = ne_week_18_pick_percent*100
                if ne_week_18_pick_percent < 0:
                    st.write(f"New England Patriots Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New England Patriots Week 18 Pick Percent: :green[{ne_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                no_week_18_pick_percent = st.slider("New Orleans Saints Week 18 Pick Percent:", -1, 100) / 100
                no_week_18_pick_percent_percent = no_week_18_pick_percent*100
                if no_week_18_pick_percent < 0:
                    st.write(f"New Orleans Saints Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New Orleans Saints Week 18 Pick Percent: :green[{no_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyg_week_18_pick_percent = st.slider("New York Giants Week 18 Pick Percent:", -1, 100) / 100
                nyg_week_18_pick_percent_percent = nyg_week_18_pick_percent*100
                if nyg_week_18_pick_percent < 0:
                    st.write(f"New York Giants Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Giants Week 18 Pick Percent: :green[{nyg_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                nyj_week_18_pick_percent = st.slider("New York Jets Week 18 Pick Percent:", -1, 100) / 100
                nyj_week_18_pick_percent_percent = nyj_week_18_pick_percent*100
                if nyj_week_18_pick_percent < 0:
                    st.write(f"New York Jets Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"New York Jets Week 18 Pick Percent: :green[{nyj_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                phi_week_18_pick_percent = st.slider("Philadelphia Eagles Week 18 Pick Percent:", -1, 100) / 100
                phi_week_18_pick_percent_percent = phi_week_18_pick_percent*100
                if phi_week_18_pick_percent < 0:
                    st.write(f"Philadelphia Eagles Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Philadelphia Eagles Week 18 Pick Percent: :green[{phi_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                pit_week_18_pick_percent = st.slider("Pittsburgh Steelers Week 18 Pick Percent:", -1, 100) / 100
                pit_week_18_pick_percent_percent = pit_week_18_pick_percent*100
                if pit_week_18_pick_percent < 0:
                    st.write(f"Pittsburgh Steelers Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Pittsburgh Steelers Week 18 Pick Percent: :green[{pit_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sf_week_18_pick_percent = st.slider("San Francisco 49ers Week 18 Pick Percent:", -1, 100) / 100
                sf_week_18_pick_percent_percent = sf_week_18_pick_percent*100
                if sf_week_18_pick_percent < 0:
                    st.write(f"San Francisco 49ers Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"San Francisco 49ers Week 18 Pick Percent: :green[{sf_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                sea_week_18_pick_percent = st.slider("Seattle Seahawks Week 18 Pick Percent:", -1, 100) / 100
                sea_week_18_pick_percent_percent = sea_week_18_pick_percent*100
                if sea_week_18_pick_percent < 0:
                    st.write(f"Seattle Seahawks Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Seattle Seahawks Week 18 Pick Percent: :green[{sea_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                tb_week_18_pick_percent = st.slider("Tampa Bay Buccaneers Week 18 Pick Percent:", -1, 100) / 100
                tb_week_18_pick_percent_percent = tb_week_18_pick_percent*100
                if tb_week_18_pick_percent < 0:
                    st.write(f"Tampa Bay Buccaneers Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tampa Bay Buccaneers Week 18 Pick Percent: :green[{tb_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                ten_week_18_pick_percent = st.slider("Tennessee Titans Week 18 Pick Percent:", -1, 100) / 100
                ten_week_18_pick_percent_percent = ten_week_18_pick_percent*100
                if ten_week_18_pick_percent < 0:
                    st.write(f"Tennessee Titans Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Tennessee Titans Week 18 Pick Percent: :green[{ten_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
        	    
                was_week_18_pick_percent = st.slider("Washington Commanders Week 18 Pick Percent:", -1, 100) / 100
                was_week_18_pick_percent_percent = was_week_18_pick_percent*100
                if was_week_18_pick_percent < 0:
                    st.write(f"Washington Commanders Week 18 Pick Percent: :red[Automatically Calculated]")
                else:
                    st.write(f"Washington Commanders Week 18 Pick Percent: :green[{was_week_18_pick_percent_percent}%]")
                st.write("")
                st.write("")
            st.write('')
            st.write('')
        if selected_contest == 'Circa':
            if starting_week <= 19 and ending_week > 19:
                week_19_pick_percents = st.checkbox('Add Week 19 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 19.")
                if week_19_pick_percents:
                    st.write('')
                    st.subheader('Week 19 Estimated Pick Percentages')
                    st.write('')
                    az_week_19_pick_percent = st.slider("Arizona Cardinals Week 19 Pick Percent:", -1, 100) / 100
                    az_week_19_pick_percent_percent = az_week_19_pick_percent*100
                    if az_week_19_pick_percent < 0:
                        st.write(f"Arizona Cardinals Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Arizona Cardinals Week 19 Pick Percent: :green[{az_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    atl_week_19_pick_percent = st.slider("Atlanta Falcons Week 19 Pick Percent:", -1, 100) / 100
                    atl_week_19_pick_percent_percent = atl_week_19_pick_percent*100
                    if atl_week_19_pick_percent < 0:
                        st.write(f"Atlanta Falcons Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Atlanta Falcons Week 19 Pick Percent: :green[{atl_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    bal_week_19_pick_percent = st.slider("Baltimore Ravens Week 19 Pick Percent:", -1, 100) / 100
                    bal_week_19_pick_percent_percent = bal_week_19_pick_percent*100
                    if bal_week_19_pick_percent < 0:
                        st.write(f"Baltimore Ravens Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Baltimore Ravenss Week 19 Pick Percent: :green[{bal_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    buf_week_19_pick_percent = st.slider("Buffalo Bills Week 19 Pick Percent:", -1, 100) / 100
                    buf_week_19_pick_percent_percent = buf_week_19_pick_percent*100
                    if buf_week_19_pick_percent < 0:
                        st.write(f"Buffalo Bills Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Buffalo Bills Week 19 Pick Percent: :green[{buf_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    car_week_19_pick_percent = st.slider("Carolina Panthers Week 19 Pick Percent:", -1, 100) / 100
                    car_week_19_pick_percent_percent = car_week_19_pick_percent*100
                    if car_week_19_pick_percent < 0:
                        st.write(f"Carolina Panthers Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Carolina Panthers Week 19 Pick Percent: :green[{car_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    chi_week_19_pick_percent = st.slider("Chicago Bears Week 19 Pick Percent:", -1, 100) / 100
                    chi_week_19_pick_percent_percent = chi_week_19_pick_percent*100
                    if chi_week_19_pick_percent < 0:
                        st.write(f"Chicago Bears Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Chicago Bears Week 19 Pick Percent: :green[{chi_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    cin_week_19_pick_percent = st.slider("Cincinnati Bengals Week 19 Pick Percent:", -1, 100) / 100
                    cin_week_19_pick_percent_percent = cin_week_19_pick_percent*100
                    if cin_week_19_pick_percent < 0:
                        st.write(f"Cincinnati Bengals Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Cincinnati Bengals Week 19 Pick Percent: :green[{cin_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    cle_week_19_pick_percent = st.slider("Cleveland Browns Week 19 Pick Percent:", -1, 100) / 100
                    cle_week_19_pick_percent_percent = cle_week_19_pick_percent*100
                    if cle_week_19_pick_percent < 0:
                        st.write(f"Cleveland Browns Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Cleveland Browns Week 19 Pick Percent: :green[{cle_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    dal_week_19_pick_percent = st.slider("Dallas Cowboys Week 19 Pick Percent:", -1, 100) / 100
                    dal_week_19_pick_percent_percent = dal_week_19_pick_percent*100
                    if dal_week_19_pick_percent < 0:
                        st.write(f"Dallas Cowboys Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Dallas Cowboys Week 19 Pick Percent: :green[{dal_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    den_week_19_pick_percent = st.slider("Denver Broncos Week 19 Pick Percent:", -1, 100) / 100
                    den_week_19_pick_percent_percent = den_week_19_pick_percent*100
                    if den_week_19_pick_percent < 0:
                        st.write(f"Denver Broncos Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Denver Broncos Week 19 Pick Percent: :green[{den_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    det_week_19_pick_percent = st.slider("Detroit Lions Week 19 Pick Percent:", -1, 100) / 100
                    det_week_19_pick_percent_percent = det_week_19_pick_percent*100
                    if det_week_19_pick_percent < 0:
                        st.write(f"Detroit Lions Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Detroit Lions Week 19 Pick Percent: :green[{det_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    gb_week_19_pick_percent = st.slider("Green Bay Packers Week 19 Pick Percent:", -1, 100) / 100
                    gb_week_19_pick_percent_percent = gb_week_19_pick_percent*100
                    if gb_week_19_pick_percent < 0:
                        st.write(f"Green Bay Packers Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Green Bay Packers Week 19 Pick Percent: :green[{gb_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    hou_week_19_pick_percent = st.slider("Houston Texans Week 19 Pick Percent:", -1, 100) / 100
                    hou_week_19_pick_percent_percent = hou_week_19_pick_percent*100
                    if hou_week_19_pick_percent < 0:
                        st.write(f"Houston Texans Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Houston Texans Week 19 Pick Percent: :green[{hou_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    ind_week_19_pick_percent = st.slider("Indianapoils Colts Week 19 Pick Percent:", -1, 100) / 100
                    ind_week_19_pick_percent_percent = ind_week_19_pick_percent*100
                    if ind_week_19_pick_percent < 0:
                        st.write(f"Indianapoils Colts Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Indianapoils Colts Week 19 Pick Percent: :green[{ind_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    jax_week_19_pick_percent = st.slider("Jacksonville Jaguars Week 19 Pick Percent:", -1, 100) / 100
                    jax_week_19_pick_percent_percent = jax_week_19_pick_percent*100
                    if jax_week_19_pick_percent < 0:
                        st.write(f"Jacksonville Jaguars Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Jacksonville Jaguars Week 19 Pick Percent: :green[{jax_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    kc_week_19_pick_percent = st.slider("Kansas City Chiefs Week 19 Pick Percent:", -1, 100) / 100
                    kc_week_19_pick_percent_percent = kc_week_19_pick_percent*100
                    if kc_week_19_pick_percent < 0:
                        st.write(f"Kansas City Chiefs Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Kansas City Chiefs Week 19 Pick Percent: :green[{kc_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    lv_week_19_pick_percent = st.slider("Las Vegas Raiders Week 19 Pick Percent:", -1, 100) / 100
                    lv_week_19_pick_percent_percent = lv_week_19_pick_percent*100
                    if lv_week_19_pick_percent < 0:
                        st.write(f"Las Vegas Raiders Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Las vegas Raiders Week 19 Pick Percent: :green[{lv_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    lac_week_19_pick_percent = st.slider("Los Angeles Chargers Week 19 Pick Percent:", -1, 100) / 100
                    lac_week_19_pick_percent_percent = lac_week_19_pick_percent*100
                    if lac_week_19_pick_percent < 0:
                        st.write(f"Los Angeles Chargers Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Los angeles Chargers Week 19 Pick Percent: :green[{lac_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    lar_week_19_pick_percent = st.slider("Los Angeles Rams Week 19 Pick Percent:", -1, 100) / 100
                    lar_week_19_pick_percent_percent = lar_week_19_pick_percent*100
                    if lar_week_19_pick_percent < 0:
                        st.write(f"Los Angeles Rams Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Los Angeles Rams Week 19 Pick Percent: :green[{lar_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    mia_week_19_pick_percent = st.slider("Miami Dolphins Week 19 Pick Percent:", -1, 100) / 100
                    mia_week_19_pick_percent_percent = mia_week_19_pick_percent*100
                    if mia_week_19_pick_percent < 0:
                        st.write(f"Miami Dolphins Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Miami Dolphins Week 19 Pick Percent: :green[{mia_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    min_week_19_pick_percent = st.slider("Minnesota Vikings Week 19 Pick Percent:", -1, 100) / 100
                    min_week_19_pick_percent_percent = min_week_19_pick_percent*100
                    if min_week_19_pick_percent < 0:
                        st.write(f"Minnesota Vikings Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Minnesota Vikings Week 19 Pick Percent: :green[{min_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    ne_week_19_pick_percent = st.slider("New England Patriots Week 19 Pick Percent:", -1, 100) / 100
                    ne_week_19_pick_percent_percent = ne_week_19_pick_percent*100
                    if ne_week_19_pick_percent < 0:
                        st.write(f"New England Patriots Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"New England Patriots Week 19 Pick Percent: :green[{ne_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    no_week_19_pick_percent = st.slider("New Orleans Saints Week 19 Pick Percent:", -1, 100) / 100
                    no_week_19_pick_percent_percent = no_week_19_pick_percent*100
                    if no_week_19_pick_percent < 0:
                        st.write(f"New Orleans Saints Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"New Orleans Saints Week 19 Pick Percent: :green[{no_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    nyg_week_19_pick_percent = st.slider("New York Giants Week 19 Pick Percent:", -1, 100) / 100
                    nyg_week_19_pick_percent_percent = nyg_week_19_pick_percent*100
                    if nyg_week_19_pick_percent < 0:
                        st.write(f"New York Giants Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"New York Giants Week 19 Pick Percent: :green[{nyg_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    nyj_week_19_pick_percent = st.slider("New York Jets Week 19 Pick Percent:", -1, 100) / 100
                    nyj_week_19_pick_percent_percent = nyj_week_19_pick_percent*100
                    if nyj_week_19_pick_percent < 0:
                        st.write(f"New York Jets Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"New York Jets Week 19 Pick Percent: :green[{nyj_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    phi_week_19_pick_percent = st.slider("Philadelphia Eagles Week 19 Pick Percent:", -1, 100) / 100
                    phi_week_19_pick_percent_percent = phi_week_19_pick_percent*100
                    if phi_week_19_pick_percent < 0:
                        st.write(f"Philadelphia Eagles Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Philadelphia Eagles Week 19 Pick Percent: :green[{phi_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    pit_week_19_pick_percent = st.slider("Pittsburgh Steelers Week 19 Pick Percent:", -1, 100) / 100
                    pit_week_19_pick_percent_percent = pit_week_19_pick_percent*100
                    if pit_week_19_pick_percent < 0:
                        st.write(f"Pittsburgh Steelers Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Pittsburgh Steelers Week 19 Pick Percent: :green[{pit_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    sf_week_19_pick_percent = st.slider("San Francisco 49ers Week 19 Pick Percent:", -1, 100) / 100
                    sf_week_19_pick_percent_percent = sf_week_19_pick_percent*100
                    if sf_week_19_pick_percent < 0:
                        st.write(f"San Francisco 49ers Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"San Francisco 49ers Week 19 Pick Percent: :green[{sf_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    sea_week_19_pick_percent = st.slider("Seattle Seahawks Week 19 Pick Percent:", -1, 100) / 100
                    sea_week_19_pick_percent_percent = sea_week_19_pick_percent*100
                    if sea_week_19_pick_percent < 0:
                        st.write(f"Seattle Seahawks Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Seattle Seahawks Week 19 Pick Percent: :green[{sea_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    tb_week_19_pick_percent = st.slider("Tampa Bay Buccaneers Week 19 Pick Percent:", -1, 100) / 100
                    tb_week_19_pick_percent_percent = tb_week_19_pick_percent*100
                    if tb_week_19_pick_percent < 0:
                        st.write(f"Tampa Bay Buccaneers Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Tampa Bay Buccaneers Week 19 Pick Percent: :green[{tb_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    ten_week_19_pick_percent = st.slider("Tennessee Titans Week 19 Pick Percent:", -1, 100) / 100
                    ten_week_19_pick_percent_percent = ten_week_19_pick_percent*100
                    if ten_week_19_pick_percent < 0:
                        st.write(f"Tennessee Titans Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Tennessee Titans Week 19 Pick Percent: :green[{ten_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    was_week_19_pick_percent = st.slider("Washington Commanders Week 19 Pick Percent:", -1, 100) / 100
                    was_week_19_pick_percent_percent = was_week_19_pick_percent*100
                    if was_week_19_pick_percent < 0:
                        st.write(f"Washington Commanders Week 19 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Washington Commanders Week 19 Pick Percent: :green[{was_week_19_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
                st.write('')
                st.write('')
        if selected_contest == 'Circa':
            if starting_week <= 20 and ending_week > 20:
                week_20_pick_percents = st.checkbox('Add Week 20 Pick Percentages?', help = "Check this box if you want to adjust the estimated pick percentages for each team in week 20.")
                if week_20_pick_percents:
                    st.write('')
                    st.subheader('Week 20 Estimated Pick Percentages')
                    st.write('')
                    az_week_20_pick_percent = st.slider("Arizona Cardinals Week 20 Pick Percent:", -1, 100) / 100
                    az_week_20_pick_percent_percent = az_week_20_pick_percent*100
                    if az_week_20_pick_percent < 0:
                        st.write(f"Arizona Cardinals Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Arizona Cardinals Week 20 Pick Percent: :green[{az_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    atl_week_20_pick_percent = st.slider("Atlanta Falcons Week 20 Pick Percent:", -1, 100) / 100
                    atl_week_20_pick_percent_percent = atl_week_20_pick_percent*100
                    if atl_week_20_pick_percent < 0:
                        st.write(f"Atlanta Falcons Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Atlanta Falcons Week 20 Pick Percent: :green[{atl_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    bal_week_20_pick_percent = st.slider("Baltimore Ravens Week 20 Pick Percent:", -1, 100) / 100
                    bal_week_20_pick_percent_percent = bal_week_20_pick_percent*100
                    if bal_week_20_pick_percent < 0:
                        st.write(f"Baltimore Ravens Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Baltimore Ravenss Week 20 Pick Percent: :green[{bal_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    buf_week_20_pick_percent = st.slider("Buffalo Bills Week 20 Pick Percent:", -1, 100) / 100
                    buf_week_20_pick_percent_percent = buf_week_20_pick_percent*100
                    if buf_week_20_pick_percent < 0:
                        st.write(f"Buffalo Bills Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Buffalo Bills Week 20 Pick Percent: :green[{buf_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    car_week_20_pick_percent = st.slider("Carolina Panthers Week 20 Pick Percent:", -1, 100) / 100
                    car_week_20_pick_percent_percent = car_week_20_pick_percent*100
                    if car_week_20_pick_percent < 0:
                        st.write(f"Carolina Panthers Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Carolina Panthers Week 20 Pick Percent: :green[{car_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    chi_week_20_pick_percent = st.slider("Chicago Bears Week 20 Pick Percent:", -1, 100) / 100
                    chi_week_20_pick_percent_percent = chi_week_20_pick_percent*100
                    if chi_week_20_pick_percent < 0:
                        st.write(f"Chicago Bears Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Chicago Bears Week 20 Pick Percent: :green[{chi_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    cin_week_20_pick_percent = st.slider("Cincinnati Bengals Week 20 Pick Percent:", -1, 100) / 100
                    cin_week_20_pick_percent_percent = cin_week_20_pick_percent*100
                    if cin_week_20_pick_percent < 0:
                        st.write(f"Cincinnati Bengals Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Cincinnati Bengals Week 20 Pick Percent: :green[{cin_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    cle_week_20_pick_percent = st.slider("Cleveland Browns Week 20 Pick Percent:", -1, 100) / 100
                    cle_week_20_pick_percent_percent = cle_week_20_pick_percent*100
                    if cle_week_20_pick_percent < 0:
                        st.write(f"Cleveland Browns Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Cleveland Browns Week 20 Pick Percent: :green[{cle_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    dal_week_20_pick_percent = st.slider("Dallas Cowboys Week 20 Pick Percent:", -1, 100) / 100
                    dal_week_20_pick_percent_percent = dal_week_20_pick_percent*100
                    if dal_week_20_pick_percent < 0:
                        st.write(f"Dallas Cowboys Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Dallas Cowboys Week 20 Pick Percent: :green[{dal_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    den_week_20_pick_percent = st.slider("Denver Broncos Week 20 Pick Percent:", -1, 100) / 100
                    den_week_20_pick_percent_percent = den_week_20_pick_percent*100
                    if den_week_20_pick_percent < 0:
                        st.write(f"Denver Broncos Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Denver Broncos Week 20 Pick Percent: :green[{den_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    det_week_20_pick_percent = st.slider("Detroit Lions Week 20 Pick Percent:", -1, 100) / 100
                    det_week_20_pick_percent_percent = det_week_20_pick_percent*100
                    if det_week_20_pick_percent < 0:
                        st.write(f"Detroit Lions Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Detroit Lions Week 20 Pick Percent: :green[{det_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    gb_week_20_pick_percent = st.slider("Green Bay Packers Week 20 Pick Percent:", -1, 100) / 100
                    gb_week_20_pick_percent_percent = gb_week_20_pick_percent*100
                    if gb_week_20_pick_percent < 0:
                        st.write(f"Green Bay Packers Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Green Bay Packers Week 20 Pick Percent: :green[{gb_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    hou_week_20_pick_percent = st.slider("Houston Texans Week 20 Pick Percent:", -1, 100) / 100
                    hou_week_20_pick_percent_percent = hou_week_20_pick_percent*100
                    if hou_week_20_pick_percent < 0:
                        st.write(f"Houston Texans Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Houston Texans Week 20 Pick Percent: :green[{hou_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    ind_week_20_pick_percent = st.slider("Indianapoils Colts Week 20 Pick Percent:", -1, 100) / 100
                    ind_week_20_pick_percent_percent = ind_week_20_pick_percent*100
                    if ind_week_20_pick_percent < 0:
                        st.write(f"Indianapoils Colts Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Indianapoils Colts Week 20 Pick Percent: :green[{ind_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    jax_week_20_pick_percent = st.slider("Jacksonville Jaguars Week 20 Pick Percent:", -1, 100) / 100
                    jax_week_20_pick_percent_percent = jax_week_20_pick_percent*100
                    if jax_week_20_pick_percent < 0:
                        st.write(f"Jacksonville Jaguars Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Jacksonville Jaguars Week 20 Pick Percent: :green[{jax_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    kc_week_20_pick_percent = st.slider("Kansas City Chiefs Week 20 Pick Percent:", -1, 100) / 100
                    kc_week_20_pick_percent_percent = kc_week_20_pick_percent*100
                    if kc_week_20_pick_percent < 0:
                        st.write(f"Kansas City Chiefs Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Kansas City Chiefs Week 20 Pick Percent: :green[{kc_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    lv_week_20_pick_percent = st.slider("Las Vegas Raiders Week 20 Pick Percent:", -1, 100) / 100
                    lv_week_20_pick_percent_percent = lv_week_20_pick_percent*100
                    if lv_week_20_pick_percent < 0:
                        st.write(f"Las Vegas Raiders Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Las vegas Raiders Week 20 Pick Percent: :green[{lv_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    lac_week_20_pick_percent = st.slider("Los Angeles Chargers Week 20 Pick Percent:", -1, 100) / 100
                    lac_week_20_pick_percent_percent = lac_week_20_pick_percent*100
                    if lac_week_20_pick_percent < 0:
                        st.write(f"Los Angeles Chargers Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Los angeles Chargers Week 20 Pick Percent: :green[{lac_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    lar_week_20_pick_percent = st.slider("Los Angeles Rams Week 20 Pick Percent:", -1, 100) / 100
                    lar_week_20_pick_percent_percent = lar_week_20_pick_percent*100
                    if lar_week_20_pick_percent < 0:
                        st.write(f"Los Angeles Rams Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Los Angeles Rams Week 20 Pick Percent: :green[{lar_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    mia_week_20_pick_percent = st.slider("Miami Dolphins Week 20 Pick Percent:", -1, 100) / 100
                    mia_week_20_pick_percent_percent = mia_week_20_pick_percent*100
                    if mia_week_20_pick_percent < 0:
                        st.write(f"Miami Dolphins Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Miami Dolphins Week 20 Pick Percent: :green[{mia_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    min_week_20_pick_percent = st.slider("Minnesota Vikings Week 20 Pick Percent:", -1, 100) / 100
                    min_week_20_pick_percent_percent = min_week_20_pick_percent*100
                    if min_week_20_pick_percent < 0:
                        st.write(f"Minnesota Vikings Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Minnesota Vikings Week 20 Pick Percent: :green[{min_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    ne_week_20_pick_percent = st.slider("New England Patriots Week 20 Pick Percent:", -1, 100) / 100
                    ne_week_20_pick_percent_percent = ne_week_20_pick_percent*100
                    if ne_week_20_pick_percent < 0:
                        st.write(f"New England Patriots Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"New England Patriots Week 20 Pick Percent: :green[{ne_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    no_week_20_pick_percent = st.slider("New Orleans Saints Week 20 Pick Percent:", -1, 100) / 100
                    no_week_20_pick_percent_percent = no_week_20_pick_percent*100
                    if no_week_20_pick_percent < 0:
                        st.write(f"New Orleans Saints Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"New Orleans Saints Week 20 Pick Percent: :green[{no_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    nyg_week_20_pick_percent = st.slider("New York Giants Week 20 Pick Percent:", -1, 100) / 100
                    nyg_week_20_pick_percent_percent = nyg_week_20_pick_percent*100
                    if nyg_week_20_pick_percent < 0:
                        st.write(f"New York Giants Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"New York Giants Week 20 Pick Percent: :green[{nyg_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    nyj_week_20_pick_percent = st.slider("New York Jets Week 20 Pick Percent:", -1, 100) / 100
                    nyj_week_20_pick_percent_percent = nyj_week_20_pick_percent*100
                    if nyj_week_20_pick_percent < 0:
                        st.write(f"New York Jets Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"New York Jets Week 20 Pick Percent: :green[{nyj_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    phi_week_20_pick_percent = st.slider("Philadelphia Eagles Week 20 Pick Percent:", -1, 100) / 100
                    phi_week_20_pick_percent_percent = phi_week_20_pick_percent*100
                    if phi_week_20_pick_percent < 0:
                        st.write(f"Philadelphia Eagles Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Philadelphia Eagles Week 20 Pick Percent: :green[{phi_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    pit_week_20_pick_percent = st.slider("Pittsburgh Steelers Week 20 Pick Percent:", -1, 100) / 100
                    pit_week_20_pick_percent_percent = pit_week_20_pick_percent*100
                    if pit_week_20_pick_percent < 0:
                        st.write(f"Pittsburgh Steelers Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Pittsburgh Steelers Week 20 Pick Percent: :green[{pit_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    sf_week_20_pick_percent = st.slider("San Francisco 49ers Week 20 Pick Percent:", -1, 100) / 100
                    sf_week_20_pick_percent_percent = sf_week_20_pick_percent*100
                    if sf_week_20_pick_percent < 0:
                        st.write(f"San Francisco 49ers Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"San Francisco 49ers Week 20 Pick Percent: :green[{sf_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    sea_week_20_pick_percent = st.slider("Seattle Seahawks Week 20 Pick Percent:", -1, 100) / 100
                    sea_week_20_pick_percent_percent = sea_week_20_pick_percent*100
                    if sea_week_20_pick_percent < 0:
                        st.write(f"Seattle Seahawks Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Seattle Seahawks Week 20 Pick Percent: :green[{sea_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    tb_week_20_pick_percent = st.slider("Tampa Bay Buccaneers Week 20 Pick Percent:", -1, 100) / 100
                    tb_week_20_pick_percent_percent = tb_week_20_pick_percent*100
                    if tb_week_20_pick_percent < 0:
                        st.write(f"Tampa Bay Buccaneers Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Tampa Bay Buccaneers Week 20 Pick Percent: :green[{tb_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    ten_week_20_pick_percent = st.slider("Tennessee Titans Week 20 Pick Percent:", -1, 100) / 100
                    ten_week_20_pick_percent_percent = ten_week_20_pick_percent*100
                    if ten_week_20_pick_percent < 0:
                        st.write(f"Tennessee Titans Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Tennessee Titans Week 20 Pick Percent: :green[{ten_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
            	    
                    was_week_20_pick_percent = st.slider("Washington Commanders Week 20 Pick Percent:", -1, 100) / 100
                    was_week_20_pick_percent_percent = was_week_20_pick_percent*100
                    if was_week_20_pick_percent < 0:
                        st.write(f"Washington Commanders Week 20 Pick Percent: :red[Automatically Calculated]")
                    else:
                        st.write(f"Washington Commanders Week 20 Pick Percent: :green[{was_week_20_pick_percent_percent}%]")
                    st.write("")
                    st.write("")
    	
    
    dk_pick_percentages = {
        'Arizona Cardinals': [az_week_1_pick_percent, az_week_2_pick_percent, az_week_3_pick_percent, az_week_4_pick_percent, az_week_5_pick_percent, az_week_6_pick_percent, az_week_7_pick_percent, az_week_8_pick_percent, az_week_9_pick_percent, az_week_10_pick_percent, az_week_11_pick_percent, az_week_12_pick_percent, az_week_13_pick_percent, az_week_14_pick_percent, az_week_15_pick_percent, az_week_16_pick_percent, az_week_17_pick_percent, az_week_18_pick_percent],
        'Atlanta Falcons': [atl_week_1_pick_percent, atl_week_2_pick_percent, atl_week_3_pick_percent, atl_week_4_pick_percent, atl_week_5_pick_percent, atl_week_6_pick_percent, atl_week_7_pick_percent, atl_week_8_pick_percent, atl_week_9_pick_percent, atl_week_10_pick_percent, atl_week_11_pick_percent, atl_week_12_pick_percent, atl_week_13_pick_percent, atl_week_14_pick_percent, atl_week_15_pick_percent, atl_week_16_pick_percent, atl_week_17_pick_percent, atl_week_18_pick_percent],
        'Baltimore Ravens': [bal_week_1_pick_percent, bal_week_2_pick_percent, bal_week_3_pick_percent, bal_week_4_pick_percent, bal_week_5_pick_percent, bal_week_6_pick_percent, bal_week_7_pick_percent, bal_week_8_pick_percent, bal_week_9_pick_percent, bal_week_10_pick_percent, bal_week_11_pick_percent, bal_week_12_pick_percent, bal_week_13_pick_percent, bal_week_14_pick_percent, bal_week_15_pick_percent, bal_week_16_pick_percent, bal_week_17_pick_percent, bal_week_18_pick_percent],
        'Buffalo Bills': [buf_week_1_pick_percent, buf_week_2_pick_percent, buf_week_3_pick_percent, buf_week_4_pick_percent, buf_week_5_pick_percent, buf_week_6_pick_percent, buf_week_7_pick_percent, buf_week_8_pick_percent, buf_week_9_pick_percent, buf_week_10_pick_percent, buf_week_11_pick_percent, buf_week_12_pick_percent, buf_week_13_pick_percent, buf_week_14_pick_percent, buf_week_15_pick_percent, buf_week_16_pick_percent, buf_week_17_pick_percent, buf_week_18_pick_percent],
        'Carolina Panthers': [car_week_1_pick_percent, car_week_2_pick_percent, car_week_3_pick_percent, car_week_4_pick_percent, car_week_5_pick_percent, car_week_6_pick_percent, car_week_7_pick_percent, car_week_8_pick_percent, car_week_9_pick_percent, car_week_10_pick_percent, car_week_11_pick_percent, car_week_12_pick_percent, car_week_13_pick_percent, car_week_14_pick_percent, car_week_15_pick_percent, car_week_16_pick_percent, car_week_17_pick_percent, car_week_18_pick_percent],
        'Chicago Bears': [chi_week_1_pick_percent, chi_week_2_pick_percent, chi_week_3_pick_percent, chi_week_4_pick_percent, chi_week_5_pick_percent, chi_week_6_pick_percent, chi_week_7_pick_percent, chi_week_8_pick_percent, chi_week_9_pick_percent, chi_week_10_pick_percent, chi_week_11_pick_percent, chi_week_12_pick_percent, chi_week_13_pick_percent, chi_week_14_pick_percent, chi_week_15_pick_percent, chi_week_16_pick_percent, chi_week_17_pick_percent, chi_week_18_pick_percent],
        'Cincinnati Bengals': [cin_week_1_pick_percent, cin_week_2_pick_percent, cin_week_3_pick_percent, cin_week_4_pick_percent, cin_week_5_pick_percent, cin_week_6_pick_percent, cin_week_7_pick_percent, cin_week_8_pick_percent, cin_week_9_pick_percent, cin_week_10_pick_percent, cin_week_11_pick_percent, cin_week_12_pick_percent, cin_week_13_pick_percent, cin_week_14_pick_percent, cin_week_15_pick_percent, cin_week_16_pick_percent, cin_week_17_pick_percent, cin_week_18_pick_percent],
        'Cleveland Browns': [cle_week_1_pick_percent, cle_week_2_pick_percent, cle_week_3_pick_percent, cle_week_4_pick_percent, cle_week_5_pick_percent, cle_week_6_pick_percent, cle_week_7_pick_percent, cle_week_8_pick_percent, cle_week_9_pick_percent, cle_week_10_pick_percent, cle_week_11_pick_percent, cle_week_12_pick_percent, cle_week_13_pick_percent, cle_week_14_pick_percent, cle_week_15_pick_percent, cle_week_16_pick_percent, cle_week_17_pick_percent, cle_week_18_pick_percent],
        'Dallas Cowboys': [dal_week_1_pick_percent, dal_week_2_pick_percent, dal_week_3_pick_percent, dal_week_4_pick_percent, dal_week_5_pick_percent, dal_week_6_pick_percent, dal_week_7_pick_percent, dal_week_8_pick_percent, dal_week_9_pick_percent, dal_week_10_pick_percent, dal_week_11_pick_percent, dal_week_12_pick_percent, dal_week_13_pick_percent, dal_week_14_pick_percent, dal_week_15_pick_percent, dal_week_16_pick_percent, dal_week_17_pick_percent, dal_week_18_pick_percent],
        'Denver Broncos': [den_week_1_pick_percent, den_week_2_pick_percent, den_week_3_pick_percent, den_week_4_pick_percent, den_week_5_pick_percent, den_week_6_pick_percent, den_week_7_pick_percent, den_week_8_pick_percent, den_week_9_pick_percent, den_week_10_pick_percent, den_week_11_pick_percent, den_week_12_pick_percent, den_week_13_pick_percent, den_week_14_pick_percent, den_week_15_pick_percent, den_week_16_pick_percent, den_week_17_pick_percent, den_week_18_pick_percent],
        'Detroit Lions': [det_week_1_pick_percent, det_week_2_pick_percent, det_week_3_pick_percent, det_week_4_pick_percent, det_week_5_pick_percent, det_week_6_pick_percent, det_week_7_pick_percent, det_week_8_pick_percent, det_week_9_pick_percent, det_week_10_pick_percent, det_week_11_pick_percent, det_week_12_pick_percent, det_week_13_pick_percent, det_week_14_pick_percent, det_week_15_pick_percent, det_week_16_pick_percent, det_week_17_pick_percent, det_week_18_pick_percent],
        'Green Bay Packers': [gb_week_1_pick_percent, gb_week_2_pick_percent, gb_week_3_pick_percent, gb_week_4_pick_percent, gb_week_5_pick_percent, gb_week_6_pick_percent, gb_week_7_pick_percent, gb_week_8_pick_percent, gb_week_9_pick_percent, gb_week_10_pick_percent, gb_week_11_pick_percent, gb_week_12_pick_percent, gb_week_13_pick_percent, gb_week_14_pick_percent, gb_week_15_pick_percent, gb_week_16_pick_percent, gb_week_17_pick_percent, gb_week_18_pick_percent],
        'Houston Texans': [hou_week_1_pick_percent, hou_week_2_pick_percent, hou_week_3_pick_percent, hou_week_4_pick_percent, hou_week_5_pick_percent, hou_week_6_pick_percent, hou_week_7_pick_percent, hou_week_8_pick_percent, hou_week_9_pick_percent, hou_week_10_pick_percent, hou_week_11_pick_percent, hou_week_12_pick_percent, hou_week_13_pick_percent, hou_week_14_pick_percent, hou_week_15_pick_percent, hou_week_16_pick_percent, hou_week_17_pick_percent, hou_week_18_pick_percent],
        'Indianapolis Colts': [ind_week_1_pick_percent, ind_week_2_pick_percent, ind_week_3_pick_percent, ind_week_4_pick_percent, ind_week_5_pick_percent, ind_week_6_pick_percent, ind_week_7_pick_percent, ind_week_8_pick_percent, ind_week_9_pick_percent, ind_week_10_pick_percent, ind_week_11_pick_percent, ind_week_12_pick_percent, ind_week_13_pick_percent, ind_week_14_pick_percent, ind_week_15_pick_percent, ind_week_16_pick_percent, ind_week_17_pick_percent, ind_week_18_pick_percent],
        'Jacksonville Jaguars': [jax_week_1_pick_percent, jax_week_2_pick_percent, jax_week_3_pick_percent, jax_week_4_pick_percent, jax_week_5_pick_percent, jax_week_6_pick_percent, jax_week_7_pick_percent, jax_week_8_pick_percent, jax_week_9_pick_percent, jax_week_10_pick_percent, jax_week_11_pick_percent, jax_week_12_pick_percent, jax_week_13_pick_percent, jax_week_14_pick_percent, jax_week_15_pick_percent, jax_week_16_pick_percent, jax_week_17_pick_percent, jax_week_18_pick_percent],
        'Kansas City Chiefs': [kc_week_1_pick_percent, kc_week_2_pick_percent, kc_week_3_pick_percent, kc_week_4_pick_percent, kc_week_5_pick_percent, kc_week_6_pick_percent, kc_week_7_pick_percent, kc_week_8_pick_percent, kc_week_9_pick_percent, kc_week_10_pick_percent, kc_week_11_pick_percent, kc_week_12_pick_percent, kc_week_13_pick_percent, kc_week_14_pick_percent, kc_week_15_pick_percent, kc_week_16_pick_percent, kc_week_17_pick_percent, kc_week_18_pick_percent],
        'Las Vegas Raiders': [lv_week_1_pick_percent, lv_week_2_pick_percent, lv_week_3_pick_percent, lv_week_4_pick_percent, lv_week_5_pick_percent, lv_week_6_pick_percent, lv_week_7_pick_percent, lv_week_8_pick_percent, lv_week_9_pick_percent, lv_week_10_pick_percent, lv_week_11_pick_percent, lv_week_12_pick_percent, lv_week_13_pick_percent, lv_week_14_pick_percent, lv_week_15_pick_percent, lv_week_16_pick_percent, lv_week_17_pick_percent, lv_week_18_pick_percent],
        'Los Angeles Chargers': [lac_week_1_pick_percent, lac_week_2_pick_percent, lac_week_3_pick_percent, lac_week_4_pick_percent, lac_week_5_pick_percent, lac_week_6_pick_percent, lac_week_7_pick_percent, lac_week_8_pick_percent, lac_week_9_pick_percent, lac_week_10_pick_percent, lac_week_11_pick_percent, lac_week_12_pick_percent, lac_week_13_pick_percent, lac_week_14_pick_percent, lac_week_15_pick_percent, lac_week_16_pick_percent, lac_week_17_pick_percent, lac_week_18_pick_percent],
        'Los Angeles Rams': [lar_week_1_pick_percent, lar_week_2_pick_percent, lar_week_3_pick_percent, lar_week_4_pick_percent, lar_week_5_pick_percent, lar_week_6_pick_percent, lar_week_7_pick_percent, lar_week_8_pick_percent, lar_week_9_pick_percent, lar_week_10_pick_percent, lar_week_11_pick_percent, lar_week_12_pick_percent, lar_week_13_pick_percent, lar_week_14_pick_percent, lar_week_15_pick_percent, lar_week_16_pick_percent, lar_week_17_pick_percent, lar_week_18_pick_percent],
        'Miami Dolphins': [mia_week_1_pick_percent, mia_week_2_pick_percent, mia_week_3_pick_percent, mia_week_4_pick_percent, mia_week_5_pick_percent, mia_week_6_pick_percent, mia_week_7_pick_percent, mia_week_8_pick_percent, mia_week_9_pick_percent, mia_week_10_pick_percent, mia_week_11_pick_percent, mia_week_12_pick_percent, mia_week_13_pick_percent, mia_week_14_pick_percent, mia_week_15_pick_percent, mia_week_16_pick_percent, mia_week_17_pick_percent, mia_week_18_pick_percent],
        'Minnesota Vikings': [min_week_1_pick_percent, min_week_2_pick_percent, min_week_3_pick_percent, min_week_4_pick_percent, min_week_5_pick_percent, min_week_6_pick_percent, min_week_7_pick_percent, min_week_8_pick_percent, min_week_9_pick_percent, min_week_10_pick_percent, min_week_11_pick_percent, min_week_12_pick_percent, min_week_13_pick_percent, min_week_14_pick_percent, min_week_15_pick_percent, min_week_16_pick_percent, min_week_17_pick_percent, min_week_18_pick_percent],
        'New England Patriots': [ne_week_1_pick_percent, ne_week_2_pick_percent, ne_week_3_pick_percent, ne_week_4_pick_percent, ne_week_5_pick_percent, ne_week_6_pick_percent, ne_week_7_pick_percent, ne_week_8_pick_percent, ne_week_9_pick_percent, ne_week_10_pick_percent, ne_week_11_pick_percent, ne_week_12_pick_percent, ne_week_13_pick_percent, ne_week_14_pick_percent, ne_week_15_pick_percent, ne_week_16_pick_percent, ne_week_17_pick_percent, ne_week_18_pick_percent],
        'New Orleans Saints': [no_week_1_pick_percent, no_week_2_pick_percent, no_week_3_pick_percent, no_week_4_pick_percent, no_week_5_pick_percent, no_week_6_pick_percent, no_week_7_pick_percent, no_week_8_pick_percent, no_week_9_pick_percent, no_week_10_pick_percent, no_week_11_pick_percent, no_week_12_pick_percent, no_week_13_pick_percent, no_week_14_pick_percent, no_week_15_pick_percent, no_week_16_pick_percent, no_week_17_pick_percent, no_week_18_pick_percent],
        'New York Giants': [nyg_week_1_pick_percent, nyg_week_2_pick_percent, nyg_week_3_pick_percent, nyg_week_4_pick_percent, nyg_week_5_pick_percent, nyg_week_6_pick_percent, nyg_week_7_pick_percent, nyg_week_8_pick_percent, nyg_week_9_pick_percent, nyg_week_10_pick_percent, nyg_week_11_pick_percent, nyg_week_12_pick_percent, nyg_week_13_pick_percent, nyg_week_14_pick_percent, nyg_week_15_pick_percent, nyg_week_16_pick_percent, nyg_week_17_pick_percent, nyg_week_18_pick_percent],
        'New York Jets': [nyj_week_1_pick_percent, nyj_week_2_pick_percent, nyj_week_3_pick_percent, nyj_week_4_pick_percent, nyj_week_5_pick_percent, nyj_week_6_pick_percent, nyj_week_7_pick_percent, nyj_week_8_pick_percent, nyj_week_9_pick_percent, nyj_week_10_pick_percent, nyj_week_11_pick_percent, nyj_week_12_pick_percent, nyj_week_13_pick_percent, nyj_week_14_pick_percent, nyj_week_15_pick_percent, nyj_week_16_pick_percent, nyj_week_17_pick_percent, nyj_week_18_pick_percent],
        'Philadelphia Eagles': [phi_week_1_pick_percent, phi_week_2_pick_percent, phi_week_3_pick_percent, phi_week_4_pick_percent, phi_week_5_pick_percent, phi_week_6_pick_percent, phi_week_7_pick_percent, phi_week_8_pick_percent, phi_week_9_pick_percent, phi_week_10_pick_percent, phi_week_11_pick_percent, phi_week_12_pick_percent, phi_week_13_pick_percent, phi_week_14_pick_percent, phi_week_15_pick_percent, phi_week_16_pick_percent, phi_week_17_pick_percent, phi_week_18_pick_percent],
        'Pittsburgh Steelers': [pit_week_1_pick_percent, pit_week_2_pick_percent, pit_week_3_pick_percent, pit_week_4_pick_percent, pit_week_5_pick_percent, pit_week_6_pick_percent, pit_week_7_pick_percent, pit_week_8_pick_percent, pit_week_9_pick_percent, pit_week_10_pick_percent, pit_week_11_pick_percent, pit_week_12_pick_percent, pit_week_13_pick_percent, pit_week_14_pick_percent, pit_week_15_pick_percent, pit_week_16_pick_percent, pit_week_17_pick_percent, pit_week_18_pick_percent],
        'San Francisco 49ers': [sf_week_1_pick_percent, sf_week_2_pick_percent, sf_week_3_pick_percent, sf_week_4_pick_percent, sf_week_5_pick_percent, sf_week_6_pick_percent, sf_week_7_pick_percent, sf_week_8_pick_percent, sf_week_9_pick_percent, sf_week_10_pick_percent, sf_week_11_pick_percent, sf_week_12_pick_percent, sf_week_13_pick_percent, sf_week_14_pick_percent, sf_week_15_pick_percent, sf_week_16_pick_percent, sf_week_17_pick_percent, sf_week_18_pick_percent],
        'Seattle Seahawks': [sea_week_1_pick_percent, sea_week_2_pick_percent, sea_week_3_pick_percent, sea_week_4_pick_percent, sea_week_5_pick_percent, sea_week_6_pick_percent, sea_week_7_pick_percent, sea_week_8_pick_percent, sea_week_9_pick_percent, sea_week_10_pick_percent, sea_week_11_pick_percent, sea_week_12_pick_percent, sea_week_13_pick_percent, sea_week_14_pick_percent, sea_week_15_pick_percent, sea_week_16_pick_percent, sea_week_17_pick_percent, sea_week_18_pick_percent],
        'Tampa Bay Buccaneers': [tb_week_1_pick_percent, tb_week_2_pick_percent, tb_week_3_pick_percent, tb_week_4_pick_percent, tb_week_5_pick_percent, tb_week_6_pick_percent, tb_week_7_pick_percent, tb_week_8_pick_percent, tb_week_9_pick_percent, tb_week_10_pick_percent, tb_week_11_pick_percent, tb_week_12_pick_percent, tb_week_13_pick_percent, tb_week_14_pick_percent, tb_week_15_pick_percent, tb_week_16_pick_percent, tb_week_17_pick_percent, tb_week_18_pick_percent],
        'Tennessee Titans': [ten_week_1_pick_percent, ten_week_2_pick_percent, ten_week_3_pick_percent, ten_week_4_pick_percent, ten_week_5_pick_percent, ten_week_6_pick_percent, ten_week_7_pick_percent, ten_week_8_pick_percent, ten_week_9_pick_percent, ten_week_10_pick_percent, ten_week_11_pick_percent, ten_week_12_pick_percent, ten_week_13_pick_percent, ten_week_14_pick_percent, ten_week_15_pick_percent, ten_week_16_pick_percent, ten_week_17_pick_percent, ten_week_18_pick_percent],
         'Washington Commanders': [was_week_1_pick_percent, was_week_2_pick_percent, was_week_3_pick_percent, was_week_4_pick_percent, was_week_5_pick_percent, was_week_6_pick_percent, was_week_7_pick_percent, was_week_8_pick_percent, was_week_9_pick_percent, was_week_10_pick_percent, was_week_11_pick_percent, was_week_12_pick_percent, was_week_13_pick_percent, was_week_14_pick_percent, was_week_15_pick_percent, was_week_16_pick_percent, was_week_17_pick_percent, was_week_18_pick_percent]
    }
    splash_pick_percentages = {
        'Arizona Cardinals': [az_week_1_pick_percent, az_week_2_pick_percent, az_week_3_pick_percent, az_week_4_pick_percent, az_week_5_pick_percent, az_week_6_pick_percent, az_week_7_pick_percent, az_week_8_pick_percent, az_week_9_pick_percent, az_week_10_pick_percent, az_week_11_pick_percent, az_week_12_pick_percent, az_week_13_pick_percent, az_week_14_pick_percent, az_week_15_pick_percent, az_week_16_pick_percent, az_week_17_pick_percent, az_week_18_pick_percent],
        'Atlanta Falcons': [atl_week_1_pick_percent, atl_week_2_pick_percent, atl_week_3_pick_percent, atl_week_4_pick_percent, atl_week_5_pick_percent, atl_week_6_pick_percent, atl_week_7_pick_percent, atl_week_8_pick_percent, atl_week_9_pick_percent, atl_week_10_pick_percent, atl_week_11_pick_percent, atl_week_12_pick_percent, atl_week_13_pick_percent, atl_week_14_pick_percent, atl_week_15_pick_percent, atl_week_16_pick_percent, atl_week_17_pick_percent, atl_week_18_pick_percent],
        'Baltimore Ravens': [bal_week_1_pick_percent, bal_week_2_pick_percent, bal_week_3_pick_percent, bal_week_4_pick_percent, bal_week_5_pick_percent, bal_week_6_pick_percent, bal_week_7_pick_percent, bal_week_8_pick_percent, bal_week_9_pick_percent, bal_week_10_pick_percent, bal_week_11_pick_percent, bal_week_12_pick_percent, bal_week_13_pick_percent, bal_week_14_pick_percent, bal_week_15_pick_percent, bal_week_16_pick_percent, bal_week_17_pick_percent, bal_week_18_pick_percent],
        'Buffalo Bills': [buf_week_1_pick_percent, buf_week_2_pick_percent, buf_week_3_pick_percent, buf_week_4_pick_percent, buf_week_5_pick_percent, buf_week_6_pick_percent, buf_week_7_pick_percent, buf_week_8_pick_percent, buf_week_9_pick_percent, buf_week_10_pick_percent, buf_week_11_pick_percent, buf_week_12_pick_percent, buf_week_13_pick_percent, buf_week_14_pick_percent, buf_week_15_pick_percent, buf_week_16_pick_percent, buf_week_17_pick_percent, buf_week_18_pick_percent],
        'Carolina Panthers': [car_week_1_pick_percent, car_week_2_pick_percent, car_week_3_pick_percent, car_week_4_pick_percent, car_week_5_pick_percent, car_week_6_pick_percent, car_week_7_pick_percent, car_week_8_pick_percent, car_week_9_pick_percent, car_week_10_pick_percent, car_week_11_pick_percent, car_week_12_pick_percent, car_week_13_pick_percent, car_week_14_pick_percent, car_week_15_pick_percent, car_week_16_pick_percent, car_week_17_pick_percent, car_week_18_pick_percent],
        'Chicago Bears': [chi_week_1_pick_percent, chi_week_2_pick_percent, chi_week_3_pick_percent, chi_week_4_pick_percent, chi_week_5_pick_percent, chi_week_6_pick_percent, chi_week_7_pick_percent, chi_week_8_pick_percent, chi_week_9_pick_percent, chi_week_10_pick_percent, chi_week_11_pick_percent, chi_week_12_pick_percent, chi_week_13_pick_percent, chi_week_14_pick_percent, chi_week_15_pick_percent, chi_week_16_pick_percent, chi_week_17_pick_percent, chi_week_18_pick_percent],
        'Cincinnati Bengals': [cin_week_1_pick_percent, cin_week_2_pick_percent, cin_week_3_pick_percent, cin_week_4_pick_percent, cin_week_5_pick_percent, cin_week_6_pick_percent, cin_week_7_pick_percent, cin_week_8_pick_percent, cin_week_9_pick_percent, cin_week_10_pick_percent, cin_week_11_pick_percent, cin_week_12_pick_percent, cin_week_13_pick_percent, cin_week_14_pick_percent, cin_week_15_pick_percent, cin_week_16_pick_percent, cin_week_17_pick_percent, cin_week_18_pick_percent],
        'Cleveland Browns': [cle_week_1_pick_percent, cle_week_2_pick_percent, cle_week_3_pick_percent, cle_week_4_pick_percent, cle_week_5_pick_percent, cle_week_6_pick_percent, cle_week_7_pick_percent, cle_week_8_pick_percent, cle_week_9_pick_percent, cle_week_10_pick_percent, cle_week_11_pick_percent, cle_week_12_pick_percent, cle_week_13_pick_percent, cle_week_14_pick_percent, cle_week_15_pick_percent, cle_week_16_pick_percent, cle_week_17_pick_percent, cle_week_18_pick_percent],
        'Dallas Cowboys': [dal_week_1_pick_percent, dal_week_2_pick_percent, dal_week_3_pick_percent, dal_week_4_pick_percent, dal_week_5_pick_percent, dal_week_6_pick_percent, dal_week_7_pick_percent, dal_week_8_pick_percent, dal_week_9_pick_percent, dal_week_10_pick_percent, dal_week_11_pick_percent, dal_week_12_pick_percent, dal_week_13_pick_percent, dal_week_14_pick_percent, dal_week_15_pick_percent, dal_week_16_pick_percent, dal_week_17_pick_percent, dal_week_18_pick_percent],
        'Denver Broncos': [den_week_1_pick_percent, den_week_2_pick_percent, den_week_3_pick_percent, den_week_4_pick_percent, den_week_5_pick_percent, den_week_6_pick_percent, den_week_7_pick_percent, den_week_8_pick_percent, den_week_9_pick_percent, den_week_10_pick_percent, den_week_11_pick_percent, den_week_12_pick_percent, den_week_13_pick_percent, den_week_14_pick_percent, den_week_15_pick_percent, den_week_16_pick_percent, den_week_17_pick_percent, den_week_18_pick_percent],
        'Detroit Lions': [det_week_1_pick_percent, det_week_2_pick_percent, det_week_3_pick_percent, det_week_4_pick_percent, det_week_5_pick_percent, det_week_6_pick_percent, det_week_7_pick_percent, det_week_8_pick_percent, det_week_9_pick_percent, det_week_10_pick_percent, det_week_11_pick_percent, det_week_12_pick_percent, det_week_13_pick_percent, det_week_14_pick_percent, det_week_15_pick_percent, det_week_16_pick_percent, det_week_17_pick_percent, det_week_18_pick_percent],
        'Green Bay Packers': [gb_week_1_pick_percent, gb_week_2_pick_percent, gb_week_3_pick_percent, gb_week_4_pick_percent, gb_week_5_pick_percent, gb_week_6_pick_percent, gb_week_7_pick_percent, gb_week_8_pick_percent, gb_week_9_pick_percent, gb_week_10_pick_percent, gb_week_11_pick_percent, gb_week_12_pick_percent, gb_week_13_pick_percent, gb_week_14_pick_percent, gb_week_15_pick_percent, gb_week_16_pick_percent, gb_week_17_pick_percent, gb_week_18_pick_percent],
        'Houston Texans': [hou_week_1_pick_percent, hou_week_2_pick_percent, hou_week_3_pick_percent, hou_week_4_pick_percent, hou_week_5_pick_percent, hou_week_6_pick_percent, hou_week_7_pick_percent, hou_week_8_pick_percent, hou_week_9_pick_percent, hou_week_10_pick_percent, hou_week_11_pick_percent, hou_week_12_pick_percent, hou_week_13_pick_percent, hou_week_14_pick_percent, hou_week_15_pick_percent, hou_week_16_pick_percent, hou_week_17_pick_percent, hou_week_18_pick_percent],
        'Indianapolis Colts': [ind_week_1_pick_percent, ind_week_2_pick_percent, ind_week_3_pick_percent, ind_week_4_pick_percent, ind_week_5_pick_percent, ind_week_6_pick_percent, ind_week_7_pick_percent, ind_week_8_pick_percent, ind_week_9_pick_percent, ind_week_10_pick_percent, ind_week_11_pick_percent, ind_week_12_pick_percent, ind_week_13_pick_percent, ind_week_14_pick_percent, ind_week_15_pick_percent, ind_week_16_pick_percent, ind_week_17_pick_percent, ind_week_18_pick_percent],
        'Jacksonville Jaguars': [jax_week_1_pick_percent, jax_week_2_pick_percent, jax_week_3_pick_percent, jax_week_4_pick_percent, jax_week_5_pick_percent, jax_week_6_pick_percent, jax_week_7_pick_percent, jax_week_8_pick_percent, jax_week_9_pick_percent, jax_week_10_pick_percent, jax_week_11_pick_percent, jax_week_12_pick_percent, jax_week_13_pick_percent, jax_week_14_pick_percent, jax_week_15_pick_percent, jax_week_16_pick_percent, jax_week_17_pick_percent, jax_week_18_pick_percent],
        'Kansas City Chiefs': [kc_week_1_pick_percent, kc_week_2_pick_percent, kc_week_3_pick_percent, kc_week_4_pick_percent, kc_week_5_pick_percent, kc_week_6_pick_percent, kc_week_7_pick_percent, kc_week_8_pick_percent, kc_week_9_pick_percent, kc_week_10_pick_percent, kc_week_11_pick_percent, kc_week_12_pick_percent, kc_week_13_pick_percent, kc_week_14_pick_percent, kc_week_15_pick_percent, kc_week_16_pick_percent, kc_week_17_pick_percent, kc_week_18_pick_percent],
        'Las Vegas Raiders': [lv_week_1_pick_percent, lv_week_2_pick_percent, lv_week_3_pick_percent, lv_week_4_pick_percent, lv_week_5_pick_percent, lv_week_6_pick_percent, lv_week_7_pick_percent, lv_week_8_pick_percent, lv_week_9_pick_percent, lv_week_10_pick_percent, lv_week_11_pick_percent, lv_week_12_pick_percent, lv_week_13_pick_percent, lv_week_14_pick_percent, lv_week_15_pick_percent, lv_week_16_pick_percent, lv_week_17_pick_percent, lv_week_18_pick_percent],
        'Los Angeles Chargers': [lac_week_1_pick_percent, lac_week_2_pick_percent, lac_week_3_pick_percent, lac_week_4_pick_percent, lac_week_5_pick_percent, lac_week_6_pick_percent, lac_week_7_pick_percent, lac_week_8_pick_percent, lac_week_9_pick_percent, lac_week_10_pick_percent, lac_week_11_pick_percent, lac_week_12_pick_percent, lac_week_13_pick_percent, lac_week_14_pick_percent, lac_week_15_pick_percent, lac_week_16_pick_percent, lac_week_17_pick_percent, lac_week_18_pick_percent],
        'Los Angeles Rams': [lar_week_1_pick_percent, lar_week_2_pick_percent, lar_week_3_pick_percent, lar_week_4_pick_percent, lar_week_5_pick_percent, lar_week_6_pick_percent, lar_week_7_pick_percent, lar_week_8_pick_percent, lar_week_9_pick_percent, lar_week_10_pick_percent, lar_week_11_pick_percent, lar_week_12_pick_percent, lar_week_13_pick_percent, lar_week_14_pick_percent, lar_week_15_pick_percent, lar_week_16_pick_percent, lar_week_17_pick_percent, lar_week_18_pick_percent],
        'Miami Dolphins': [mia_week_1_pick_percent, mia_week_2_pick_percent, mia_week_3_pick_percent, mia_week_4_pick_percent, mia_week_5_pick_percent, mia_week_6_pick_percent, mia_week_7_pick_percent, mia_week_8_pick_percent, mia_week_9_pick_percent, mia_week_10_pick_percent, mia_week_11_pick_percent, mia_week_12_pick_percent, mia_week_13_pick_percent, mia_week_14_pick_percent, mia_week_15_pick_percent, mia_week_16_pick_percent, mia_week_17_pick_percent, mia_week_18_pick_percent],
        'Minnesota Vikings': [min_week_1_pick_percent, min_week_2_pick_percent, min_week_3_pick_percent, min_week_4_pick_percent, min_week_5_pick_percent, min_week_6_pick_percent, min_week_7_pick_percent, min_week_8_pick_percent, min_week_9_pick_percent, min_week_10_pick_percent, min_week_11_pick_percent, min_week_12_pick_percent, min_week_13_pick_percent, min_week_14_pick_percent, min_week_15_pick_percent, min_week_16_pick_percent, min_week_17_pick_percent, min_week_18_pick_percent],
        'New England Patriots': [ne_week_1_pick_percent, ne_week_2_pick_percent, ne_week_3_pick_percent, ne_week_4_pick_percent, ne_week_5_pick_percent, ne_week_6_pick_percent, ne_week_7_pick_percent, ne_week_8_pick_percent, ne_week_9_pick_percent, ne_week_10_pick_percent, ne_week_11_pick_percent, ne_week_12_pick_percent, ne_week_13_pick_percent, ne_week_14_pick_percent, ne_week_15_pick_percent, ne_week_16_pick_percent, ne_week_17_pick_percent, ne_week_18_pick_percent],
        'New Orleans Saints': [no_week_1_pick_percent, no_week_2_pick_percent, no_week_3_pick_percent, no_week_4_pick_percent, no_week_5_pick_percent, no_week_6_pick_percent, no_week_7_pick_percent, no_week_8_pick_percent, no_week_9_pick_percent, no_week_10_pick_percent, no_week_11_pick_percent, no_week_12_pick_percent, no_week_13_pick_percent, no_week_14_pick_percent, no_week_15_pick_percent, no_week_16_pick_percent, no_week_17_pick_percent, no_week_18_pick_percent],
        'New York Giants': [nyg_week_1_pick_percent, nyg_week_2_pick_percent, nyg_week_3_pick_percent, nyg_week_4_pick_percent, nyg_week_5_pick_percent, nyg_week_6_pick_percent, nyg_week_7_pick_percent, nyg_week_8_pick_percent, nyg_week_9_pick_percent, nyg_week_10_pick_percent, nyg_week_11_pick_percent, nyg_week_12_pick_percent, nyg_week_13_pick_percent, nyg_week_14_pick_percent, nyg_week_15_pick_percent, nyg_week_16_pick_percent, nyg_week_17_pick_percent, nyg_week_18_pick_percent],
        'New York Jets': [nyj_week_1_pick_percent, nyj_week_2_pick_percent, nyj_week_3_pick_percent, nyj_week_4_pick_percent, nyj_week_5_pick_percent, nyj_week_6_pick_percent, nyj_week_7_pick_percent, nyj_week_8_pick_percent, nyj_week_9_pick_percent, nyj_week_10_pick_percent, nyj_week_11_pick_percent, nyj_week_12_pick_percent, nyj_week_13_pick_percent, nyj_week_14_pick_percent, nyj_week_15_pick_percent, nyj_week_16_pick_percent, nyj_week_17_pick_percent, nyj_week_18_pick_percent],
        'Philadelphia Eagles': [phi_week_1_pick_percent, phi_week_2_pick_percent, phi_week_3_pick_percent, phi_week_4_pick_percent, phi_week_5_pick_percent, phi_week_6_pick_percent, phi_week_7_pick_percent, phi_week_8_pick_percent, phi_week_9_pick_percent, phi_week_10_pick_percent, phi_week_11_pick_percent, phi_week_12_pick_percent, phi_week_13_pick_percent, phi_week_14_pick_percent, phi_week_15_pick_percent, phi_week_16_pick_percent, phi_week_17_pick_percent, phi_week_18_pick_percent],
        'Pittsburgh Steelers': [pit_week_1_pick_percent, pit_week_2_pick_percent, pit_week_3_pick_percent, pit_week_4_pick_percent, pit_week_5_pick_percent, pit_week_6_pick_percent, pit_week_7_pick_percent, pit_week_8_pick_percent, pit_week_9_pick_percent, pit_week_10_pick_percent, pit_week_11_pick_percent, pit_week_12_pick_percent, pit_week_13_pick_percent, pit_week_14_pick_percent, pit_week_15_pick_percent, pit_week_16_pick_percent, pit_week_17_pick_percent, pit_week_18_pick_percent],
        'San Francisco 49ers': [sf_week_1_pick_percent, sf_week_2_pick_percent, sf_week_3_pick_percent, sf_week_4_pick_percent, sf_week_5_pick_percent, sf_week_6_pick_percent, sf_week_7_pick_percent, sf_week_8_pick_percent, sf_week_9_pick_percent, sf_week_10_pick_percent, sf_week_11_pick_percent, sf_week_12_pick_percent, sf_week_13_pick_percent, sf_week_14_pick_percent, sf_week_15_pick_percent, sf_week_16_pick_percent, sf_week_17_pick_percent, sf_week_18_pick_percent],
        'Seattle Seahawks': [sea_week_1_pick_percent, sea_week_2_pick_percent, sea_week_3_pick_percent, sea_week_4_pick_percent, sea_week_5_pick_percent, sea_week_6_pick_percent, sea_week_7_pick_percent, sea_week_8_pick_percent, sea_week_9_pick_percent, sea_week_10_pick_percent, sea_week_11_pick_percent, sea_week_12_pick_percent, sea_week_13_pick_percent, sea_week_14_pick_percent, sea_week_15_pick_percent, sea_week_16_pick_percent, sea_week_17_pick_percent, sea_week_18_pick_percent],
        'Tampa Bay Buccaneers': [tb_week_1_pick_percent, tb_week_2_pick_percent, tb_week_3_pick_percent, tb_week_4_pick_percent, tb_week_5_pick_percent, tb_week_6_pick_percent, tb_week_7_pick_percent, tb_week_8_pick_percent, tb_week_9_pick_percent, tb_week_10_pick_percent, tb_week_11_pick_percent, tb_week_12_pick_percent, tb_week_13_pick_percent, tb_week_14_pick_percent, tb_week_15_pick_percent, tb_week_16_pick_percent, tb_week_17_pick_percent, tb_week_18_pick_percent],
        'Tennessee Titans': [ten_week_1_pick_percent, ten_week_2_pick_percent, ten_week_3_pick_percent, ten_week_4_pick_percent, ten_week_5_pick_percent, ten_week_6_pick_percent, ten_week_7_pick_percent, ten_week_8_pick_percent, ten_week_9_pick_percent, ten_week_10_pick_percent, ten_week_11_pick_percent, ten_week_12_pick_percent, ten_week_13_pick_percent, ten_week_14_pick_percent, ten_week_15_pick_percent, ten_week_16_pick_percent, ten_week_17_pick_percent, ten_week_18_pick_percent],
         'Washington Commanders': [was_week_1_pick_percent, was_week_2_pick_percent, was_week_3_pick_percent, was_week_4_pick_percent, was_week_5_pick_percent, was_week_6_pick_percent, was_week_7_pick_percent, was_week_8_pick_percent, was_week_9_pick_percent, was_week_10_pick_percent, was_week_11_pick_percent, was_week_12_pick_percent, was_week_13_pick_percent, was_week_14_pick_percent, was_week_15_pick_percent, was_week_16_pick_percent, was_week_17_pick_percent, was_week_18_pick_percent]
    }  
    circa_pick_percentages = {
        'Arizona Cardinals': [az_week_1_pick_percent, az_week_2_pick_percent, az_week_3_pick_percent, az_week_4_pick_percent, az_week_5_pick_percent, az_week_6_pick_percent, az_week_7_pick_percent, az_week_8_pick_percent, az_week_9_pick_percent, az_week_10_pick_percent, az_week_11_pick_percent, az_week_12_pick_percent, az_week_13_pick_percent, az_week_14_pick_percent, az_week_15_pick_percent, az_week_16_pick_percent, az_week_17_pick_percent, az_week_18_pick_percent, az_week_19_pick_percent, az_week_20_pick_percent],
        'Atlanta Falcons': [atl_week_1_pick_percent, atl_week_2_pick_percent, atl_week_3_pick_percent, atl_week_4_pick_percent, atl_week_5_pick_percent, atl_week_6_pick_percent, atl_week_7_pick_percent, atl_week_8_pick_percent, atl_week_9_pick_percent, atl_week_10_pick_percent, atl_week_11_pick_percent, atl_week_12_pick_percent, atl_week_13_pick_percent, atl_week_14_pick_percent, atl_week_15_pick_percent, atl_week_16_pick_percent, atl_week_17_pick_percent, atl_week_18_pick_percent, atl_week_19_pick_percent, atl_week_20_pick_percent],
        'Baltimore Ravens': [bal_week_1_pick_percent, bal_week_2_pick_percent, bal_week_3_pick_percent, bal_week_4_pick_percent, bal_week_5_pick_percent, bal_week_6_pick_percent, bal_week_7_pick_percent, bal_week_8_pick_percent, bal_week_9_pick_percent, bal_week_10_pick_percent, bal_week_11_pick_percent, bal_week_12_pick_percent, bal_week_13_pick_percent, bal_week_14_pick_percent, bal_week_15_pick_percent, bal_week_16_pick_percent, bal_week_17_pick_percent, bal_week_18_pick_percent, bal_week_19_pick_percent, bal_week_20_pick_percent],
        'Buffalo Bills': [buf_week_1_pick_percent, buf_week_2_pick_percent, buf_week_3_pick_percent, buf_week_4_pick_percent, buf_week_5_pick_percent, buf_week_6_pick_percent, buf_week_7_pick_percent, buf_week_8_pick_percent, buf_week_9_pick_percent, buf_week_10_pick_percent, buf_week_11_pick_percent, buf_week_12_pick_percent, buf_week_13_pick_percent, buf_week_14_pick_percent, buf_week_15_pick_percent, buf_week_16_pick_percent, buf_week_17_pick_percent, buf_week_18_pick_percent, buf_week_19_pick_percent, buf_week_20_pick_percent],
        'Carolina Panthers': [car_week_1_pick_percent, car_week_2_pick_percent, car_week_3_pick_percent, car_week_4_pick_percent, car_week_5_pick_percent, car_week_6_pick_percent, car_week_7_pick_percent, car_week_8_pick_percent, car_week_9_pick_percent, car_week_10_pick_percent, car_week_11_pick_percent, car_week_12_pick_percent, car_week_13_pick_percent, car_week_14_pick_percent, car_week_15_pick_percent, car_week_16_pick_percent, car_week_17_pick_percent, car_week_18_pick_percent, car_week_19_pick_percent, car_week_20_pick_percent],
        'Chicago Bears': [chi_week_1_pick_percent, chi_week_2_pick_percent, chi_week_3_pick_percent, chi_week_4_pick_percent, chi_week_5_pick_percent, chi_week_6_pick_percent, chi_week_7_pick_percent, chi_week_8_pick_percent, chi_week_9_pick_percent, chi_week_10_pick_percent, chi_week_11_pick_percent, chi_week_12_pick_percent, chi_week_13_pick_percent, chi_week_14_pick_percent, chi_week_15_pick_percent, chi_week_16_pick_percent, chi_week_17_pick_percent, chi_week_18_pick_percent, chi_week_19_pick_percent, chi_week_20_pick_percent],
        'Cincinnati Bengals': [cin_week_1_pick_percent, cin_week_2_pick_percent, cin_week_3_pick_percent, cin_week_4_pick_percent, cin_week_5_pick_percent, cin_week_6_pick_percent, cin_week_7_pick_percent, cin_week_8_pick_percent, cin_week_9_pick_percent, cin_week_10_pick_percent, cin_week_11_pick_percent, cin_week_12_pick_percent, cin_week_13_pick_percent, cin_week_14_pick_percent, cin_week_15_pick_percent, cin_week_16_pick_percent, cin_week_17_pick_percent, cin_week_18_pick_percent, cin_week_19_pick_percent, cin_week_20_pick_percent],
        'Cleveland Browns': [cle_week_1_pick_percent, cle_week_2_pick_percent, cle_week_3_pick_percent, cle_week_4_pick_percent, cle_week_5_pick_percent, cle_week_6_pick_percent, cle_week_7_pick_percent, cle_week_8_pick_percent, cle_week_9_pick_percent, cle_week_10_pick_percent, cle_week_11_pick_percent, cle_week_12_pick_percent, cle_week_13_pick_percent, cle_week_14_pick_percent, cle_week_15_pick_percent, cle_week_16_pick_percent, cle_week_17_pick_percent, cle_week_18_pick_percent, cle_week_19_pick_percent, cle_week_20_pick_percent],
        'Dallas Cowboys': [dal_week_1_pick_percent, dal_week_2_pick_percent, dal_week_3_pick_percent, dal_week_4_pick_percent, dal_week_5_pick_percent, dal_week_6_pick_percent, dal_week_7_pick_percent, dal_week_8_pick_percent, dal_week_9_pick_percent, dal_week_10_pick_percent, dal_week_11_pick_percent, dal_week_12_pick_percent, dal_week_13_pick_percent, dal_week_14_pick_percent, dal_week_15_pick_percent, dal_week_16_pick_percent, dal_week_17_pick_percent, dal_week_18_pick_percent, dal_week_19_pick_percent, dal_week_20_pick_percent],
        'Denver Broncos': [den_week_1_pick_percent, den_week_2_pick_percent, den_week_3_pick_percent, den_week_4_pick_percent, den_week_5_pick_percent, den_week_6_pick_percent, den_week_7_pick_percent, den_week_8_pick_percent, den_week_9_pick_percent, den_week_10_pick_percent, den_week_11_pick_percent, den_week_12_pick_percent, den_week_13_pick_percent, den_week_14_pick_percent, den_week_15_pick_percent, den_week_16_pick_percent, den_week_17_pick_percent, den_week_18_pick_percent, den_week_19_pick_percent, den_week_20_pick_percent],
        'Detroit Lions': [det_week_1_pick_percent, det_week_2_pick_percent, det_week_3_pick_percent, det_week_4_pick_percent, det_week_5_pick_percent, det_week_6_pick_percent, det_week_7_pick_percent, det_week_8_pick_percent, det_week_9_pick_percent, det_week_10_pick_percent, det_week_11_pick_percent, det_week_12_pick_percent, det_week_13_pick_percent, det_week_14_pick_percent, det_week_15_pick_percent, det_week_16_pick_percent, det_week_17_pick_percent, det_week_18_pick_percent, det_week_19_pick_percent, det_week_20_pick_percent],
        'Green Bay Packers': [gb_week_1_pick_percent, gb_week_2_pick_percent, gb_week_3_pick_percent, gb_week_4_pick_percent, gb_week_5_pick_percent, gb_week_6_pick_percent, gb_week_7_pick_percent, gb_week_8_pick_percent, gb_week_9_pick_percent, gb_week_10_pick_percent, gb_week_11_pick_percent, gb_week_12_pick_percent, gb_week_13_pick_percent, gb_week_14_pick_percent, gb_week_15_pick_percent, gb_week_16_pick_percent, gb_week_17_pick_percent, gb_week_18_pick_percent, gb_week_19_pick_percent, gb_week_20_pick_percent],
        'Houston Texans': [hou_week_1_pick_percent, hou_week_2_pick_percent, hou_week_3_pick_percent, hou_week_4_pick_percent, hou_week_5_pick_percent, hou_week_6_pick_percent, hou_week_7_pick_percent, hou_week_8_pick_percent, hou_week_9_pick_percent, hou_week_10_pick_percent, hou_week_11_pick_percent, hou_week_12_pick_percent, hou_week_13_pick_percent, hou_week_14_pick_percent, hou_week_15_pick_percent, hou_week_16_pick_percent, hou_week_17_pick_percent, hou_week_18_pick_percent, hou_week_19_pick_percent, hou_week_20_pick_percent],
        'Indianapolis Colts': [ind_week_1_pick_percent, ind_week_2_pick_percent, ind_week_3_pick_percent, ind_week_4_pick_percent, ind_week_5_pick_percent, ind_week_6_pick_percent, ind_week_7_pick_percent, ind_week_8_pick_percent, ind_week_9_pick_percent, ind_week_10_pick_percent, ind_week_11_pick_percent, ind_week_12_pick_percent, ind_week_13_pick_percent, ind_week_14_pick_percent, ind_week_15_pick_percent, ind_week_16_pick_percent, ind_week_17_pick_percent, ind_week_18_pick_percent, ind_week_19_pick_percent, ind_week_20_pick_percent],
        'Jacksonville Jaguars': [jax_week_1_pick_percent, jax_week_2_pick_percent, jax_week_3_pick_percent, jax_week_4_pick_percent, jax_week_5_pick_percent, jax_week_6_pick_percent, jax_week_7_pick_percent, jax_week_8_pick_percent, jax_week_9_pick_percent, jax_week_10_pick_percent, jax_week_11_pick_percent, jax_week_12_pick_percent, jax_week_13_pick_percent, jax_week_14_pick_percent, jax_week_15_pick_percent, jax_week_16_pick_percent, jax_week_17_pick_percent, jax_week_18_pick_percent, jax_week_19_pick_percent, jax_week_20_pick_percent],
        'Kansas City Chiefs': [kc_week_1_pick_percent, kc_week_2_pick_percent, kc_week_3_pick_percent, kc_week_4_pick_percent, kc_week_5_pick_percent, kc_week_6_pick_percent, kc_week_7_pick_percent, kc_week_8_pick_percent, kc_week_9_pick_percent, kc_week_10_pick_percent, kc_week_11_pick_percent, kc_week_12_pick_percent, kc_week_13_pick_percent, kc_week_14_pick_percent, kc_week_15_pick_percent, kc_week_16_pick_percent, kc_week_17_pick_percent, kc_week_18_pick_percent, kc_week_19_pick_percent, kc_week_20_pick_percent],
        'Las Vegas Raiders': [lv_week_1_pick_percent, lv_week_2_pick_percent, lv_week_3_pick_percent, lv_week_4_pick_percent, lv_week_5_pick_percent, lv_week_6_pick_percent, lv_week_7_pick_percent, lv_week_8_pick_percent, lv_week_9_pick_percent, lv_week_10_pick_percent, lv_week_11_pick_percent, lv_week_12_pick_percent, lv_week_13_pick_percent, lv_week_14_pick_percent, lv_week_15_pick_percent, lv_week_16_pick_percent, lv_week_17_pick_percent, lv_week_18_pick_percent, lv_week_19_pick_percent, lv_week_20_pick_percent],
        'Los Angeles Chargers': [lac_week_1_pick_percent, lac_week_2_pick_percent, lac_week_3_pick_percent, lac_week_4_pick_percent, lac_week_5_pick_percent, lac_week_6_pick_percent, lac_week_7_pick_percent, lac_week_8_pick_percent, lac_week_9_pick_percent, lac_week_10_pick_percent, lac_week_11_pick_percent, lac_week_12_pick_percent, lac_week_13_pick_percent, lac_week_14_pick_percent, lac_week_15_pick_percent, lac_week_16_pick_percent, lac_week_17_pick_percent, lac_week_18_pick_percent, lac_week_19_pick_percent, lac_week_20_pick_percent],
        'Los Angeles Rams': [lar_week_1_pick_percent, lar_week_2_pick_percent, lar_week_3_pick_percent, lar_week_4_pick_percent, lar_week_5_pick_percent, lar_week_6_pick_percent, lar_week_7_pick_percent, lar_week_8_pick_percent, lar_week_9_pick_percent, lar_week_10_pick_percent, lar_week_11_pick_percent, lar_week_12_pick_percent, lar_week_13_pick_percent, lar_week_14_pick_percent, lar_week_15_pick_percent, lar_week_16_pick_percent, lar_week_17_pick_percent, lar_week_18_pick_percent, lar_week_19_pick_percent, lar_week_20_pick_percent],
        'Miami Dolphins': [mia_week_1_pick_percent, mia_week_2_pick_percent, mia_week_3_pick_percent, mia_week_4_pick_percent, mia_week_5_pick_percent, mia_week_6_pick_percent, mia_week_7_pick_percent, mia_week_8_pick_percent, mia_week_9_pick_percent, mia_week_10_pick_percent, mia_week_11_pick_percent, mia_week_12_pick_percent, mia_week_13_pick_percent, mia_week_14_pick_percent, mia_week_15_pick_percent, mia_week_16_pick_percent, mia_week_17_pick_percent, mia_week_18_pick_percent, mia_week_19_pick_percent, mia_week_20_pick_percent],
        'Minnesota Vikings': [min_week_1_pick_percent, min_week_2_pick_percent, min_week_3_pick_percent, min_week_4_pick_percent, min_week_5_pick_percent, min_week_6_pick_percent, min_week_7_pick_percent, min_week_8_pick_percent, min_week_9_pick_percent, min_week_10_pick_percent, min_week_11_pick_percent, min_week_12_pick_percent, min_week_13_pick_percent, min_week_14_pick_percent, min_week_15_pick_percent, min_week_16_pick_percent, min_week_17_pick_percent, min_week_18_pick_percent, min_week_19_pick_percent, min_week_20_pick_percent],
        'New England Patriots': [ne_week_1_pick_percent, ne_week_2_pick_percent, ne_week_3_pick_percent, ne_week_4_pick_percent, ne_week_5_pick_percent, ne_week_6_pick_percent, ne_week_7_pick_percent, ne_week_8_pick_percent, ne_week_9_pick_percent, ne_week_10_pick_percent, ne_week_11_pick_percent, ne_week_12_pick_percent, ne_week_13_pick_percent, ne_week_14_pick_percent, ne_week_15_pick_percent, ne_week_16_pick_percent, ne_week_17_pick_percent, ne_week_18_pick_percent, ne_week_19_pick_percent, ne_week_20_pick_percent],
        'New Orleans Saints': [no_week_1_pick_percent, no_week_2_pick_percent, no_week_3_pick_percent, no_week_4_pick_percent, no_week_5_pick_percent, no_week_6_pick_percent, no_week_7_pick_percent, no_week_8_pick_percent, no_week_9_pick_percent, no_week_10_pick_percent, no_week_11_pick_percent, no_week_12_pick_percent, no_week_13_pick_percent, no_week_14_pick_percent, no_week_15_pick_percent, no_week_16_pick_percent, no_week_17_pick_percent, no_week_18_pick_percent, no_week_19_pick_percent, no_week_20_pick_percent],
        'New York Giants': [nyg_week_1_pick_percent, nyg_week_2_pick_percent, nyg_week_3_pick_percent, nyg_week_4_pick_percent, nyg_week_5_pick_percent, nyg_week_6_pick_percent, nyg_week_7_pick_percent, nyg_week_8_pick_percent, nyg_week_9_pick_percent, nyg_week_10_pick_percent, nyg_week_11_pick_percent, nyg_week_12_pick_percent, nyg_week_13_pick_percent, nyg_week_14_pick_percent, nyg_week_15_pick_percent, nyg_week_16_pick_percent, nyg_week_17_pick_percent, nyg_week_18_pick_percent, nyg_week_19_pick_percent, nyg_week_20_pick_percent],
        'New York Jets': [nyj_week_1_pick_percent, nyj_week_2_pick_percent, nyj_week_3_pick_percent, nyj_week_4_pick_percent, nyj_week_5_pick_percent, nyj_week_6_pick_percent, nyj_week_7_pick_percent, nyj_week_8_pick_percent, nyj_week_9_pick_percent, nyj_week_10_pick_percent, nyj_week_11_pick_percent, nyj_week_12_pick_percent, nyj_week_13_pick_percent, nyj_week_14_pick_percent, nyj_week_15_pick_percent, nyj_week_16_pick_percent, nyj_week_17_pick_percent, nyj_week_18_pick_percent, nyj_week_19_pick_percent, nyj_week_20_pick_percent],
        'Philadelphia Eagles': [phi_week_1_pick_percent, phi_week_2_pick_percent, phi_week_3_pick_percent, phi_week_4_pick_percent, phi_week_5_pick_percent, phi_week_6_pick_percent, phi_week_7_pick_percent, phi_week_8_pick_percent, phi_week_9_pick_percent, phi_week_10_pick_percent, phi_week_11_pick_percent, phi_week_12_pick_percent, phi_week_13_pick_percent, phi_week_14_pick_percent, phi_week_15_pick_percent, phi_week_16_pick_percent, phi_week_17_pick_percent, phi_week_18_pick_percent, phi_week_19_pick_percent, phi_week_20_pick_percent],
        'Pittsburgh Steelers': [pit_week_1_pick_percent, pit_week_2_pick_percent, pit_week_3_pick_percent, pit_week_4_pick_percent, pit_week_5_pick_percent, pit_week_6_pick_percent, pit_week_7_pick_percent, pit_week_8_pick_percent, pit_week_9_pick_percent, pit_week_10_pick_percent, pit_week_11_pick_percent, pit_week_12_pick_percent, pit_week_13_pick_percent, pit_week_14_pick_percent, pit_week_15_pick_percent, pit_week_16_pick_percent, pit_week_17_pick_percent, pit_week_18_pick_percent, pit_week_19_pick_percent, pit_week_20_pick_percent],
        'San Francisco 49ers': [sf_week_1_pick_percent, sf_week_2_pick_percent, sf_week_3_pick_percent, sf_week_4_pick_percent, sf_week_5_pick_percent, sf_week_6_pick_percent, sf_week_7_pick_percent, sf_week_8_pick_percent, sf_week_9_pick_percent, sf_week_10_pick_percent, sf_week_11_pick_percent, sf_week_12_pick_percent, sf_week_13_pick_percent, sf_week_14_pick_percent, sf_week_15_pick_percent, sf_week_16_pick_percent, sf_week_17_pick_percent, sf_week_18_pick_percent, sf_week_19_pick_percent, sf_week_20_pick_percent],
        'Seattle Seahawks': [sea_week_1_pick_percent, sea_week_2_pick_percent, sea_week_3_pick_percent, sea_week_4_pick_percent, sea_week_5_pick_percent, sea_week_6_pick_percent, sea_week_7_pick_percent, sea_week_8_pick_percent, sea_week_9_pick_percent, sea_week_10_pick_percent, sea_week_11_pick_percent, sea_week_12_pick_percent, sea_week_13_pick_percent, sea_week_14_pick_percent, sea_week_15_pick_percent, sea_week_16_pick_percent, sea_week_17_pick_percent, sea_week_18_pick_percent, sea_week_19_pick_percent, sea_week_20_pick_percent],
        'Tampa Bay Buccaneers': [tb_week_1_pick_percent, tb_week_2_pick_percent, tb_week_3_pick_percent, tb_week_4_pick_percent, tb_week_5_pick_percent, tb_week_6_pick_percent, tb_week_7_pick_percent, tb_week_8_pick_percent, tb_week_9_pick_percent, tb_week_10_pick_percent, tb_week_11_pick_percent, tb_week_12_pick_percent, tb_week_13_pick_percent, tb_week_14_pick_percent, tb_week_15_pick_percent, tb_week_16_pick_percent, tb_week_17_pick_percent, tb_week_18_pick_percent, tb_week_19_pick_percent, tb_week_20_pick_percent],
        'Tennessee Titans': [ten_week_1_pick_percent, ten_week_2_pick_percent, ten_week_3_pick_percent, ten_week_4_pick_percent, ten_week_5_pick_percent, ten_week_6_pick_percent, ten_week_7_pick_percent, ten_week_8_pick_percent, ten_week_9_pick_percent, ten_week_10_pick_percent, ten_week_11_pick_percent, ten_week_12_pick_percent, ten_week_13_pick_percent, ten_week_14_pick_percent, ten_week_15_pick_percent, ten_week_16_pick_percent, ten_week_17_pick_percent, ten_week_18_pick_percent, ten_week_19_pick_percent, ten_week_20_pick_percent],
        'Washington Commanders': [was_week_1_pick_percent, was_week_2_pick_percent, was_week_3_pick_percent, was_week_4_pick_percent, was_week_5_pick_percent, was_week_6_pick_percent, was_week_7_pick_percent, was_week_8_pick_percent, was_week_9_pick_percent, was_week_10_pick_percent, was_week_11_pick_percent, was_week_12_pick_percent, was_week_13_pick_percent, was_week_14_pick_percent, was_week_15_pick_percent, was_week_16_pick_percent, was_week_17_pick_percent, was_week_18_pick_percent, was_week_19_pick_percent, was_week_20_pick_percent]
    }

    
    st.write('')
    st.write('')
    st.write('')
    
    use_cached_expected_value = 0
    use_live_sportsbook_odds = 1
    
#    if yes_i_have_customized_rankings:
#    	st.subheader('Use Saved Expected Value')
#    	st.write('Warning, this data may not be nup to date.')
#    	st.write('- Checking this box will ensure the process is fast, (Less than 1 minute, compared to 5-10 mins) and will prevent the risk of crashing')
#    	st.write('- This will not use your customized rankings in the EV calculation process')
#    	st.write('- This will NOT have an impact on your customized ranking output, just the EV output')
#    	st.write('Last Update: :green[01/01/2025]')
#    	use_cached_expected_value = 1 if st.checkbox('Use Cached Expected Value') else 0
#    	st.write('')
#    	st.write('')
#    	st.write('')
#    	if use_cached_expected_value == 1:
#                use_live_sportsbook_odds = 1 if st.checkbox('Use Live Sportsbook Odds to calculate win probability (If Available?') else 0
#                st.write('')
#                st.write("If this is checked, we will use odds from DraftKings to determine a team's win probability. For games where live odds from DraftKings are unavailable, we will use your own internal rankings to determine the predicted spread and win probability.")	
#                st.write('If this is left unchecked, we will use your own internal rankings to determine the predicted spread and win probability for all games.')	
#    	st.write('')
#    	st.write('')
#    	st.write('')
    st.subheader('Get Optimized Survivor Picks')
    number_of_solutions_options = [
        1,5,10,25,50,100
    ]

    st.write('How many solutions would you like from each solver?')
    number_solutions = st.selectbox('Number of Solutions', options = number_of_solutions_options, help = "How many solutions do you want to produce? We recommend 10.")
    double_number_solutions = number_solutions * 2

    with st.expander("More Information"):
        st.write(f"""This button will find the best picks for each week. It will pump out :green[{double_number_solutions} solutions].
	\n- The first {number_solutions} solutions will be :red[based purely on EV] that is a complicated formula based on their predicted pick percentage of each team in each week, and each team's chances of winning that week.
 	\n- This will use the rankings defined above (or within our system) to determine win probability and thus pick percentage for each team.
  	\n- The remaining {number_solutions} solutions will be based on the :red[rankings and constraints you provided]. 
  	\n- This method finds the teams that are just straight up most likely to win. While helpful, thsi does not provide you with much of a competitive advantage.
   	\n- All solutions will abide by the constraints you've provided
   	\n- If you have too many constraints, or the solution is impossible, you will see an error
       	\n- :green[Mathematically, EV is most likely to win. However, using your own rankings has advantages as well, which is why we provide both solutions (Sometimes it's just preposterous to pick the Jets)]
        """)
    
    
    st.write('')
    st.write('')
    schedule_data_retrieved = False #Initialize on first run
    
    if st.button("Get Optimized Survivor Picks"):
        st.write("Step 1/6: Fetching Schedule Data...")
        schedule_table, schedule_rows = get_schedule() # Call the function   
        if schedule_table:
            st.write("Step 1 Completed: Schedule Data Retrieved!")
            schedule_data_retrieved = True #Set Flag to True after retrieval
        else:
            st.write("Error. Could not find the table.")
            schedule_data_retrieved = False #Set flag to False on error
             
        if schedule_rows:
            st.write(f"Number of Schedule Rows: {len(schedule_rows)}") #Display row length
            st.write("Step 2/6: Collecting Travel, Ranking, Odds, and Rest Data...")
            collect_schedule_travel_ranking_data_df = collect_schedule_travel_ranking_data(pd)
            st.write("Step 2 Completed: Travel, Ranking, Odds, and Rest Data Retrieved!")
            st.write("Step 3/6: Predicting Future Pick Percentages of Public...")
        if use_cached_expected_value == 0:
            nfl_schedule_pick_percentages_df = get_predicted_pick_percentages(pd)
            st.write("Step 3 Completed: Public Pick Percentages Predicted")
            #nfl_schedule_circa_df_2 = manually_adjust_pick_predictions()
        if use_cached_expected_value == 0:
            if selected_contest == 'Circa':
                st.write("Step 3a: Predicting Pick Percentages based on Team Availability...")
                nfl_schedule_pick_percentages_df = get_predicted_pick_percentages_with_availability(pd)
        if use_cached_expected_value == 1:
            nfl_schedule_pick_percentages_df = get_predicted_pick_percentages(pd)
            if selected_contest == 'Circa':
                full_df_with_ev = nfl_schedule_pick_percentages_df_circa #pd.read_csv('NFL Schedule with full ev_circa.csv')
            elif selected_contest == 'Splash Sports':
                full_df_with_ev = nfl_schedule_pick_percentages_df_splash #pd.read_csv('NFL Schedule with full ev_circa.csv'
            else:
                full_df_with_ev = nfl_schedule_pick_percentages_df_dk #pd.read_csv('NFL Schedule with full ev_dk.csv')
            st.write("Step 3 Completed: Public Pick Percentages Predicted")
            st.write('- Using Cached Expected Values...')
        else:
            st.write("Step 4/6: Calculating Live Expected Value (Will take 5-10 mins)...")
            with st.spinner('Processing...'):
                full_df_with_ev = calculate_ev(nfl_schedule_pick_percentages_df, starting_week, ending_week, selected_contest, use_cached_expected_value)
                st.write("Processing Complete!")
                #st.dataframe(full_df_with_ev)
        st.write("Step 4 Completed: Expected Value Calculated")
        st.subheader('Full Dataset')
        st.write(full_df_with_ev)
        st.write('Step 5/6: Calculating Best Combination of Picks Based on EV...')
        st.write('')
    
    
        ending_week_2 = ending_week - 1	
        if selected_contest == 'Circa':
            st.subheader(f':blue[Optimal Picks for Circa: Weeks {starting_week} through {ending_week_2}]')
            st.write('')
        elif selected_contest == 'Splash Sports':
            st.subheader(f':yellow[Optimal Picks for Splash Sports: Weeks {starting_week} through {ending_week_2}]')
            st.write('')
        else:
            st.subheader(f':green[Optimal Picks for Draftkings: Weeks {starting_week} through {ending_week_2}]')
            st.write('')
            st.subheader('Expected Value Optimized Picks')
    
        get_survivor_picks_based_on_ev()
        st.write('Step 5 Completed: Top Picks Determined Based on EV')
        if yes_i_have_customized_rankings:
            st.write('Step 6/6: Calculating Best Combination of Picks Based on Customized Rankings...')
            st.write("---------------------------------------------------------------------------------------------------------------")
            st.write("---------------------------------------------------------------------------------------------------------------")
            st.write("---------------------------------------------------------------------------------------------------------------")
            st.subheader('Customized Ranking Optimized Picks')
            get_survivor_picks_based_on_internal_rankings()
            st.write('Step 6 Completed: Top Picks Determined Based on Customized Rankings')
        else:
            st.write('Step 6/6: Calculating Best Combination of Picks Based on Default Rankings...')
            st.write("---------------------------------------------------------------------------------------------------------------")
            st.write("---------------------------------------------------------------------------------------------------------------")
            st.write("---------------------------------------------------------------------------------------------------------------")
            st.subheader('Default Ranking Optimized Picks')
            get_survivor_picks_based_on_internal_rankings()
            st.write('Step 6 Completed: Top Picks Determined Based on Default Rankings')
    else:
        schedule_data_retrieved = False #Set flag to False on error
