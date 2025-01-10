import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import pytz
from dateutil.parser import parse
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from ortools.linear_solver import pywraplp

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

def collect_schedule_travel_ranking_data_circa(pd):
    data = []
    # Initialize a variable to hold the last valid date and week
    last_date = None
    start_date_str = 'September 4, 2024'
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
                    date = date.replace(year=2025)
                else:
                    date = date.replace(year=2024)
                # Adjust week for games on or after November 30th
                if date >= pd.Timestamp('2024-11-30'):
                    week += 1
                # Adjust week for games on or after December 27th
                if date >= pd.Timestamp('2024-12-27'):
                    week += 1
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
            cols_text.insert(0, {week})

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

    df['Date'] = df['Date'].str.replace(r'(\w+)\s(\w+)\s(\d+)', r'\2 \3, 2024', regex=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    # Adjust January games to 2025 in the DataFrame
    df['Date'] = df['Date'].apply(lambda x: x.replace(year=2025) if x.month == 1 else x)
    #df['Week_Num'] = df['Week'].str.replace('Week ', '').astype(int)
    df['Week'] = df['Week'].astype(int)
    

    # Increment 'Week' for games on or after 2024-11-30
    df.loc[df['Date'] >= pd.to_datetime('2024-11-30'), 'Week'] += 1
    df.loc[df['Date'] >= pd.to_datetime('2024-12-27'), 'Week'] += 1

    # Convert 'Week' back to string format if needed
    #df['Week'] = 'Week ' + df['Week'].astype(str)
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


    def get_preseason_odds():
        url = "https://sportsbook.draftkings.com/leagues/football/nfl"

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.text, 'html.parser')


        team_name_mapping = {
            "ARI Cardinals" : "Arizona Cardinals",
            "ATL Falcons" : "Atlanta Falcons",
            "BAL Ravens" : "Baltimore Ravens",
            "BUF Bills" : "Buffalo Bills",
            "CAR Panthers" : "Carolina Panthers",
            "CHI Bears" : "Chicago Bears",
            "CIN Bengals" : 'Cincinnati Bengals',
            "CLE Browns" : 'Cleveland Browns',
            "DAL Cowboys" : 'Dallas Cowboys',
            "DEN Broncos" : 'Denver Broncos',
            "DET Lions" : 'Detroit Lions',
            "GB Packers" : 'Green Bay Packers',
            "HOU Texans" : 'Houston Texans',
            "IND Colts" : 'Indianapolis Colts',
            "JAX Jaguars" : 'Jacksonville Jaguars',
            "KC Chiefs" : 'Kansas City Chiefs',
            "LV Raiders" : 'Las Vegas Raiders',
            "LA Chargers" : 'Los Angeles Chargers',
            "LA Rams" : 'Los Angeles Rams',
            "MIA Dolphins" : 'Miami Dolphins',
            "MIN Vikings" : 'Minnesota Vikings',
            "NE Patriots" : 'New England Patriots',
            "NO Saints" : 'New Orleans Saints',
            "NY Giants" : 'New York Giants',
            "NY Jets" : 'New York Jets',
            "PHI Eagles" : 'Philadelphia Eagles',
            "PIT Steelers" : 'Pittsburgh Steelers',
            "SF 49ers" : 'San Francisco 49ers',
            "SEA Seahawks" : 'Seattle Seahawks',
            "TB Buccaneers" : 'Tampa Bay Buccaneers',
            "TEN Titans" : 'Tennessee Titans',
            "WAS Commanders" : 'Washington Commanders'
        }

        # Find all the table rows containing game data
        game_rows = soup.find_all('tr', class_=['break-line', ''])

        games = []
        game_data = {}  # Temporary dictionary to store game data

        for i, row in enumerate(game_rows):
            # Extract time only from the first row of a game
            if 'break-line' in row['class']:
                time = row.find('span', class_='event-cell__start-time').text
                game_data['Time'] = time

            # Extract team and odds - handle potential missing elements
            team = row.find('div', class_='event-cell__name-text')
            if team:
                team = team.text.strip()
                team = team_name_mapping.get(team, team)
            else:
                team = None  # Set team to None if not found

            odds_element = row.find('span', class_='sportsbook-odds american no-margin default-color')
            if odds_element:
                odds = odds_element.text.strip().replace('−', '-')
                odds = int(odds)
            else:
                odds = None  # Set odds to None if not found

            # Assign team and odds to the appropriate key in the game_data dictionary
            if i % 2 == 0:  # Even index: Away Team
                game_data['Away Team'] = team
                game_data['Away Odds'] = odds
            else:  # Odd index: Home Team
                game_data['Home Team'] = team
                game_data['Home Odds'] = odds

                # Append complete game data to the games list and reset game_data
                games.append(game_data)
                game_data = {}

        # Create pandas DataFrame from the extracted data
        df = pd.DataFrame(games)

        print(df)
        df.to_csv('Live Scraped Odds.csv', index=False)

        live_scraped_odds_nfl_df = df

        return live_scraped_odds_nfl_df
            
    live_scraped_odds_df = get_preseason_odds()    
    
    
    def add_odds_to_main_csv():
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
            0: [-110, -110],
            .5: [-116, -104],    
            1: [-122, 101],
            1.5: [-128, 105],
            2: [-131, 108],
            2.5: [-142, 117],
            3: [-164, 135],
            3.5: [-191, 156],
            4: [-211, 171],
            4.5: [-224, 181],
            5: [-234, 188],
            5.5: [-244, 195],
            6: [-261, 208],
            6.5: [-282, 224],
            7: [-319, 249],
            7.5: [-346, 268],
            8: [-366, 282],
            8.5: [-397, 302],
            9: [-416, 314],
            9.5: [-436, 327],
            10: [-483, 356],
            10.5: [-538, 389],
            11: [-567, 406],
            11.5: [-646, 450],
            12: [-660, 458],
            12.5: [-675, 466],
            13: [-729, 494],
            13.5: [-819, 539],
            14: [-890, 573],
            14.5: [-984, 615],
            15: [-1134, 677],
            15.5: [-1197, 702],
            16: [-1266, 728],
            16.5: [-1267, 728],
            17: [-1381, 769],
            17.5: [-1832, 906],
            18: [-2149, 986],
            18.5: [-2590, 1079],
            19: [-3245, 1190],
            19.5: [-4323, 1324],
            20: [-4679, 1359],
            20.5: [-5098, 1396],
            21: [-5597, 1434],
            21.5: [-6000, 1500],
            22: [-6500, 1600],
            22.5: [-7000, 1650],
            23: [-7500, 1700],
            23.5: [-8000, 1750],
            24: [-8500, 1800],
            24.5: [-9000, 1850],
            25: [-9500, 1900],
            25.5: [-10000, 2000],
            26: [-10000, 2000],
            26.5: [-10000, 2000],
            27: [-10000, 2000],
            27.5: [-10000, 2000],
            28: [-10000, 2000],
            28.5: [-10000, 2000],
            29: [-10000, 2000],
            29.5: [-10000, 2000],
            30: [-10000, 2000]
        }

        live_odds_df = live_scraped_odds_df
        

        #df.to_csv('TEST Manual Odds.csv', index = False)
        # Load the CSV data
        csv_df = df
        # Update CSV data with scraped odds
        for index, row in csv_df.iterrows():
            matching_row = live_odds_df[
                (live_odds_df['Away Team'] == row['Away Team']) & (live_odds_df['Home Team'] == row['Home Team'])
            ]
            if not matching_row.empty:
                csv_df.loc[index, 'Away Team Moneyline'] = matching_row.iloc[0]['Away Odds']
                csv_df.loc[index, 'Home Team Moneyline'] = matching_row.iloc[0]['Home Odds']
                csv_df.loc[index, 'Favorite'] = csv_df.loc[index, 'Home Team'] if csv_df.loc[index, 'Home Team Moneyline'] <= -110 else csv_df.loc[index, 'Away Team']
                csv_df.loc[index, 'Underdog'] = csv_df.loc[index, 'Home Team'] if csv_df.loc[index, 'Home Team Moneyline'] > -110 else csv_df.loc[index, 'Away Team']
		
                # Create the mask for where there is no 'Home Odds'
        mask = csv_df['Home Team Moneyline'].isna()
        # Only apply calculations if the 'Home Odds' column is empty
        def get_moneyline_masked(row, odds, team_type):
            spread = round(row['Adjusted Spread'] * 2) / 2
            try:
                moneyline_tuple = odds[spread]
                if team_type == 'home':
                    if row['Favorite'] == row['Home Team']:
                        return moneyline_tuple[0]
                    else:
                        return moneyline_tuple[1]
                elif team_type == 'away':
                    if row['Favorite'] == row['Away Team']:
                        return moneyline_tuple[0]
                    else:
                        return moneyline_tuple[1]

            except KeyError:
                if team_type == 'home':
                    if row['Favorite'] == row['Home Team']:
                        return -10000
                    else:
                        return 2000
                elif team_type == 'away':
                    if row['Favorite'] == row['Away Team']:
                        return -10000
                    else:
                        return 2000

        if mask.any():
            # Adjust Average Points Difference for Favorite/Underdog Determination
            csv_df['Adjusted Home Points'] = csv_df['Home Team Adjusted Current Rank']
            csv_df['Adjusted Away Points'] = csv_df['Away Team Adjusted Current Rank']

            csv_df['Preseason Spread'] = abs(csv_df['Away Team Adjusted Preseason Rank'] - csv_df['Home Team Adjusted Preseason Rank'])

            # Determine Favorite and Underdog
            csv_df['Favorite'] = csv_df.apply(lambda row: row['Home Team'] if row['Home Team Adjusted Current Rank'] >= row['Away Team Adjusted Current Rank'] else row['Away Team'], axis=1)
            csv_df['Underdog'] = csv_df.apply(lambda row: row['Home Team'] if row['Home Team Adjusted Current Rank'] < row['Away Team Adjusted Current Rank'] else row['Away Team'], axis=1)

            # Adjust Spread based on Favorite
            csv_df['Adjusted Spread'] = abs(csv_df['Away Team Adjusted Current Rank'] - csv_df['Home Team Adjusted Current Rank'])

            # Overwrite Odds based on Spread and Favorite/Underdog
            csv_df['Home Team Moneyline'] = csv_df.apply(
               lambda row: get_moneyline_masked(row, odds, 'home'), axis=1
            )
            csv_df['Away Team Moneyline'] = csv_df.apply(
                lambda row: get_moneyline_masked(row, odds, 'away'), axis=1
            )

        def get_moneyline(row, odds, team_type):
            spread = round(row['Adjusted Spread'] * 2) / 2
            try:
                moneyline_tuple = odds[spread]
                if team_type == 'home':
                    if row['Favorite'] == row['Home Team']:
                        return moneyline_tuple[0]
                    else:
                        return moneyline_tuple[1]
                elif team_type == 'away':
                    if row['Favorite'] == row['Away Team']:
                        return moneyline_tuple[0]
                    else:
                        return moneyline_tuple[1]

            except KeyError:
                if team_type == 'home':
                    if row['Favorite'] == row['Home Team']:
                        return -10000
                    else:
                        return 2000
                elif team_type == 'away':
                    if row['Favorite'] == row['Away Team']:
                        return -10000
                    else:
                        return 2000

        csv_df['Internal Ranking Home Team Moneyline'] = csv_df.apply(
            lambda row: get_moneyline(row, odds, 'home'), axis=1
        )

        csv_df['Internal Ranking Away Team Moneyline'] = csv_df.apply(
            lambda row: get_moneyline(row, odds, 'away'), axis=1
        )

        for index, row in csv_df.iterrows():
            # Implied Odds
            if row['Away Team Moneyline'] > 0:
                csv_df.loc[index, 'Away Team Implied Odds to Win'] = 100 / (row['Away Team Moneyline'] + 100)
            else:
                csv_df.loc[index, 'Away Team Implied Odds to Win'] = abs(row['Away Team Moneyline']) / (abs(row['Away Team Moneyline']) + 100)

            if row['Home Team Moneyline'] > 0:
                csv_df.loc[index, 'Home team Implied Odds to Win'] = 100 / (row['Home Team Moneyline'] + 100)
            else:
                csv_df.loc[index, 'Home team Implied Odds to Win'] = abs(row['Home Team Moneyline']) / (abs(row['Home Team Moneyline']) + 100)

            # Fair Odds
            away_implied_odds = csv_df.loc[index, 'Away Team Implied Odds to Win']
            home_implied_odds = csv_df.loc[index, 'Home team Implied Odds to Win']
            csv_df.loc[index, 'Away Team Fair Odds'] = away_implied_odds / (away_implied_odds + home_implied_odds)
            csv_df.loc[index, 'Home Team Fair Odds'] = home_implied_odds / (away_implied_odds + home_implied_odds)

            # Convert to percentage and round to 2 decimal places
            csv_df.loc[index, 'Away Team Implied Odds to Win'] = round(csv_df.loc[index, 'Away Team Implied Odds to Win'], 4)
            csv_df.loc[index, 'Home team Implied Odds to Win'] = round(csv_df.loc[index, 'Home team Implied Odds to Win'], 4)
            csv_df.loc[index, 'Away Team Fair Odds'] = round(csv_df.loc[index, 'Away Team Fair Odds'], 4)
            csv_df.loc[index, 'Home Team Fair Odds'] = round(csv_df.loc[index, 'Home Team Fair Odds'], 4)

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
    consolidated_csv_file = "nfl_schedule_circa.csv"
    consolidated_df.to_csv(consolidated_csv_file, index=False)    
    collect_schedule_travel_ranking_data_nfl_schedule_df = consolidated_df
    
    return collect_schedule_travel_ranking_data_nfl_schedule_df

def collect_schedule_travel_ranking_data_draftkings(pd):
    data = []
    # Initialize a variable to hold the last valid date and week
    last_date = None
    start_date_str = 'September 4, 2024'
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
    for schedule_row in schedule_table.find_all('tr'):
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
                    date = date.replace(year=2025)
                else:
                    date = date.replace(year=2024)

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
            cols_text.insert(0, {week})

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

    df['Date'] = df['Date'].str.replace(r'(\w+)\s(\w+)\s(\d+)', r'\2 \3, 2024', regex=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    # Adjust January games to 2025 in the DataFrame
    df['Date'] = df['Date'].apply(lambda x: x.replace(year=2025) if x.month == 1 else x)
    df['Week_Num'] = df['Week'].str.replace('Week ', '').astype(int)
    #df['Week'] = df['Week'].str.replace('Week ', '', regex=False).astype(int)
    

    # Convert 'Week' back to string format if needed
    #df['Week'] = 'Week ' + df['Week'].astype(str)
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


    def get_preseason_odds():
        url = "https://sportsbook.draftkings.com/leagues/football/nfl"

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.text, 'html.parser')


        team_name_mapping = {
            "ARI Cardinals" : "Arizona Cardinals",
            "ATL Falcons" : "Atlanta Falcons",
            "BAL Ravens" : "Baltimore Ravens",
            "BUF Bills" : "Buffalo Bills",
            "CAR Panthers" : "Carolina Panthers",
            "CHI Bears" : "Chicago Bears",
            "CIN Bengals" : 'Cincinnati Bengals',
            "CLE Browns" : 'Cleveland Browns',
            "DAL Cowboys" : 'Dallas Cowboys',
            "DEN Broncos" : 'Denver Broncos',
            "DET Lions" : 'Detroit Lions',
            "GB Packers" : 'Green Bay Packers',
            "HOU Texans" : 'Houston Texans',
            "IND Colts" : 'Indianapolis Colts',
            "JAX Jaguars" : 'Jacksonville Jaguars',
            "KC Chiefs" : 'Kansas City Chiefs',
            "LV Raiders" : 'Las Vegas Raiders',
            "LA Chargers" : 'Los Angeles Chargers',
            "LA Rams" : 'Los Angeles Rams',
            "MIA Dolphins" : 'Miami Dolphins',
            "MIN Vikings" : 'Minnesota Vikings',
            "NE Patriots" : 'New England Patriots',
            "NO Saints" : 'New Orleans Saints',
            "NY Giants" : 'New York Giants',
            "NY Jets" : 'New York Jets',
            "PHI Eagles" : 'Philadelphia Eagles',
            "PIT Steelers" : 'Pittsburgh Steelers',
            "SF 49ers" : 'San Francisco 49ers',
            "SEA Seahawks" : 'Seattle Seahawks',
            "TB Buccaneers" : 'Tampa Bay Buccaneers',
            "TEN Titans" : 'Tennessee Titans',
            "WAS Commanders" : 'Washington Commanders'
        }

        # Find all the table rows containing game data
        game_rows = soup.find_all('tr', class_=['break-line', ''])

        games = []
        game_data = {}  # Temporary dictionary to store game data

        for i, row in enumerate(game_rows):
            # Extract time only from the first row of a game
            if 'break-line' in row['class']:
                time = row.find('span', class_='event-cell__start-time').text
                game_data['Time'] = time

            # Extract team and odds - handle potential missing elements
            team = row.find('div', class_='event-cell__name-text')
            if team:
                team = team.text.strip()
                team = team_name_mapping.get(team, team)
            else:
                team = None  # Set team to None if not found

            odds_element = row.find('span', class_='sportsbook-odds american no-margin default-color')
            if odds_element:
                odds = odds_element.text.strip().replace('−', '-')
                odds = int(odds)
            else:
                odds = None  # Set odds to None if not found

            # Assign team and odds to the appropriate key in the game_data dictionary
            if i % 2 == 0:  # Even index: Away Team
                game_data['Away Team'] = team
                game_data['Away Odds'] = odds
            else:  # Odd index: Home Team
                game_data['Home Team'] = team
                game_data['Home Odds'] = odds

                # Append complete game data to the games list and reset game_data
                games.append(game_data)
                game_data = {}

        # Create pandas DataFrame from the extracted data
        df = pd.DataFrame(games)

        print(df)
        df.to_csv('Live Scraped Odds.csv', index=False)

        live_scraped_odds_nfl_df = df

        return live_scraped_odds_nfl_df
            
    live_scraped_odds_df = get_preseason_odds()    
    
    
    def add_odds_to_main_csv():
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
            0: [-110, -110],
            .5: [-116, -104],    
            1: [-122, 101],
            1.5: [-128, 105],
            2: [-131, 108],
            2.5: [-142, 117],
            3: [-164, 135],
            3.5: [-191, 156],
            4: [-211, 171],
            4.5: [-224, 181],
            5: [-234, 188],
            5.5: [-244, 195],
            6: [-261, 208],
            6.5: [-282, 224],
            7: [-319, 249],
            7.5: [-346, 268],
            8: [-366, 282],
            8.5: [-397, 302],
            9: [-416, 314],
            9.5: [-436, 327],
            10: [-483, 356],
            10.5: [-538, 389],
            11: [-567, 406],
            11.5: [-646, 450],
            12: [-660, 458],
            12.5: [-675, 466],
            13: [-729, 494],
            13.5: [-819, 539],
            14: [-890, 573],
            14.5: [-984, 615],
            15: [-1134, 677],
            15.5: [-1197, 702],
            16: [-1266, 728],
            16.5: [-1267, 728],
            17: [-1381, 769],
            17.5: [-1832, 906],
            18: [-2149, 986],
            18.5: [-2590, 1079],
            19: [-3245, 1190],
            19.5: [-4323, 1324],
            20: [-4679, 1359],
            20.5: [-5098, 1396],
            21: [-5597, 1434],
            21.5: [-6000, 1500],
            22: [-6500, 1600],
            22.5: [-7000, 1650],
            23: [-7500, 1700],
            23.5: [-8000, 1750],
            24: [-8500, 1800],
            24.5: [-9000, 1850],
            25: [-9500, 1900],
            25.5: [-10000, 2000],
            26: [-10000, 2000],
            26.5: [-10000, 2000],
            27: [-10000, 2000],
            27.5: [-10000, 2000],
            28: [-10000, 2000],
            28.5: [-10000, 2000],
            29: [-10000, 2000],
            29.5: [-10000, 2000],
            30: [-10000, 2000]
        }

        live_odds_df = live_scraped_odds_df
        

        #df.to_csv('TEST Manual Odds.csv', index = False)
        # Load the CSV data
        csv_df = df
        # Update CSV data with scraped odds
        for index, row in csv_df.iterrows():
            matching_row = live_odds_df[
                (live_odds_df['Away Team'] == row['Away Team']) & (live_odds_df['Home Team'] == row['Home Team'])
            ]
            if not matching_row.empty:
                csv_df.loc[index, 'Away Team Moneyline'] = matching_row.iloc[0]['Away Odds']
                csv_df.loc[index, 'Home Team Moneyline'] = matching_row.iloc[0]['Home Odds']
                csv_df.loc[index, 'Favorite'] = csv_df.loc[index, 'Home Team'] if csv_df.loc[index, 'Home Team Moneyline'] <= -110 else csv_df.loc[index, 'Away Team']
                csv_df.loc[index, 'Underdog'] = csv_df.loc[index, 'Home Team'] if csv_df.loc[index, 'Home Team Moneyline'] > -110 else csv_df.loc[index, 'Away Team']
		
                # Create the mask for where there is no 'Home Odds'
        mask = csv_df['Home Team Moneyline'].isna()
        # Only apply calculations if the 'Home Odds' column is empty

        def get_moneyline_masked(row, odds, team_type):
            spread = round(row['Adjusted Spread'] * 2) / 2
            try:
                moneyline_tuple = odds[spread]
                if team_type == 'home':
                    if row['Favorite'] == row['Home Team']:
                        return moneyline_tuple[0]
                    else:
                        return moneyline_tuple[1]
                elif team_type == 'away':
                    if row['Favorite'] == row['Away Team']:
                        return moneyline_tuple[0]
                    else:
                        return moneyline_tuple[1]

            except KeyError:
                if team_type == 'home':
                    if row['Favorite'] == row['Home Team']:
                        return -10000
                    else:
                        return 2000
                elif team_type == 'away':
                    if row['Favorite'] == row['Away Team']:
                        return -10000
                    else:
                        return 2000

        if mask.any():
            # Adjust Average Points Difference for Favorite/Underdog Determination
            csv_df['Adjusted Home Points'] = csv_df['Home Team Adjusted Current Rank']
            csv_df['Adjusted Away Points'] = csv_df['Away Team Adjusted Current Rank']

            csv_df['Preseason Spread'] = abs(csv_df['Away Team Adjusted Preseason Rank'] - csv_df['Home Team Adjusted Preseason Rank'])

            # Determine Favorite and Underdog
            csv_df['Favorite'] = csv_df.apply(lambda row: row['Home Team'] if row['Home Team Adjusted Current Rank'] >= row['Away Team Adjusted Current Rank'] else row['Away Team'], axis=1)
            csv_df['Underdog'] = csv_df.apply(lambda row: row['Home Team'] if row['Home Team Adjusted Current Rank'] < row['Away Team Adjusted Current Rank'] else row['Away Team'], axis=1)

            # Adjust Spread based on Favorite
            csv_df['Adjusted Spread'] = abs(csv_df['Away Team Adjusted Current Rank'] - csv_df['Home Team Adjusted Current Rank'])

            # Overwrite Odds based on Spread and Favorite/Underdog
            csv_df['Home Team Moneyline'] = csv_df.apply(
               lambda row: get_moneyline_masked(row, odds, 'home'), axis=1
            )
            csv_df['Away Team Moneyline'] = csv_df.apply(
                lambda row: get_moneyline_masked(row, odds, 'away'), axis=1
            )

        def get_moneyline(row, odds, team_type):
            spread = round(row['Adjusted Spread'] * 2) / 2
            try:
                moneyline_tuple = odds[spread]
                if team_type == 'home':
                    if row['Favorite'] == row['Home Team']:
                        return moneyline_tuple[0]
                    else:
                        return moneyline_tuple[1]
                elif team_type == 'away':
                    if row['Favorite'] == row['Away Team']:
                        return moneyline_tuple[0]
                    else:
                        return moneyline_tuple[1]

            except KeyError:
                if team_type == 'home':
                    if row['Favorite'] == row['Home Team']:
                        return -10000
                    else:
                        return 2000
                elif team_type == 'away':
                    if row['Favorite'] == row['Away Team']:
                        return -10000
                    else:
                        return 2000

        csv_df['Internal Ranking Home Team Moneyline'] = csv_df.apply(
            lambda row: get_moneyline(row, odds, 'home'), axis=1
        )

        csv_df['Internal Ranking Away Team Moneyline'] = csv_df.apply(
            lambda row: get_moneyline(row, odds, 'away'), axis=1
        )

        for index, row in csv_df.iterrows():
            # Implied Odds
            if row['Away Team Moneyline'] > 0:
                csv_df.loc[index, 'Away Team Implied Odds to Win'] = 100 / (row['Away Team Moneyline'] + 100)
            else:
                csv_df.loc[index, 'Away Team Implied Odds to Win'] = abs(row['Away Team Moneyline']) / (abs(row['Away Team Moneyline']) + 100)

            if row['Home Team Moneyline'] > 0:
                csv_df.loc[index, 'Home team Implied Odds to Win'] = 100 / (row['Home Team Moneyline'] + 100)
            else:
                csv_df.loc[index, 'Home team Implied Odds to Win'] = abs(row['Home Team Moneyline']) / (abs(row['Home Team Moneyline']) + 100)
            

            # Fair Odds
            away_implied_odds = csv_df.loc[index, 'Away Team Implied Odds to Win']
            home_implied_odds = csv_df.loc[index, 'Home team Implied Odds to Win']
            csv_df.loc[index, 'Away Team Fair Odds'] = away_implied_odds / (away_implied_odds + home_implied_odds)
            csv_df.loc[index, 'Home Team Fair Odds'] = home_implied_odds / (away_implied_odds + home_implied_odds)

            # Convert to percentage and round to 2 decimal places
            csv_df.loc[index, 'Away Team Implied Odds to Win'] = round(csv_df.loc[index, 'Away Team Implied Odds to Win'], 4)
            csv_df.loc[index, 'Home team Implied Odds to Win'] = round(csv_df.loc[index, 'Home team Implied Odds to Win'], 4)
            csv_df.loc[index, 'Away Team Fair Odds'] = round(csv_df.loc[index, 'Away Team Fair Odds'], 4)
            csv_df.loc[index, 'Home Team Fair Odds'] = round(csv_df.loc[index, 'Home Team Fair Odds'], 4)

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
    consolidated_csv_file = "nfl_schedule_circa.csv"
    consolidated_df.to_csv(consolidated_csv_file, index=False)    
    collect_schedule_travel_ranking_data_nfl_schedule_df = consolidated_df
    
    return collect_schedule_travel_ranking_data_nfl_schedule_df

def get_predicted_pick_percentages_circa(pd):
    # Load your historical data (replace 'historical_pick_data_FV_circa.csv' with your actual file path)
    if selected_contest == 'Circa':
        df = pd.read_csv('Circa_historical_data.csv')
    else:
        df = pd.read_csv('historical_pick_data_FV.csv')
    df.rename(columns={"Week": "Date"}, inplace=True)
    # Remove percentage sign and convert to float
    #df['Win %'] = df['Win %'].str.rstrip('%').astype(float) / 100
    #df['Pick %'] = df['Pick %'].str.rstrip('%').astype(float) / 100
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
    
    df = collect_schedule_travel_ranking_data_df

    # Create a new DataFrame with selected columns
    selected_columns = ['Week', 'Away Team', 'Home Team', 'Away Team Fair Odds',
                        'Home Team Fair Odds', 'Away Team Star Rating', 'Home Team Star Rating', 'Divisional Matchup Boolean', 'Away Team Thanksgiving Favorite', 'Home Team Thanksgiving Favorite', 'Away Team Christmas Favorite', 'Home Team Christmas Favorite']
    new_df = df[selected_columns]

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
    away_df['Year'] = 2024
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
    home_df['Year'] = 2024
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
        lambda row: row["Pick %"] / 4 if row["Home Team Thanksgiving Favorite"] else row["Pick %"],
        axis=1
    )

    pick_predictions_df["Pick %"] = pick_predictions_df.apply(
        lambda row: row["Pick %"] / 4 if row["Away Team Thanksgiving Favorite"] else row["Pick %"],
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
    #nfl_schedule_df['Week'] = nfl_schedule_df['Week'].str.extract(r'(\d+)').astype(int)
    # Merge the DataFrames based on matching columns
    nfl_schedule_df = pd.merge(nfl_schedule_df, away_df, 
                               left_on=['Week', 'Away Team', 'Home Team'],
                               right_on=['Week', 'Away Team', 'Home Team'],
                               how='left')
    nfl_schedule_df = pd.merge(nfl_schedule_df, home_df, 
                               left_on=['Week', 'Away Team', 'Home Team'],
                               right_on=['Week', 'Away Team', 'Home Team'],
                               how='left')

    #print(nfl_schedule_circa_df)

    # Add 'Home Team EV' and 'Away Team EV' columns to nfl_schedule_circa_df
    nfl_schedule_df['Home Team EV'] = 0.0  # Initialize with 0.0
    nfl_schedule_df['Away Team EV'] = 0.0  # Initialize with 0.0


    nfl_schedule_df.to_csv("Circa_Predicted_Pick_%.csv", index=False)
    return nfl_schedule_df


def calculate_ev():
    def calculate_all_scenarios(week_df):
        """
        Calculates the expected value (EV) for each team in a given week,
        considering all possible game outcomes. EV is calculated as the 
        inverse of the survival probability of all teams in that scenario.

        Args:
            week_df: DataFrame for a single week, including 'Home Team', 
                     'Away Team', 'Home Team Fair Odds', 'Away Team Fair Odds', 
                     'Home Pick %', and 'Away Pick %' columns.

        Returns:
            A DataFrame with EVs for all scenarios, all outcomes, and scenario weights.
        """

        def generate_outcomes(games):
            if not games:
                return [[]]
            else:
                outcomes = []
                for outcome in ['Home Win', 'Away Win']:
                    new_games = games[1:]
                    new_outcomes = generate_outcomes(new_games)
                    for new_outcome in new_outcomes:
                        outcomes.append([outcome] + new_outcome)
                return outcomes

        all_outcomes = generate_outcomes(list(range(len(week_df))))
        ev_df = pd.DataFrame(columns=week_df['Home Team'].tolist() + week_df['Away Team'].tolist())

        scenario_weights = []  # Calculate scenario weights directly
        st.write("Full Season Progress")
        weekly_progress_bar = st.progress(0)
        total_scenarios = len(all_outcomes)
        for i, outcome in enumerate(tqdm(all_outcomes, desc="Calculating Scenarios", leave=False)):  
            scenario_ev = {team: 0 for team in week_df['Home Team'].unique().tolist() + week_df['Away Team'].unique().tolist()} 
            surviving_entries = 0
            scenario_weight = 1.0  # Calculate weight for the current scenario

            # Calculate surviving entries for ALL teams in the scenario
            for j, game_outcome in enumerate(outcome):
                if game_outcome == 'Home Win':
                    winning_team = week_df.iloc[j]['Home Team']
                    surviving_entries += week_df.iloc[j]['Home Pick %']
                    scenario_weight *= week_df.iloc[j]['Home Team Fair Odds'] 
                else:
                    winning_team = week_df.iloc[j]['Away Team']
                    surviving_entries += week_df.iloc[j]['Away Pick %']
                    scenario_weight *= week_df.iloc[j]['Away Team Fair Odds']

            # Calculate EV for EACH team in the scenario
            for j, game_outcome in enumerate(outcome):
                if game_outcome == 'Home Win':
                    winning_team = week_df.iloc[j]['Home Team']
                    if surviving_entries > 0:
                        scenario_ev[winning_team] = 1 / surviving_entries
                else:
                    winning_team = week_df.iloc[j]['Away Team']
                    if surviving_entries > 0:
                        scenario_ev[winning_team] = 1 / surviving_entries


            for team, ev in scenario_ev.items():
                ev_df.loc[i, team] = ev

            scenario_weights.append(scenario_weight) # Append weight for the scenario

            # --- Option 2: Update progress bar ---
            progress_percent = int((i / total_scenarios) * 100)
            progress_bar.progress(progress_percent)

        # Calculate weighted average EV
        weighted_avg_ev = {}
        for team in ev_df.columns:
            weighted_evs_for_team = ev_df[team] * scenario_weights 
            weighted_avg_ev[team] = sum(weighted_evs_for_team) / sum(scenario_weights)

        # Update week_df with weighted average EVs using .loc
        for i in range(len(week_df)):
            week = week_df.iloc[i]['Week']
            home_team = week_df.iloc[i]['Home Team']
            away_team = week_df.iloc[i]['Away Team']

            # Find the weighted average EV for the home team
            if home_team in weighted_avg_ev:
                 week_df.loc[(week_df['Week'] == week) & (week_df['Home Team'] == home_team), 'Home Team EV'] = weighted_avg_ev[home_team]

            # Find the weighted average EV for the away team
            if away_team in weighted_avg_ev:
                 week_df.loc[(week_df['Week'] == week) & (week_df['Away Team'] == away_team), 'Away Team EV'] = weighted_avg_ev[away_team]

        # Return updated week_df and other values
        return week_df, all_outcomes, scenario_weights 

    # Add "Week" to the beginning of each value in the 'Week' column
    nfl_schedule_pick_percentages_df['Week'] = nfl_schedule_pick_percentages_df['Week'].apply(lambda x: f"Week {x}")
    
    # --- Option 1: Using st.empty for text updates ---
    #progress_bar = st.empty()  # Create an empty placeholder

    # --- Option 2: Using st.progress for a bar ---
    st.write("Current Week Progress")
    progress_bar = st.progress(0)  # Initialize progress bar at 0%

    for week in tqdm(range(starting_week, ending_week), desc="Processing Weeks", leave=False): #########SET THE RANGE TO (1, 21) TO PROCESS THE WHOLE SEASON, or (2,3) to process ONLY WEEK . The rest you can figure out 
        week_df = nfl_schedule_pick_percentages_df[nfl_schedule_pick_percentages_df['Week'] == week]
        week_df, all_outcomes, scenario_weights = calculate_all_scenarios(week_df)

        # Update nfl_schedule_circa_df_2 using the 'update' method
        nfl_schedule_pick_percentages_df.update(week_df[['Home Team EV', 'Away Team EV']])

        # --- Option 1: Update progress text ---
        #progress_bar.write(f"Processing Week: {week}/{total_weeks}")

        # --- Option 2: Update progress bar ---
        progress_percent = int((week / ending_week) * 100)
        progress_bar.progress(progress_percent)
    if selected_contest == 'Circa':
        nfl_schedule_pick_percentages_df.to_csv("NFL Schedule with full ev_circa.csv", index=False)
    else:
        nfl_schedule_pick_percentages_df.to_csv("NFL Schedule with full ev_DraftKings.csv", index=False)
    return nfl_schedule_pick_percentages_df

def get_survivor_picks_based_on_ev():
    # Loop through 100 iterations
    for iteration in range(number_solutions):
        df = full_df_with_ev
		

        #Number of weeks that have already been played
        #weeks_completed = starting_week -1

        # Teams already picked - Team name in quotes and separated by commas

        # Filter out weeks that have already been played and reset index
        st.write(df)
        df = df[(df['Week'] >= starting_week) & (df['Week'] <= ending_week)].reset_index(drop=True)

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
        for week in df['Week'].unique():
            # One team per week
            solver.Add(solver.Sum([picks[i] for i in range(len(df)) if df.loc[i, 'Week'] == week]) == 1)

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


            
            if df.loc[i, 'Adjusted Current Winner'] == 'Arizona Cardinals' and df.loc[i, 'Week'] in az_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Atlanta Falcons' and df.loc[i, 'Week'] in atl_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Baltimore Ravens' and df.loc[i, 'Week'] in bal_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Buffalo Bills' and df.loc[i, 'Week'] in buf_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Carolina Panthers' and df.loc[i, 'Week'] in car_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Chicago Bears' and df.loc[i, 'Week'] in chi_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Cincinnati Bengals' and df.loc[i, 'Week'] in cin_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Cleveland Browns' and df.loc[i, 'Week'] in cle_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Dallas Cowboys' and df.loc[i, 'Week'] in dal_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Denver Broncos' and df.loc[i, 'Week'] in den_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Detroit Lions' and df.loc[i, 'Week'] in det_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'AGreen Bay Packers' and df.loc[i, 'Week'] in gb_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Houston Texans' and df.loc[i, 'Week'] in hou_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Indianapolis Colts' and df.loc[i, 'Week'] in ind_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Jacksonville Jaguars' and df.loc[i, 'Week'] in jax_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Kansas City Chiefs' and df.loc[i, 'Week'] in kc_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Las vegas Raiders' and df.loc[i, 'Week'] in lv_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Chargers' and df.loc[i, 'Week'] in lac_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Rams' and df.loc[i, 'Week'] in lar_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Miami Dolphins' and df.loc[i, 'Week'] in mia_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Minnesota Vikings' and df.loc[i, 'Week'] in min_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'New England Patriots' and df.loc[i, 'Week'] in ne_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'New Orleans Saints' and df.loc[i, 'Week'] in no_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'New York Giants' and df.loc[i, 'Week'] in nyg_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'New York Jets' and df.loc[i, 'Week'] in nyj_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Philadelphia Eagles' and df.loc[i, 'Week'] in phi_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Pittsburgh Steelers' and df.loc[i, 'Week'] in pit_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Seattle Seahawks' and df.loc[i, 'Week'] in sea_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Tampa Bay Buccaneers' and df.loc[i, 'Week'] in tb_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Tennessee Titans' and df.loc[i, 'Week'] in ten_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Washington Commanders' and df.loc[i, 'Week'] in was_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'San Francisco 49ers' and df.loc[i, 'Week'] in sf_excluded_weeks:
                    solver.Add(picks[i] == 0)


            if df.loc[i, 'Adjusted Current Winner'] in picked_teams:
                solver.Add(picks[i] == 0)
        # Dynamically create the forbidden solution list
        forbidden_solutions_1 = []
        if iteration > 0: 
            for previous_iteration in range(iteration):
                # Load the picks from the previous iteration
                previous_picks_df = pd.read_csv(f"picks_ev_{previous_iteration + 1}.csv")

                # Extract the forbidden solution for this iteration
                forbidden_solution_1 = previous_picks_df['Adjusted Current Winner'].tolist()
                forbidden_solutions_1.append(forbidden_solution_1)

        # Add constraints for all forbidden solutions
        for forbidden_solution_1 in forbidden_solutions_1:
            # Get the indices of the forbidden solution in the DataFrame
            forbidden_indices_1 = []
            for i in range(len(df)):
                # Calculate the relative week number within the forbidden solution
                df_week = df.loc[i, 'Week']
                relative_week = df_week - starting_week  # Adjust week to be relative to starting week

                #Check if the week is within the range and the solution is forbidden
                if 0 <= relative_week < len(forbidden_solution_1) and df_week >= starting_week and df_week <= ending_week: #Added this to make sure we are only looking at the range
                    if (df.loc[i, 'Adjusted Current Winner'] == forbidden_solution_1[relative_week]):
                        forbidden_indices_1.append(i)

            # Add the constraint
            solver.Add(solver.Sum([1 - picks[i] for i in forbidden_indices_1]) >= 1)


        # Add the constraint for San Francisco 49ers in week 11
        if sf_req_week > 0:        
            for i in range(len(df)):
                if (df.loc[i, 'Home Team'] == 'San Francisco 49ers' or df.loc[i, 'Away Team'] == 'San Francisco 49ers') and df.loc[i, 'Week'] == sf_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Arizona Cardinals' or df.loc[i, 'Away Team'] == 'Arizona Cardinals') and df.loc[i, 'Week'] == az_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Atlanta Falcons' or df.loc[i, 'Away Team'] == 'Atlanta Falcons') and df.loc[i, 'Week'] == atl_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Baltimore Ravens' or df.loc[i, 'Away Team'] == 'Baltimore Ravens') and df.loc[i, 'Week'] == bal_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Buffalo Bills' or df.loc[i, 'Away Team'] == 'Buffalo Bills') and df.loc[i, 'Week'] == buf_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Carolina Panthers' or df.loc[i, 'Away Team'] == 'Carolina Panthers') and df.loc[i, 'Week'] == car_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Chicago Bears' or df.loc[i, 'Away Team'] == 'Chicago Bears') and df.loc[i, 'Week'] == chi_req_week:
                    if df.loc[i, 'Adjusted Current Winner'] == 'Chicago Bears':
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Home Team'] == 'Chicago Bears':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Away Team'] == 'Chicago Bears':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                        solver.Add(picks[i] == 1)
        if cin_req_week > 0:        
                if (df.loc[i, 'Home Team'] == 'Cincinnati Bengals' or df.loc[i, 'Away Team'] == 'Cincinnati Bengals') and df.loc[i, 'Week'] == cin_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Cleveland Browns' or df.loc[i, 'Away Team'] == 'Cleveland Browns') and df.loc[i, 'Week'] == cle_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Dallas Cowboys' or df.loc[i, 'Away Team'] == 'Dallas Cowboys') and df.loc[i, 'Week'] == dal_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Denver Broncos' or df.loc[i, 'Away Team'] == 'Denver Broncos') and df.loc[i, 'Week'] == den_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Detroit Lions' or df.loc[i, 'Away Team'] == 'Detroit Lions') and df.loc[i, 'Week'] == det_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Green Bay Packers' or df.loc[i, 'Away Team'] == 'Green Bay Packers') and df.loc[i, 'Week'] == gb_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Houston Texans' or df.loc[i, 'Away Team'] == 'Houston Texans') and df.loc[i, 'Week'] == hou_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Indianapolis Colts' or df.loc[i, 'Away Team'] == 'Indianapolis Colts') and df.loc[i, 'Week'] == ind_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Jacksonville Jaguars' or df.loc[i, 'Away Team'] == 'Jacksonville Jaguars') and df.loc[i, 'Week'] == jax_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Kansas City Chiefs' or df.loc[i, 'Away Team'] == 'Kansas City Chiefs') and df.loc[i, 'Week'] == kc_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Las Vegas Raiders' or df.loc[i, 'Away Team'] == 'Las Vegas Raiders') and df.loc[i, 'Week'] == lv_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Los Angeles Chargers' or df.loc[i, 'Away Team'] == 'Los Angeles Chargers') and df.loc[i, 'Week'] == lac_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Los Angeles Rams' or df.loc[i, 'Away Team'] == 'Los Angeles Rams') and df.loc[i, 'Week'] == lar_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Miami Dolphins' or df.loc[i, 'Away Team'] == 'Miami Dolphins') and df.loc[i, 'Week'] == mia_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Minnesota Vikings' or df.loc[i, 'Away Team'] == 'Minnesota Vikings') and df.loc[i, 'Week'] == min_req_week:
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
                if (df.loc[i, 'Home Team'] == 'New England Patriots' or df.loc[i, 'Away Team'] == 'New England Patriots') and df.loc[i, 'Week'] == ne_req_week:
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
                if (df.loc[i, 'Home Team'] == 'New Orleans Saints' or df.loc[i, 'Away Team'] == 'New Orleans Saints') and df.loc[i, 'Week'] == no_req_week:
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
                if (df.loc[i, 'Home Team'] == 'New York Giants' or df.loc[i, 'Away Team'] == 'New York Giants') and df.loc[i, 'Week'] == nyg_req_week:
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
                if (df.loc[i, 'Home Team'] == 'New York Jets' or df.loc[i, 'Away Team'] == 'New York Jets') and df.loc[i, 'Week'] == nyj_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Philadelphia Eagles' or df.loc[i, 'Away Team'] == 'Philadelphia Eagles') and df.loc[i, 'Week'] == phi_req_week:
                    if df.loc[i, 'Adjusted Current Winner'] == 'Philadelphia Eagles':
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Home Team'] == 'Philadelphia Eagles':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Away Team'] == 'Philadelphia Eagles':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                        solver.Add(picks[i] == 1)
        if pit_req_week > 0:        
                if (df.loc[i, 'Home Team'] == 'Pittsburgh Steelers' or df.loc[i, 'Away Team'] == 'Pittsburgh Steelers') and df.loc[i, 'Week'] == pit_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Seattle Seahawks' or df.loc[i, 'Away Team'] == 'Seattle Seahawks') and df.loc[i, 'Week'] == sea_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Tampa Bay Buccaneers' or df.loc[i, 'Away Team'] == 'Tampa Bay Buccaneers') and df.loc[i, 'Week'] == tb_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Tennessee Titans' or df.loc[i, 'Away Team'] == 'Tennessee Titans') and df.loc[i, 'Week'] == ten_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Washington Commanders' or df.loc[i, 'Away Team'] == 'Washington Commanders') and df.loc[i, 'Week'] == was_req_week:
                    if df.loc[i, 'Adjusted Current Winner'] == 'Washington Commanders':
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Home Team'] == 'Washington Commanders':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Away Team'] == 'Washington Commanders':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                        solver.Add(picks[i] == 1)

        # Objective: maximize the sum of Adjusted Current Difference of each game picked
        solver.Maximize(solver.Sum([picks[i] * (df.loc[i, 'Home Team EV'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team EV']) for i in range(len(df))]))

        # Solve the problem and print the solution
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            st.write(f'**Solution Based on EV: {iteration + 1}**')
            st.write('')
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
                    divisional_game = 'Divisional' if df.loc[i, 'Divisional Matchup Boolean'] == '1' else ''
                    home_team = '(Home Team)' if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else '(Away Team)'
                    weekly_rest = df.loc[i, 'Home Team Weekly Rest'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Weekly Rest']
                    weekly_rest_advantage = df.loc[i, 'Weekly Home Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Weekly Away Rest Advantage']
                    cumulative_rest = df.loc[i, 'Home Cumulative Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Cumulative Rest Advantage']
                    cumulative_rest_advantage = df.loc[i, 'Home Team Current Week Cumulative Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Current Week Cumulative Rest Advantage']
                    travel_advantage = df.loc[i, 'Home Travel Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Travel Advantage']
                    back_to_back_away_games = 'True' if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Away Team'] and df.loc[i, 'Back to Back Away Games'] == 'True' else 'False'
                    thursday_night_game = 'Thursday Night Game' if df.loc[i, "Thursday Night Game"] == 'True' else 'Sunday/Monday Game'
                    international_game = 'International Game' if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else 'Domestic Game'
                    previous_opponent = df.loc[i, 'Home Team Previous Opponent'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Previous Opponent']
                    previous_game_location = df.loc[i, 'Home Team Previous Location'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Previous Location']
                    next_opponent = df.loc[i, 'Home Team Next Opponent'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Next Opponent']
                    next_game_location = df.loc[i, 'Home Team Next Location'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Next Location']
                    win_odds = df.loc[i, 'Home Team Fair Odds'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Away Team'] else df.loc[i, 'Away Team Fair Odds']
                    

                    # Get differences
                    preseason_difference = df.loc[i, 'Preseason Difference']
                    adjusted_preseason_difference = df.loc[i, 'Adjusted Preseason Difference']
                    current_difference = df.loc[i, 'Current Difference']
                    adjusted_current_difference = df.loc[i, 'Adjusted Current Difference']
                    # Calculate EV for this game
                    ev = (df.loc[i, 'Home Team EV'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team EV'])


                    print('Week %i: Pick %s %s %s (%i, %i, %i, %i, %.4f)' % (df.loc[i, 'Week'], df.loc[i, 'Adjusted Current Winner'], divisional_game, home_team,
                                                                       preseason_difference, adjusted_preseason_difference,
                                                                       current_difference, adjusted_current_difference, ev))
                    new_row_2 = {
                        'Week': week,
                        'Pick': pick,
                        'Opponent': opponent,
                        'EV': ev,
                        'Win Odds': win_odds,
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
            st.write(f'Total EV: :red[{sum_ev}]')
        else:
            st.write('No solution found. Consider using fewer constraints. Or you may just be fucked')
            st.write('No solution found. Consider using fewer constraints. Or you may just be fucked')
        st.write("---------------------------------------------------------------------------------------------------------------")
        st.write("")
        st.write("")	    
        st.write("")

            # Save the picks to a CSV file for the current iteration
        picks_df.to_csv(f'picks_ev_{iteration + 1}.csv', index=False)
        summarized_picks_df.to_csv(f'picks_ev_subset_{iteration + 1}.csv', index=False)
        
        # Append the new forbidden solution to the list
        forbidden_solutions_1.append(picks_df['Adjusted Current Winner'].tolist())
        #print(forbidden_solutions)


def get_survivor_picks_based_on_internal_rankings():
    # Loop through 100 iterations
    for iteration in range(number_solutions):
        df = full_df_with_ev

        #Number of weeks that have already been played
        #weeks_completed = 20 - starting_week

        # Teams already picked - Team name in quotes and separated by commas

        # Filter out weeks that have already been played and reset index
        df = df[(df['Week'] >= starting_week) & (df['Week'] < ending_week)].reset_index(drop=True)

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
        for week in df['Week'].unique():
            # One team per week
            solver.Add(solver.Sum([picks[i] for i in range(len(df)) if df.loc[i, 'Week'] == week]) == 1)

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


            
            if df.loc[i, 'Adjusted Current Winner'] == 'Arizona Cardinals' and df.loc[i, 'Week'] in az_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Atlanta Falcons' and df.loc[i, 'Week'] in atl_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Baltimore Ravens' and df.loc[i, 'Week'] in bal_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Buffalo Bills' and df.loc[i, 'Week'] in buf_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Carolina Panthers' and df.loc[i, 'Week'] in car_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Chicago Bears' and df.loc[i, 'Week'] in chi_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Cincinnati Bengals' and df.loc[i, 'Week'] in cin_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Cleveland Browns' and df.loc[i, 'Week'] in cle_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Dallas Cowboys' and df.loc[i, 'Week'] in dal_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Denver Broncos' and df.loc[i, 'Week'] in den_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Detroit Lions' and df.loc[i, 'Week'] in det_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'AGreen Bay Packers' and df.loc[i, 'Week'] in gb_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Houston Texans' and df.loc[i, 'Week'] in hou_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Indianapolis Colts' and df.loc[i, 'Week'] in ind_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Jacksonville Jaguars' and df.loc[i, 'Week'] in jax_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Kansas City Chiefs' and df.loc[i, 'Week'] in kc_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Las vegas Raiders' and df.loc[i, 'Week'] in lv_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Chargers' and df.loc[i, 'Week'] in lac_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Los Angeles Rams' and df.loc[i, 'Week'] in lar_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Miami Dolphins' and df.loc[i, 'Week'] in mia_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Minnesota Vikings' and df.loc[i, 'Week'] in min_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'New England Patriots' and df.loc[i, 'Week'] in ne_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'New Orleans Saints' and df.loc[i, 'Week'] in no_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'New York Giants' and df.loc[i, 'Week'] in nyg_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'New York Jets' and df.loc[i, 'Week'] in nyj_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Philadelphia Eagles' and df.loc[i, 'Week'] in phi_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Pittsburgh Steelers' and df.loc[i, 'Week'] in pit_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Seattle Seahawks' and df.loc[i, 'Week'] in sea_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Tampa Bay Buccaneers' and df.loc[i, 'Week'] in tb_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Tennessee Titans' and df.loc[i, 'Week'] in ten_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'Washington Commanders' and df.loc[i, 'Week'] in was_excluded_weeks:
                    solver.Add(picks[i] == 0)
            if df.loc[i, 'Adjusted Current Winner'] == 'San Francisco 49ers' and df.loc[i, 'Week'] in sf_excluded_weeks:
                    solver.Add(picks[i] == 0)


            if df.loc[i, 'Adjusted Current Winner'] in picked_teams:
                solver.Add(picks[i] == 0)
        # Dynamically create the forbidden solution list
        forbidden_solutions_1 = []
        if iteration > 0: 
            for previous_iteration in range(iteration):
                # Load the picks from the previous iteration
                previous_picks_df = pd.read_csv(f"picks_ev_{previous_iteration + 1}.csv")

                # Extract the forbidden solution for this iteration
                forbidden_solution_1 = previous_picks_df['Adjusted Current Winner'].tolist()
                forbidden_solutions_1.append(forbidden_solution_1)

        # Add constraints for all forbidden solutions
        for forbidden_solution_1 in forbidden_solutions_1:
            # Get the indices of the forbidden solution in the DataFrame
            forbidden_indices_1 = []
            for i in range(len(df)):
                week_index = df.loc[i, 'Week'] - (weeks_completed + 1)
                if week_index >= 0 and week_index < len(forbidden_solution_1):
                    if (df.loc[i, 'Adjusted Current Winner'] == forbidden_solution_1[week_index]):
                        forbidden_indices_1.append(i)

            # Add the constraint that at least one of these picks should not be selected
            solver.Add(solver.Sum([1 - picks[i] for i in forbidden_indices_1]) >= 1)


        # Add the constraint for San Francisco 49ers in week 11
        if sf_req_week > 0:        
            for i in range(len(df)):
                if (df.loc[i, 'Home Team'] == 'San Francisco 49ers' or df.loc[i, 'Away Team'] == 'San Francisco 49ers') and df.loc[i, 'Week'] == sf_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Arizona Cardinals' or df.loc[i, 'Away Team'] == 'Arizona Cardinals') and df.loc[i, 'Week'] == az_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Atlanta Falcons' or df.loc[i, 'Away Team'] == 'Atlanta Falcons') and df.loc[i, 'Week'] == atl_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Baltimore Ravens' or df.loc[i, 'Away Team'] == 'Baltimore Ravens') and df.loc[i, 'Week'] == bal_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Buffalo Bills' or df.loc[i, 'Away Team'] == 'Buffalo Bills') and df.loc[i, 'Week'] == buf_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Carolina Panthers' or df.loc[i, 'Away Team'] == 'Carolina Panthers') and df.loc[i, 'Week'] == car_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Chicago Bears' or df.loc[i, 'Away Team'] == 'Chicago Bears') and df.loc[i, 'Week'] == chi_req_week:
                    if df.loc[i, 'Adjusted Current Winner'] == 'Chicago Bears':
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Home Team'] == 'Chicago Bears':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Away Team'] == 'Chicago Bears':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                        solver.Add(picks[i] == 1)
        if cin_req_week > 0:        
                if (df.loc[i, 'Home Team'] == 'Cincinnati Bengals' or df.loc[i, 'Away Team'] == 'Cincinnati Bengals') and df.loc[i, 'Week'] == cin_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Cleveland Browns' or df.loc[i, 'Away Team'] == 'Cleveland Browns') and df.loc[i, 'Week'] == cle_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Dallas Cowboys' or df.loc[i, 'Away Team'] == 'Dallas Cowboys') and df.loc[i, 'Week'] == dal_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Denver Broncos' or df.loc[i, 'Away Team'] == 'Denver Broncos') and df.loc[i, 'Week'] == den_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Detroit Lions' or df.loc[i, 'Away Team'] == 'Detroit Lions') and df.loc[i, 'Week'] == det_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Green Bay Packers' or df.loc[i, 'Away Team'] == 'Green Bay Packers') and df.loc[i, 'Week'] == gb_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Houston Texans' or df.loc[i, 'Away Team'] == 'Houston Texans') and df.loc[i, 'Week'] == hou_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Indianapolis Colts' or df.loc[i, 'Away Team'] == 'Indianapolis Colts') and df.loc[i, 'Week'] == ind_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Jacksonville Jaguars' or df.loc[i, 'Away Team'] == 'Jacksonville Jaguars') and df.loc[i, 'Week'] == jax_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Kansas City Chiefs' or df.loc[i, 'Away Team'] == 'Kansas City Chiefs') and df.loc[i, 'Week'] == kc_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Las Vegas Raiders' or df.loc[i, 'Away Team'] == 'Las Vegas Raiders') and df.loc[i, 'Week'] == lv_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Los Angeles Chargers' or df.loc[i, 'Away Team'] == 'Los Angeles Chargers') and df.loc[i, 'Week'] == lac_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Los Angeles Rams' or df.loc[i, 'Away Team'] == 'Los Angeles Rams') and df.loc[i, 'Week'] == lar_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Miami Dolphins' or df.loc[i, 'Away Team'] == 'Miami Dolphins') and df.loc[i, 'Week'] == mia_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Minnesota Vikings' or df.loc[i, 'Away Team'] == 'Minnesota Vikings') and df.loc[i, 'Week'] == min_req_week:
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
                if (df.loc[i, 'Home Team'] == 'New England Patriots' or df.loc[i, 'Away Team'] == 'New England Patriots') and df.loc[i, 'Week'] == ne_req_week:
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
                if (df.loc[i, 'Home Team'] == 'New Orleans Saints' or df.loc[i, 'Away Team'] == 'New Orleans Saints') and df.loc[i, 'Week'] == no_req_week:
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
                if (df.loc[i, 'Home Team'] == 'New York Giants' or df.loc[i, 'Away Team'] == 'New York Giants') and df.loc[i, 'Week'] == nyg_req_week:
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
                if (df.loc[i, 'Home Team'] == 'New York Jets' or df.loc[i, 'Away Team'] == 'New York Jets') and df.loc[i, 'Week'] == nyj_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Philadelphia Eagles' or df.loc[i, 'Away Team'] == 'Philadelphia Eagles') and df.loc[i, 'Week'] == phi_req_week:
                    if df.loc[i, 'Adjusted Current Winner'] == 'Philadelphia Eagles':
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Home Team'] == 'Philadelphia Eagles':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Away Team'] == 'Philadelphia Eagles':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                        solver.Add(picks[i] == 1)
        if pit_req_week > 0:        
                if (df.loc[i, 'Home Team'] == 'Pittsburgh Steelers' or df.loc[i, 'Away Team'] == 'Pittsburgh Steelers') and df.loc[i, 'Week'] == pit_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Seattle Seahawks' or df.loc[i, 'Away Team'] == 'Seattle Seahawks') and df.loc[i, 'Week'] == sea_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Tampa Bay Buccaneers' or df.loc[i, 'Away Team'] == 'Tampa Bay Buccaneers') and df.loc[i, 'Week'] == tb_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Tennessee Titans' or df.loc[i, 'Away Team'] == 'Tennessee Titans') and df.loc[i, 'Week'] == ten_req_week:
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
                if (df.loc[i, 'Home Team'] == 'Washington Commanders' or df.loc[i, 'Away Team'] == 'Washington Commanders') and df.loc[i, 'Week'] == was_req_week:
                    if df.loc[i, 'Adjusted Current Winner'] == 'Washington Commanders':
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Home Team'] == 'Washington Commanders':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Home Team']
                        solver.Add(picks[i] == 1)
                    elif df.loc[i, 'Away Team'] == 'Washington Commanders':
                        df.loc[i, 'Adjusted Current Winner'] = df.loc[i, 'Away Team']
                        solver.Add(picks[i] == 1)

        # Objective: maximize the sum of Adjusted Current Difference of each game picked
        solver.Maximize(solver.Sum([picks[i] * df.loc[i, 'Adjusted Current Difference'] for i in range(len(df))]))

        # Solve the problem and print the solution
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
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
                    divisional_game = 'Divisional' if df.loc[i, 'Divisional Matchup Boolean'] == '1' else ''
                    home_team = '(Home Team)' if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else '(Away Team)'
                    weekly_rest = df.loc[i, 'Home Team Weekly Rest'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Weekly Rest']
                    weekly_rest_advantage = df.loc[i, 'Weekly Home Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Weekly Away Rest Advantage']
                    cumulative_rest = df.loc[i, 'Home Cumulative Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Cumulative Rest Advantage']
                    cumulative_rest_advantage = df.loc[i, 'Home Team Current Week Cumulative Rest Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Current Week Cumulative Rest Advantage']
                    travel_advantage = df.loc[i, 'Home Travel Advantage'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Travel Advantage']
                    back_to_back_away_games = 'True' if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Away Team'] and df.loc[i, 'Back to Back Away Games'] == 'True' else 'False'
                    thursday_night_game = 'Thursday Night Game' if df.loc[i, "Thursday Night Game"] == 'True' else 'Sunday/Monday Game'
                    international_game = 'International Game' if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else 'Domestic Game'
                    previous_opponent = df.loc[i, 'Home Team Previous Opponent'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Previous Opponent']
                    previous_game_location = df.loc[i, 'Home Team Previous Location'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Previous Location']
                    next_opponent = df.loc[i, 'Home Team Next Opponent'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Next Opponent']
                    next_game_location = df.loc[i, 'Home Team Next Location'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team Next Location']
                    win_odds = df.loc[i, 'Home Team Fair Odds'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Away Team'] else df.loc[i, 'Away Team Fair Odds']
                    

                    # Get differences
                    preseason_difference = df.loc[i, 'Preseason Difference']
                    adjusted_preseason_difference = df.loc[i, 'Adjusted Preseason Difference']
                    current_difference = df.loc[i, 'Current Difference']
                    adjusted_current_difference = df.loc[i, 'Adjusted Current Difference']
                    # Calculate EV for this game
                    ev = (df.loc[i, 'Home Team EV'] if df.loc[i, 'Adjusted Current Winner'] == df.loc[i, 'Home Team'] else df.loc[i, 'Away Team EV'])


                    print('Week %i: Pick %s %s %s (%i, %i, %i, %i, %.4f)' % (df.loc[i, 'Week'], df.loc[i, 'Adjusted Current Winner'], divisional_game, home_team,
                                                                       preseason_difference, adjusted_preseason_difference,
                                                                       current_difference, adjusted_current_difference, ev))
                    new_row_2 = {
                        'Week': week,
                        'Pick': pick,
                        'Opponent': opponent,
                        'EV': ev,
                        'Win Odds': win_odds,
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
   





            # Add row to picks_df
            #picks_df = pd.concat([picks_df, df.loc[[i]]], ignore_index=True)
            #print(picks_df)
            # Print sums

            st.write(f'**Solution Based on EV: {iteration}**')

            st.write(summarized_picks_df)
            st.write('')
            st.write('\nPreseason Difference:', sum_preseason_difference)
            st.write('Adjusted Preseason Difference:', sum_adjusted_preseason_difference)
            st.write('Current Difference:', sum_current_difference)
            st.write('Adjusted Current Difference:', sum_adjusted_current_difference)
            st.write(f'Total EV: :red[{sum_ev}]')
        else:
            st.write('No solution found. Consider using fewer constraints. Or you may just be fucked')
        st.write("---------------------------------------------------------------------------------------------------------------")
        st.write("")
        st.write("")	    
        st.write("")
            # Save the picks to a CSV file for the current iteration
        picks_df.to_csv(f'picks_ev_{iteration + 1}.csv', index=False)
        summarized_picks_df.to_csv(f'picks_ev_subset_{iteration + 1}.csv', index=False)
        
        # Append the new forbidden solution to the list
        forbidden_solutions_1.append(picks_df['Adjusted Current Winner'].tolist())
        #print(forbidden_solutions)

picked_teams = []

default_az_rank = -.5
default_atl_rank = .5
default_bal_rank = 9
default_buf_rank = 8
default_car_rank = -6
default_chi_rank = -5.5
default_cin_rank = 2.5
default_cle_rank = -9.5
default_dal_rank = -5
default_den_rank = -.5
default_det_rank = 5.5
default_gb_rank = 5.5
default_hou_rank = 0
default_ind_rank = -3.5
default_jax_rank = -7
default_kc_rank = 5.5
default_lv_rank = -6
default_lac_rank = 2.5
default_lar_rank = 1.5
default_mia_rank = 2
default_min_rank = 5
default_ne_rank = -5
default_no_rank = -8.5
default_nyg_rank = -7
default_nyj_rank = -4
default_phi_rank = 8.5
default_pit_rank = 0
default_sf_rank = -.5
default_sea_rank = 0
default_tb_rank = 2.5
default_ten_rank = -7
default_was_rank = 2

preseason_az_rank = -2
preseason_atl_rank = 1
preseason_bal_rank = 3.5
preseason_buf_rank = 3
preseason_car_rank = -5
preseason_chi_rank = .5
preseason_cin_rank = 2
preseason_cle_rank = -1
preseason_dal_rank = -2
preseason_den_rank = -3.5
preseason_det_rank = 3.5
preseason_gb_rank = 2
preseason_hou_rank = 2
preseason_ind_rank = -.5
preseason_jax_rank = .5
preseason_kc_rank = 5.5
preseason_lv_rank = -2.5
preseason_lac_rank = 0
preseason_lar_rank = .5
preseason_mia_rank = 1.5
preseason_min_rank = -2
preseason_ne_rank = -6
preseason_no_rank = -1
preseason_nyg_rank = -3.5
preseason_nyj_rank = -2
preseason_phi_rank = 4
preseason_pit_rank = -.5
preseason_sf_rank = 4.5
preseason_sea_rank = -.5
preseason_tb_rank = -1
preseason_ten_rank = -3
preseason_was_rank = -2.5


az_home_adv = 1.5
atl_home_adv = 2.3
bal_home_adv = 3.8
buf_home_adv = 3.6
car_home_adv = 1.9
chi_home_adv = 1.5
cin_home_adv = 2.1
cle_home_adv = 1.3
dal_home_adv = 3.7
den_home_adv = 2.6
det_home_adv = 2.1
gb_home_adv = 3.8
hou_home_adv = 1.9
ind_home_adv = 2.6
jax_home_adv = 1.4
kc_home_adv = 3.8
lv_home_adv = 1.4
lac_home_adv = 2.6
lar_home_adv = 2.6
mia_home_adv = 2.3
min_home_adv = 3.1
ne_home_adv = 3.9
no_home_adv = 3.1
nyg_home_adv = 1.1
nyj_home_adv = 1.2
phi_home_adv = 3.3
pit_home_adv = 3.5
sf_home_adv = 3.6
sea_home_adv = 2.6
tb_home_adv = 2.0
ten_home_adv = 2.1
was_home_adv = 1.3


az_away_adj = -.3
atl_away_adj = .2
bal_away_adj = -1.5
buf_away_adj = -1.1
car_away_adj = .5
chi_away_adj = 1
cin_away_adj = -.2
cle_away_adj = 1.5
dal_away_adj = -1.2
den_away_adj = .6
det_away_adj = .7
gb_away_adj = -.1
hou_away_adj = .7
ind_away_adj = -.2
jax_away_adj = 1.6
kc_away_adj = -1.6
lv_away_adj = -.3
lac_away_adj = -.8
lar_away_adj = 1.3
mia_away_adj = 1.4
min_away_adj = -.5
ne_away_adj = -1.8
no_away_adj = -1.6
nyg_away_adj = .9
nyj_away_adj = 1.9
phi_away_adj = -.2
pit_away_adj = -.2
sf_away_adj = -1.1
sea_away_adj = -.4
tb_away_adj = -.1
ten_away_adj = .4
was_away_adj = .6

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
number_solutions = 5
selected_contest = 'Circa'
starting_week = 1
if selected_contest == 'Circa':
	ending_week = 21
else:
	ending_week = 19


st.title("NFL Survivor Optimization")
st.subheader("The second best Circa Survivor Contest optimizer")
contest_options = [
    "Circa", "DraftKings"
]
st.write("Alright, clowns. This site is built to help you optimize your picks for the Circa Survivor contest (Eventually other contests). :red[This tool is just for informational use. It does not take into account injuries or certain other factors. Do not use this tool as your only source of information.] Simply input which week you're in, your team rankings, constraints, etc... and the algorithm will do the rest.")
st.write('Caluclating EV will take the longest in this process. For a full season, this step will take up to 5 hours or more. For that reason, we recommend using the saved Expected Value Calculations Good luck!')
st.write('')
st.write('')
st.subheader('Select Contest')
st.write('Choose the contest you are using this algorithm for: Circa (Advanced) or Draftkings (Traditional and Pathetic)')
st.write('The biggest differences between the two contests:')
st.write('- Circa has 20 Weeks (Christmas and Thanksgiving/Black Friday act as their own indivdual weeks)')
st.write('- Thanksgiving/Black Friday week will be Week 13 on this site (If you select Circa)')
st.write('- Christmas Week will be week 18 on this site (If you select Circa)')
st.write('- In Circa, a tie eliminates you, but in Draftkings, you move on with a tie')
st.write('- Players in Circa tend to be sharper, making it more difficult to play contrarian')
selected_contest = st.selectbox('Choose Contest:', options = contest_options)
if selected_contest == "DraftKings":
	ending_week = 19
else:
	ending_week = 21
st.write('')
st.write('')
st.write('')
st.write('')
st.subheader('Picked Teams:')
yes_i_have_picked_teams = st.checkbox('Have you already used any teams in the contest, or want to prevent the alogorithm from using any specific teams?')

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
    st.write('Select the teams that you have already used in the Survivor contest, or teams that you just do not want to pick in the enirety of the contest')
    selected_teams = st.multiselect("Prohibited Picks:", options=nfl_teams)
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
    st.text('Select the upcoming week for the starting week. Select the week you want the algorithm to stop at. If you select one week, Calculating EV can take up to 30-45 minutes (If you do not use the Saved EV calculations). All 20 weeks will take 5-6 hours. Ending Week must be greater than or equal to Starting Week.')
    if selected_contest == "DraftKings":
        starting_week = st.selectbox("Select Starting Week:", options=range(1, 19))
    else:
        st.write(":red[Week 13 is Thanksgiving/Black Friday Week and Week 18 is Christmas Week]")
        starting_week = st.selectbox("Select Starting Week:", options=range(1, 21))
    #if starting_week:
        #st.write(f"Selected Starting Week: {starting_week}")
    # Create a dynamic range for ending week based on starting week
    st.write('')
    if selected_contest == "DraftKings":
        ending_week_options = range(starting_week, 19)
    else:
        ending_week_options = range(starting_week, 21)
    ending_week = st.selectbox("Select Ending Week:", options=ending_week_options)
    ending_week = ending_week + 1
#return starting_week, ending_week, picked_teams
    #if ending_week:
        #st.write(f"Selected Ending Week: {ending_week}")
#if __name__ == "__main__":
#    starting_week, ending_week, picked_teams = create_nfl_app()
st.write('')
st.write('')
st.write('')
st.subheader('Teams That Have to Be Picked')
yes_i_have_a_required_team = st.checkbox('Do you have a team that you requirew to be used in a specific week?')
if yes_i_have_a_required_team:
    st.write('Select the week in which the algorithm has to pick that team. If you do not want the team to be :red[required] to be used, select 0')
    required_week_options = [0] + list(range(starting_week, ending_week))
    
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
yes_i_have_prohibited_teams = st.checkbox('Do you have teams that you want to prohibit the alogrithm from choosing ina specifc week?')
if yes_i_have_prohibited_teams:
    st.write('Choose which week you do :red[NOT] want a team to be picked. If, for example, you think the 49ers have a bad matchup in Week 15, and you do not want them to be used then, select "15" for the San Francisco 49ers')

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
    cin_excluded_weeks = cin_prohibited_week if cin_prohibited_weeks else []
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
yes_i_have_customized_rankings = st.checkbox('Wopuld you like to use custoimized rankings instead of our default rankings?')
if yes_i_have_customized_rankings:
    st.write('The Ranking represents :red[how much a team would either win (positive number) or lose (negative number) by to an average NFL team] on a neutral field. 0 means the team is perfectly average. If you leave the "Default" value, the default rankings will be used.')
    st.write('')
    
    
    team_rankings = [
        "Default", 0,-15,-14.5,-14,-13.5,-13,-12.5,-12,-11.5,-11,-10.5,-10,-9.5,-9,-8.5,-8,-7.5,
        -7,-6.5,-6,-5.5,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-.5,.5,1,1.5,2,2.5,3,3.5,4,
        4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15
    ]
    az_rank = st.selectbox("Arizona Cardinals Ranking:", options=team_rankings)
    if az_rank == "Default":
        az_rank = default_atl_rank
    st.write(f'Current Arizona Cardinals Ranking: {az_rank}')
    st.write('')
    atl_rank = st.selectbox("Atlanta Falcons Ranking:", options=team_rankings)
    if atl_rank == "Default":
        atl_rank = default_atl_rank
    st.write(f'Current Atlanta Falcons Ranking: {atl_rank}')
    st.write('')
    bal_rank = st.selectbox("Baltimore Ravens Ranking:", options=team_rankings)
    if bal_rank == "Default":
        bal_rank = default_bal_rank
    st.write(f'Current Baltimore Ravens Ranking: {bal_rank}')
    st.write('')
    buf_rank = st.selectbox("Buffalo Bills Ranking:", options=team_rankings)
    if buf_rank == "Default":
        buf_rank = default_buf_rank
    st.write(f'Current Buffalo Bills Ranking: {buf_rank}')
    st.write('')
    car_rank = st.selectbox("Carolina Panthers Ranking:", options=team_rankings)
    if car_rank == "Default":
        car_rank = default_car_rank
    st.write(f'Current Carolina Panthers Ranking: {car_rank}')
    st.write('')
    chi_rank = st.selectbox("Chicago Bears Ranking:", options=team_rankings)
    if chi_rank == "Default":
        chi_rank = default_chi_rank
    st.write(f'Current Chicago Bears Ranking: {chi_rank}')
    st.write('')
    cin_rank = st.selectbox("Cincinnati Bengals Ranking:", options=team_rankings)
    if cin_rank == "Default":
        cin_rank = default_cin_rank
    st.write(f'Current Cincinnati Bengals Ranking: {cin_rank}')
    st.write('')
    cle_rank = st.selectbox("Cleveland Browns Ranking:", options=team_rankings)
    if cle_rank == "Default":
        cle_rank = default_cle_rank
    st.write(f'Current Cleveland Browns Ranking: {cle_rank}')
    st.write('')
    dal_rank = st.selectbox("Dallas Cowboys Ranking:", options=team_rankings)
    if dal_rank == "Default":
        dal_rank = default_dal_rank
    st.write(f'Current Dallas Cowboys Ranking: {dal_rank}')
    st.write('')
    den_rank = st.selectbox("Denver Broncos Ranking:", options=team_rankings)
    if den_rank == "Default":
        den_rank = default_den_rank
    st.write(f'Current Denver Broncos Ranking: {den_rank}')
    st.write('')
    det_rank = st.selectbox("Detroit Lions Ranking:", options=team_rankings)
    if det_rank == "Default":
        det_rank = default_det_rank
    st.write(f'Current Detroit Lions Ranking: {det_rank}')
    st.write('')
    gb_rank = st.selectbox("Green Bay Packers Ranking:", options=team_rankings)
    if gb_rank == "Default":
        gb_rank = default_gb_rank
    st.write(f'Current Green Bay Packers Ranking: {gb_rank}')
    st.write('')
    hou_rank = st.selectbox("Houston Texans Ranking:", options=team_rankings)
    if hou_rank == "Default":
        hou_rank = default_hou_rank
    st.write(f'Current Houston Texans Ranking: {hou_rank}')
    st.write('')
    ind_rank = st.selectbox("Indianapoils Colts Ranking:", options=team_rankings)
    if ind_rank == "Default":
        ind_rank = default_ind_rank
    st.write(f'Current Indianapoils Colts Ranking: {ind_rank}')
    st.write('')
    jax_rank = st.selectbox("Jacksonville Jaguars Ranking:", options=team_rankings)
    if jax_rank == "Default":
        jax_rank = default_jax_rank
    st.write(f'Current Jacksonville Jaguars Ranking: {jax_rank}')
    st.write('')
    kc_rank = st.selectbox("Kansas City Chiefs Ranking:", options=team_rankings)
    if kc_rank == "Default":
        kc_rank = default_kc_rank
    st.write(f'Current Kansas City Chiefs Ranking: {kc_rank}')
    st.write('')
    lv_rank = st.selectbox("Las Vegas Raiders Ranking:", options=team_rankings)
    if lv_rank == "Default":
        lv_rank = default_lv_rank
    st.write(f'Current Las Vegas Raiders Ranking: {lv_rank}')
    st.write('')
    lac_rank = st.selectbox("Los Angeles Chargers Ranking:", options=team_rankings)
    if lac_rank == "Default":
        lac_rank = default_lac_rank
    st.write(f'Current Los Angeles Chargers Ranking: {lac_rank}')
    st.write('')
    lar_rank = st.selectbox("Los Angeles Rams Ranking:", options=team_rankings)
    if lar_rank == "Default":
        lar_rank = default_lar_rank
    st.write(f'Current Los Angeles Rams Ranking: {lar_rank}')
    st.write('')
    mia_rank = st.selectbox("Miami Dolphins Ranking:", options=team_rankings)
    if mia_rank == "Default":
        mia_rank = default_mia_rank
    st.write(f'Current Miami Dolphins Ranking: {mia_rank}')
    st.write('')
    min_rank = st.selectbox("Minnesota Vikings Ranking:", options=team_rankings)
    if min_rank == "Default":
        min_rank = default_min_rank
    st.write(f'Current Minnesota Vikings Ranking: {min_rank}')
    st.write('')
    ne_rank = st.selectbox("New England Patriots Ranking:", options=team_rankings)
    if ne_rank == "Default":
        ne_rank = default_ne_rank
    st.write(f'Current New England Patriots Ranking: {ne_rank}')
    st.write('')
    no_rank = st.selectbox("New Orleans Saints Ranking:", options=team_rankings)
    if no_rank == "Default":
        no_rank = default_no_rank
    st.write(f'Current New Orleans Saints Ranking: {no_rank}')
    st.write('')
    nyg_rank = st.selectbox("New York Giants Ranking:", options=team_rankings)
    if nyg_rank == "Default":
        nyg_rank = default_nyg_rank
    st.write(f'Current New York Giants Ranking: {nyg_rank}')
    st.write('')
    nyj_rank = st.selectbox("New York Jets Ranking:", options=team_rankings)
    if nyj_rank == "Default":
        nyj_rank = default_nyj_rank
    st.write(f'Current New York Jets Ranking: {nyj_rank}')
    st.write('')
    phi_rank = st.selectbox("Philadelphia Eagles Ranking:", options=team_rankings)
    if phi_rank == "Default":
        phi_rank = default_phi_rank
    st.write(f'Current Philadelphia Eagles Ranking: {phi_rank}')
    st.write('')
    pit_rank = st.selectbox("Pittsburgh Steelers Ranking:", options=team_rankings)
    if pit_rank == "Default":
        pit_rank = default_pit_rank
    st.write(f'Current Pittsburgh Steelers Ranking: {pit_rank}')
    st.write('')
    sf_rank = st.selectbox("San Francisco 49ers Ranking:", options=team_rankings)
    if sf_rank == "Default":
        sf_rank = default_sf_rank
    st.write(f'Current San Francisco 49ers Ranking: {sf_rank}')
    st.write('')
    sea_rank = st.selectbox("Seattle Seahawks Ranking:", options=team_rankings)
    if sea_rank == "Default":
        sea_rank = default_sea_rank
    st.write(f'Current Seattle Seahawks Ranking: {sea_rank}')
    st.write('')
    tb_rank = st.selectbox("Tampa Bay Buccaneers Ranking:", options=team_rankings)
    if tb_rank == "Default":
        tb_rank = default_tb_rank
    st.write(f'Current Tampa Bay Buccaneers Ranking: {tb_rank}')
    st.write('')
    ten_rank = st.selectbox("Tennessee Titans Ranking:", options=team_rankings)
    if ten_rank == "Default":
        ten_rank = default_ten_rank
    st.write(f'Current Tennessee Titans Ranking: {ten_rank}')
    st.write('')
    was_rank = st.selectbox("Washington Commanders Ranking:", options=team_rankings)
    if was_rank == "Default":
        was_rank = default_was_rank
    st.write(f'Current Washington Commanders Ranking: {was_rank}')

st.write('')
st.write('')
st.write('')
st.subheader('Select Constraints')
yes_i_have_constraints = st.checkbox('Would youy like to add constraints? For example, "Avoid Teams on Short Rest"')
if yes_i_have_constraints:
    st.write('These constraints will not work 100% of the time (For example in week 18, all Games are divisional matchups). However, it will require a team to be so heavily favored that the impact of the constrained factor should be minimal.')
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
    
    bayesian_rest_travel_constraint = st.selectbox('Bayesian, Rest, and Travel Impact:', options = bayesian_and_travel_options)

st.write('')
st.write('')
st.write('')

use_cached_expected_value = 1

if yes_i_have_customized_rankings:
	st.subheader('Use Saved Expected Value')
	st.write('Warning, this data may not be nup to date.')
	st.write('- Checking this box will ensure the process is fast, (Less than 1 minute, compared to 5+ hours) and will prevent the risk of crashing')
	st.write('- This will not use your customized rankings in the EV calculation process')
	st.write('- This will NOT have an impact on your customized ranking output, just the EV output')
	st.write('Last Update: :green[9/20/2024]')
	use_cached_expected_value = 1 if st.checkbox('Use Cached Expected Value') else 0
	st.write('')
	st.write('')
	st.write('')
st.subheader('Get Optimized Survivor Picks')
number_of_solutions_options = [
    1,5,10,25,50,100
]
st.write('How many solutions would you like from each solver?')
number_solutions = st.selectbox('Number of Solutions', options = number_of_solutions_options)
double_number_solutions = number_solutions * 2
st.text(f'This button will find the best picks for each week. It will pump out {double_number_solutions} solutions.')
st.write(f'- The first {number_solutions} will be :red[based purely on EV] that is a complicated formula based on predicted pick percentage of each team in each week, and their chances of winning that week.')
st.write('- This will use the rankings defined above to determine win probability and thus pick percentage for each team. If you provide your own rankings, :red[you CANNOT use the cached version of EV]. If you use the default rankings, the cached version will be fine') 
st.write(f'- The second {number_solutions} solutions will be based on the :red[rankings and constraints you provided]')
st.write('- This will use the rankings defined above to determine win probability for each team. Because this :red[does not use predicted pick percentage nor EV], you can use the cached version of EV to speed things up.') 
st.write('- All solutions will abide by the prohibited teams and the weeks you selected')
st.write('- If you have too many constraints, or the solution is impossible, you will see an error')
st.write("- :green[Mathematically, EV is most likely to win, however, using your own rankings has advantages as well, which is why we provide both solutions (Sometimes it's just preposterous to Pick the Jets)]")


st.write('')
st.write('')
schedule_data_retrieved = False #Initialize on first run

if st.button("Get Optimized Survivor Picks"):
    st.write("Step 1/9: Fetching Schedule Data...")
    schedule_table, schedule_rows = get_schedule() # Call the function   
    if schedule_table:
        st.write("Step 1 Completed: Schedule Data Retrieved!")
        schedule_data_retrieved = True #Set Flag to True after retrieval
    else:
        st.write("Error. Could not find the table.")
        schedule_data_retrieved = False #Set flag to False on error
         
    if schedule_rows:
        st.write(f"Number of Schedule Rows: {len(schedule_rows)}") #Display row length
        st.write("Step 2/9: Collecting Travel, Ranking, Odds, and Rest Data...")
        collect_schedule_travel_ranking_data_df = collect_schedule_travel_ranking_data_circa(pd)
        st.write("Step 2 Complete: Travel, Ranking, Odds, and Rest Data Retrieved!")
        st.write(collect_schedule_travel_ranking_data_df)
        st.write("Step 3/9: Predicting Future Pick Percentages of Public...")
    if use_cached_expected_value == 0:
        nfl_schedule_pick_percentages_df = get_predicted_pick_percentages_circa(pd)
        st.write("Step 3 Completed: Public Pick Percentages Predicted")
        #nfl_schedule_circa_df_2 = manually_adjust_pick_predictions()
        st.write("Step 4/9: Calculating Expected Value (Could take several hours)...")
    if use_cached_expected_value == 1:
        st.write('- Using Cached Expected Values...')
        full_df_with_ev = pd.read_csv('NFL Schedule with full ev_circa.csv')
    else:
        st.write('- Calculating Live EV...')
        with st.spinner('Processing...'):
            full_df_with_ev = calculate_ev()
            st.write("Processing Complete!")
            st.dataframe(full_df_with_ev)
    st.write("Step 4 Completed: Expected Value Calculated")
    st.write(full_df_with_ev)
    st.write('Step 5/9: Calculating Best Comnbination of Picks Based on EV...')
    ending_week_2 = ending_week - 1	
    if selected_contest == 'Circa':
        st.subheader(f'Optimal Picks for Circa: Weeks {starting_week} through {ending_week_2}')
    else:
        st.subheader(f'Optimal Picks for Draftkings: Weeks {starting_week} through {ending_week_2}')
    get_survivor_picks_based_on_ev()
    st.write('Step 5 Completed: Top Picks Determined Based on EV')
    if yes_i_have_customized_rankings:
        st.write('Step 6/6: Calculating Best Comnbination of Picks Based on Customized Rankings...')
        st.subheader('Customized Ranking Calculations')
        get_survivor_picks_based_on_internal_rankings()
        st.write('Step 6 Completed: Top Picks Determined Based on Customized Rankings')
    else:
        st.write('Step 6/6: Calculating Best Comnbination of Picks Based on Default Rankings...')
        st.subheader('Default Ranking Calculations')
        get_survivor_picks_based_on_internal_rankings()
        st.write('Step 6 Completed: Top Picks Determined Based on Default Rankings')
else:
    st.write("Error. Could not find the rows")
    schedule_data_retrieved = False #Set flag to False on error
