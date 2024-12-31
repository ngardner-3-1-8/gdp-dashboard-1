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
from tqdm.notebook import tqdm
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
    print("Schedule Data Retrieved")
    return table, rows

def collect_schedule_travel_ranking_data(pd):
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
    # 0: Stadium | 1: Lattitude | 2: Longitude | 3: Timezone | 4: Division | 5: Start of 2023 Season Rank | 6: Current Rank | 7: Average points better than Average Team (Used for Spread and Odds Calculation)
    stadiums = {
        'Arizona Cardinals': ['State Farm Stadium', 33.5277, -112.262608, 'America/Denver', 'NFC West', 26, 24, -.5],
        'Atlanta Falcons': ['Mercedez-Benz Stadium', 33.757614, -84.400972, 'America/New_York', 'NFC South', 13, 18, 0],
        'Baltimore Ravens': ['M&T Stadium', 39.277969, -76.622767, 'America/New_York', 'AFC North', 3, 3, 3],
        'Buffalo Bills': ['Highmark Stadium', 42.773739, -78.786978, 'America/New_York', 'AFC East', 5, 7, 3.5],
        'Carolina Panthers': ['Bank of America Stadium', 35.225808, -80.852861, 'America/New_York', 'NFC South', 32, 32, -7],
        'Chicago Bears': ['Soldier Field', 41.862306, -87.616672, 'America/Chicago', 'NFC North', 15, 19, -2],
        'Cincinnati Bengals': ['Paycor Stadium', 39.095442, -84.516039, 'America/New_York', 'AFC North', 6, 11, 2],
        'Cleveland Browns': ['Cleveland Browns Stadium', 41.506022, -81.699564, 'America/New_York', 'AFC North', 17, 20, 0],
        'Dallas Cowboys': ['AT&T Stadium', 32.747778, -97.092778, 'America/Chicago', 'NFC East', 9, 6, 1.5],
        'Denver Broncos': ['Empower Field at Mile High', 39.743936, -105.020097, 'America/Denver', 'AFC West', 29, 29, -5.5],
        'Detroit Lions': ['Ford Field', 42.340156, -83.045808, 'America/New_York', 'NFC North', 4, 5, 3],
        'Green Bay Packers': ['Lambeau Field', 44.501306, -88.062167, 'America/Chicago', 'NFC North', 10, 12, -4],
        'Houston Texans': ['NRG Stadium', 29.684781, -95.410956, 'America/Chicago', 'AFC South', 7, 8, 3.5],
        'Indianapolis Colts': ['Lucas Oil Stadium', 39.760056, -86.163806, 'America/New_York', 'AFC South', 20, 19, -2],
        'Jacksonville Jaguars': ['Everbank Stadium', 30.323925, -81.637356, 'America/New_York', 'AFC South', 18, 17, -.5],
        'Kansas City Chiefs': ['Arrowhead Stadium', 39.048786, -94.484566, 'America/Chicago', 'AFC West', 1, 1, 5],
        'Las Vegas Raiders': ['Allegiant Stadium', 36.090794, -115.183952, 'America/Los_Angeles', 'AFC West', 28, 26, -3],
        'Los Angeles Chargers': ['SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'AFC West', 14, 17, 1.5],
        'Los Angeles Rams': ['SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'NFC West', 16, 15, -3.5],
        'Miami Dolphins': ['Hard Rock Stadium', 25.957919, -80.238842, 'America/New_York', 'AFC East', 12, 11, -4],
        'Minnesota Vikings': ['U.S Bank Stadium', 44.973881, -93.258094, 'America/Chicago', 'NFC North', 24, 22, .5],
        'New England Patriots': ['Gillette Stadium', 42.090925, -71.26435, 'America/New_York', 'AFC East', 31, 27, -4.5],
        'New Orleans Saints': ['Caesars Superdome', 29.950931, -90.081364, 'America/Chicago', 'NFC South', 23, 16, 2],
        'New York Giants': ['MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'NFC East', 27, 31, -5],
        'New York Jets': ['MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'AFC East', 11, 13, 1],
        'Philadelphia Eagles': ['Lincoln Financial Field', 39.900775, -75.167453, 'America/New_York', 'NFC East', 8, 4, 3],
        'Pittsburgh Steelers': ['Acrisure Stadium', 40.446786, -80.015761, 'America/New_York', 'AFC North', 19, 16, .5],
        'San Francisco 49ers': ['Levi\'s Stadium', 37.713486, -122.386256, 'America/Los_Angeles', 'NFC West', 2, 1.5, 4.5],
        'Seattle Seahawks': ['Lumen Field', 47.595153, -122.331625, 'America/Los_Angeles', 'NFC West', 22, 19, .5],
        'Tampa Bay Buccaneers': ['Raymomd James Stadium', 27.975967, -82.50335, 'America/New_York', 'NFC South', 21, 21, 0],
        'Tennessee Titans': ['Nissan Stadium', 36.166461, -86.771289, 'America/Chicago', 'AFC South', 20, 24, -2.5],
        'Washington Commanders': ['FedExField', 38.907697, -76.864517, 'America/New_York', 'NFC East', 25, 28, -3.5]
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

    df['Date'] = df['Date'].str.replace(r'(\w+)\s(\w+)\s(\d+)', r'\2 \3, 2024', regex=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    # Adjust January games to 2025 in the DataFrame
    df['Date'] = df['Date'].apply(lambda x: x.replace(year=2025) if x.month == 1 else x)
    df['Week'] = df['Week'].str.replace('Week ', '', regex=False).astype(int)

    # Increment 'Week' for games on or after 2024-11-30
    df.loc[df['Date'] >= pd.to_datetime('2024-11-30'), 'Week'] += 1
    df.loc[df['Date'] >= pd.to_datetime('2024-12-27'), 'Week'] += 1

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

    df['Preseason Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Preseason Rank'] < row['Home Team Preseason Rank'] else (row['Home Team'] if row['Away Team Preseason Rank'] > row['Home Team Preseason Rank'] else 'Tie'), axis=1)
    df['Preseason Difference'] = abs(df['Away Team Preseason Rank'] - df['Home Team Preseason Rank'])

    df['Away Team Adjusted Preseason Rank'] = df['Away Team'].map(lambda team: stadiums[team][5]) + np.where((df['Away Travel Advantage'] < -100) & (df['Home Stadium'] == df['Actual Stadium']), 1.5, 0) - pd.to_numeric(df['Away Timezone Advantage'], errors='coerce').fillna(0) - pd.to_numeric(df['Weekly Away Rest Advantage'], errors='coerce').fillna(0) - .125*df['Away Team Current Week Cumulative Rest Advantage']
    df['Home Team Adjusted Preseason Rank'] = df['Home Team'].map(lambda team: stadiums[team][5]) - np.where((df['Away Travel Advantage'] < -100) & (df['Home Stadium'] == df['Actual Stadium']), 1.5, 0) - pd.to_numeric(df['Home Timezone Advantage'], errors='coerce').fillna(0) - pd.to_numeric(df['Weekly Home Rest Advantage'], errors='coerce').fillna(0) - .125*df['Home Team Current Week Cumulative Rest Advantage']

    df['Adjusted Preseason Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Adjusted Preseason Rank'] < row['Home Team Adjusted Preseason Rank'] else (row['Home Team'] if row['Away Team Adjusted Preseason Rank'] > row['Home Team Adjusted Preseason Rank'] else 'Tie'), axis=1)
    df['Adjusted Preseason Difference'] = abs(df['Away Team Adjusted Preseason Rank'] - df['Home Team Adjusted Preseason Rank'])

    df['Away Team Current Rank'] = df['Away Team'].map(lambda team: stadiums[team][6] if team in stadiums else 'NA')
    df['Home Team Current Rank'] = df['Home Team'].map(lambda team: stadiums[team][6] if team in stadiums else 'NA')

    df['Current Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Current Rank'] < row['Home Team Current Rank'] else (row['Home Team'] if row['Away Team Current Rank'] > row['Home Team Current Rank'] else 'Tie'), axis=1)
    df['Current Difference'] = abs(df['Away Team Current Rank'] - df['Home Team Current Rank'])

    df['Away Team Adjusted Current Rank'] = df['Away Team'].map(lambda team: stadiums[team][6]) + np.where((df['Away Travel Advantage'] < -100) & (df['Home Stadium'] == df['Actual Stadium']), 1.5, 0) - pd.to_numeric(df['Away Timezone Advantage'], errors='coerce').fillna(0)-pd.to_numeric(df['Weekly Away Rest Advantage'], errors='coerce').fillna(0)-.125*df['Away Team Current Week Cumulative Rest Advantage']
    df['Home Team Adjusted Current Rank'] = df['Home Team'].map(lambda team: stadiums[team][6]) - np.where((df['Away Travel Advantage'] < -100) & (df['Home Stadium'] == df['Actual Stadium']), 1.5, 0) - pd.to_numeric(df['Home Timezone Advantage'], errors='coerce').fillna(0)-pd.to_numeric(df['Weekly Home Rest Advantage'], errors='coerce').fillna(0)-.125*df['Home Team Current Week Cumulative Rest Advantage']

    df['Adjusted Current Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Adjusted Current Rank'] < row['Home Team Adjusted Current Rank'] else (row['Home Team'] if row['Away Team Adjusted Current Rank'] > row['Home Team Adjusted Current Rank'] else 'Tie'), axis=1)
    df['Adjusted Current Difference'] = abs(df['Away Team Adjusted Current Rank'] - df['Home Team Adjusted Current Rank'])

    df['Same Winner?'] = df.apply(lambda row: 'Same' if row['Preseason Winner'] == row['Adjusted Preseason Winner'] == row['Current Winner'] == row['Adjusted Current Winner'] else 'Different', axis=1)
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
            'Arizona Cardinals': ['State Farm Stadium', 33.5277, -112.262608, 'America/Denver', 'NFC West', 26, 24, -.5],
            'Atlanta Falcons': ['Mercedez-Benz Stadium', 33.757614, -84.400972, 'America/New_York', 'NFC South', 13, 18, 0],
            'Baltimore Ravens': ['M&T Stadium', 39.277969, -76.622767, 'America/New_York', 'AFC North', 3, 3, 3],
            'Buffalo Bills': ['Highmark Stadium', 42.773739, -78.786978, 'America/New_York', 'AFC East', 5, 7, 3.5],
            'Carolina Panthers': ['Bank of America Stadium', 35.225808, -80.852861, 'America/New_York', 'NFC South', 32, 32, -7],
            'Chicago Bears': ['Soldier Field', 41.862306, -87.616672, 'America/Chicago', 'NFC North', 15, 19, -2],
            'Cincinnati Bengals': ['Paycor Stadium', 39.095442, -84.516039, 'America/New_York', 'AFC North', 6, 11, 2],
            'Cleveland Browns': ['Cleveland Browns Stadium', 41.506022, -81.699564, 'America/New_York', 'AFC North', 17, 20, 0],
            'Dallas Cowboys': ['AT&T Stadium', 32.747778, -97.092778, 'America/Chicago', 'NFC East', 9, 6, 1.5],
            'Denver Broncos': ['Empower Field at Mile High', 39.743936, -105.020097, 'America/Denver', 'AFC West', 29, 29, -5.5],
            'Detroit Lions': ['Ford Field', 42.340156, -83.045808, 'America/New_York', 'NFC North', 4, 5, 3],
            'Green Bay Packers': ['Lambeau Field', 44.501306, -88.062167, 'America/Chicago', 'NFC North', 10, 12, -4],
            'Houston Texans': ['NRG Stadium', 29.684781, -95.410956, 'America/Chicago', 'AFC South', 7, 8, 3.5],
            'Indianapolis Colts': ['Lucas Oil Stadium', 39.760056, -86.163806, 'America/New_York', 'AFC South', 20, 19, -2],
            'Jacksonville Jaguars': ['Everbank Stadium', 30.323925, -81.637356, 'America/New_York', 'AFC South', 18, 17, -.5],
            'Kansas City Chiefs': ['Arrowhead Stadium', 39.048786, -94.484566, 'America/Chicago', 'AFC West', 1, 1, 5],
            'Las Vegas Raiders': ['Allegiant Stadium', 36.090794, -115.183952, 'America/Los_Angeles', 'AFC West', 28, 26, -3],
            'Los Angeles Chargers': ['SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'AFC West', 14, 17, 1.5],
            'Los Angeles Rams': ['SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'NFC West', 16, 15, -3.5],
            'Miami Dolphins': ['Hard Rock Stadium', 25.957919, -80.238842, 'America/New_York', 'AFC East', 12, 11, -4],
            'Minnesota Vikings': ['U.S Bank Stadium', 44.973881, -93.258094, 'America/Chicago', 'NFC North', 24, 22, .5],
            'New England Patriots': ['Gillette Stadium', 42.090925, -71.26435, 'America/New_York', 'AFC East', 31, 27, -4.5],
            'New Orleans Saints': ['Caesars Superdome', 29.950931, -90.081364, 'America/Chicago', 'NFC South', 23, 16, 2],
            'New York Giants': ['MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'NFC East', 27, 31, -5],
            'New York Jets': ['MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'AFC East', 11, 13, 1],
            'Philadelphia Eagles': ['Lincoln Financial Field', 39.900775, -75.167453, 'America/New_York', 'NFC East', 8, 4, 3],
            'Pittsburgh Steelers': ['Acrisure Stadium', 40.446786, -80.015761, 'America/New_York', 'AFC North', 19, 16, .5],
            'San Francisco 49ers': ['Levi\'s Stadium', 37.713486, -122.386256, 'America/Los_Angeles', 'NFC West', 2, 1.5, 4.5],
            'Seattle Seahawks': ['Lumen Field', 47.595153, -122.331625, 'America/Los_Angeles', 'NFC West', 22, 19, .5],
            'Tampa Bay Buccaneers': ['Raymomd James Stadium', 27.975967, -82.50335, 'America/New_York', 'NFC South', 21, 21, 0],
            'Tennessee Titans': ['Nissan Stadium', 36.166461, -86.771289, 'America/Chicago', 'AFC South', 20, 24, -2.5],
            'Washington Commanders': ['FedExField', 38.907697, -76.864517, 'America/New_York', 'NFC East', 25, 28, -3.5]
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
            21: [-5597, 1434]
        }

        live_odds_df = live_scraped_odds_df
        
        print(live_odds_df)

        #df.to_csv('TEST Manual Odds.csv', index = False)
        # Load the CSV data
        csv_df = df
        print(csv_df)
        # Update CSV data with scraped odds
        for index, row in csv_df.iterrows():
            matching_row = live_odds_df[
                (live_odds_df['Away Team'] == row['Away Team']) & (live_odds_df['Home Team'] == row['Home Team'])
            ]
            if not matching_row.empty:
                csv_df.loc[index, 'Away Team Moneyline'] = matching_row.iloc[0]['Away Odds']
                csv_df.loc[index, 'Home Team Moneyline'] = matching_row.iloc[0]['Home Odds']
                # Create the mask for where there is no 'Home Odds'
        mask = csv_df['Home Team Moneyline'].isna()
        # Only apply calculations if the 'Home Odds' column is empty
        if mask.any():
            # Adjust Average Points Difference for Favorite/Underdog Determination
            csv_df['Adjusted Home Points'] = csv_df.apply(lambda row: stadiums[row['Home Team']][7] + 1, axis=1)
            csv_df['Adjusted Away Points'] = csv_df.apply(lambda row: stadiums[row['Away Team']][7] - 1, axis=1)

            csv_df['Spread'] = csv_df.apply(lambda row: abs(stadiums[row['Away Team']][7] - stadiums[row['Home Team']][7]), axis=1)

            # Determine Favorite and Underdog
            csv_df['Favorite'] = csv_df.apply(lambda row: row['Home Team'] if row['Adjusted Home Points'] > row['Adjusted Away Points'] else row['Away Team'], axis=1)
            csv_df['Underdog'] = csv_df.apply(lambda row: row['Home Team'] if row['Adjusted Home Points'] < row['Adjusted Away Points'] else row['Away Team'], axis=1)

            # Adjust Spread based on Favorite
            csv_df['Adjusted Spread'] = csv_df.apply(lambda row: row['Spread'] + 2 if row['Favorite'] == row['Home Team'] else row['Spread'] - 2, axis=1)

            # Overwrite Odds based on Spread and Favorite/Underdog
            csv_df['Home Team Moneyline'] = csv_df.apply(lambda row: odds[row['Adjusted Spread']][0] if row['Favorite'] == row['Home Team'] else odds[row['Adjusted Spread']][1], axis=1)
            csv_df['Away Team Moneyline'] = csv_df.apply(lambda row: odds[row['Adjusted Spread']][1] if row['Favorite'] == row['Home Team'] else odds[row['Adjusted Spread']][0], axis=1)
        # Calculate Implied Odds and Fair Odds
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
        # Save the updated CSV
        csv_df.to_csv('nfl_schedule_circa.csv', index=False)
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
    
    collect_schedule_travel_ranking_data_nfl_schedule_circa_df = consolidated_df
    
    return collect_schedule_travel_ranking_data_nfl_schedule_circa_df









st.title("NFL Survivor Optimization")
st.subheader("The second best Circa Survivor Contest optimizer")
st.text("Alright, clowns. This site is built to help you optimize your picks for the Circa Survivor contest (Eventually other contests). Simply input which week you're in, your team rankings, and the algorithm will do the rest. For a full season, Running the algorith will take up to 5 hours or more. Good luck!")

schedule_data_retrieved = False #Initialize on first run

if st.button("Get Optimized Survivor Picks"):
    st.write("Step 1/9: Fetching Schedule Data...")
    schedule_table, schedule_rows = get_schedule() # Call the function   
    if schedule_table:
        st.write("Step 1 Completed: Schedule Data Retrieved")
        schedule_data_retrieved = True #Set Flag to True after retrieval
        schedule_table = schedule_table #Store table data into session state
        schedule_rows = schedule_rows #Store rows data into session state # ADD THIS LINE
    else:
         st.write("Error. Could not find the table.")
         schedule_data_retrieved = False #Set flag to False on error
         
    if schedule_rows:
        st.write(f"Number of Schedule Rows: {len(schedule_rows)}") #Display row length
        st.write("Step 2/9: Collecting Travel, Ranking, Odds, and Rest Data...")
        collect_schedule_travel_ranking_data_df = collect_schedule_travel_ranking_data(pd)
        st.write("Step 2 Complete: Travel, Ranking, Odds, and Rest Data Retrieved")
        st.write(collect_schedule_travel_ranking_data_df)
        #live_scraped_odds_df = get_preseason_odds()
        #schedule_df_with_odds_df = add_odds_to_main_csv()
        #nfl_schedule_circa_pick_percentages_df = get_predicted_pick_percentages(pd)
        #nfl_schedule_circa_df_2 = manually_adjust_pick_predictions()
        #full_df_with_ev = calculate_ev()
        #get_survivor_picks_based_on_ev()
        #get_survivor_picks_based_on_internal_rankings()
    else:
         st.write("Error. Could not find the rows")
         schedule_data_retrieved = False #Set flag to False on error


