#Stand: 01-01-2024

import pandas as pd
import numpy as np
# pd.set_option("display.max_columns", 8)

club_name = "Tottenham" #The main name of your team e.g. Tottenham, Leipzig,...
season = 2022 #The season you want to analyze

competitions = pd.read_csv(r'C:\Users\anhvi\OneDrive\Desktop\Football Data\datasets\competitions.csv')
appearances = pd.read_csv(r"C:\Users\anhvi\OneDrive\Desktop\Football Data\datasets\appearances.csv")
club_games = pd.read_csv(r'C:\Users\anhvi\OneDrive\Desktop\Football Data\datasets\club_games.csv') 
clubs = pd.read_csv(r'C:\Users\anhvi\OneDrive\Desktop\Football Data\datasets\clubs.csv')
game_events = pd.read_csv(r'C:\Users\anhvi\OneDrive\Desktop\Football Data\datasets\game_events.csv')
game_lineups = pd.read_csv(r'C:\Users\anhvi\OneDrive\Desktop\Football Data\datasets\game_lineups.csv')
games = pd.read_csv(r'C:\Users\anhvi\OneDrive\Desktop\Football Data\datasets\games.csv')
players = pd.read_csv(r'C:\Users\anhvi\OneDrive\Desktop\Football Data\datasets\players.csv')
player_values = pd.read_csv(r'C:\Users\anhvi\OneDrive\Desktop\Football Data\datasets\player_valuations.csv')

#For later comparison with other clubs
game = games
club = clubs
player_value = player_values


#Get id club
clubs = clubs[clubs['name'].str.contains(club_name, case=False)]
club_id = clubs['club_id'].iat[0]

#League_id 
league_id = clubs['domestic_competition_id'].iat[0]
league = competitions[competitions['competition_id'] == league_id]


club_country = clubs.merge(competitions[['country_name', 'domestic_league_code']], 
                                                   left_on='domestic_competition_id' , 
                                                   right_on='domestic_league_code', how='inner')['country_name'].iat[0]


'''Clean and make use of data'''
#Get all the appearances of season (sofar)
appearances['date'] = pd.to_datetime(appearances['date'])
appearances['season'] = appearances['date'].apply(lambda x: f'{x.year}-{x.year + 1}' if x.month >= 8 else f'{x.year - 1}-{x.year}')
appearances['season_start_year'] = appearances['season'].str.split('-').str[0].astype(int)
appearances['season_end_year'] = appearances['season'].str.split('-').str[1].astype(int)
appearances['season_start_month_year'] = appearances['season_start_year'].apply(lambda x: f"08 {x}")

games['date'] = pd.to_datetime(games['date'])


#Filter layers and create an order after positions
order = ['Goalkeeper', 'Defender', 'Midfield', 'Attack']
players['position'] = pd.Categorical(players['position'], categories=order, ordered=True)
players['date_of_birth'] = pd.to_datetime(players['date_of_birth'])
players['is_foreigner'] = np.where(players['country_of_citizenship'] != club_country, 1, 0)
players['age'] = season - players['date_of_birth'].dt.year

player_values['date'] = pd.to_datetime(player_values['date'])
player_values['month_year'] = player_values['date'].apply(lambda x: f"08 {x.year}" if x >= pd.Timestamp(x.year, 7, 1) else f"08 {x.year-1}")
player_values['season'] = player_values['date'].apply(lambda x: f"{x.year}-{x.year+1}" if x >= pd.Timestamp(x.year, 7, 1) else f"{x.year-1}-{x.year}")
player_values = player_values.loc[player_values.groupby(['player_id','season'])['date'].idxmin()] #Get value only at start of season


'''Filter data'''
#Get all domestic games in that season
games = games[((games['home_club_id'] == club_id) | (games['away_club_id'] == club_id)) & (games['season'] == season)]
games_id = games[games['competition_id'] == league_id].sort_values(by='round')['game_id'].tolist()
domestic_game = games[games['competition_id'] == league_id].sort_values(by='date')

#Filter all table to get tottenham in a season (if current season then sofar) only
club_games = club_games[((club_games['club_id'] == club_id)) & (club_games['game_id'].isin(games_id))]
club_games['total_goals'] = club_games['own_goals'] + club_games['opponent_goals']
club_games['clean_sheets'] = club_games.apply(
    lambda row: 1 if (row['opponent_id'] == club_id and row['own_goals'] == 0) | (row['club_id'] == club_id and row['opponent_goals'] == 0) else 0,
    axis=1
)

game_events = game_events[(game_events['club_id'] == club_id ) & (game_events['game_id'].isin(games_id))]
game_lineups = game_lineups[(game_lineups['club_id'] == club_id ) & (game_lineups['game_id'].isin(games_id))]