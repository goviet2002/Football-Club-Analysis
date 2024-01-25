from Club_Data import club_id, club_name, season, competitions, league_id, club, game, player_value, \
                    clubs, players, club_games, game_events, games, appearances, \
                    games_id, player_values, domestic_game
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import warnings

# suppress the warning message
warnings.filterwarnings('ignore')

#Get all players in the season
appearance = appearances #For later when analyzing player
appearances = appearances[appearances['player_club_id'] == club_id]
appearances = appearances[(appearances['game_id'].isin(games_id)) & (appearances['player_club_id'] == club_id)]
get_all_players = set(appearances['player_id'].to_list())

players_current_season = players[players['player_id'].isin(get_all_players)]
players_current_season['season_start'] = f'08 {season}'

#Get the value of the following season
players_current_season = players_current_season.merge(player_values[['player_id', 'market_value_in_eur', 'month_year']], 
                left_on=['season_start', 'player_id'], right_on=['month_year', 'player_id'])
attribute = ['name', 'country_of_citizenship', 'date_of_birth', 'sub_position', 'foot', 'height_in_cm', 'market_value_in_eur_y']


'''Name, Stadium and attendance'''
name = clubs['name'].iat[0]
stadium = games[games['home_club_id'] == club_id]['stadium'].iat[0]
attendance = games['attendance'].mean()
print(f'{name} plays in {stadium} with average attendance of {round(attendance, 2)} seats per match\n')


'''Club link on transfermarkt'''
url = clubs[clubs['name'].str.contains(club_name)]['url'].iat[0]
print(f"Club URL on Transfermarkt: {url}\n")


'''Coach'''
a = set(club_games[club_games['club_id'] == club_id]['own_manager_name'].tolist())
b = set(club_games[club_games['opponent_id'] == club_id]['opponent_manager_name'].tolist())
coach = list(set(a|b))
if len(coach) == 1:
    coach = coach[0]
print(f'Manager in {season}/{season+1} season is {coach}\n')


'''Total market value'''
total_market_value = players_current_season['market_value_in_eur_y'].sum()
print(f'In {season}/{season+1} the club has a total market value of {total_market_value}€\n')


'''Most valuable player'''
mvp = players_current_season[attribute][players_current_season['market_value_in_eur_y'] == players_current_season['market_value_in_eur_y'].max()]
print(f"Most valuable player of the team:\n{mvp}\n")


# #Club captain
# club_captain_id = game_lineups[(game_lineups['club_id'] == club_id )].groupby(['player_id'])['team_captain'].sum().sort_values(ascending = False).index[0]
# club_captain = players_current_season[attribute][players_current_season['player_id'] == club_captain_id]
# print(f"Team Captain:\n {club_captain}")


'''Average Age, oldest, youngest players'''
avg_age = players_current_season['age'].mean()
print(f'Average age of the team is {round(avg_age, 2)} years old\n')

oldest = players_current_season[['name', 'age']][players_current_season['age'] == players_current_season['age'].max()]
youngest = players_current_season[['name', 'age']][players_current_season['age'] == players_current_season['age'].min()]
print(f"The youngest player of the team are:\n {youngest}\n")
print(f"The oldest player of the team are:\n {oldest}")


'''most played'''
most_minutes = appearances.groupby('player_name')['minutes_played'].sum().sort_values(ascending=False).head(1)
print(f"\nPlayer with the most minutes\n {most_minutes}")


#Player value ranking in that season
sns.barplot(data=players_current_season.sort_values(by='market_value_in_eur_y', ascending=False), x='market_value_in_eur_y', y='name', edgecolor="none")
def format_func(value, tick_number):
    return f'{value:.0f}'

pvl = plt.gca()
pvl.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

plt.xlabel('Market Value (€)')
plt.xticks(rotation=30)

plt.ylabel('Name')
plt.title("Player's market value")
plt.show()


'''"Most used formation (games) based on minutes'''
xi_id = appearances[(appearances['game_id'].isin(games_id))].groupby('player_id')['minutes_played'].sum(). \
        sort_values(ascending = False).index[:11].tolist()
xi = players_current_season[players_current_season['player_id'].isin(xi_id)].sort_values(by='position')[attribute]
print(f"\nThe most used XI of the season:\n {xi} \n")


'''Describe goals scored, conceded'''
#Goals per Player
goal_assist = appearances.groupby('player_name')[['goals', 'assists']].sum()
goal = goal_assist[goal_assist['goals'] > 0].sort_values(by='goals', ascending = False).reset_index()
sns.barplot(data=goal, y='player_name', x='goals')
plt.title('Goal per Player')
plt.xlabel('Number of goals')
plt.ylabel('Player')
plt.show()

#Assists per Player
assist = goal_assist[goal_assist['assists'] > 0].sort_values(by='assists', ascending = False).reset_index()
sns.barplot(data=assist, y='player_name', x='assists')
plt.title('Assist per Player')
plt.xlabel('Number of assists')
plt.ylabel('Player')
plt.show()

#Player with most goal contributions
goal_assist['total_contribution'] = goal_assist['goals'] + goal_assist['assists']
goal_assist.reset_index(inplace=True)
player_most_contribution = (goal_assist[goal_assist['total_contribution'] == goal_assist['total_contribution'].max()] \
                            [['player_name', 'total_contribution']].values.tolist())
player_most_contribution_list = [player[0] for player in player_most_contribution]

print(f"Player with the most goal contribution:")
for i in range(0, len(player_most_contribution)):
    print(f"{i+1}. {player_most_contribution[i][0]} with {player_most_contribution[i][1]} goals and assists")
    
#Goal Distribution
bins = list(range(0, 101, 10))
label = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-']
game_events['interval'] = pd.cut(game_events['minute'], bins=bins, labels=label)
total_goals = club_games['own_goals'].sum()
goals = game_events[(game_events['type'] == 'Goals') & (game_events['club_id'] == club_id)]

sns.histplot(data=goals, x='interval', kde=True)

plt.title('Goal Distribution')
plt.ylabel('Number of goals')
plt.xlabel('Time Interval')

plt.show()

goal_col = club_games['own_goals']
bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
bin_labels = ['0.5', '1.5', '2.5', '3.5', '4.5', '5.5']

# Calculate percentage distribution of total goals
lower = [np.sum(goal_col < bin) / len(goal_col) * 100 for bin in bins]
higher = [np.sum(goal_col > bin) / len(goal_col) * 100 for bin in bins]

fig, plot_percentage = plt.subplots()
plot_percentage.barh(bin_labels, lower, label='Lower')
plot_percentage.barh(bin_labels, higher, left=lower, label='Higher')

plot_percentage.set_xlabel('Percentage')
plot_percentage.set_ylabel('Goal Interval')
plot_percentage.set_title('Percentage distribution of goals')
plot_percentage.legend(bbox_to_anchor=(1, 1), loc='upper left')

#Conceded Goals
conceed_col = club_games['opponent_goals']
lower = [np.sum(conceed_col < bin) / len(goal_col) * 100 for bin in bins]
higher = [np.sum(conceed_col > bin) / len(goal_col) * 100 for bin in bins]

figu, plot_percentages = plt.subplots()
plot_percentages.barh(bin_labels, lower, label='Lower')
plot_percentages.barh(bin_labels, higher, left=lower, label='Higher')

plot_percentages.set_xlabel('Percentage')
plot_percentages.set_ylabel('Goal Interval')
plot_percentages.set_title('Percentage distribution of conceded goals')
plot_percentages.legend(bbox_to_anchor=(1, 1), loc='upper left')

plt.show()

    
'''Foreigner Rate'''
foreign_rate = players_current_season[players_current_season['is_foreigner'] == 1]['is_foreigner'].sum()/ len(players_current_season) * 100
print(f'\n{club_name} has a foreign rate of {round(foreign_rate,2)}%\n')


'''box plot of foreign players with goal contributions'''
player_contribution = players_current_season.merge(goal_assist, left_on='name', right_on='player_name', how='left')
player_contribution['total_contribution'].fillna(0, inplace=True)

sns.boxplot(data=player_contribution, x='is_foreigner', y='total_contribution')
plt.xlabel('Is Foreign')
plt.ylabel('Goal Contributions')
plt.show()


'''points per match/ standing'''
results = {'win': 0, 'draw': 0, 'lose': 0}

# Check for wins, draws, and losses
for index, row in domestic_game.iterrows():
    if row['home_club_id'] == club_id:
        if row['home_club_goals'] > row['away_club_goals']:
            results['win'] += 1
        elif row['home_club_goals'] == row['away_club_goals']:
            results['draw'] += 1
        else:
            results['lose'] += 1
    elif row['away_club_id'] == club_id:
        if row['away_club_goals'] > row['home_club_goals']:
            results['win'] += 1
        elif row['away_club_goals'] == row['home_club_goals']:
            results['draw'] += 1
        else:
            results['lose'] += 1
print(f"Results in the season: {results}")


'''Home, Away results'''
home = {'win': 0, 'draw': 0, 'lose': 0}
for index, row in domestic_game.iterrows():
    if row['home_club_id'] == club_id:
        if row['home_club_goals'] > row['away_club_goals']:
            home['win'] += 1
        elif row['home_club_goals'] == row['away_club_goals']:
            home['draw'] += 1
        else:
            home['lose'] += 1
print(f"\nresults when playing at home: {home}")

away = {'win': 0, 'draw': 0, 'lose': 0}
for index, row in domestic_game.iterrows():
    if row['away_club_id'] == club_id:
        if row['away_club_goals'] > row['home_club_goals']:
            away['win'] += 1
        elif row['away_club_goals'] == row['home_club_goals']:
            away['draw'] += 1
        else:
            away['lose'] += 1
print(f"\nresults when playing away from home: {away}")

#Points per game
points_per_game = (results['win'] * 3 + results['draw']) / (results['win'] + results['draw'] + results['lose'])
print(f"\nAfter {results['win'] + results['draw'] + results['lose']} matches the team gets an average of {round(points_per_game, 2)} points per game")


#Clean sheets
clean_sheets = club_games['clean_sheets'].sum()
print(f'\nThe club has {clean_sheets} matches where they conceeded no goal ({round(clean_sheets/len(club_games) * 100, 2)}%)')


'''Position Changes '''
position = []
round = []
for index, row in domestic_game[['round','home_club_id', 'away_club_id', 'home_club_position', 'away_club_position']].iterrows():
    round.append(row['round'])
    if row['home_club_id'] == club_id:
        position.append(row['home_club_position'])
    elif row['away_club_id'] == club_id:
        position.append(row['away_club_position'])
        
plt.plot(round, position)
plt.gca().invert_yaxis()
plt.title(f'Table Position in {season}/{season+1} season (sofar)')

plt.xlabel('Match Round')
plt.xticks(rotation=45)

plt.ylabel('Position')
plt.show()


'''Market Value to poistion compare to other club'''
clubs_id = set(game[(game['competition_id'] == league_id) & (game['season'] == season)]['home_club_id'].tolist())
team_name = []
market_value = []

for id in clubs_id: #It displays the full market value for all team in that season
    team_name.append(club[club['club_id'] == id]['name'].iat[0])
    
    game_team = game[((game['home_club_id'] == id) | (game['away_club_id'] == id)) & (game['season'] == season)]
    gameid = game_team[game_team['competition_id'] == league_id].sort_values(by='round')['game_id'].tolist()
    app = appearance[(appearance['game_id'].isin(gameid)) & (appearance['player_club_id'] == id)]
    get_all = set(app['player_id'].to_list())
    players_current = players[players['player_id'].isin(get_all)]
    players_current['season_start'] = f'08 {season}'
    players_current = players_current.merge(player_value[['player_id', 'market_value_in_eur', 'month_year']], 
                    left_on=['season_start', 'player_id'], right_on=['month_year', 'player_id'])
    players_current.drop_duplicates(subset=['player_id'], inplace=True)
    
    market_value.append(players_current['market_value_in_eur_y'].sum())

team_name = [name.replace(" Football Club", "") for name in team_name] #Shorten the name but only applied for English club

#pair sort the team and their values
team_and_market_value = list(zip(team_name, market_value))
team_and_market_value.sort(key=lambda x: x[1])
team_name, market_value = zip(*team_and_market_value)

plt.barh(team_name, market_value)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

plt.xlabel('Market Value (€)')
plt.xticks(rotation=30)

plt.ylabel('Team')
plt.show()


'''Correlation position and market value'''
#Map team name and market value to 2 lists
league = game[(game['competition_id'] == league_id) & (game['season'] == season)]
last_current_round = league.sort_values(by='date', ascending=False)['round'].iat[0]
position_home = league[league['round'] == last_current_round][['home_club_name', 'home_club_position']].values.tolist()
position_away = league[league['round'] == last_current_round][['away_club_name', 'away_club_position']].values.tolist()
league_position = position_home + position_away

# Make the club name shorter (only for English clubs)
league_position = dict([[club[0].replace(" Football Club", ""), club[1]] for club in league_position])
team_and_market_value = dict([[club[0].replace(" Football Club", ""), club[1]] for club in team_and_market_value]) 

values = [team_and_market_value[club] for club in league_position]
positions = [int(league_position[club]) for club in league_position]

plt.scatter(values, positions)
sns.regplot(x=values, y=positions, scatter=False, color='red')
plot_value_pos = plt.gca()

plt.title('Scatter Plot between Market Value and Position')
plt.xlabel('Market Value (€)')
plt.xticks(rotation=30)# This is to make the position 1 appear at the top
plot_value_pos.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

plt.ylabel('Position on Table')
plt.yticks(range(min(positions), max(positions) + 1))
plt.gca().invert_yaxis()

plt.show()


'''cards distribution'''
cards = game_events[(game_events['type'] == 'Cards') & (game_events['club_id'] == club_id)]
conditions = [(cards['description'].str.contains('Yellow card', case=False)), (cards['description'].str.contains('Red|Second yellow', case=False))]
choice = ['Yellow', 'Red']
cards['card_type'] = np.select(conditions, choice)

#Total Cards: Red and Yellow
total_yellow = appearances['yellow_cards'].sum()
total_red = appearances['red_cards'].sum()
total_cards = total_yellow + total_red
print(f"The team has received {total_yellow} yellow and {total_red} red cards")
print(f"The team has received average {total_yellow/ len(club_games)} yellow cards per match")

#Player with most yellow cards
most_yellow = appearances.groupby('player_id')['yellow_cards'].sum()
no_yellow = most_yellow.max()
player_most_yellow_id = most_yellow[most_yellow == no_yellow].index.tolist()
player_most_yellow = players_current_season[attribute][players_current_season['player_id'].isin(player_most_yellow_id)]
player_most_yellow = player_most_yellow.assign(Number_of_yellow_cards=no_yellow)
print(f"Players with the most yellow cards:\n {player_most_yellow}\n")

#Player with most red cards
most_red = appearances.groupby('player_id')['red_cards'].sum().rename("sum")
no_red = most_red.max()
player_most_red_id = most_red[most_red == no_red].index.tolist()
player_most_red = players_current_season[attribute][players_current_season['player_id'].isin(player_most_red_id)]
player_most_red = player_most_red.assign(Number_of_red_cards=no_red)
print(f"Players with the most red cards:\n {player_most_red}\n")

if not cards.empty:
    card_hist = sns.histplot(data=cards, x='interval', hue='card_type', palette=['yellow', 'red'])
    plt.title('Card Distribution')
    plt.ylabel('Number of cards')
    plt.xlabel('Time Interval')
    plt.show()
else: pass

#Red cards affect, get the match
game_id_red_card = appearances[appearances['red_cards'] >= 1]['game_id'].tolist()
game_red = games[games['game_id'].isin(set(game_id_red_card))]
df_ids = pd.DataFrame(game_id_red_card, columns=['id'])
df_ids['Number of cards'] = df_ids.groupby('id')['id'].transform('count')
df_ids = df_ids.drop_duplicates()
game_red = game_red.merge(df_ids, left_on='game_id', right_on='id')[['season', 'round', 'date', 'home_club_name', 'aggregate', 'away_club_name', 'Number of cards']]
print(game_red)


'''Top Scorer analysis'''
for player_most_contribution_name in player_most_contribution_list: #in case there are more than 1 player
    player_most_contribution_id = players_current_season[players_current_season['name'] == player_most_contribution_name]['player_id'].iat[0]
    print(f"\n{players_current_season[players_current_season['name'] == player_most_contribution_name][attribute]}\n")
    
    #Market value over time until the analyzed season
    value_over_time = player_values[player_values['date'] <= f'{season+1}-06-30'] \
            .merge(players_current_season[players_current_season['name'] == player_most_contribution_name][['player_id', 'name']], on='player_id')
    sns.lineplot(data=value_over_time, x='month_year', y='market_value_in_eur')

    plt.xlabel('Date')
    plt.xticks(rotation=75)

    plt.ylabel('Market Value (€')
    value_time_plot = plt.gca()
    value_time_plot.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    plt.title(f'Market Value over time of {player_most_contribution_name}')
    plt.show()


    #Reload apperances again to get stats of previous season
    #Get his game contributions
    his_data = appearance[(appearance['player_id'] == player_most_contribution_id) 
                    & (appearance['player_club_id'] == club_id)
                    & (appearance['competition_id'] == league_id)
                    & (appearance['season_start_year'] <= season)]
    
    his_contributions= his_data.groupby('season').agg({
        'goals': ['sum', 'mean'],
        'assists': ['sum', 'mean'],
        'yellow_cards': ['sum', 'mean'],
        'red_cards': ['sum', 'mean'],
        'minutes_played': ['sum', 'mean']
    }).reset_index()

    for column in ['goals', 'assists', 'yellow_cards', 'red_cards', 'minutes_played']:
        his_contributions[(column, 'mean')] = his_contributions[(column, 'mean')].round(2)
        
    goals_sum = his_contributions[('goals', 'sum')].tolist()
    assists_sum = his_contributions[('assists', 'sum')].tolist()
    seasons = his_contributions['season'].tolist()
    
    # Plot goals
    plt.plot(seasons, goals_sum, marker='o', label='Goals')

    # Plot assists
    plt.plot(seasons, assists_sum, marker='o', label='Assists')

    plt.xlabel('Season')
    plt.xticks(rotation=30)

    plt.ylabel('Total Numbers')
    plt.title(f'Total of Goals and Assists per Season of {player_most_contribution_name} until {season}-{season+1} season')
    plt.legend()

    plt.show()
