Datasets link: https://www.kaggle.com/datasets/davidcariboo/player-scores/data?select=games.csv

The Data consists of 9 CSV files that record data about football players and clubs including the data from matches that occur with them. 

I will use these files to extract important information about the team's performance. With the code I provide in Python, you can choose any clubs and seasons since the 2012-2023 season due to the dates available among files.

You can set the variable name of the club and the season you are interested in. For example: club_name = "Tottenham" and season = 2023. The name doesn't need to be the full name because we later use the .contains() function to get the full club name and id from it.

Throughout this analysis you will be able to understand the following:

1. Clubs' main information: Link on Transfermarkt, Name, Stadium, Coach, Total Market Value...

2. Players's information of that season

3. Goals and assists analysis

4. Games results and position in the table

5. Cards analysis and how it affects the game's result (The card distribution plot only works with the current season due to the lack of card data in the 'games_lineup.csv' file which only has card data in the current season).

6. Analysis of players with the most contributions in the season

Also, remember to adjust the file directory path correctly ("Club_data.py")

I also provide an attached Power BI report which is dynamic to choose clubs and season, in which you just need to choose the clubs filter for all pages (currently Tottenham) and type the name of the club you're interested in and use slicer in Overview page to filter the season.

