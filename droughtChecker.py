import pandas as pd
import heapq
import duckdb
import math

#Takes input from statistically interesting performances (player -> (score, columns)) and checks how long it has been since
#the feat was last done.
#Returns the last player who performed the feat, and the date.
#This is then scored by a main class / function

#Returns results, along with the executed Query where clause to help write better text via AI
MIN_MINUTES = 5
BAD_MINUTES_ADDITION = 10
GOOD_IS_LOW = {"turnovers", "foulsPersonal", "defRating", "eDefRating"}


def check_drought(stats_dict, observation, total, zscores):
    #Checks how long since stat line last achieved, returns last player, and the dated

    #Expects observation to not exist within df.

    results = {} #personID key -> (personId (who last did it), date)
    wheres = {}

    for player in stats_dict.keys():
        where = build_where_query(player, stats_dict[player][1], observation, zscores)
        result = duckdb.sql(f"SELECT personId, gameDateTimeEst FROM total {where} ORDER BY gameDateTimeEst DESC LIMIT 2").df()
        if result.empty:
            results[player] = None
            wheres[player] = where
        else:
            #Check if first entry matches observation. If it does, move on to the second entry.
            if observation[observation["personId"] == player]["gameDateTimeEst"].iloc[0] == result["gameDateTimeEst"].iloc[0]:
                if len(result) > 1:
                    #They match, use the second entry
                    results[player] = (result["personId"].iloc[1], result["gameDateTimeEst"].iloc[1])
                    wheres[player] = where
                else:
                    #didn't return a long enough result.

                    results[player] = None  
                    wheres[player] = where
            else:
                #They don't match, use the first entry
                results[player] = (result["personId"].iloc[0], result["gameDateTimeEst"].iloc[0])
                wheres[player] = where

    return results, wheres

def check_num_times_done(stats_dict, observation, total, zscores):
    #Checks how many times a specific statline has been achieved and returns count

    counts = {}
    wheres = {}

    for player in stats_dict.keys():
        where = build_where_query(player, stats_dict[player][1], observation, zscores)
        result = duckdb.sql(f"SELECT COUNT(*) FROM total {where}").df()
        if result.empty:
            counts[player] = 0
            wheres[player] = where
        else:
            counts[player] = result.iloc[0,0]
            wheres[player] = where

    return counts, wheres

def get_previous_players(stats_dict, observation, total, zscores):
    #Gets a list of distinct players who have achieved the stat line and returns player names list

        players = {}
        wheres = {}

        for player in stats_dict.keys():
            where = build_where_query(player, stats_dict[player][1], observation, zscores)
            result = duckdb.sql(f"SELECT DISTINCT(personId) FROM total {where}").df()
            if result.empty:
                players[player] = None
                wheres[player] = where
            else:
                players[player] = result["personId"].to_list()
                wheres[player] = where

        return players, wheres

def build_where_query(personId, columns, observation, zscores):
    #Builds a duckDB query based on observation and columns.
    #Think about how to handle comparisons though. Sometimes we may want points >= observed points.
    #Other times, we may want to say this many assists, rebounds, with less than x points so it would be <= observed points.
    #Can use z scores to determine this. If below average / above can map to operations. Still points would be opp of turnovers, etc.
    observation_values = observation[observation["personId"] == personId].iloc[0]
    zscore_values = zscores[zscores["personId"] == personId].iloc[0]

    operations = map_columns_to_operations(columns, zscore_values)
    #Check if any of the columns have 0
    #If column is 0, make operation "==", not <= or >=.
    bad = False
    #z score low, col in good is low, <= (good)
    #z score low, col not in good is low, <= (bad)
    #z score high, col in good is low >= (bad)
    #z score high, col not in good is low >= (good)
    for i, column in enumerate(columns):
        #print(observation_values[column])
        if observation_values[column] == 0:
            operations[i] = "=="
    for column in columns:
        if column in GOOD_IS_LOW:
            if zscore_values[column] > 0:
                bad = True
        else:
            if zscore_values[column] < 0:
                bad = True
    
    query = "WHERE"
    for i, column in enumerate(columns):
        query += f" {column} {operations[i]} {observation_values[column]} AND"
    min_minutes = MIN_MINUTES
    if bad:
        min_minutes = MIN_MINUTES+BAD_MINUTES_ADDITION
    query += f" numMinutes >= {min_minutes}"
    return query

def map_columns_to_operations(columns, zscores):
    #Takes z scores, columns, personId to figure out what operations to use in SQL. (==, <=, >=)
    #Needs input df from before abs value (as compared to population, not player)

    #Starting with a simple version. If Z score < 0, then assume it is bad. so, use <= (works for points/ turnovers)
    
    operations = []
    for column in columns:
        #z score low, col in good is low, <= (good)
        #z score low, col not in good is low, <= (bad)
        #z score high, col in good is low >= (bad)
        #z score high, col not in good is low >= (good)
        operator = ">=" if zscores[column] >= 0 else "<="
        operations.append(operator)
    return operations


    #Ultimately, not sure we ever need to stray from the z score >= 0 vs not method.
    #For instance, if z score for points > 0 - find players who scored a lot. Vice versa for zscore <0
    #For turnovers, if z score is > 0, find players who turned the ball over a lot. Turnovers shouldn't
    #Make it into the interesting columns unless the player had a LOT of turnovers. So, treat it as a meme stat.