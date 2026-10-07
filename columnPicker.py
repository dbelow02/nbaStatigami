import pandas as pd
import heapq
import sys
from pathlib import Path
from droughtChecker import check_drought, check_num_times_done, get_previous_players, MIN_MINUTES
from saveResults import saveResults, saveToCloud
from datetime import datetime, timedelta
from pdf import PDF, PDFCollector
from scipy import stats
import numpy as np
from statsToText import getLLMText
from publish import send_headline_email
import os


COOL_THRESHOLD = 8 #5 for basic, #6 for interesting, #7 for impressive
MAX_COLUMNS = 8
MAX_STAT_LINES = 20
COLUMN_MULTIPLIER = 0.5
PLAYER_ZSCORE_WEIGHT = 0.2
POPULATION_ZSCORE_WEIGHT = 0.8
FIT_PDFS_FLAG = False
DEBUG = False
METHOD = "AWS"

DATA_DIR = os.environ.get("NBA_DATA_DIR", r"D:\PythonProjects2026\NBA\Data")
COMBINED_PATH = os.path.join(DATA_DIR, "combineddatasetPOPULATED.csv")
MODEL_SAVE_PATH = os.environ.get("NBA_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

#PERCENTILE_COLUMNS = ["blocks", "steals", "fieldGoalsMade", "fieldGoalsAttempted","threePointersAttempted", "threePointersMade", "freeThrowsAttempted", "freeThrowsMade", "reboundsTotal", "reboundsOffensive", "reboundsDefensive"]
#TRIM_ZERO_COLUMNS = ["efgPct", "tsPct", "fieldGoalsPercentage", "threePointersPercentage", "freeThrowsPercentage"]

#Will also need to handle these in the operations portion of drought checker.
#Remove percentile columns from z score columns global, then add them back in in a new combine function
#Also make it so that in droughtChecker, blocks, steals, points are always >= instead of rely on z score to determine

COLUMN_TO_PDF = {
    # Discrete counting stats — overdispersed (variance > mean), zero-inflated
    # Negative binomial handles the long right tail and excess zeros
    "blocks":                   "nbinom",
    "steals":                   "nbinom",
    "threePointersMade":        "nbinom",
    "threePointersAttempted":   "nbinom",
    "reboundsOffensive":        "nbinom",
    "turnovers":                "nbinom",
    "foulsPersonal":            "nbinom",
    "fieldGoalsMade":           "nbinom",
    "fieldGoalsAttempted":      "nbinom",
    "freeThrowsMade":           "nbinom",
    "freeThrowsAttempted":      "nbinom",
    "reboundsDefensive":        "nbinom",
    "reboundsTotal":            "nbinom",

    # Points and assists — discrete but wide range, roughly normal at scale
    # Normal is a reasonable approximation here
    "points":                   "gamma",
    "assists":                  "nbinom",
    "numMinutes":               "normal",
    "plusMinusPoints":          "normal",  # can be negative, so normal is the only real option

    # Percentage stats defined on [0,1] — beta is the natural fit
    # Caller is responsible for filtering zeros (TRIM_ZERO_COLUMNS) before fitting
    "fieldGoalsPercentage":     "beta",
    "threePointersPercentage":  "beta",
    "freeThrowsPercentage":     "beta",
    "efgPct":                   "gamma",
    "tsPct":                    "gamma",
    "orebPct":                  "beta",
    "usgPct":                   "beta",
    "eUsgPct":                  "beta",
    "pie":                      "normal",

    # Proportion/percentage breakdown stats — all on [0,1]
    "pctAst2Pm":                "beta",
    "pctAst3Pm":                "beta",
    "pctAstFgm":                "beta",
    "pctFga2Pt":                "beta",
    "pctFga3Pt":                "beta",
    "pctPts2Pt":                "beta",
    "pctPts2PtMr":              "beta",
    "pctPts3Pt":                "beta",
    "pctPtsFb":                 "beta",
    "pctPtsFt":                 "beta",
    "pctPtsOffTov":             "beta",
    "pctPtsPaint":              "beta",
    "pctUast2Pm":               "beta",
    "pctUast3Pm":               "beta",
    "pctUastFgm":               "beta",

    # Advanced ratings — continuous, roughly normal, can be negative
    "netRating":                "normal",
    "eNetRating":               "normal",
    "offRating":                "normal",
    "eOffRating":               "normal",
    "defRating":                "normal",
    "eDefRating":               "normal",

    # Pace/possession — continuous, right-skewed, strictly positive -> gamma
    "poss":                     "normal",
}

COLUMN_WEIGHTS = {
    # Core box score — highest weight, most intuitive "wow" stats
    "points":                   1.0,
    "assists":                  0.8,
    "reboundsTotal":            0.8,
    "steals":                   0.9,
    "blocks":                   0.8,
    "plusMinusPoints":          0.9,

    # Shooting efficiency — important but secondary to raw production
    "fieldGoalsPercentage":     0.6,
    "threePointersPercentage":  0.45,
    "freeThrowsPercentage":     0.4,
    "efgPct":                   0.6,
    "tsPct":                    0.6,

    # Volume shooting — less exciting than efficiency
    "fieldGoalsAttempted":      0.4,
    "fieldGoalsMade":           0.5,
    "threePointersAttempted":   0.4,
    "threePointersMade":        0.6,
    "freeThrowsAttempted":      0.3,
    "freeThrowsMade":           0.4,

    # Rebounds breakdown — total already captured above
    "reboundsDefensive":        0.4,
    "reboundsOffensive":        0.6,  # offensive boards are more exciting

    # Turnovers/fouls — negative contribution to "cool"
    "turnovers":               0.5,
    "foulsPersonal":           0.1,

    # Advanced ratings — useful signal but less legible
    "netRating":                0.5,
    "eNetRating":               0.5,
    "offRating":                0.4,
    "eOffRating":               0.4,
    "defRating":                0.4,
    "eDefRating":               0.4,

    # Usage/pace — contextual, low direct weight
    "numMinutes":               0.3,
    "usgPct":                   0.3,
    "eUsgPct":                  0.2,
    "poss":                     0.2,
    "pie":                      0.5,
    "orebPct":                  0.2,

    # Assist/shot distribution breakdowns — low, redundant with assists/shooting
    "pctAst2Pm":                0.1,
    "pctAst3Pm":                0.1,
    "pctAstFgm":                0.1,
    "pctFga2Pt":                0.1,
    "pctFga3Pt":                0.1,
    "pctPts2Pt":                0.1,
    "pctPts2PtMr":              0.1,
    "pctPts3Pt":                0.15,
    "pctPtsFb":                 0.2,  # fastbreak points are exciting
    "pctPtsFt":                 0.15,
    "pctPtsOffTov":             0.2,  # points off turnovers also exciting
    "pctPtsPaint":              0.1,
    "pctUast2Pm":               0.1,
    "pctUast3Pm":               0.1,
    "pctUastFgm":               0.1,
}

def getYesterday(date: datetime.date) -> datetime.date:
    return date - timedelta(days=1)

def cdf_to_zscore(cdf_value):
    # Converts percentile [0,1] to a z-score-like value [-inf, inf]
    # 0.5 -> 0, 0.95 -> 1.645, 0.99 -> 2.326
    cdf_value = np.clip(cdf_value, 1e-6, 1 - 1e-6)  # avoid log(0)
    return stats.norm.ppf(cdf_value)

def get_zscore_all(observation, total, columns):
    #Observation: a subset df of total
    #Total: all data
    #Columns: subset of columns to perform Z-score on
    #Returns a new, symmetrical df with the zscore of each column for each player / gameid
    
    historical_means = total[columns].mean()
    historical_stds = total[columns].std().replace(0, float("nan")) #need the nan to avoid div by 0 later on.

    observation_z_scores = (observation[columns] - historical_means) / historical_stds
    observation_z_scores = observation_z_scores.fillna(0)
    observation_z_scores["personId"] = observation["personId"]
    return observation_z_scores


def get_pdf_scores_all(observation, PDFS, columns):
    #Observation: a subset df of total
    #PDFs: a PDFCollector object with fitted PDFs for each column
    #Columns: subset of columns to perform PDF scoring on
    #Returns a new, symmetrical df with the pdf score of each column for each player / gameid

    pdf_scores = pd.DataFrame(index=observation.index)

    for col in columns:

        pdf_scores[col] = observation[col].apply(
            lambda x: cdf_to_zscore(PDFS[col].getProbability(x))
        )

    pdf_scores = pdf_scores.fillna(0)
    pdf_scores["personId"] = observation["personId"]
    return pdf_scores

def get_zscore_player(observation, total, columns):
    #Observation: a subset df of total
    #Total: all data
    #Columns: subset of columns to perform Z-score on
    #Returns a new, symmetrical df with the zscore of each column for each player / gameid

    historical_means_by_player = total.groupby("personId")[columns].agg("mean")
    historical_stds_by_player = total.groupby("personId")[columns].agg("std").replace(0, float("nan"))

    #Check observation contains more than one game per player
    if observation["personId"].duplicated().any():
        print(observation)
        raise ValueError("Passed dataframe for player zscores contains more than one entry per player.")
    
    player_ids = observation["personId"]
    historical_means_by_player = historical_means_by_player.loc[player_ids].values
    historical_stds_by_player = historical_stds_by_player.loc[player_ids].values
    #Need to be .values since the indexes don't match. However, we know they should line up!
    #Trim to only the id's we care about too.
    observation_z_scores = (observation[columns].values - historical_means_by_player) / historical_stds_by_player
    observation_z_scores_df = pd.DataFrame(observation_z_scores, columns=columns, index=observation.index)

    observation_z_scores_df = observation_z_scores_df.fillna(0)
    observation_z_scores_df["personId"] = observation["personId"]
    #Here, need to join back player Id's
    return observation_z_scores_df


def create_new_PDFs(columns: list):
    PDFS = PDFCollector()
    featureDict = {}
    for column in columns:
        #dict[featurename] = (newBool, filepath, filename)
        featureDict[column] = (True, MODEL_SAVE_PATH, f"{column}.pkl")
    PDFS.load(featureDict)
    return PDFS

def fit_PDFs(PDFS: PDFCollector, total: pd.DataFrame, columns: list):
    #First, trim all data for players who played 0 minutes. Don't want to include.
    totalTrimmed = total[total["numMinutes"] > MIN_MINUTES]
    totalTrimmed = totalTrimmed[(totalTrimmed["eUsgPct"] <= 1) | (totalTrimmed["eUsgPct"].isna())]
    totalTrimmed = totalTrimmed[(totalTrimmed["pace"] <= 250) | (totalTrimmed["pace"].isna())]
    for column in columns:
        PDFS[column].setData(totalTrimmed[column])
        PDFtype = COLUMN_TO_PDF[column]
        PDFS[column].setPDFType(PDFtype)
        PDFS[column].fit()
    return

def load_PDFs(columns: list):
    PDFS = PDFCollector()
    featureDict = {}
    for column in columns:
        #dict[featurename] = (newBool, filepath, filename)
        featureDict[column] = (False, MODEL_SAVE_PATH, f"{column}.pkl")
    PDFS.load(featureDict)
    return PDFS

def pick_columns(observation, total, all_z_scores, player_z_scores):
    #MINIMIZE NUMBER OF COLUMNS TO HIT THRESHOLD OF RARITY
    #all_z_scores = get_zscore_all(observation, total, Z_SCORE_COLUMNS)
    #player_z_scores = get_zscore_player(observation, total, Z_SCORE_COLUMNS)
    if player_z_scores.empty:
        all_z_scores = all_z_scores.drop(columns="personId")
        z_scores = all_z_scores.abs() + 1
    else:
        all_z_scores = all_z_scores.drop(columns="personId")
        player_z_scores = player_z_scores.drop(columns="personId")
        z_scores = ((POPULATION_ZSCORE_WEIGHT * all_z_scores.abs()) + (PLAYER_ZSCORE_WEIGHT * player_z_scores.abs())) + 1
        # Take average of all z scores and then player z score too. Want emphasis on bad player playing good.

    #Now, apply weights to each of the columns.
    weight_series = pd.Series(COLUMN_WEIGHTS)

    scores = z_scores.mul(weight_series).abs()
    z_scores["personId"] = observation["personId"]
    scores["personId"] = observation["personId"]





    #Now, handle percentages like fg% when 0 shots taken. Fix score to be 0
    scores.loc[observation["freeThrowsAttempted"].values == 0, "freeThrowsPercentage"] = 0
    scores.loc[observation["fieldGoalsAttempted"].values == 0, "fieldGoalsPercentage"] = 0
    scores.loc[observation["threePointersAttempted"].values == 0, "threePointersPercentage"] = 0

    scores = scores[observation["numMinutes"] >= MIN_MINUTES]

    #Eventually, scores will be a combination of several models, not just a zscore.
    #Eventually, scores don't necessarily even need to be column wise.

    #For example, a certain player could have a higher score coefficient at all times
    #(Lebron James, Wemby, etc. vs some rando)

    #Now, use scores to pick the interesting columns needed for each row.
    #If length of columns needed > 8, we don't want it.

    #Eventually, can use some sort of regression model to pick stat lines. (need engagement / results first)
    #Or, could use an XG boost on labeled "cool" or "not cool" data
    columns_per_player = {}
    #Bring back personId now (can guarantee observation is aligned on same index)

    for player in scores["personId"].to_list(): 
        score = 0
        selected_columns = []
        player_scores = scores[scores["personId"] == player].drop(columns="personId").iloc[0]
        top_columns_player = player_scores.nlargest(MAX_COLUMNS)  # Series of col -> score, sorted desc        
        i = 0
        while score < COOL_THRESHOLD and i < len(top_columns_player):
            col = top_columns_player.index[i]
            selected_columns.append(col)
            score += top_columns_player.iloc[i]
            i += 1
        columns_per_player[player] = (score, selected_columns)
            
    topstatlinesheap = []
    for key in columns_per_player.keys():
        combined_score = columns_per_player[key][0] - (COLUMN_MULTIPLIER * len(columns_per_player[key][1]))
        if len(topstatlinesheap) > MAX_STAT_LINES:
            #Check if score > min score in heap.
            #If >, kick from heap and replace with new score.
            #print(topstatlinesheap[0])
            #print(combined_score)
            if combined_score > topstatlinesheap[0][0]:
                #New score > min score in heap.
                heapq.heappop(topstatlinesheap)
                heapq.heappush(topstatlinesheap, (combined_score, key))
            else:
                #New score still less than min in heap, continue
                continue
        else:
            #Heap not full yet, push player in.
            heapq.heappush(topstatlinesheap, (combined_score, key))
        
        #Need to weight high scores with low columns. So, add regularization to num columns.

    top_players = sorted(topstatlinesheap, reverse=True)
    #Now, reconstruct the players -> columns dict.
    players_out = {}
    for tuple in top_players:
        combined_score = tuple[0]
        columns = columns_per_player[tuple[1]][1]
        players_out[tuple[1]] = (combined_score, columns)
    
    return players_out

def printTopStatlines(player_columns, observation):
    #Takes input dict where key is personId, value[0] is model score, value[1] is columns.
    if observation["personId"].duplicated().any():
        print("Passed dataframe for printing player statlines contains more than one entry per player.")
        return
    for key in player_columns:
        firstname = observation[observation["personId"] == key]["firstName"]
        lastname = observation[observation["personId"] == key]["lastName"]
        print(f"{firstname} {lastname} scored {player_columns[key][0]}")
        print(f"With columns: {player_columns[key][1]}")
        desired_columns = observation[observation["personId"] == key][player_columns[key][1]]
        print(desired_columns)
        print()
    return


def printDroughts(drought_dict, count_dict, observation, total):
    if observation["personId"].duplicated().any():
        print("Passed dataframe for printing droughts contains more than one entry per player.")
        return

    for player, info in drought_dict.items():
        firstname = observation[observation["personId"] == player]["firstName"].iloc[0]
        lastname = observation[observation["personId"] == player]["lastName"].iloc[0]
        count = count_dict[player]

        if info is None:
            print(f"{firstname} {lastname} has never had this statline before. (achieved {count} times historically)")
            print()
            continue

        prev_player_id, date = info

        prev_player_row = total[total["personId"] == prev_player_id]
        prev_firstname = prev_player_row["firstName"].iloc[0]
        prev_lastname = prev_player_row["lastName"].iloc[0]

        print(f"{firstname} {lastname}'s statline was last achieved by {prev_firstname} {prev_lastname} on {date}. (achieved {count} times historically)")
        print()

    return

def printStatlineReport(player_columns, drought_dict, count_dict, observation, total, wheres):
    if observation["personId"].duplicated().any():
        raise ValueError("Passed dataframe for printing contains more than one entry per player.")

    for key in player_columns:
        firstname = observation[observation["personId"] == key]["firstName"].iloc[0]
        lastname = observation[observation["personId"] == key]["lastName"].iloc[0]
        score = player_columns[key][0]
        columns = player_columns[key][1]
        count = count_dict[key]
        where = wheres[key]

        print(f"{firstname} {lastname} scored {score}")
        print(f"With columns: {columns}")
        desired_columns = observation[observation["personId"] == key][columns]
        print(desired_columns)

        info = drought_dict[key]
        print(f"With a SQL where clause of {where}")
        if info is None:
            print(f"{firstname} {lastname} achieved a statline never done before. (achieved {count} times historically)")
        else:
            prev_player_id, date = info
            prev_player_row = total[total["personId"] == prev_player_id]
            prev_firstname = prev_player_row["firstName"].iloc[0]
            prev_lastname = prev_player_row["lastName"].iloc[0]
            print(f"This statline was last achieved by {prev_firstname} {prev_lastname} on {date}. (achieved {count} times historically)")

        print()

    return


def buildStatlineReport(player_columns, drought_dict, count_dict, observation, total, wheres):
    if observation["personId"].duplicated().any():
        raise ValueError("Passed dataframe for printing contains more than one entry per player.")

    lines = []

    for key in player_columns:
        firstname = observation[observation["personId"] == key]["firstName"].iloc[0]
        lastname = observation[observation["personId"] == key]["lastName"].iloc[0]
        score = player_columns[key][0]
        columns = player_columns[key][1]
        count = count_dict[key]
        where = wheres[key]

        lines.append(f"{firstname} {lastname} scored {score}")
        lines.append(f"With columns: {columns}")
        desired_columns = observation[observation["personId"] == key][columns]
        lines.append(desired_columns.to_string())

        info = drought_dict[key]
        lines.append(f"With a SQL where clause of {where}")
        if info is None:
            lines.append(f"{firstname} {lastname} achieved a statline never done before. (achieved {count} times historically)")
        else:
            prev_player_id, date = info
            prev_player_row = total[total["personId"] == prev_player_id]
            prev_firstname = prev_player_row["firstName"].iloc[0]
            prev_lastname = prev_player_row["lastName"].iloc[0]
            lines.append(f"This statline was last achieved by {prev_firstname} {prev_lastname} on {date}. (achieved {count} times historically)")

        lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":

    combined = pd.read_csv(COMBINED_PATH, dtype={'gameId': str})
    date = getYesterday(datetime.today().date())
    dates = pd.to_datetime(combined["gameDateTimeEst"], format="mixed").dt.date
    observation = combined[dates == date]
    if len(observation) == 0:
        if DEBUG:
            print(f"No games on {date}; using Finals game 5 for testing.")
            observation = combined[pd.to_numeric(combined["gameId"]) == 42500405]
        else:
            print(f"No games on {date}; nothing to send.")
            sys.exit(0)

    #print("Creating PDFS")
    #PDFS = create_new_PDFs(list(COLUMN_TO_PDF.keys()))
    print("Loading PDFS")
    PDFS = load_PDFs(list(COLUMN_TO_PDF.keys()))
    if FIT_PDFS_FLAG:
        print("Fitting PDFS (flip FIT_PDFS_FLAG if don't want to retrain.)")
        fit_PDFs(PDFS, combined, list(COLUMN_TO_PDF.keys()))

    all_z_scores = get_pdf_scores_all(observation, PDFS, list(COLUMN_TO_PDF.keys()))


    player_z_scores = get_zscore_player(observation, combined, list(COLUMN_TO_PDF.keys()))
    results = pick_columns(observation, combined, all_z_scores, player_z_scores)
    if not results:
        print("No statlines selected; nothing to send.")
        sys.exit(0)
    #printTopStatlines(results, observation)
    time1 = datetime.now()
    drought_dict, drought_wheres = check_drought(results, observation, combined, all_z_scores)
    time2 = datetime.now()
    print(f"It took {time2-time1} to check drought")
    numtimesdone_dict, numtimesdone_wheres = check_num_times_done(results, observation, combined, all_z_scores)
    time1 = datetime.now()
    print(f"It took {time1-time2} to check number of times done")
    previousplayers_dict, previousplayers_wheres = get_previous_players(results, observation, combined, all_z_scores)
    time2 = datetime.now()
    print(f"It took {time2-time1} to check previous players")
    #printDroughts(drought_dict, numtimesdone_dict, observation, combined)

    #All wheres dicts should be the same.

    printStatlineReport(results, drought_dict, numtimesdone_dict, observation, combined, drought_wheres)
    statlineOutput = buildStatlineReport(results, drought_dict, numtimesdone_dict, observation, combined, drought_wheres)

    finalPosts, posts = getLLMText(statlineOutput)
    print(finalPosts)
    if finalPosts is not None:
        if METHOD == "EMAIL":
            send_headline_email(body=finalPosts)
        elif METHOD == "AWS":
            path = saveResults(posts)
            saveToCloud(path)
        else:
            print("Unknown method! Not sending results anywhere")
    else:
        print("finalPosts from LLM output is None. Double check that the model returned an answer and that the answer was formatted correctly.")
    if FIT_PDFS_FLAG:
        print("Saving PDFs:")
        PDFS.save()
    





#Why do 0 steals, 0 blocks keep getting added in? Shouldn't be that much z score
#Both are extremely logarithmic (many many many at 0, not so many after.)
#Low standard deviation, high weight = biased towards

#Likely has to do with the following feature weights:

#Can use percentile on non-normal distribution columns? - long term
#Otherwise, can mask out columns where 0 doesn't matter (just wipe the z score to 0 if blocks = 0)

#Blocks is giving >= instead of <=, why?
