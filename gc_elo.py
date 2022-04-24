"""
Driver script for generating elo rating output for GC results.
"""

from datetime import date
import json
import pandas as pd

# local
import utils
from ratings.CyclElo import Velo
import ratings.rider_selection as rs

# ===== Script Parameters ===== #
VERBOSE = True
RESULTS_DATA_PATH = 'data/men_velodata.csv'
RACE_WEIGHTS_PATH = 'data/men_races_data.json'  # path to weights for each race
TIMEGAP_MULTIPLIER = 1  # weight given to margin of victory
NEW_SEASON_REGRESS_WEIGHT = 0.4  # weight the degree to which rider scores converge to 1500 during off season
RIDER_SELECTION_METHOD = rs.select_all  # how the system determines which riders to save/print data for in the system
RAW_RESULT_NUM_PRINTED = 15  # number of finishers printed in raw data per race if VERBOSE = True

# weight classes taken from the RACE_WEIGHTS_PATH file, and these values give the actual weights for
# the 12 (slightly) subjective race classifications I use
WEIGHT_CLASSES = {
    0: 25,  # tour
    1: 17.5,  # giro/vuelta
    2: 7.5,  # dauphine/suisse
    3: 7,  # paris-nice/tirreno
    4: 6,  # mid level world tour stage race (basque country, catalunya)
    5: 5,  # lower world tour stage race (with climbs)
    6: 3.5,  # lower world tour stage race (w/o major climbs)
    7: 2.5,  # top level pro race (algarve, provence, tour of the alps)
    8: 2,  # misc .1 or .pro which isn't good enough to be a 7
    9: 1.25,
    10: 1,
    11: 0.5,
    12: 0.25  # stage race, but no climbs or generally uncompetitive and thus not particularly informative
}


# ===== Script Variable Setup ===== #

# results data
DATA = pd.read_csv(RESULTS_DATA_PATH)

# store race weight variables here... to be converted to actual weights using WEIGHT_CLASSES variable
with open(RACE_WEIGHTS_PATH) as f:
    RACE_WEIGHTS = json.load(f)
f.close()

# init elo system variable
elo = Velo()

# loop through each year in the gc data
begin_year = 2022
end_year = 2023  # doesn't include this year
for year in range(begin_year, end_year):
    
    print(f'\n====={year}=====\n')
    
    # prepare and isolate data for the given year
    year_data = utils.prepare_year_data(DATA, year, typ = 'gc')
    
    # loop through each race in the current year's data
    for race in year_data['name'].unique():

        # if race is not contained within the weight data, skip
        if race not in RACE_WEIGHTS['stage-races'][str(year)]:
            continue

        # prepare and isolate data for the given race
        race_data = utils.prepare_race_data(year_data, race)

        # get race weight
        race_weight = WEIGHT_CLASSES[RACE_WEIGHTS['stage-races'][str(year)][race]]
        
        # get the race's date as date object
        race_date = utils.get_race_date(race_data)

        # save the elo system in a dictionary
        elo.save_system(race_date)
        
        # simulate the race and add it to the rankings
        rider_order = elo.simulate_race(race, race_data, race_weight, TIMEGAP_MULTIPLIER)
        
        # apply changes to rider elos
        elo.apply_all_deltas(race, race_weight, race_date)
        
        if VERBOSE:
            
            # print raw results for the current race
            print(f'\n==={race} - {year} (weight = {race_weight})===\n')
            print(race_data[['place', 'rider', 'time', 'team']].iloc[0: RAW_RESULT_NUM_PRINTED, :])

            # print the elo system after this race is added
            elo.print_system(year, RIDER_SELECTION_METHOD, min_rating = 1500)
        
        elo.save_system(race_date)
    
    # regress scores back to the mean of 1500 at the start of each season
    if year < end_year - 1:
        elo.new_season_regression(year, regression_to_mean_weight = NEW_SEASON_REGRESS_WEIGHT)

elo.save_system_data('gc_men')
