"""
Driver script for generating elo rating output.
"""

from datetime import date
import json
import pandas as pd
import sys

# local
from ratings import utils
from ratings.Velo import Velo

# ===== Script Parameters ===== #
ACCEPTED_RACE_TYPE_CATS = ['gc', 'one-day-race', 'itt']
ACCEPTED_GENDER_CATS = ['men', 'women']
VERBOSE = True
TIMEGAP_MULTIPLIER = None  # weight given to margin of victory
NEW_SEASON_REGRESS_WEIGHT = 0.4  # weight the degree to which rider scores converge to 1500 during off season
RAW_RESULT_NUM_PRINTED = 15  # number of finishers printed in raw data per race if VERBOSE = True
MIN_YEAR = 2007
MAX_YEAR = 2023
SAVE_RESULTS = False

# ===== Get Script Args ===== #
EXPECTED_ARGS = 5
if len(sys.argv) < EXPECTED_ARGS:
    raise Exception(f'Script expects {EXPECTED_ARGS - 1} user args. {EXPECTED_ARGS - 1} were given.')

# handle given year args
try: beg_year = int(sys.argv[1])
except: raise Exception(f'Given {beg_year} must be an integer.')
if beg_year < MIN_YEAR or beg_year > MAX_YEAR:
    raise Exception(f'Year must be between {MIN_YEAR} and {MAX_YEAR}. {beg_year} given.')
try: end_year = int(sys.argv[2])
except: raise Exception(f'Given {end_year} must be an integer.')
if end_year < MIN_YEAR or end_year > MAX_YEAR:
    raise Exception(f'Year must be between {MIN_YEAR} and {MAX_YEAR}. {end_year} given.')

# accept gender arg
gender = sys.argv[3]
if gender not in ACCEPTED_GENDER_CATS:
    raise Exception(f'Gender must be given as one of {ACCEPTED_GENDER_CATS} for data collection.')

# accept race type arg
race_type = sys.argv[4]
if race_type not in ACCEPTED_RACE_TYPE_CATS:
    raise Exception(f'Race type must be given as one of {ACCEPTED_RACE_TYPE_CATS} for data collection.')

# ===== Establish data paths for the script ===== #
WEIGHTS_PATH = 'data/race_weight_data.json'
RESULTS_DATA_PATH = 'data/men_velodata.csv' if gender == 'men' else 'data/women_velodata.csv'
RACE_CLASSES_PATH = 'data/men_races_data.json' if gender == 'men' else 'data/women_races_data.json'

# results data
DATA = pd.read_csv(RESULTS_DATA_PATH)

# store race class variables here... to be converted to actual weights using WEIGHTS variable
with open(RACE_CLASSES_PATH) as f:
    RACE_CLASSES = json.load(f)
f.close()
with open(WEIGHTS_PATH) as f:
    WEIGHTS = json.load(f)
f.close()

# init elo system variable
elo = Velo(decay_alpha = 1.5, decay_beta = 1.8)

# loop through each year in the gc data
for year in range(beg_year, end_year):
    
    # prepare and isolate data for the given year
    year_data = utils.prepare_year_data(DATA, year, race_type = race_type)
    if len(year_data) == 0:
        continue

    print(f'\n====={year}=====\n')
    
    # loop through each race in the current year's data
    for race in year_data['name'].unique():

        # if race is not contained within the weight data, skip
        if race not in RACE_CLASSES[race_type][str(year)]:
            continue
        
        stages_data = utils.prepare_race_data(year_data, race)
        for stage_data in stages_data:

            # get the stage number
            stage_name = stage_data['stage'].iloc[0]

            # get race weight
            race_weight = WEIGHTS[gender][race_type][str(RACE_CLASSES[race_type][str(year)][race])]
            
            # get the race's date as date object
            stage_date = utils.get_race_date(stage_data)

            # save the elo system in a dictionary
            elo.save_system(stage_date)
            
            # simulate the race and add it to the rankings
            elo.simulate_race(race, stage_data, race_weight, TIMEGAP_MULTIPLIER)
            
            # apply changes to rider elos
            elo.apply_all_deltas(race, race_weight, stage_date)
            
            if VERBOSE:

                # print raw results for the current race
                print(f'\n==={race} - {year} - {stage_name} - (weight = {race_weight})===\n')
                print(stage_data[['place', 'rider', 'time', 'team']].iloc[0: RAW_RESULT_NUM_PRINTED, :])

                # print the elo system after this race is added
                elo.print_system(year, min_rating = 1500)

            if SAVE_RESULTS:
                elo.save_system(stage_date)
    
    # regress scores back to the mean of 1500 at the start of each season
    if year < end_year - 1:
        elo.new_season_regression(year, regression_to_mean_weight = NEW_SEASON_REGRESS_WEIGHT)

if SAVE_RESULTS:
    elo.save_system_data(f'{race_type}_{gender}')
