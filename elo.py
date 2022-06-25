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
ACCEPTED_RACE_TYPE_CATS = ['gc', 'one-day-race', 'itt', 'sprints']
ACCEPTED_GENDER_CATS = ['men', 'women']
VERBOSE = True
MIN_YEAR = 1995
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
RESULTS_DATA_PATH = 'data/men_velodata.csv' if gender == 'men' else 'data/women_velodata.csv'

# ===== Load settings for race type and gender ===== #
with open(f'data/settings/{gender}-{race_type}.json') as f:
    settings = json.load(f)
f.close()

# establish race alpha and beta
DECAY_ALPHA = settings['decay-alpha']
DECAY_BETA = settings['decay-beta']
TIMEGAP_MULT = settings['timegap-multiplier']
RACE_CLASSES = settings['race-classes']
WEIGHTS = settings['race-class-weights']
NEW_SEASON_REGRESS_WEIGHT = settings['new-season-regression']
TT_LENGTH_ADJUSTOR = settings['tt-length-adjustor'] if race_type == 'itt' else None
TT_VERT_ADJUSTOR = settings['tt-vert-adjustor'] if race_type == 'itt' else None

# results data
DATA = pd.read_csv(RESULTS_DATA_PATH)

utils.elo_driver(
    DATA, RACE_CLASSES, WEIGHTS, 
    beg_year, end_year, gender, race_type,
    timegap_multiplier = TIMEGAP_MULT,
    decay_alpha = DECAY_ALPHA,
    decay_beta = DECAY_BETA,
    new_season_regress_weight = NEW_SEASON_REGRESS_WEIGHT,
    given_tt_length_adjustor = TT_LENGTH_ADJUSTOR,
    given_tt_vert_adjustor = TT_VERT_ADJUSTOR,
    eval_races = [
        'tour-de-france', 'giro-d-italia', 'vuelta-a-espana',
        'paris-nice', 'tirreno-adriatico', 'dauphine', 'volta-a-catalunya',
        'itzulia-basque-country', 'tour-de-suisse'
    ], 
    save_results = True
)
