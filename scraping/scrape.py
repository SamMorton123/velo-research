import json
import sys

# local
import scrape_lib_v2 as scrl
import utils

# arg checking constants
EXPECTED_N_ARGS = 4
MIN_YEAR = 2007
MAX_YEAR = 2023
ACCEPTED_RACE_TYPE_CATS = ['stage-races', 'one-day-races']
ACCEPTED_GENDER_CATS = ['men', 'women']

# argument structure as follows:
# (1) [year] (2) [gender] (2) [race type] (4...) [individual race names]
# begin and end year are 4-digit integers; range includes begin year, but excludes end year
# gender argument must be given as 'men' or 'women' for the purpose of race result scraping
# race type must be given as 'stage-race' or 'one-day'
# individual race names must be given in their hyphenated format as seen in pcs links
# for example, the tour de france would be given as tour-de-france
# ex call 1: python scrape.py 2022 2023 women one-day
# ex call 2: python scrape.py 2021 2022 men stage-race tour-de-france tirreno-adriatico
# if no individual race names are given, then the race names will be pulled
# from the race name data for the given gender, race type, and year
if len(sys.argv) < EXPECTED_N_ARGS:
    raise Exception(f'Expected at least {EXPECTED_N_ARGS} arguments.')

try: year = int(sys.argv[1])
except: raise Exception(f'Given {year} must be an integer.')
if year < MIN_YEAR or year > MAX_YEAR:
    raise Exception(f'Year must be between {MIN_YEAR} and {MAX_YEAR}. {year} given.')

gender = sys.argv[2]
if gender not in ACCEPTED_GENDER_CATS:
    raise Exception(f'Gender must be given as one of {ACCEPTED_GENDER_CATS} for data collection.')

race_type = sys.argv[3]
if race_type not in ACCEPTED_RACE_TYPE_CATS:
    raise Exception(f'Race type must be given as one of {ACCEPTED_RACE_TYPE_CATS} for data collection.')

# race names not checked individually because scraping code will skip them if they're not
# in the proper format
if len(sys.argv) > EXPECTED_N_ARGS:
    race_names = [sys.argv[i] for i in range(EXPECTED_N_ARGS, len(sys.argv))]
else:
    race_names = []


M_RACES_PATH = '../data/men_races_data.json'
W_RACES_PATH = '../data/women_races_data.json'
M_DATA_PATH = '../data/men_velodata.csv'
W_DATA_PATH = '../data/women_velodata.csv'


# script obtains list of races to scrape if they aren't given in args
if len(race_names) == 0:
    
    races_fname = M_RACES_PATH if gender == 'men' else W_RACES_PATH
    with open(races_fname) as f:
        # load data for race type
        race_data = json.load(f)[race_type][str(year)] 
    f.close()

    # add races from race data to race_names
    for race in race_data:
        race_names.append(race)


# set race type bool
race_type_bool = race_type == 'one-day-races'

# populate list to package data for input to PCS scraping driver
race_info = []
for race in race_names:
    race_info.append((race, utils.create_link(race, year), race_type_bool))

# run scraper
data_path = M_DATA_PATH if gender == 'men' else W_DATA_PATH
scrl.get_race_data(race_info, fname = data_path)
