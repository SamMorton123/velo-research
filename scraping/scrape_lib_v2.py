'''
Consolidated API for scraping data from ProCyclingStats.
'''

from bs4 import BeautifulSoup
from datetime import datetime
import json
import os
import pandas as pd
import requests
import time
import sys
from tqdm import tqdm

# local
sys.path.append('/Users/samuelhmorton/indiv_projects/work/velosight/')
from db.cyclingdb import CyclingDB

# constants
DBNAME = 'cycling'  # name of cycling db
SLEEP = 5
RESULTS_TABLE_NAME = 'test12'

# pcs links
PCS_BASE = 'https://www.procyclingstats.com/'
RACES_LINK_BASE = 'https://www.procyclingstats.com/races.php'
RACES_LINK_TAIL = '&filter=Filter&p=me&s=races'
DATA_COLUMNS = [
    'name', 'stage', 'points_scale', 'parcours_type', 'year', 'month', 'day',
    'type', 'length', 'profile_score', 'avg_speed', 'vertical_meters', 'won_how',
    'place', 'rider', 'team', 'age', 'time'
]

RACE_SPEED_IDX = 2
POINTS_SCALE_IDX = 5
PARCOURS_TYPE_IDX = 6
PROFILE_SCORE_IDX = 7
VERT_IDX = 8

# ===== Public Methods ===== #
def get_soup(link, sleep = SLEEP):
    '''
    Handles the actual scraping from PCS, and forces
    all scraping methods to take a short pause after
    each url scrape, just in case PCS ever institutes
    some kind of limits to the amount people can
    scrape.
    '''

    # get the html
    page = requests.get(link)
    soup = BeautifulSoup(page.content, 'html.parser')

    time.sleep(sleep)
    return soup


def get_race_data(race_info, fname = 'main_dataset.csv'):
    '''
    Given the necessary data, scrape the results of races. race_info is a list
    of tuples in the format [(race name, race PCS link, whether or not the race is a one-day race (bool))].
    The user can also specify the table in which to save collected data.
    '''

    # load the dataset
    try:
        data = pd.read_csv(fname)
    except:
        data = pd.DataFrame(data = {col: [] for col in DATA_COLUMNS})
    new_data = {col: [] for col in DATA_COLUMNS}

    # loop through each tuple containing race info
    for i in tqdm(range(len(race_info))):

        # for tracking the time taken to scrape the current race
        t1 = datetime.today()

        # unpack data for current race
        name, link, one_day = race_info[i]
        
        # decide which data collection script to call
        # NOTE: code currently just saves race classification as 'NA'
        if one_day:
            _get_one_day_data(new_data, name, link, 'NA')
        else:
            _get_stage_race_data(new_data, name, link, 'NA')
        
        print(f'=== Collected data for {race_info[i][0]} in {datetime.today() - t1}.')
    
    data = pd.concat([data, pd.DataFrame(new_data)])
    data.drop_duplicates(inplace = True)
    data.to_csv(fname, index = False)

def get_from_table(table_name):
    '''
    Simple method to simply return the given table in the database as
    a Pandas DF.
    '''

    # init cycling db object
    cdb = CyclingDB(DBNAME)

    return cdb.get_data(f'SELECT * FROM {table_name};')

# ===== Helpers ===== #
def _create_races_link(year, classification):
    return f'{RACES_LINK_BASE}?year={year}&circuit=1&class={classification}&filter=Filter'

def _get_one_day_data(data, name, link, classification, stage_name = 'NA'):
    '''
    Given the race name and link, scrape the results of the one day
    race and add all the results to the given table name. Acts as a driver
    method for collecting data, as the processes for collecting data for
    both one day races and stage races go through this method.
    '''

    # get soup
    soup = get_soup(link)

    # get date
    try: (year, month, day) = _get_race_date(soup)
    except:
        print(f'{name} {year} {stage_name} not collected; date could not be retreived from soup.')
        return
    
    # get race type (e.g. itt, one day race, etc.)
    try: race_type = _get_race_type(soup, stage_name)
    except:
        print(f'{name} {year} {stage_name} not collected; race type could not be retreived from soup.')
        return
    
    # save placeholder if the race was a ttt, which aren't useful to me at this time
    if race_type == 'ttt':
        _insert_ttt_placeholder(data, name, stage_name, classification, year, month, day, race_type)
        return
    
    # get race distance
    try: race_distance = _get_race_distance(soup, stage_name)
    except:
        print(f'{name} {year} {stage_name} not collected; race distance could not be retreived from soup.')
        return
    
    # get profile score
    try: profile_score = _get_race_profile_score(soup, stage_name)
    except:
        print(f'{name} {year} {stage_name} not collected; profile score could not be retreived from soup.')
        return
    
    # get vertical meters
    try: vert = _get_vert(soup, stage_name)
    except:
        print(f'{name} {year} {stage_name} not collected; vert could not be retreived from soup.')
        return
    
    try: avg_speed = _get_avg_speed(soup)
    except:
        print(f'{name} {year} {stage_name} not collected; avg speed could not be retreived from soup.')
        return
    
    try: parcours_type = _get_parcours_type(soup, stage_name)
    except:
        print(f'{name} {year} {stage_name} not collected; parcours type could not be retreived from soup.')
        return
    
    try: points_scale = _get_points_scale(soup)
    except:
        print(f'{name} {year} {stage_name} not collected; points scale could not be retreived from soup.')
        return
    
    # get the race finish scenerio
    try: won_how = _get_race_finish_scenerio(soup, stage_name)
    except:
        print(f'{name} {year} {stage_name} not collected; race finish scenerio could not be retreived from soup.')
        return

    # get rows in the results table
    try:
        if stage_name == 'final-gc':
            table_rows = soup.find_all('tbody')[1].find_all('tr')
        else:
            table_rows = soup.find_all('tbody')[0].find_all('tr')
    except:
        print(f'{name} {year} {stage_name} not collected; results table rows could not be retreived from soup.')
        return
    

    # loop through each row in the table
    prev_time = None  # to ensure that all riders that finish on the same time are given the same time in results
    place = 0  # track the finish position of the riders
    for row in table_rows:

        # init dict to hold data for the current row
        row_data = {}

        # get rider data
        listed_place = _get_place(row)

        # if the place is -1, then skip this rider and move to the next row
        # NOTE: -1 is just a catchall to indicate that we shouldn't record this rider's result
        if listed_place == -1:
            continue
        else:
            place += 1

        # get finisher data from soup (errors handled in the helper methods themselves)
        rider = _get_rider(soup, row, stage_name)
        team = _get_team(soup, row, stage_name)
        age = _get_age(soup, row, stage_name)
        time = _get_score(soup, row, place, prev_time, stage_name, race_type)

        # populate row data
        row_data = {
            'name': name,
            'stage': stage_name,
            'points_scale': points_scale,
            'parcours_type': parcours_type,
            'year': year,
            'month': month,
            'day': day,
            'type': race_type,
            'length': race_distance,
            'profile_score': profile_score,
            'avg_speed': avg_speed,
            'vertical_meters': vert,
            'won_how': won_how,
            'place': place,
            'rider': rider,
            'team': team,
            'age': age,
            'time': time,
        }

        # add row to the db
        for key in row_data:
            data[key].append(row_data[key])

        # track the previous time for catching finishers who are given the same time
        prev_time = time

def _insert_ttt_placeholder(data, name, stage_name, classification, year, month, day, race_type):

    # create row data
    row_data = {
        'name': name,
        'stage': stage_name,
        'points_scale': 'NA',
        'parcours_type': 'NA',
        'year': year,
        'month': month,
        'day': day,
        'type': race_type,
        'vertical_meters': 'NA',
        'avg_speed': 'NA',
        'length': 'NA',
        'profile_score': 'NA',
        'won_how': 'NA',
        'place': 'NA',
        'rider': 'NA',
        'team': 'NA',
        'age': 'NA',
        'time': 'NA',
    }
    
    for key in row_data:
        data[key].append(row_data[key])

def _get_stage_race_data(data, name, link, classification):
    '''
    Driver for collecting stage race data. Primarily sets up params to call
    the method for collecting one day race data.
    '''

    # get soup
    soup = get_soup(link)

    # get list of stage names
    try:
        stages = _get_pages(soup)
    except:
        stages = None
        print(f'===== SKIPPED {name} =====')

    if stages is not None:
        for (stage_name, stage_link) in stages:
            print(f'Collecting stage {stage_name}.')
            _get_one_day_data(data, name, f'{PCS_BASE}{stage_link}', classification, stage_name = stage_name)

def _get_info_rows(soup):
    '''
    A few methods will all try to access the info panel in the
    top right of the race page. This has that functionality all
    go through one place.
    '''
    return soup.find_all('ul', class_ = 'infolist')[0].find_all('li')
    
def _get_race_date(soup):
    '''
    Extract the date of a race as a (year, month, day) tuple.
    '''

    # get info panel rows
    info_rows = _get_info_rows(soup)

    # long form date
    long_date = info_rows[0].find_all('div')[1].text

    # get date in YYYY-MM-DD
    year = int(long_date.split('-')[0]) if '-' in long_date else int(long_date.split(',')[0].split(' ')[-1])
    month = int(long_date.split('-')[1]) if '-' in long_date else datetime.strptime(long_date.split(',')[0].split(' ')[1], '%B').month
    day = int(long_date.split('-')[2]) if '-' in long_date else int(long_date.split(',')[0].split(' ')[0])
    return (year, month, day)

def _get_avg_speed(soup):

    # get info panel rows
    info_rows = _get_info_rows(soup)

    # get speed
    try:
        speed = info_rows[RACE_SPEED_IDX].find_all('div')[1].text
        return speed
    except:
        return 'NA'

def _get_points_scale(soup):

    # get info panel rows
    info_rows = _get_info_rows(soup)

    try:
        scale = info_rows[POINTS_SCALE_IDX].find_all('div')[1].find_all('a')[0].text
        return scale
    except:
        return 'NA'

def _get_race_type(soup, stage_name):
    '''
    Get the race type from the given soup.
    '''

    if 'gc' in stage_name:
        return 'gc'
    if 'points' in stage_name:
        return 'points'
    if 'mountains' in stage_name:
        return 'kom'

    # get the race type from the top left of the screen
    race_type = soup.find_all('span', class_ = 'blue')[0].text.lower()

    # race type is either 'prologue', 'itt', 'ttt', 'one-day-race', or 'standard'
    if 'prologue' in race_type:
        return 'prologue'
    elif 'itt' in race_type:
        return 'itt'
    elif 'ttt' in race_type:
        return 'ttt'
    elif 'one day race' in race_type:
        return 'one-day-race'
    else:
        return 'standard'

def _get_race_distance(soup, stage_name):
    '''
    Get the length of the race from the given soup.
    '''

    if 'gc' in stage_name:
        return 'NA'
    if 'points' in stage_name:
        return 'NA'
    if 'mountains' in stage_name:
        return 'NA'

    # get info panel rows
    info_rows = _get_info_rows(soup)

    # loop through each info row
    for row in info_rows:

        # get the divs available in the row; return 'NA' if there's an issue
        divs = row.find_all('div')
        if len(divs) < 2:
            return 'NA'
        
        # return the race distance
        if divs[0].text.strip().lower() == 'distance:':
            dist_w_units = divs[1].text
            return float(dist_w_units.split(' ')[0])
    
    return 'NA'

def _get_vert(soup, stage_name):

    if 'gc' in stage_name:
        return 'NA'
    if 'points' in stage_name:
        return 'NA'
    if 'mountains' in stage_name:
        return 'NA'

    # get info panel rows
    info_rows = _get_info_rows(soup)

    # get vert
    try:
        vert = int(info_rows[VERT_IDX].find_all('div')[1].text)
        return vert
    except:
        return 'NA'

def _get_parcours_type(soup, stage_name):

    if 'gc' in stage_name:
        return 'NA'
    if 'points' in stage_name:
        return 'NA'
    if 'mountains' in stage_name:
        return 'NA'

    # get info panel rows
    info_rows = _get_info_rows(soup)

    # get profile score
    try:
        ptype = info_rows[PARCOURS_TYPE_IDX].find_all('div')[1].find_all('span')[0]['class'][2]
        return ptype
    except:
        return 'NA'

def _get_race_profile_score(soup, stage_name):
    '''
    Get race profile score (quantification of course difficulty) from the
    given soup.
    '''

    if 'gc' in stage_name:
        return 'NA'
    if 'points' in stage_name:
        return 'NA'
    if 'mountains' in stage_name:
        return 'NA'

    # get info panel rows
    info_rows = _get_info_rows(soup)

    # get profile score
    try:
        pscore = int(info_rows[PROFILE_SCORE_IDX].find_all('div')[1].text)
        return pscore
    except:
        return 'NA'

def _get_race_finish_scenerio(soup, stage_name):
    '''
    Get the race finish scenerio (how it was won) from the
    given soup.
    '''


    if 'gc' in stage_name:
        return 'NA'
    if 'points' in stage_name:
        return 'NA'
    if 'mountains' in stage_name:
        return 'NA'

    # get info panel rows
    info_rows = _get_info_rows(soup)

    # loop through each info row
    for row in info_rows:

        # get all the divs in the row
        divs = row.find_all('div')
        if len(divs) < 2:
            return 'NA'
        
        # get the finish scenerio if it exists in the row
        if divs[0].text.strip().lower() == 'won how:':
            a_lst = row.find_all('a')
            if len(a_lst) == 1:
                return a_lst[0].text.strip().lower()
            else:
                return row.find_all('div')[1].text.strip().lower()
    
    return 'NA'

def _get_place(row):
    '''
    Get the rider finish place from the given row in the finish table.
    '''

    listed_place = row.find_all('td')[0]
    
    # return -1 if the rider was popped for doping
    if len(listed_place.find_all('s')) > 0:
        return -1
    
    # return -1 if there's a problem
    try:
        return int(listed_place.text)
    except:
        return -1

def _get_rider(soup, row, stage_name):
    '''
    Get the name of the rider in a given row.
    '''

    # get the position of the rider name within the row
    rider_idx = _get_row_index(soup, stage_name, 'rider')
    
    # if the rider position isn't found, return 'NA'
    if rider_idx is None:
        return 'NA'
    
    return row.find_all('td')[rider_idx].find_all('a')[0].text.strip()

def _get_team(soup, row, stage_name):
    '''
    Get the name of the team a rider is on.
    '''

    # get the position of the team name within each row
    team_idx = _get_row_index(soup, stage_name, 'team')
    
    if team_idx is None:
        return 'NA'
    
    # try to get the team name
    try:
        return row.find_all('td')[team_idx].find_all('a')[0].text.strip()
    except:
        return 'NA'

def _get_age(soup, row, stage_name):
    '''
    Get the age of the rider.
    '''

    # get the position of the rider's age in each row
    age_idx = _get_row_index(soup, stage_name, 'age')
    
    if age_idx is None:
        return 'NA'
    
    try:
        return row.find_all('td')[age_idx].text.strip()
    except:
        return 'NA'

def _convert_time_to_seconds(tm):
    '''
    Given a string timegap, convert it to seconds (as an int).
    '''

    # split time by colons
    tm_split = tm.split(':')

    if len(tm_split) == 2:
        return (int(tm_split[0]) * 60) + int(tm_split[1])
    else:
        return (int(tm_split[0]) * 3600) + (int(tm_split[1]) * 60) + int(tm_split[2])

def _get_score(soup, row, place, prev_time, stage_name, race_type):
    '''
    Get the time as in seconds as an int. Be mindful that times may be given as ",,",
    in which case they need to be put as equal to that above it.
    '''

    # account for points and KOM competitions
    if 'points' in stage_name:
        return int(row.find_all('td')[8].text.strip())
    if 'mountains' in stage_name:
        return int(row.find_all('td')[8].text.strip())

    if place == 1:
        return 0

    # get the listed time
    try:

        # get the location of the given time in the row
        time_idx = _get_row_index(soup, stage_name, 'time')
        
        if time_idx is None:
            return 'NA'
        listed_time = row.find_all('td')[time_idx].find_all('span')[0].text.strip()
        
        # if the listed time is ',,' set the time equal to that of the previous row
        if ',,' in listed_time:
            return prev_time
        else:
            return _convert_time_to_seconds(listed_time)
    
    except:
        return 'NA'

def _get_row_index(soup, stage_name, column):
    '''
    Given a column name, give the index it's found in the results table
    of a link.
    '''

    # specify which index the desired table is located in
    thead_idx = 1 if stage_name == 'final-gc' else 0

    # find the index of the given column, if it is present in the table
    trs = soup.find_all('thead')[thead_idx].find_all('th')
    for i in range(len(trs)):
        if trs[i].text.strip().lower() == column:
            return i
    return None

def _get_pages(soup):
    '''
    Get a list of the results pages included for a given race. For example,
    it might return 5 stages and a final-gc as the possible pages to scrape.
    '''

    # isolate the select widget
    select = soup.find_all('div', class_ = 'pageSelectNav')[1].find_all('select')[0]

    # get all the options inside the select widget
    options = select.find_all('option')

    # get a list in the form [(stage name, link)]
    stages = [
        (option.text.split(' | ')[0].lower().replace(' ', '-'), option['value'])
        for option in options if option.text not in [
            'Points classification', 'Mountains classification', 'Youth classification', 'Teams classification'
        ]
    ]

    return stages
