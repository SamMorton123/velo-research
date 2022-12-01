"""
Support module for Elo system.
"""

from bs4 import BeautifulSoup
from datetime import date
import json
import math
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from tqdm import tqdm

# local
from ratings import RaceSelection
from ratings.Velo import Velo

PLACES_COL = 'place'
RIDER_COL = 'rider'
TIME_COL = 'time'
LENGTH_COL = 'length'
VERT_COL = 'vertical_meters'
WORLD_OLYMPICS_ITT_NAMES = [
    'world-championship-itt', 'olympic-games-itt',
    'world-championship-itt-we', 'olympic-games-we-itt'
]
RAW_RESULT_NUM_PRINTED = 15  # number of finishers printed in raw data per race if VERBOSE = True
NEW_SEASON_REGRESS_WEIGHT = 0.4  # weight the degree to which rider scores converge to 1500 during off season
EVAL_COLLECTION_LIM = 100  # amount of riders considered when evaluating predictive performance of rankings
SPRINT_NUM_SAME_TIME_FINISH_THRESH = 30


def elo_driver(main_data, race_data, settings):
    """
    Given necessary params, init an Elo system and run through all the applicable data,
    tracking changes to rider ratings over time. Returns an Elo object.

    This method is a little long; need to modularize soon.
    """

    # init elo system variable
    elo = Velo(
        decay_alpha = settings['decay_alpha'],
        decay_beta = settings['decay_beta'],
        season_turnover_default = settings['new_season_regression_weight']
    )

    # loop through each year in the gc data
    yrange = range(settings['begin_year'], settings['end_year'])
    yrange_in_use = yrange if settings['verbose'] else tqdm(yrange)
    for year in yrange_in_use:
        
        # prepare and isolate data for the given year
        year_data = prepare_year_data(main_data, year, race_type = settings['race_type'])
        if len(year_data) == 0:
            continue  # if the data for the current year doesn't exist, move on

        if settings['verbose']:
            print(f'\n====={year}=====\n')
        
        for race in year_data['name'].unique():
            
            # extract data for individual stages
            stages_data = prepare_race_data(year_data, race)
            
            for stage_data in stages_data:

                # get some stage details
                stage_name = stage_data['stage'].iloc[0]
                profile_score = stage_data['profile_score'].iloc[0]
                points_scale = stage_data['points_scale'].iloc[0]

                # compute the stage weight, or pass on it in some cases
                race_weight = _compute_race_weight(
                    stage_data, race_data, settings, 
                    race, year, profile_score, points_scale
                )
                if race_weight is None:
                    continue
                
                # get the race's date as date object
                stage_date = get_race_date(stage_data)
                
                # simulate the race and add it to the rankings
                elo.simulate_race(race, stage_data, race_weight, settings['timegap_multiplier'])
                
                # apply changes to rider elos
                elo.apply_all_deltas(race, settings['race_type'], race_weight, stage_date)
                
                if settings['verbose']:

                    # print raw results for the current race
                    print(f'\n==={race} - {year} - {stage_name} - (weight = {race_weight})===\n')
                    print(stage_data[['place', 'rider', 'time', 'team']].iloc[0: RAW_RESULT_NUM_PRINTED, :])

                    # print the elo system after this race is added
                    elo.print_system(year, min_rating = 1500)

                if settings['save_results']:
                    elo.save_system(stage_date)
        
        # regress scores back to the mean of 1500 at the start of each season
        if year < settings['end_year'] - 1:
            elo.new_season_regression(year)

    elo.print_system(year, min_rating = 1500)

def get_elo_probabilities(rider1_rating, rider2_rating, q_base, q_exponent_denom):
    """
    Rider Elo deltas are dependent on the ratings of each of the riders prior to
    the head-to-head result. This method returns pseudo-probabilities of victory
    based on the rider ratings.
    """

    # get each competitor's Q value
    rider1_q = np.power(q_base, rider1_rating / q_exponent_denom)
    rider2_q = np.power(q_base, rider2_rating / q_exponent_denom)

    # get the probabilities and return
    rider1_p = rider1_q / (rider1_q + rider2_q)
    rider2_p = rider2_q / (rider1_q + rider2_q)
    return (rider1_p, rider2_p)

def get_marg_victory_factor(rider1, rider1_time, rider2, rider2_time, time_gap_multiplier):
    '''
    ASSUMES rider1 is the winning rider and rider2 is the losing rider. Calculates a multiplier
    to have rating changes take margin of victory into account.

    Formula taken from FiveThirtyEight.com's NFL Elo 
    ratings: https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
    '''

    # if given timegap multiplier is None, then return None, so that timegaps are not
    # taken into account
    if time_gap_multiplier is None:
        return 1

    # expand time difference with multiplier
    time_factor = abs(rider2_time.seconds - rider1_time.seconds) * time_gap_multiplier
    
    # get difference in ratings
    rating_diff = rider1.rating - rider2.rating
    return math.log(time_factor + 2) * (2.2 / ((rating_diff * 0.001) + 2.2))

def prepare_year_data(data, year, race_type = 'gc', sort = True):
    """
    Given a dataframe and a year, return a dataframe containing GC data for the
    given year.
    """

    # extract data for the given year
    year_data = data[data['year'] == year]
    
    if race_type == 'itt':

        race_type_lst = [race_type, 'prologue']

        worlds_olympics_itts = year_data[year_data['name'].isin(WORLD_OLYMPICS_ITT_NAMES)]
        year_data = year_data[year_data['type'].isin(race_type_lst)]

        year_data = pd.concat([year_data, worlds_olympics_itts])
    
    elif race_type in ['gc', 'one-day-race']:
        race_type_lst = [race_type]
        year_data = year_data[year_data['type'].isin(race_type_lst)]
    
    elif race_type == 'sprints':
        year_data = year_data[year_data['won_how'] == 'sprint of large group']
    
    if sort:
        year_data.sort_values(by = ['month', 'day'], inplace = True)
    
    return year_data

def prepare_race_data(data, race):
    """
    Given a dataframe containing race results for a given year, isolate just the
    data for the given race, split by different stages.
    """

    race_data = data[data['name'] == race]
    
    stages_data = []
    stages = pd.unique(race_data['stage'])
    
    if len(stages) > 1:
        for stage in stages:
            stages_data.append(race_data[race_data['stage'] == stage])
    else:
        stages_data.append(race_data)

    for i in range(len(stages_data)):
        
        # ensure there is no duplicate data by checking rider results placings are
        # listed in the proper order
        for j in range(1, len(stages_data[i].index)):

            if stages_data[i].loc[stages_data[i].index[j], 'place'] < stages_data[i].loc[stages_data[i].index[j - 1], 'place']:
                stages_data[i] = stages_data[i].iloc[0: j, :]
                break

        stages_data[i] = remove_banned_riders(stages_data[i])

    return stages_data

def _compute_race_weight(stage_data, race_data, settings, race, year, profile_score, points_scale):

    # sprint ranking must make sure appropriate number of riders
    # finished on the same time
    if settings['race_type'] == 'sprints':
        
        # get the number of riders that finished on zero seconds; this is the number 
        # of riders that finished together in the main group
        num_same_time = 0
        for t in stage_data['time']:
            if t == 0: 
                num_same_time += 1
            else: 
                break
        
        # do not consider this stage if too few riders finish in the
        # main group; this is to further exclude finishes that are
        # not necessarily bunch sprints
        if (num_same_time < settings['sprinting_bunch_finish_thresh']
            or profile_score >= settings['sprinting_max_profile_score']):
            return None

    # get race weight for gc results
    elif settings['race_type'] == 'gc':
        
        # for GC rankings, the race must be in the dict
        # to be considered for the rankings
        if str(year) in race_data['race_weights'] and race in race_data['race_weights'][str(year)]:
            
            # return the race weight
            return race_data['race_classes'][str(race_data['race_weights'][str(year)][race])]
        
        else: 
            return None
    
    else:
        
        # this allows us to ignore races with a points scale of nan
        if isinstance(points_scale, float): 
            return None
        
        # verify the points scale is amongst those the rating system should consider
        points_scale = points_scale.strip()
        if points_scale not in race_data['race_weights']:
            return None
        
        # get initial race weight
        race_weight = race_data['race_weights'][points_scale]

        # adjust race weight if race_type is TT and types given
        # this allows us to adjust TT ratings to give more or less
        # weight to different types of TTs
        if settings['race_type'] == 'itt':
            race_weight = RaceSelection.weight_itt_by_type(
                stage_data, (settings['tt_length_adjustor'], settings['tt_vert_adjustor']), race_weight
            )

def get_race_date(race_data):
    """
    Given dataframe of race data, return data as datetime date object.
    """

    race_year = int(race_data['year'].iloc[0])
    month = int(race_data['month'].iloc[0])
    day = int(race_data['day'].iloc[0])
    return date(year = race_year, month = month, day = day)

def remove_banned_riders(df):
    """
    Remove riders from the results who was DQ'd from the race.
    """

    # reset the index of the df
    df = df.reset_index()
    
    # init list to track indices to drop from the df
    to_drop = []
    
    # if a rider has been DQ'd, the following rider in the table will
    # have the same place as them, so this is a way to detect who to
    # remove from results
    prior_place = None
    for i in range(len(df.index)):
        
        # get the current rider's place
        place = df['place'].iloc[i]
        
        # compare with the previous rider's place (if i == 0 this just won't catch)
        if place == prior_place:
            to_drop.append(i - 1)
        
        prior_place = place

    # return the df with the DQ'd riders removed
    return df.drop(labels = to_drop)

def get_startlist(name, year):
    """
    Given the name of a race and the year, return a list of riders on the startlist
    for the race.
    
    NOTE: name param must be given in the format which ProCyclingStats uses in its
    links. For example, if you'd like the startlist for the Tour de Suisse, you would
    given 'tour-de-suisse' as the name param.
    """
    
    # get page html
    link = f'https://www.procyclingstats.com/race/{name}/{year}/gc/startlist'
    page = requests.get(link)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # get the riders from each team starting the race
    teams = {}
    lis = soup.find_all('li', class_ = 'team')
    for li in lis:
        team = li.find_all('a')[0].text
        uls = li.find_all('ul')[0]
        roster = [rider.find_all('a')[0].text for rider in uls.find_all('li')]
        teams[team] = roster
    
    # return the startlist as a list of rider names
    startlist = [rider for team in teams for rider in teams[team]]
    return startlist

