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

def elo_driver(data_main, race_classes, race_weights, beg_year, end_year, gender, race_type,
        timegap_multiplier = None, new_season_regress_weight = NEW_SEASON_REGRESS_WEIGHT,
        decay_alpha = 1.5, decay_beta = 1.8, given_tt_length_adjustor = None,
        given_tt_vert_adjustor = None, save_results = False, verbose = True,
        eval_races = []):
    """
    Given necessary params, init an Elo system and run through all the applicable data,
    tracking changes to rider ratings over time. Returns an Elo object.

    This method is a little long; need to modularize soon.
    """

    # init elo system variable
    elo = Velo(
        decay_alpha = decay_alpha,
        decay_beta = decay_beta,
        season_turnover_default = new_season_regress_weight
    )

    # dict to track the elo ratings of riders at the onset of
    # specific races; saving data here will make it easier to
    # prepare data for future ML applications
    eval_results = {}

    # loop through each year in the gc data
    for year in range(beg_year, end_year):
        
        # prepare and isolate data for the given year
        year_data = prepare_year_data(data_main, year, race_type = race_type)
        if len(year_data) == 0:
            continue

        if verbose:
            print(f'\n====={year}=====\n')
        
        # loop through each race in the current year's data
        for race in year_data['name'].unique():
            
            # extract data for individual stages
            stages_data = prepare_race_data(year_data, race)
            
            for stage_data in stages_data:

                # get the stage number and length
                stage_name = stage_data['stage'].iloc[0]
                stage_length = stage_data['length'].iloc[0]   

                # sprint ranking must make sure appropriate number of riders
                # finished on the same time
                if race_type == 'sprints':
                    
                    # get the number of riders that finished on zero seconds
                    # this is the number of riders that finished together in
                    # the main group
                    num_same_time = 0
                    for t in stage_data['time']:
                        if t == 0: num_same_time += 1
                        else: break
                    
                    # do not consider this stage if too few riders finish in the
                    # main group; this is to further exclude finishes that are
                    # not necessarily bunch sprints
                    if num_same_time < SPRINT_NUM_SAME_TIME_FINISH_THRESH:
                        continue

                # get race weight
                if race_type == 'gc':
                    
                    # for GC rankings, the race must be in the race_classes dict
                    # to be considered for the rankings
                    if str(year) in race_weights and race in race_weights[str(year)]:
                        race_weight = race_classes[str(race_weights[str(year)][race])]
                    else:
                        continue
                else:

                    # for non-GC race types, get the race's weight using its points scale
                    points_scale = stage_data['points_scale'].iloc[0]
                    
                    # this allows us to ignore races with a points scale of nan
                    if isinstance(points_scale, float): continue
                    
                    # verify the points scale is amongst those the rating system should consider
                    points_scale = points_scale.strip()
                    if points_scale not in race_weights:
                        continue
                    
                    race_weight = race_weights[points_scale]

                # adjust race weight if race_type is TT and types given
                # this allows us to adjust TT ratings to give more or less
                # weight to different types of TTs
                if race_type == 'itt':
                    race_weight = RaceSelection.weight_itt_by_type(
                        stage_data, (given_tt_length_adjustor, given_tt_vert_adjustor), race_weight
                    )
                
                # get the race's date as date object
                stage_date = get_race_date(stage_data)

                # save the results and rating in eval_results
                eval_results[f'{race}-{year}-{stage_name}'] = {
                    'date': stage_date.isoformat(),
                    'actual': list(stage_data['rider']),
                    'timegaps': list(stage_data['time']),
                    'top_active': sorted([
                        [rider, elo.riders[rider].rating] if rider in elo.riders else [rider, 1500]
                        for rider in stage_data['rider']
                    ], key = lambda t: t[1], reverse = True),
                    'length': stage_length
                }
                
                # simulate the race and add it to the rankings
                elo.simulate_race(race, stage_data, race_weight, timegap_multiplier)
                
                # apply changes to rider elos
                elo.apply_all_deltas(race, race_weight, stage_date)
                
                if verbose:

                    # print raw results for the current race
                    print(f'\n==={race} - {year} - {stage_name} - (weight = {race_weight})===\n')
                    print(stage_data[['place', 'rider', 'time', 'team']].iloc[0: RAW_RESULT_NUM_PRINTED, :])

                    # print the elo system after this race is added
                    elo.print_system(year, min_rating = 1500)

                if save_results:
                    elo.save_system(stage_date)
        
        # regress scores back to the mean of 1500 at the start of each season
        if year < end_year - 1:
            elo.new_season_regression(year)

    # save Elo system results
    if save_results:
        
        # saves the elo system to a csv for easy access later
        elo.save_system_data(f'{race_type}_{gender}')
        
        # saves the eval data as json separately for use in creating better organized data
        # for ML purposes later
        if len(eval_races) > 0:
            with open(f'data/eval-data/{gender}-{race_type}-{beg_year}-{end_year}-{given_tt_vert_adjustor}-{given_tt_length_adjustor}-eval.json', 'w') as f:
                json.dump(eval_results, f)
            f.close()
    
    return eval_results

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

