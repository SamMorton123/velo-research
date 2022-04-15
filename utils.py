"""
Utility module to support Elo ranking scripts.
"""

from datetime import date, timedelta
import pandas as pd
from tqdm import tqdm

PLACES_COL = 'place'
RIDER_COL = 'rider'
TIME_COL = 'time'


def prepare_year_data(data, year, typ = 'gc', sort = True):
    """
    Given a dataframe and a year, return a dataframe containing GC data for the
    given year.
    """

    year_data = data[data['year'] == year]
    year_data = year_data[year_data['type'] == typ]
    
    if sort:
        year_data.sort_values(by = ['month', 'day'], inplace = True)
    
    return year_data

def prepare_one_day_year_data(data, year, sort = True):

    year_data = data[data['year'] == year]
    year_data = year_data[year_data['type'] == 'one-day-race']

    if sort:
        year_data.sort_values(by = ['month', 'day'], inplace = True)
    
    return year_data

def prepare_race_data(data, race):
    """
    Given a dataframe containing gc results for a given year, isolate the gc result
    for the given race name.
    """

    race_data = data[data['name'] == race]

    # remove riders banned for doping from the results
    return _remove_dopers(race_data)

def prepare_stage_data(data, stage):
    return data[data['stage'] == stage]

def get_worlds_and_olympics(data, year):
    year_data = data[data['year'] == year]
    return year_data[year_data['name'].isin(['world-championship-itt', 'olympic-games-itt'])]

def get_race_date(race_data):
    """
    Given dataframe of race data, return data as datetime date object.
    """

    race_year = int(race_data['year'].iloc[0])
    month = int(race_data['month'].iloc[0])
    day = int(race_data['day'].iloc[0])
    return date(year = race_year, month = month, day = day)

def _remove_dopers(df):
    '''
    Remove riders from a results df which have been popped for doping
    and DQ'd from the race, and therefore shouldn't count in the results.
    '''

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

def get_tt_gc(race_df):

    # split the separate TTs of the race if there are multiple
    rdfs = split_stage_race_tts(race_df)

    # init dict to track total seconds gained/lost across the TTs
    timegaps = {}

    # get the summed timegaps for each rider in each TT
    for rdf in rdfs:
        rdf = _remove_dopers(rdf)
        for rider_idx in range(len(rdf.index)):

            # get rider name
            rider_name = rdf[RIDER_COL].iloc[rider_idx]

            # get rider place
            rider_place = rdf[PLACES_COL].iloc[rider_idx]

            # get rider timegap
            try:
                if rider_place == 0:
                    timegap = timedelta(seconds = 0)
                else:
                    timegap = timedelta(seconds = int(rdf[TIME_COL].iloc[rider_idx]))
            except:
                timegap = timedelta(seconds = 0)
            
            if rider_name not in timegaps:
                timegaps[rider_name] = timegap
            else:
                timegaps[rider_name] += timegap
    
    # get the winner's timegap
    sorted_timegaps = sorted(list(timegaps.items()), key = lambda tup: tup[1])
    winner_gap = sorted_timegaps[0][1]

    # normalize all times so that winner's gap is 0
    if winner_gap.seconds > 0:
        for rider in timegaps:
            timegaps[rider] -= winner_gap
    
    return timegaps

def get_race_df_wo_tt(race_df, tt_timegaps):

    # loop through each row of the race df and adjust the timegap
    leader_timegap = float('inf')
    for i in range(len(race_df.index)):
        
        # get rider name
        rider_name = race_df[RIDER_COL].iloc[i]

        # get rider place
        rider_place = race_df[PLACES_COL].iloc[i]

        # get rider timegap
        try:
            if rider_place == 1:
                timegap = timedelta(seconds = 0)
            else:
                timegap = timedelta(seconds = int(race_df[TIME_COL].iloc[i]))
        except:
            timegap = timedelta(seconds = 0)
        
        if rider_name in tt_timegaps:
            new_timegap = timegap.seconds - tt_timegaps[rider_name].seconds
            race_df.at[i, TIME_COL] = new_timegap
            leader_timegap = min(leader_timegap, new_timegap)
    
    # go through all the timegaps in the df and normalize so the leader timegap is 0
    for i in race_df.index:
        race_df.at[i, TIME_COL] = int(race_df.at[i, TIME_COL]) + (-1 * leader_timegap)
    
    # sort the df by timegap
    new_race_df = race_df.sort_values(by = TIME_COL).dropna().reset_index()
    new_race_df[PLACES_COL] = [i + 1 for i in new_race_df.index]
    return new_race_df


def split_stage_race_tts(race_df):

    # if there are multiple TT stages
    if len(race_df['stage'].unique()) > 1:
        return [
            race_df[race_df['stage'] == stg]
            for stg in race_df['stage'].unique()
        ]
    else:
        return [race_df]

def isolate_gc_results(df):
    return df[df['stage'] == 'final-gc']

def isolate_tts(df):
    '''
    Given a df containing all races data, return a df
    with only TT data included.
    '''

    one_day_itts_df = df[df['name'].isin([
        'world-championship-itt', 'olympic-games-itt', 'uec-road-european-championships-itt'
    ])]
    df = df[df['type'].isin(['prologue', 'itt'])]
    df = pd.concat([df, one_day_itts_df])
    return df
