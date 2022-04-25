"""
Support module for Elo system.
"""

from datetime import date
import math
import numpy as np

PLACES_COL = 'place'
RIDER_COL = 'rider'
TIME_COL = 'time'


def k_decay(K: float, p1: int, p2: int, alpha: float = 1.5, beta: float = 1.8):
    """
    Most recent iteration of the decay function for head-to-head result weights
    in cycling results. K is the weight given to the race in question, and this
    function updates K for individual head-to-heads based on their overall
    "importance". Head-to-heads are more "important" if they (1) involve the highest
    placed rider(s) in the race or (2) involve riders who placed close to one another.
    K is first scaled based off the absolute place of the higher placed rider (p1), and then
    based on the relative placing (difference between p1 and p2).Adjust alpha and beta 
    to signify different importances of absolute placing (alpha) and relative 
    placing (beta). For both alpha and beta, higher values signify an increase in 
    the rate of decay as absolute/relative placing get larger.
    """

    # adjust K based on absolute placing
    out = K * (p1 / (p1 ** alpha))

    # adjust K based on relative placing
    out = out * ((p2 - p1) / ((p2 - p1) ** beta))

    return out

def get_elo_probabilities(rider1, rider2, q_base, q_exponent_denom):
    """
    Rider Elo deltas are dependent on the ratings of each of the riders prior to
    the head-to-head result. This method returns pseudo-probabilities of victory
    based on the rider ratings.
    """

    # get each competitor's Q value
    rider1_q = np.power(q_base, rider1.rating / q_exponent_denom)
    rider2_q = np.power(q_base, rider2.rating / q_exponent_denom)

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
    time_factor = (rider2_time.seconds - rider1_time.seconds) * time_gap_multiplier
    
    # get difference in ratings
    rating_diff = rider1.rating - rider2.rating

    return math.log(time_factor + 2) * (2.2 / ((rating_diff * 0.001) + 2.2))

def prepare_year_data(data, year, race_type = 'gc', sort = True):
    """
    Given a dataframe and a year, return a dataframe containing GC data for the
    given year.
    """

    year_data = data[data['year'] == year]
    year_data = year_data[year_data['type'] == race_type]
    
    if sort:
        year_data.sort_values(by = ['month', 'day'], inplace = True)
    
    return year_data

def prepare_race_data(data, race, stage = None):
    """
    Given a dataframe containing race results for a given year, isolate just the
    data for the given race.
    """

    race_data = data[data['name'] == race].reset_index()
    if stage is not None:
        race_data = race_data[race_data['stage'] == stage]

    # ensure there is no duplicate data by checking rider results placings are
    # listed in the proper order
    for i in range(1, len(race_data.index)):

        if race_data.loc[race_data.index[i], 'place'] < race_data.loc[race_data.index[i - 1], 'place']:
            race_data = race_data.iloc[0: i, :]
            break

    # remove riders banned for PED violations from the results
    return remove_banned_riders(race_data)

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
