"""
New gen of the cycling elo class. Changing the name to Velo (V-Elo), and consolidating
functionality to make it a much more coherent module.
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
from termcolor import colored

# local
from ratings import entities
from ratings.entities import Rider, Race
from ratings import utils

# ====== Column name constants ======
PLACES_COL = 'place'
RIDER_COL = 'rider'
AGE_COL = 'age'
TIME_COL = 'time'
YEAR_COL = 'year'
MONTH_COL = 'month'
DAY_COL = 'day'

# ===== Elo system constants ======
ELO_Q_BASE = 10
ELO_Q_EXPONENT_DENOM = 400

# ===== Constant defaults =====
ALPHA_DEFAULT = 1.5
BETA_DEFAULT = 1.8
SEASON_TURNOVER_DEFAULT = 0.4

PRINT_RIDER_NUM_TOP_PLACINGS = 1
TOP_PLACING_DEFINITION = 10

class Velo:
    
    def __init__(self, decay_alpha: float = 1.5, decay_beta: float = 1.8,
            season_turnover_default: float = SEASON_TURNOVER_DEFAULT, 
            elo_q_base: int = ELO_Q_BASE, elo_q_exponent_denom: int = ELO_Q_EXPONENT_DENOM):

        # collect args as instance variables
        self.decay_alpha = decay_alpha
        self.decay_beta = decay_beta
        self.season_turnover_default = season_turnover_default
        self.elo_q_base = elo_q_base
        self.elo_q_exponent_denom = elo_q_exponent_denom
         
        # init dict and array to track rider and race objects respectively
        self.riders = {}  # format: {rider name: rider object}
        self.races = []

        # init dictionary to track rating data over time
        self.rating_data = {'year': [], 'month': [], 'day': []}
    
    def find_rider(self, name):
        """
        Return an instance of the Rider class if the given name is found. Otherwise,
        return None.
        """

        if name in self.riders:
            return self.riders[name]
        
        return None
    
    def add_new_rider(self, name, age):
        """
        Add a new rider object to the Elo system.
        """

        # check if the given name is already in the system
        rider = self.find_rider(name)

        # if the rider isn't already in the system...
        if rider is None:
            rider = Rider(name, age = age)
            self.riders[name] = rider
        
        # otherwise update the age of the rider
        else:
            rider.age = age
        
        return rider
    
    def simulate_race(self, race_name, results, race_weight, timegap_multiplier, k_func = utils.k_decay):
        """
        Apply the affect of the given race to the Elo system. Race simulated as a series of head-to-head
        matchups evaluated as if they all occurred simultaneously.
        """

        # get race date
        race_date = date(
            year = int(results[YEAR_COL].iloc[0]),
            month = int(results[MONTH_COL].iloc[0]),
            day = int(results[DAY_COL].iloc[0])
        )

        # add race object to list of races
        self.races.append(Race(race_name, race_weight, race_date, None))

        # run each head-to-head
        for i in range(len(results.index) - 1):
            for j in range(i + 1, len(results.index)):

                # get each rider's place
                rider1_place = results[PLACES_COL].iloc[i]
                rider2_place = results[PLACES_COL].iloc[j]
                
                # if rider2 place is less than rider1 place, there's a data error
                # and the matchup is skipped
                if rider1_place >= rider2_place:
                    continue

                # get rider i and j names
                rider1_name = results[RIDER_COL].iloc[i]
                rider2_name = results[RIDER_COL].iloc[j]

                # get rider ages
                rider1_age = results[AGE_COL].iloc[i]
                rider2_age = results[AGE_COL].iloc[j]

                # get Rider objects for each rider
                rider1 = self.add_new_rider(rider1_name, rider1_age)
                rider2 = self.add_new_rider(rider2_name, rider2_age)

                # add to rider race histories
                rider1.add_new_race(race_name, race_weight, race_date, rider1_place)
                rider2.add_new_race(race_name, race_weight, race_date, rider2_place)

                # get rider timegaps in seconds
                if rider1_place == 0:
                    rider1_time = timedelta(seconds = 0)
                else:
                    try: rider1_time = timedelta(seconds = int(results[TIME_COL].iloc[i]))
                    except: rider1_time = timedelta(seconds = 0)

                try: rider2_time = timedelta(seconds = int(results[TIME_COL].iloc[j]))
                except: rider2_time = timedelta(seconds = 0)

                # get the matchup weight
                matchup_weight = k_func(
                    race_weight,
                    rider1_place,
                    rider2_place,
                    alpha = self.decay_alpha,
                    beta = self.decay_beta
                )

                # run the head to head
                self.h2h(
                    rider1, rider1_time, rider2, rider2_time, matchup_weight, timegap_multiplier
                )

    def h2h(self, rider1, rider1_time, rider2, rider2_time, matchup_weight, timegap_multiplier):
        """
        Calculate Elo delta from a single head to head and attribute the delta to each rider.
        """

        # get respective win/loss probabilities
        rider1_p, rider2_p = utils.get_elo_probabilities(
            rider1.rating, rider2.rating, self.elo_q_base, self.elo_q_exponent_denom
        )

        # set rider "scores" (assumes rider1 defeats rider2)
        rider1_score = 1
        rider2_score = 0

        # get margin of victory multiplier
        margvict_factor = utils.get_marg_victory_factor(
            rider1, rider1_time, rider2, rider2_time, timegap_multiplier
        )

        # get elo deltas for each rider
        rider1_delta = matchup_weight * margvict_factor * (rider1_score - rider1_p)
        rider2_delta = matchup_weight * margvict_factor * (rider2_score - rider2_p)

        # update rider elo deltas
        rider1.update_delta(rider1_delta)
        rider2.update_delta(rider2_delta)

        # update wins/losses
        rider1.increment_wins()
        rider2.increment_losses()

    def apply_all_deltas(self, race_name, race_weight, datestamp):
        """
        Apply all outstanding Elo deltas for riders in the system.
        """

        for name in self.riders:
            self.riders[name].resolve_delta(race_name, race_weight, datestamp)

    def new_season_regression(self, year, regression_to_mean_weight = 0.5):
        """
        Regress all rider ratings back to the default initial rating at the onset
        of a new season.
        """

        self.races.append(Race('New Season', None, None, None))
        for name in self.riders:
            self.riders[name].new_season(regression_to_mean_weight)
        
        self.save_system(date(year = year, month = 1, day = 1))
    
    def save_system(self, date):
        """
        Save rating data as a timeseries for each rider in the system.
        """

        # loop through every rider in the system and add their current rating to the cumulative data
        for name in self.riders:
            
            if name in self.rating_data:
                self.rating_data[name].append(self.riders[name].rating)
            
            # add a new key and array to the rating_data dict, and make sure that all keys
            # have corresponding arrays of the same length by filling the array with the
            # default initial Elo value and then adding their most recent rating
            else:
                self.rating_data[name] = [
                    entities.DEFAULT_INITIAL_RATING
                ] * len(self.rating_data['year']) + [self.riders[name].rating]
        
        # add a new date to the dict
        self.rating_data['year'].append(date.year)
        self.rating_data['month'].append(date.month)
        self.rating_data['day'].append(date.day)
    
    def save_system_data(self, rating_type, base_fname = 'data/system_data'):
        pd.DataFrame(data = self.rating_data).to_csv(f'{base_fname}_{rating_type}.csv', index = False)

    def print_system(self, curr_year, min_rating = entities.DEFAULT_INITIAL_RATING, printing_limit = 50):
        '''
        Rank the riders by their system and print.
        '''
        
        # sort the list of rider objects by rating
        riders_sorted = sorted(
            list(self.riders.values()),
            key = lambda rider: rider.rating,
            reverse = True
        )

        # now print each rider and their place
        place = 1
        for rider in riders_sorted:
            
            # only continue printing if the rider rating is above the threshold
            # and we haven't hit the page limit
            if rider.rating < min_rating or place > printing_limit: 
                break

            # rider must have competed in the current year or the year before in order
            # to be printed
            if rider.most_recent_active_year < curr_year - 1:
                continue

            # get the rider's most recent Elo delta, and add color
            delta = round(rider.rating_history[-1][0] - rider.rating_history[-2][0], 2)
            if delta == 0: delta = ''
            else:
                color = 'green' if delta > 0 else 'red'
                delta = colored(delta, color)

            print(
                f'{place}.', f'{rider.name} - {round(rider.rating, 2)};', 
                f'Active: {rider.most_recent_active_year}, Age: {rider.age}', delta
            )

            place += 1
