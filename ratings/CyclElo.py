'''
V2 of the cycling Elo class, but no longer a subclass of my
original Elo implementation for track and field. Contains methods
necessary for updating, managing, and printing the Elo system as
new results come in and change the rankings.
'''

from datetime import date, timedelta
import numpy as np
import pandas as pd
from termcolor import colored

# local
from ratings.decay_funcs import logistic_decay, linear2, piecewise
from ratings.Race import Race
from ratings.Rider import Rider

# column names
PLACES_COL = 'place'
RIDER_COL = 'rider'
AGE_COL = 'age'
TIME_COL = 'time'
YEAR_COL = 'year'
MONTH_COL = 'month'
DAY_COL = 'day'

# elo math constants
ELO_Q_BASE = 10
ELO_Q_EXPONENT_DENOM = 400

PRINT_RIDER_NUM_TOP_PLACINGS = 1
TOP_PLACING_DEFINITION = 10

class CyclElo:
    
    def __init__(self):
        self.riders = []
        self.races = []
        self.rating_data = {'year': [], 'month': [], 'day': []}
    
    def find_rider(self, name):
        '''
        Given the name of a rider, try to find the corresponding Rider object
        within the elo system, and return the entire object. Returns None if no
        Rider is found under the given name.
        '''

        for rider in self.riders:
            if rider.name == name:
                return rider
        return None
    
    def add_new_rider(self, name, age):
        '''
        Add new rider to the Elo system.
        '''

        # try to find the rider
        rider = self.find_rider(name)

        if rider is None:
            rider = Rider(name, age = age)
            self.riders.append(rider)
        else:
            rider.age = age
        
        return rider
    
    def simulate_race(self, race_name, results, race_weight, timegap_multiplier, k_func = linear2):
        '''
        Simulate a race as a series of head to head matchups between all participants in the race. This type
        of treatment can be used on different cycling disciplines, such as GC results and TT results.
        '''

        # get race date
        year = int(results[YEAR_COL].iloc[0])
        month = int(results[MONTH_COL].iloc[0])
        day = int(results[DAY_COL].iloc[0])
        race_date = date(year = year, month = month, day = day)

        # add race to list of races
        race_info = Race(race_name, race_weight, race_date, None)
        self.races.append(race_info)

        # init list of rider order
        rider_order = []

        # run each head-to-head
        for i in range(len(results.index)):
            for j in range(len(results.index)):

                # get rider i and j places
                rider1_place = results[PLACES_COL].iloc[i]
                rider2_place = results[PLACES_COL].iloc[j]
                
                # if rider i's place isn't less than j's, continue
                if rider1_place >= rider2_place:
                    continue

                # get rider i and j names
                rider1_name = results[RIDER_COL].iloc[i]
                try: rider1_age = int(results[AGE_COL].iloc[i])
                except: rider1_age = 0
                rider2_name = results[RIDER_COL].iloc[j]
                try: rider2_age = int(results[AGE_COL].iloc[j])
                except: rider2_age = 0

                # get rider objects
                rider1 = self.add_new_rider(rider1_name, rider1_age)
                rider2 = self.add_new_rider(rider2_name, rider2_age)

                # add rider i to order if not already in it
                if len(rider_order) == 0 or rider1 != rider_order[-1]:
                    rider_order.append(rider1)

                # add to rider race histories
                rider1.add_new_race(race_name, race_weight, race_date, rider1_place)
                rider2.add_new_race(race_name, race_weight, race_date, rider2_place)

                # get rider times
                try:
                    if rider1_place == 0:
                        rider1_time = timedelta(seconds = 0)
                    else:
                        rider1_time = timedelta(seconds = int(results[TIME_COL].iloc[i]))
                    rider2_time = timedelta(seconds = int(results[TIME_COL].iloc[j]))
                except:
                    rider1_time = 0
                    rider2_time = 0

                # get the matchup weight
                matchup_weight = k_func(race_weight, rider1_place, rider2_place)

                # run the head to head
                self.h2h(
                    rider1, rider1_time, rider2, rider2_time, matchup_weight, timegap_multiplier
                )
        
        return rider_order

    def h2h(self, rider1, rider1_time, rider2, rider2_time, matchup_weight, timegap_multiplier):
        '''
        Simulate the head to head matchup between two given riders, and adjust their elo deltas
        accordingly.
        '''

        # get respective win/loss probabilities
        rider1_p, rider2_p = _get_win_loss_probabilities(rider1, rider2)

        # set rider "scores" (assumes rider1 beat rider2 if times are equal)
        rider1_score = 1 if rider1_time <= rider2_time else 0
        rider2_score = 1 if rider2_time < rider1_time else 0

        # get margin of victory multiplier
        margvict_mult = _get_margvict_mult(rider1, rider1_time, rider2, rider2_time, timegap_multiplier)

        # get elo deltas for each rider
        rider1_delta = matchup_weight * margvict_mult * (rider1_score - rider1_p)
        rider2_delta = matchup_weight * margvict_mult * (rider2_score - rider2_p)

        # update rider elo deltas
        rider1.update_delta(rider1_delta)
        rider2.update_delta(rider2_delta)

        # update wins/losses
        if rider1_score == 1: rider1.increment_wins()
        else: rider1.increment_losses()
        if rider2_score == 1: rider2.increment_wins()
        else: rider2.increment_losses()

    def apply_all_deltas(self, race_name, race_weight, datestamp):
        '''
        The Elo delta is the accumulated change in a Rider's Elo rating within the simulation
        of a given race. The reason this is necessary is that we should be treating these head-to-heads
        within a race as if they happen simultaneously, and therefore the effects of each of the head-to-heads
        on a rider's Elo shouldn't be added until all the head-to-heads have been run.
        '''

        for rider in self.riders:
            rider.resolve_delta(race_name, race_weight, datestamp)

    def new_season_regression(self, year, regression_to_mean_weight = 0.5):
        '''
        Each new season brings changes in the relative quality of riders. This moves the Elo rating
        of each rider closer to the starting Elo rating of 1500 at the beginning of each season. This
        means that a rider that ends a season with a rating of over 1500 will have their rating go
        down, and a rider with a rating less than 1500 at the end of the season will see their
        rating go up.
        '''

        self.races.append(Race('New Season', None, None, None))
        for rider in self.riders:
            rider.new_season(regression_to_mean_weight)
        
        self.save_system(date(year = year, month = 1, day = 1))
    
    def save_system(self, date):
        '''
        Save rating data as a timeseries for each rider in the system.
        '''

        # loop through every rider in the system and add their current rating to the cumulative data
        for rider in self.riders:
            if rider.name in self.rating_data:
                self.rating_data[rider.name].append(rider.rating)
            else:
                self.rating_data[rider.name] = [1500] * len(self.rating_data['year']) + [rider.rating]
        
        # add a new date to the dict
        self.rating_data['year'].append(date.year)
        self.rating_data['month'].append(date.month)
        self.rating_data['day'].append(date.day)
    
    def save_system_data(self, rating_type, base_fname = 'system_data'):
        pd.DataFrame(data = self.rating_data).to_csv(f'{base_fname}_{rating_type}.csv', index = False)

    def print_system(self, curr_year, rider_selection_method, min_rating = 1500):
        '''
        Print the rankings as they currently stand.
        '''

        # determine if system will print rankings of all riders, or a specific list of riders
        riders_to_rank = [rider for rider in self.riders if rider_selection_method(rider)]
        
        # sort the list of riders by rating
        riders_sorted = sorted(riders_to_rank, key = lambda r: r.rating, reverse = True)

        # now print each rider and their place
        place = 1
        for rider in riders_sorted:
            if rider.rating < min_rating or place > 50: 
                break

            # continue, if the rider hasn't competed in the current year or the year before
            if rider.most_recent_active_year < curr_year - 1:
                continue
            
            # continue, if the rider hasn't placed high enough a sufficient number of times
            sorted_results = sorted(rider.race_history, key = lambda race: (race.place is None, race.place))
            if (len(sorted_results) < PRINT_RIDER_NUM_TOP_PLACINGS or
                sorted_results[PRINT_RIDER_NUM_TOP_PLACINGS - 1].place is None or
                sorted_results[PRINT_RIDER_NUM_TOP_PLACINGS - 1].place > TOP_PLACING_DEFINITION):
                continue

            # get the rider's delta, and add color
            delta = round(rider.rating_history[-1][0] - rider.rating_history[-2][0], 2)
            if delta == 0:
                delta = ''
            else:
                color = 'green' if delta > 0 else 'red'
                delta = colored(delta, color)

            print(
                f'{place}.', f'{rider.name} - {round(rider.rating, 2)};', 
                f'Active: {rider.most_recent_active_year}, Age: {rider.age}', delta,
                f'Top {TOP_PLACING_DEFINITION}s:', len([r for r in sorted_results if r.place is not None and r.place <= TOP_PLACING_DEFINITION])
            )

            place += 1
    
    def print_most_improved(self, riders_printed = 20):
        '''
        Print the most improved riders as a percentage of their rating at the start
        of the season.
        '''

        print('\nMost Improved:\n')
        

        # get riders and their percentage improvements
        riders_p_improvements = [
            (rider, _get_improvement(rider)) for rider in self.riders
        ]

        # sort by improvement
        riders_p_improvements.sort(key = lambda t: t[1], reverse = True)
        for i in range(min(riders_printed, len(riders_p_improvements))):
            print(f'{i + 1}. {riders_p_improvements[i][0].name} - ({riders_p_improvements[i][1]}%)')

    
    @staticmethod
    def top_probabilities(existing_probabilities, rider_order):
        '''
        Still in development. Goal: generating probabilities of a rider doing well in a race
        based on their rating and the ratings of the other competitors in the race.
        '''

        for quality_thresh in existing_probabilities:
            for thresh in existing_probabilities[quality_thresh]:
                for rider in rider_order:
                    if rider.rating >= quality_thresh:
                        if rider_order.index(rider) < thresh:
                            existing_probabilities[quality_thresh][thresh].append(1)
                        else:
                            existing_probabilities[quality_thresh][thresh].append(0)
        return existing_probabilities

def _get_win_loss_probabilities(rider1, rider2):
    '''
    Utility method for helping with calculating rider Elos.
    '''

    # get each competitor's Q value
    rider1_q = np.power(ELO_Q_BASE, rider1.rating / ELO_Q_EXPONENT_DENOM)
    rider2_q = np.power(ELO_Q_BASE, rider2.rating / ELO_Q_EXPONENT_DENOM)

    # get the probabilities and return
    rider1_p = rider1_q / (rider1_q + rider2_q)
    rider2_p = rider2_q / (rider1_q + rider2_q)
    return (rider1_p, rider2_p)

def _get_margvict_mult(rider1, rider1_time, rider2, rider2_time, time_gap_multiplier):
    '''
    ASSUMES rider1 is the winning rider and rider2 is the losing rider. Calculates a multiplier
    to have rating changes take margin of victory into account.
    '''

    # expand time difference with multiplier
    score_diff = (rider1_time - rider2_time) * time_gap_multiplier
    
    # get difference in ratings
    rating_diff = rider1.rating - rider2.rating
    
    # try to calculate the score multiplier
    try:
        return math.log(score_diff + 1) * (2.2 / ((rating_diff * 0.001) + 2.2))
    
    # if there is a problem, just return 1 so that the multiplier simply has no effect on calculations
    except:
        return 1
    
def _get_improvement(rider):

    # get rating from beginning of year
    for i in range(len(rider.rating_history) - 1, -1, -1):
        if rider.rating_history[i][1] == 'newseason':
            initial_rating = rider.rating_history[i][0]
            break

    return round((rider.rating - initial_rating) / initial_rating * 100, 2)
