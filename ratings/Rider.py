'''
Generic rider class intended to be useful for a cycling Elo implementation
in addition to more sophisticated rating systems in the future.
'''

from datetime import date

# local
from ratings.Race import Race

# constants
DEFAULT_INITIAL_RATING = 1500

class Rider:
    def __init__(self, name, initial_rating = DEFAULT_INITIAL_RATING, team = None, age = None):
        
        # init instance variables
        self.name = name
        self.rating = initial_rating
        self.season_wins = 0
        self.season_losses = 0
        self.team = team
        self.age = age
        self.delta = 0
        self.rating_history = [(self.rating, 'newseason')]
        self.race_history = []
        self.most_recent_active_year = 1800  # make it far in the past, so if never updated the rider never shows in rankings
    
    def __eq__(self, other_rider):
        return self.name == other_rider.name
    
    def __str__(self):
        return f'{self.name}: {self.rating}'
    
    def __repr__(self):
        return str(self)
    
    def increment_wins(self):
        self.season_wins += 1
    
    def increment_losses(self):
        self.season_losses += 1
    
    def add_new_race(self, race_name, race_weight, race_date, place, classification = None):
        new_race = Race(race_name, race_weight, race_date, place, classification = classification)
        if new_race not in self.race_history:
            self.race_history.append(new_race)
    
    def update_delta(self, delta_addition):
        '''
        Adds the given delta addition to the rider's current delta.
        '''

        self.delta += delta_addition
    
    def resolve_delta(self, race_name, race_weight, datestamp):
        '''
        When deltas are resolved, it's assumed that this marks the addition
        of a race to the rider's race history. Therefore, the current delta
        is the change in the rider's rating from the given race (aggregating
        all the individual head-to-heads that occurred from that race.)
        '''

        # add delta to rating
        self.rating += self.delta

        # reset delta
        self.delta = 0

        # add new rating to rating history
        self.rating_history.append((self.rating, datestamp))

        self.most_recent_active_year = datestamp.year
    
    def new_season(self, weight):
        new_rating = (DEFAULT_INITIAL_RATING * weight) + (self.rating * (1 - weight))
        self.race_history.append(Race('New Season', None, None, None))
        self.rating = new_rating
        self.rating_history.append((self.rating, 'newseason'))
