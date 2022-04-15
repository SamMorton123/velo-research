'''
Simple script to scrape a number of stage race results from a file detailing which
races to scrape.
'''

import json
import scrape_lib_v2 as scrl

with open('all_races.json') as f:
    RACES = json.load(f)
f.close()

def create_link(name, year, base = 'https://www.procyclingstats.com/race/'):
    return f'{base}{name}/{year}'

for year in range(2021, 2022):
    
    print(f'===== {year} =====')
    
    race_info = []
    for race in RACES['stage-races'][str(year)]:
        race_info.append((race, create_link(race, year), False))
    
    scrl.get_race_data(race_info, fname = 'new_collected_dataset.csv')
