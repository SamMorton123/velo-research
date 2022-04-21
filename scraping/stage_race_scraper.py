'''
Simple script to scrape a number of stage race results from a file detailing which
races to scrape.
'''

import json
import scrape_lib_v2 as scrl

RACES_DATA = '../data/women_race_data.json'  # races to scrape
DATA_FNAME = '../data/women_velodata.csv'  # where to save scraped data

with open(RACES_DATA) as f:
    RACES = json.load(f)
f.close()

def create_link(name, year, base = 'https://www.procyclingstats.com/race/'):
    return f'{base}{name}/{year}'

for year in range(2019, 2023):
    
    print(f'===== {year} =====')
    
    race_info = []
    for race in RACES['stage-races'][str(year)]:
        race_info.append((race, create_link(race, year), False))
    
    scrl.get_race_data(race_info, fname = DATA_FNAME)
