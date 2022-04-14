"""
Scraper specifically for scraping women's one day races.

April 14, 2022
"""

import json

# local
import scrape_lib_v2 as scrl

def create_link(name, year, base = 'https://www.procyclingstats.com/race/'):
    return f'{base}{name}/{year}'


# ===== Load One Day Races ===== #
with open('../data/women_race_data.json') as f:
    RACES = json.load(f)
f.close()

for year in range(2022, 2023):

    race_info = []
    for race in RACES[str(year)]['one-day']:
        race_info.append((race, create_link(race, year), True))


    scrl.get_race_data(race_info, fname = '../data/women_velodata.csv')