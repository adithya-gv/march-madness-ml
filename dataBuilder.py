import pandas as pd
import numpy as np

# computes ratings
def buildOne():
    # [year, winner, loser, wscore, lscore]
    games = pd.read_csv("data/MRegularSeasonCompactResults.csv")[["Season", "WTeamID", "LTeamID", "WScore", "LScore"]].to_numpy()

    # Calcuate ORTG and DRTG
    # ORTG = average points scored
    # DRTG = average points allowed

    years = {}

    def calculateRating(line, years):
        # line = [year, winner, loser, wscore, lscore]
        year = line[0]
        winner = line[1]
        loser = line[2]
        wscore = line[3]
        lscore = line[4]
        if year in years.keys():
            yearDict = years[year]
            if winner in yearDict.keys():
                wdict = yearDict[winner]
                wdict['score'] += wscore
                wdict['scoredAgainst'] += lscore
                wdict['games'] += 1
                wdict['gamesWon'] += 1
            else:
                yearDict[winner] = {'score': wscore, 'games': 1, 'gamesWon': 1, 'scoredAgainst': lscore}
            if loser in yearDict.keys():
                ldict = yearDict[loser]
                ldict['score'] += lscore
                ldict['scoredAgainst'] += wscore
                ldict['games'] += 1
                ldict['gamesWon'] += 0

            else:
                yearDict[loser] = {'score': lscore, 'games': 1, 'gamesWon': 0, 'scoredAgainst': wscore}
            years[year] = yearDict
        else:
            years[year] = {winner: {'score': wscore, 'games': 1, 'gamesWon': 1, 'scoredAgainst': lscore}, loser: {'score': lscore, 'games': 1, 'gamesWon': 0, 'scoredAgainst': wscore}}

    for line in games:
        calculateRating(line, years)

    finalStats = {}

    for year in years.keys():
        yearDict = years[year]
        for team in yearDict.keys():
            teamDict = yearDict[team]
            teamDict['ORTG'] = teamDict['score'] / teamDict['games']
            teamDict['DRTG'] = teamDict['scoredAgainst'] / teamDict['games']
            teamDict['NRTG'] = teamDict['ORTG'] - teamDict['DRTG']
            teamDict['winRate'] = teamDict['gamesWon'] / teamDict['games']
            yearDict[team] = teamDict
        finalStats[year] = yearDict

    # export stats to csv
    # year, team, ORTG, DRTG, NRTG, winRate
    stats = []
    for year in finalStats.keys():
        yearDict = finalStats[year]
        for team in yearDict.keys():
            teamDict = yearDict[team]
            stats.append([int(year), int(team), teamDict['ORTG'], teamDict['DRTG'], teamDict['NRTG'], teamDict['winRate']])
    stats = np.array(stats)
    np.savetxt("data/stats.csv", stats, delimiter=",", fmt='%s')

# computes winners of matchups
def buildTwo():
    games = pd.read_csv("data/MNCAATourneyCompactResults.csv")[["Season", "WTeamID", "LTeamID"]].to_numpy()

    # [year, winner, loser]
    winners = []
    for line in games:
        winners.append([line[0], line[1], line[2]])

    winners = np.array(winners)
    np.savetxt("data/winners.csv", winners, delimiter=",", fmt='%s')

buildTwo()
