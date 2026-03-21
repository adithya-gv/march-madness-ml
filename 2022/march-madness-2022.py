import pandas as pd
import numpy as np
import tensorflow as tf

import random
import time

limit = 40

def findWinners(games, results):
    victors = []
    i = 0
    for g in games:
        L = results[i][g[0] - 1] 
        R = results[i][g[1] - 1]
        total = L / (L + R)
        bound = total * 10000
        game = random.randint(0, 10001 + (upsetCount))
        if (game > bound):
            victors.append(g[1])
        else:
            victors.append(g[0])
        i = i + 1
    return victors

def reWrap(games):
    new_games = []
    for i in range(0, int(len(games) / 2)):
        new_games.append([games[i], games[len(games) - 1 - i]])
    new_games = np.array(new_games)
    return new_games

print("Loading Data...")
# Load Teams List
teams = pd.read_csv("data/MTeams.csv")[["TeamID", "TeamName"]].to_dict()
IDs = teams['TeamID']
names = teams['TeamName']

# Load Seeds
seeds = pd.read_csv("data/MNCAATourneySeeds.csv").to_dict()
season = seeds['Season']
seed = seeds['Seed']
seedIDs = seeds['TeamID']

# Load Results
results = pd.read_csv("data/MNCAATourneyCompactResults.csv")[["Season", "WTeamID", "LTeamID"]].to_dict()
resSeason = results['Season']
resWinID = results['WTeamID']
resLossID = results['LTeamID']

print("Mapping Data...")
# Map Years to TeamIDs
seedMap = {}
for k in season.keys():
    year = season[k]
    if year in seedMap.keys():
        val = seedMap[year]
        val.append(seedIDs[k])
        seedMap[year] = val
    else:
        val = []
        val.append(seedIDs[k])
        seedMap[year] = val

print("Pre-Processing Data...")
# Init Training Lists
matches = []
winners = []

# Generate Training Data: Map Wins and Losses to Seeds
upsetCount = 0
for k in resSeason.keys():
    year = resSeason[k]
    yearVals = seedMap[year]
    WTeam = resWinID[k]
    LTeam = resLossID[k]
    WSeed = yearVals.index(WTeam) % 16 + 1 
    LSeed = yearVals.index(LTeam) % 16 + 1
    if (LSeed < WSeed):
        upsetCount = upsetCount + 1
    sample = [WSeed, LSeed]
    matches.append(sample)
    winners.append(WSeed)

# Prep Data (Split into Training and Testing over here)

X = np.array(matches)
Y = np.ndarray(shape=(X.shape[0], 16))
j = 0
for winner in winners:
    for i in range(1, 17):
        if i == winner:
            Y[j, i - 1] = 1
        else :
            Y[j, i - 1] = 0
    j = j + 1

X_train = X[:2000]
X_test = X[2000:]

Y_train = Y[:2000]
Y_test = Y[2000:]


print("Building Model...")
# Build Neural Net
model = tf.keras.models.Sequential(layers=[
    tf.keras.layers.Flatten(input_shape=(2, 1)),
    tf.keras.layers.Dense(48, activation='sigmoid'),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Softmax()
])

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

print("Training Model...")
# Train Neural Net
model.fit(X_train, Y_train, epochs=limit, verbose=1)
model.evaluate(X_test,  Y_test, verbose=1)


# Load this year's data and run limit * 2 games without printing anything to remove noise
print("Running Demo Simulations...")
for i in range(0, limit * 2):
    if (i % 10 == 9):
        print(str(i + 1) + " sims complete.")
    conference = []
    for i in range(0, 4):
        games_22 = []
        for i in range(1, 9):
            games_22.append([i, 17 - i])

        games_22 = np.array(games_22)
        while games_22.shape[0] > 0:
            winners = findWinners(games_22, model.predict(games_22))
            if (len(winners) == 1):
                conference.append(winners[0])
                break
            games_22 = reWrap(winners)

    games_22 = reWrap(conference)
    for i in range(0, 2):
        winners = findWinners(games_22, model.predict(games_22))
        if (len(winners) == 1):
            break
        games_22 = reWrap(winners)

for i in range(0, 20):
    print()
time.sleep(2)

print("Actual Run:")
print()
# Actual Run
conference = []
for i in range(0, 4):
    print("Conference " + str(i + 1)  + ":")
    games_22 = []
    for i in range(1, 9):
        games_22.append([i, 17 - i])

    games_22 = np.array(games_22)
    while games_22.shape[0] > 0:
        winners = findWinners(games_22, model.predict(games_22))
        if (len(winners) == 1):
            print(winners[0])
            conference.append(winners[0])
            break
        else:
            print(winners)
        games_22 = reWrap(winners)
    print()

print("Final Four:")
print(conference)
games_22 = reWrap(conference)
for i in range(0, 2):
    winners = findWinners(games_22, model.predict(games_22))
    if (len(winners) == 1):
        print(winners[0])
        break
    else:
        print(winners)
    games_22 = reWrap(winners)
