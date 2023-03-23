import torch
from torch import nn 
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gc

"""
features: [Wseed, LSeed, WORTG, LORTG, WDRTG, LDRTG, WNRTG, LNRTG, WRecord, LRecord]
"""

gc.collect()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 36)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(36, 12)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(12, 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# build matchup data training set
stats = pd.read_csv("data/stats.csv").to_numpy()
matchups = pd.read_csv("data/winners.csv").to_numpy()
seeds = pd.read_csv("data/MNCAATourneySeeds.csv")[["TeamID", "Season", "Seed"]].to_numpy()

# build dictionary of seeds of teams over years
seedMap = {}
for seed in seeds:
    team = seed[0]
    year = seed[1]
    seedNo = seed[2]
    if team in seedMap.keys():
        val = seedMap[team]
        val[year] = int(seedNo[1:3])
        seedMap[team] = val
    else:
        val = {}
        val[year] = int(seedNo[1:3])
        seedMap[team] = val

# build dictionary of stats of teams over years
statsMap = {}
for stat in stats:
    team = int(stat[1])
    year = int(stat[0])
    ortg = stat[2]
    drtg = stat[3]
    nrtg = stat[4]
    record = stat[5]
    if team in statsMap.keys():
        val = statsMap[team]
        val[year] = [ortg, drtg, nrtg, record]
        statsMap[team] = val
    else:
        val = {}
        val[year] = [ortg, drtg, nrtg, record]
        statsMap[team] = val


# build dataset
dataset = []
answers = []
for matchup in matchups:
    year = matchup[0]
    winner = matchup[1]
    loser = matchup[2]


    # features: [1seed, 2Seed, 1ORTG, 2ORTG, 1DRTG, 2DRTG, 1NRTG, 2NRTG, 1Record, 2Record]
    try:
        wSeed = seedMap[winner][year]
        lSeed = seedMap[loser][year]
        wStats = statsMap[winner][year]
        lStats = statsMap[loser][year]

        # randomly permute the order of the features
        if np.random.rand() > 0.5:
            features = [wSeed, lSeed, wStats[0], lStats[0], wStats[1], lStats[1], wStats[2], lStats[2], wStats[3], lStats[3]]
            answer = [1, 0]
        else:
            features = [lSeed, wSeed, lStats[0], wStats[0], lStats[1], wStats[1], lStats[2], wStats[2], lStats[3], wStats[3]]
            answer = [0, 1]
        
        dataset.append(features)
        answers.append(answer)
    except:
        continue

dataset = np.array(dataset)
answers = np.array(answers)

# merge dataset and answers
dataset = np.concatenate((dataset, answers), axis=1)


# shuffle and split dataset
np.random.shuffle(dataset)
train = dataset[:int(0.8*len(dataset))]
test = dataset[int(0.8*len(dataset)):]

# split back into features and answers
train_features = train[:, :10]
train_answers = train[:, 10:]
test_features = test[:, :10]
test_answers = test[:, 10:]

train_features = train_features.astype(np.float32)
train_answers = train_answers.astype(np.float32)
test_features = test_features.astype(np.float32)
test_answers = test_answers.astype(np.float32)

# train model
batch_size = 64
train = DataLoader(list(zip(train_features, train_answers)), batch_size=batch_size, shuffle=True)
test = DataLoader(list(zip(test_features, test_answers)), batch_size=batch_size, shuffle=True)


model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

epochs = 650

plots = []
losses = []
local_loss = 0
for i in range(epochs):
    for features, labels in train:
        optimizer.zero_grad()
        output = model(features.float())
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    
    local_loss += loss.item()
    if (i + 1) % 10 == 0:
        plots.append(i + 1)
        losses.append(local_loss / 10)
        local_loss = 0
    print("Epoch: ", i + 1, " Loss: ", loss.item())

plt.plot(plots, losses)
plt.show()

# test model
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test:
        output = model(features.float())
        for idx, i in enumerate(output):
            if torch.argmax(i) == torch.argmax(labels[idx]):
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))


# make actual predictions
actual_data = pd.read_csv("data/test.csv")[["TeamID", "Season", "Seed"]].to_numpy()

# clean seedings
for team in actual_data:
    team[2] = int(team[2][1:3])

year = 2023

# get 2023 teams
team_list = []
seed_list = []
for team in actual_data:
    if team[1] == year:
        team_list.append(team[0])
        seed_list.append(team[2])

# get 2023 stats
stats_2023 = {}
for team in team_list:
    try:
        stats_2023[team] = statsMap[team][year]
    except:
        continue


"""
SWEET SIXTEEN
"""

# build next round of matchups
matchups = []
seeds = []
for i in range(0, 16, 4):
    k = 0
    for j in range(i, i + 4, 2):
        matchups.append([team_list[j], team_list[j + 1]])
        seeds.append([seed_list[j], seed_list[j + 1]])
        k = k + 1

print(matchups)
print(seeds)

# build dataset
dataset = []
for matchup in matchups:
    wSeed = seeds[matchups.index(matchup)][0]
    lSeed = seeds[matchups.index(matchup)][1]
    wStats = statsMap[matchup[0]][year]
    lStats = statsMap[matchup[1]][year]

    features = [wSeed, lSeed, wStats[0], lStats[0], wStats[1], lStats[1], wStats[2], lStats[2], wStats[3], lStats[3]]
    
    dataset.append(features)

dataset = np.array(dataset)
dataset = dataset.astype(np.float32)

# make tensor
dataset = torch.from_numpy(dataset)

# make predictions
alive_teams = []
alive_seeds = []
with torch.no_grad():
    i = 0
    for features in dataset:
        output = model(features)
        team = torch.argmax(output).item()
        alive_teams.append(matchups[i][team])
        alive_seeds.append(seeds[i][team])
        i = i + 1

print(alive_teams)
print(alive_seeds)

"""
ELITE EIGHT
"""

# build next round of matchups
matchups = []
seeds = []
for i in range(0, 8, 2):
    k = 0
    for j in range(i, i + 1):
        matchups.append([alive_teams[j], alive_teams[j + 1]])
        seeds.append([alive_seeds[j], alive_seeds[j + 1]])
        k = k + 1

# build dataset
dataset = []
for matchup in matchups:
    wSeed = seeds[matchups.index(matchup)][0]
    lSeed = seeds[matchups.index(matchup)][1]
    wStats = statsMap[matchup[0]][year]
    lStats = statsMap[matchup[1]][year]

    features = [wSeed, lSeed, wStats[0], lStats[0], wStats[1], lStats[1], wStats[2], lStats[2], wStats[3], lStats[3]]
    
    dataset.append(features)

dataset = np.array(dataset)
dataset = dataset.astype(np.float32)

# make tensor
dataset = torch.from_numpy(dataset)

# make predictions
alive_teams = []
alive_seeds = []
with torch.no_grad():
    i = 0
    for features in dataset:
        output = model(features)
        team = torch.argmax(output).item()
        alive_teams.append(matchups[i][team])
        alive_seeds.append(seeds[i][team])
        i = i + 1

print(alive_teams)
print(alive_seeds)

"""
FINAL FOUR
"""

# build next round of matchups
matchups = []
seeds = []
for i in range(0, 4, 2):
    matchups.append([alive_teams[i], alive_teams[i + 1]])
    seeds.append([alive_seeds[i], alive_seeds[i + 1]])


# build dataset
dataset = []
for matchup in matchups:
    wSeed = seeds[matchups.index(matchup)][0]
    lSeed = seeds[matchups.index(matchup)][1]
    wStats = statsMap[matchup[0]][year]
    lStats = statsMap[matchup[1]][year]

    features = [wSeed, lSeed, wStats[0], lStats[0], wStats[1], lStats[1], wStats[2], lStats[2], wStats[3], lStats[3]]
    
    dataset.append(features)

dataset = np.array(dataset)
dataset = dataset.astype(np.float32)

# make tensor
dataset = torch.from_numpy(dataset)

# make predictions
alive_teams = []
alive_seeds = []
with torch.no_grad():
    i = 0
    for features in dataset:
        output = model(features)
        team = torch.argmax(output).item()
        alive_teams.append(matchups[i][team])
        alive_seeds.append(seeds[i][team])
        i = i + 1

print(alive_teams)

"""
CHAMPIONSHIP
"""

# build next round of matchups
matchups = []
seeds = []
matchups.append([alive_teams[0], alive_teams[1]])
seeds.append([alive_seeds[0], alive_seeds[1]])


# build dataset
dataset = []
for matchup in matchups:
    wSeed = seeds[matchups.index(matchup)][0]
    lSeed = seeds[matchups.index(matchup)][1]
    wStats = statsMap[matchup[0]][year]
    lStats = statsMap[matchup[1]][year]

    features = [wSeed, lSeed, wStats[0], lStats[0], wStats[1], lStats[1], wStats[2], lStats[2], wStats[3], lStats[3]]
    
    dataset.append(features)

dataset = np.array(dataset)
dataset = dataset.astype(np.float32)

# make tensor
dataset = torch.from_numpy(dataset)

# make predictions
alive_teams = []
alive_seeds = []
with torch.no_grad():
    i = 0
    for features in dataset:
        output = model(features)
        team = torch.argmax(output).item()
        alive_teams.append(matchups[i][team])
        alive_seeds.append(seeds[i][team])
        i = i + 1

print(alive_teams)