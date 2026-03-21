import torch
from torch import nn 
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

import copy
import math

# Set random seed for reproducibility
torch.manual_seed(0)

# Build Map from TeamID to Team Names
teams = pd.read_csv('2024/data/Team Results.csv').to_numpy()
teamList = []

for row in teams:
    teamList.append(row[1])

# Start gathering stats
# 538 Rating
    
def roundMap(num):
    if (num == 0):
        num = 64
    return 7 - math.log(num, 2)

# Heat-Check Rating
# Format: [YEAR,TEAM NO,TEAM,SEED,ROUND,POWER,PATH,DRAW,WINS,POOL VALUE,POOL S-RANK,NCAA S-RANK,VAL Z-SCORE,POWER-PATH]

data_heat = pd.read_csv('2024/data/Heat Check Tournament Index.csv')
# Split dataset into two datasets: 2023-earlier and 2024
data_heat_2023 = data_heat[data_heat['YEAR'] != 2024]
data_heat_2024 = data_heat[data_heat['YEAR'] == 2024]

data_heat = data_heat_2023.to_numpy()
currYear = 2023
statMap_2 = {}
yearMap = {}

for row in data_heat:
    if row[0] != currYear:
        # Make a deep copy of yearMap
        yearMapCopy = copy.deepcopy(yearMap)
        statMap_2[currYear] = yearMapCopy
        currYear = row[0]
        yearMap = {}
    team = row[2]
    seed = row[3]
    round = roundMap(int(row[4]))
    power = int(row[5])
    path = int(row[6])
    draw = int(row[7])
    wins = int(row[8])
    pool_value = float(row[9])
    pool_srank = int(row[10])
    ncaa_srank = int(row[11])
    val_zscore = float(row[12])
    power_path = int(row[13])
    yearMap[team] = [power, path, draw, wins, pool_value, pool_srank, ncaa_srank, val_zscore, power_path]

statMap_2[currYear] = yearMap

# KenPom
# [YEAR,CONF,CONF ID,QUAD NO,QUAD ID,TEAM NO,TEAM ID,TEAM,SEED,ROUND,K TEMPO,K TEMPO RANK,KADJ T,KADJ T RANK,K OFF,KO RANK,KADJ O,KADJ O RANK,K DEF,KD RANK,KADJ D,KADJ D RANK,KADJ EM,KADJ EM RANK,BADJ EM,BADJ O,BADJ D,BARTHAG,GAMES,W,L,WIN%,EFG%,EFG%D,FTR,FTRD,TOV%,TOV%D,OREB%,DREB%,OP OREB%,OP DREB%,RAW T,2PT%,2PT%D,3PT%,3PT%D,BLK%,BLKED%,AST%,OP AST%,2PTR,3PTR,2PTRD,3PTRD,BADJ T,AVG HGT,EFF HGT,EXP,TALENT,FT%,OP FT%,PPPO,PPPD,ELITE SOS,WAB,BADJ EM RANK,BADJ O RANK,BADJ D RANK,BARTHAG RANK,EFG% RANK,EFGD% RANK,FTR RANK,FTRD RANK,TOV% RANK,TOV%D RANK,OREB% RANK,DREB% RANK,OP OREB% RANK,OP DREB% RANK,RAW T RANK,2PT% RANK,2PT%D RANK,3PT% RANK,3PT%D RANK,BLK% RANK,BLKED% RANK,AST% RANK,OP AST% RANK,2PTR RANK,3PTR RANK,2PTRD RANK,3PTRD RANK,BADJT RANK,AVG HGT RANK,EFF HGT RANK,EXP RANK,TALENT RANK,FT% RANK,OP FT% RANK,PPPO RANK,PPPD RANK,ELITE SOS RANK]
data_ken = pd.read_csv('2024/data/KenPom Barttorvik.csv')

# Split dataset into two datasets: 2023-earlier and 2024
data_ken_2023 = data_ken[data_ken['YEAR'] != 2024]
data_ken_2024 = data_ken[data_ken['YEAR'] == 2024]

data_ken = data_ken_2023.to_numpy()

currYear = 2023
statMap_3 = {}
yearMap = {}

for row in data_ken:
    if row[0] != currYear:
        # Make a deep copy of yearMap
        yearMapCopy = copy.deepcopy(yearMap)
        statMap_3[currYear] = yearMapCopy
        currYear = row[0]
        yearMap = {}
        exists = False

    team = row[7]
    seed = row[8]
    round = roundMap(int(row[9]))
    k_tempo = float(row[10])
    k_tempo_rank = int(row[11])
    k_adj_t = float(row[12])
    k_adj_t_rank = int(row[13])
    k_off = float(row[14])
    ko_rank = int(row[15])
    k_adj_o = float(row[16])
    k_adj_o_rank = int(row[17])
    k_def = float(row[18])
    kd_rank = int(row[19])
    k_adj_d = float(row[20])
    kadj_d_rank = int(row[21])
    kadj_em = float(row[22])
    kadj_em_rank = int(row[23])
    badj_em = float(row[24])
    badj_o = float(row[25])
    badj_d = float(row[26])
    barthag = float(row[27])
    games = int(row[28])
    w = int(row[29])
    l = int(row[30])
    win_percent = float(row[31])
    efg_percent = float(row[32])
    efgd_percent = float(row[33])
    ftr = float(row[34])
    ftrd = float(row[35])
    tov_percent = float(row[36])
    tovd_percent = float(row[37])
    oreb_percent = float(row[38])
    dreb_percent = float(row[39])
    op_oreb_percent = float(row[40])
    op_dreb_percent = float(row[41])
    raw_t = float(row[42])
    twopt_percent = float(row[43])
    twoptd_percent = float(row[44])
    threept_percent = float(row[45])
    threeptd_percent = float(row[46])
    blk_percent = float(row[47])
    blked_percent = float(row[48])
    ast_percent = float(row[49])
    op_ast_percent = float(row[50])
    twoptr = float(row[51])
    threeptr = float(row[52])
    twoptrd = float(row[53])
    threeptd = float(row[54])
    badjt = float(row[55])
    avg_hgt = float(row[56])
    eff_hgt = float(row[57])
    exp = float(row[58])
    talent = float(row[59])
    ft_percent = float(row[60])
    op_ft_percent = float(row[61])
    pp_po = float(row[62])
    pp_pd = float(row[63])
    elite_sos = float(row[64])
    wab = float(row[65])
    badj_em_rank = int(row[66])
    badj_o_rank = int(row[67])
    badj_d_rank = int(row[68])
    barthag_rank = int(row[69])
    efg_percent_rank = int(row[70])
    efgd_percent_rank = int(row[71])
    ftr_rank = int(row[72])
    ftrd_rank = int(row[73])
    tov_percent_rank = int(row[74])
    tovd_percent_rank = int(row[75])
    oreb_percent_rank = int(row[76])
    dreb_percent_rank = int(row[77])
    op_oreb_percent_rank = int(row[78])
    op_dreb_percent_rank = int(row[79])
    raw_t_rank = int(row[80])
    twopt_percent_rank = int(row[81])
    twoptd_percent_rank = int(row[82])
    threept_percent_rank = int(row[83])
    threeptd_percent_rank = int(row[84])
    blk_percent_rank = int(row[85])
    blked_percent_rank = int(row[86])
    ast_percent_rank = int(row[87])
    op_ast_percent_rank = int(row[88])
    twoptr_rank = int(row[89])
    threeptr_rank = int(row[90])
    twoptrd_rank = int(row[91])
    threeptd_rank = int(row[92])
    badjt_rank = int(row[93])
    avg_hgt_rank = int(row[94])
    eff_hgt_rank = int(row[95])
    exp_rank = int(row[96])
    talent_rank = int(row[97])
    ft_percent_rank = int(row[98])
    op_ft_percent_rank = int(row[99])
    pp_po_rank = int(row[100])
    pp_pd_rank = int(row[101])
    elite_sos_rank = int(row[102])
    yearMap[team] = [k_tempo, k_tempo_rank, k_adj_t, k_adj_t_rank, k_off, ko_rank, k_adj_o, k_adj_o_rank, k_def, kd_rank, k_adj_d, kadj_d_rank, kadj_em, kadj_em_rank, badj_em, badj_o, badj_d, barthag, games, w, l, win_percent, efg_percent, efgd_percent, ftr, ftrd, tov_percent, tovd_percent, oreb_percent, dreb_percent, op_oreb_percent, op_dreb_percent, raw_t, twopt_percent, twoptd_percent, threept_percent, threeptd_percent, blk_percent, blked_percent, ast_percent, op_ast_percent, twoptr, threeptr, twoptrd, threeptd, badjt, avg_hgt, eff_hgt, exp, talent, ft_percent, op_ft_percent, pp_po, pp_pd, elite_sos, wab, badj_em_rank, badj_o_rank, badj_d_rank, barthag_rank, efg_percent_rank, efgd_percent_rank, ftr_rank, ftrd_rank, tov_percent_rank, tovd_percent_rank, oreb_percent_rank, dreb_percent_rank, op_oreb_percent_rank, op_dreb_percent_rank, raw_t_rank, twopt_percent_rank, twoptd_percent_rank, threept_percent_rank, threeptd_percent_rank, blk_percent_rank, blked_percent_rank, ast_percent_rank, op_ast_percent_rank, twoptr_rank, threeptr_rank, twoptrd_rank, threeptd_rank, badjt_rank, avg_hgt_rank, eff_hgt_rank, exp_rank, talent_rank, ft_percent_rank, op_ft_percent_rank, pp_po_rank, pp_pd_rank, elite_sos_rank]

statMap_3[currYear] = yearMap

# Combine all the stats
realStatsMap = {}
for team in teamList:
    # create array for each team from each statmap
    for year in range(2008, 2024, 1):
        teamArray = []
        if year in statMap_2:
            if team in statMap_2[year]:
                teamArray += statMap_2[year][team]
            else:
                teamArray += [0 for i in range(9)]
        else:
            teamArray += [0 for i in range(9)]
        if year in statMap_3:
            if team in statMap_3[year]:
                teamArray += statMap_3[year][team]
            else:
                teamArray += [0 for i in range(93)]
        else:
            teamArray += [0 for i in range(93)]
        realStatsMap[(year, team)] = teamArray

# Build Matchup Data
matchups = pd.read_csv('2024/data/Tournament Matchups.csv').to_numpy()
matchupData = []
i = 0

for i in range(0, len(matchups), 2):
    matchupA = []
    year = matchups[i][0]
    teamA = matchups[i][4]
    teamB = matchups[i+1][4]
    round = roundMap(matchups[i][-2])
    scoreA = matchups[i][-1]
    scoreB = matchups[i + 1][-1]
    winner = 0
    if scoreA < scoreB:
        winner = 1
    teamAStats = realStatsMap[(year, teamA)]
    teamBStats = realStatsMap[(year, teamB)]
    # add each element from statMap to matchupA
    matchupA.append(year)
    matchupA.append(round)
    for j in range(len(teamAStats)):
        matchupA.append(teamAStats[j])
    for j in range(len(teamBStats)):
        matchupA.append(teamBStats[j])
    matchupA.append(winner)
    matchupData.append(matchupA)

    # Invert the problem
    matchupB = []
    winner = 1 - winner
    matchupB.append(year)
    matchupB.append(round)
    for j in range(len(teamBStats)):
        matchupB.append(teamBStats[j])
    for j in range(len(teamAStats)):
        matchupB.append(teamAStats[j])
    matchupB.append(winner)

    matchupData.append(matchupB)

# Split into training and testing data
matchupData = np.array(matchupData)
X = matchupData[:, 2:-1]
y = matchupData[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(204, 500)
        self.fc2 = nn.Linear(500, 75)
        self.fc3 = nn.Linear(75, 25)
        self.fc4 = nn.Linear(25, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=0.03)

# Cast to tensor
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# Change to float
X_train = X_train.float()
X_test = X_test.float()

# changte y_type to long
y_train = y_train.long()
y_test = y_test.long()

# Build data loaders
train_data = []
for i in range(len(X_train)):
    train_data.append([X_train[i], y_train[i]])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = []
for i in range(len(X_test)):
    test_data.append([X_test[i], y_test[i]])
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


# Train the model
epochs = 100
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# Test the model
    
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    # Print accuracy
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

# Make Predictions!
    
# Gather 2024 Data from KenPom, and Heat Check
currYear = 2024
statMap_2 = {}
yearMap = {}

for row in data_heat_2024.to_numpy():
    if row[0] != currYear:
        # Make a deep copy of yearMap
        yearMapCopy = copy.deepcopy(yearMap)
        statMap_2[currYear] = yearMapCopy
        currYear = row[0]
        yearMap = {}
    team = row[2]
    seed = row[3]
    round = roundMap(int(row[4]))
    power = int(row[5])
    path = int(row[6])
    draw = 0
    wins = int(row[8])
    pool_value = float(row[9])
    pool_srank = int(row[10])
    ncaa_srank = int(row[11])
    val_zscore = 0
    power_path = int(row[13])
    yearMap[team] = [power, path, draw, wins, pool_value, pool_srank, ncaa_srank, val_zscore, power_path]

statMap_2[currYear] = yearMap
    
currYear = 2024
statMap_3 = {}
yearMap = {}

for row in data_ken_2024.to_numpy():
    if row[0] != currYear:
        # Make a deep copy of yearMap
        yearMapCopy = copy.deepcopy(yearMap)
        statMap_3[currYear] = yearMapCopy
        currYear = row[0]
        yearMap = {}
        exists = False

    team = row[7]
    seed = row[8]
    round = roundMap(int(row[9]))
    k_tempo = float(row[10])
    k_tempo_rank = int(row[11])
    k_adj_t = float(row[12])
    k_adj_t_rank = int(row[13])
    k_off = float(row[14])
    ko_rank = int(row[15])
    k_adj_o = float(row[16])
    k_adj_o_rank = int(row[17])
    k_def = float(row[18])
    kd_rank = int(row[19])
    k_adj_d = float(row[20])
    kadj_d_rank = int(row[21])
    kadj_em = float(row[22])
    kadj_em_rank = int(row[23])
    badj_em = float(row[24])
    badj_o = float(row[25])
    badj_d = float(row[26])
    barthag = float(row[27])
    games = int(row[28])
    w = int(row[29])
    l = int(row[30])
    win_percent = float(row[31])
    efg_percent = float(row[32])
    efgd_percent = float(row[33])
    ftr = float(row[34])
    ftrd = float(row[35])
    tov_percent = float(row[36])
    tovd_percent = float(row[37])
    oreb_percent = float(row[38])
    dreb_percent = float(row[39])
    op_oreb_percent = float(row[40])
    op_dreb_percent = float(row[41])
    raw_t = float(row[42])
    twopt_percent = float(row[43])
    twoptd_percent = float(row[44])
    threept_percent = float(row[45])
    threeptd_percent = float(row[46])
    blk_percent = float(row[47])
    blked_percent = float(row[48])
    ast_percent = float(row[49])
    op_ast_percent = float(row[50])
    twoptr = float(row[51])
    threeptr = float(row[52])
    twoptrd = float(row[53])
    threeptd = float(row[54])
    badjt = float(row[55])
    avg_hgt = float(row[56])
    eff_hgt = float(row[57])
    exp = float(row[58])
    talent = float(row[59])
    ft_percent = float(row[60])
    op_ft_percent = float(row[61])
    pp_po = float(row[62])
    pp_pd = float(row[63])
    elite_sos = float(row[64])
    wab = float(row[65])
    badj_em_rank = int(row[66])
    badj_o_rank = int(row[67])
    badj_d_rank = int(row[68])
    barthag_rank = int(row[69])
    efg_percent_rank = int(row[70])
    efgd_percent_rank = int(row[71])
    ftr_rank = int(row[72])
    ftrd_rank = int(row[73])
    tov_percent_rank = int(row[74])
    tovd_percent_rank = int(row[75])
    oreb_percent_rank = int(row[76])
    dreb_percent_rank = int(row[77])
    op_oreb_percent_rank = int(row[78])
    op_dreb_percent_rank = int(row[79])
    raw_t_rank = int(row[80])
    twopt_percent_rank = int(row[81])
    twoptd_percent_rank = int(row[82])
    threept_percent_rank = int(row[83])
    threeptd_percent_rank = int(row[84])
    blk_percent_rank = int(row[85])
    blked_percent_rank = int(row[86])
    ast_percent_rank = int(row[87])
    op_ast_percent_rank = int(row[88])
    twoptr_rank = int(row[89])
    threeptr_rank = int(row[90])
    twoptrd_rank = int(row[91])
    threeptd_rank = int(row[92])
    badjt_rank = int(row[93])
    avg_hgt_rank = int(row[94])
    eff_hgt_rank = int(row[95])
    exp_rank = int(row[96])
    talent_rank = int(row[97])
    ft_percent_rank = int(row[98])
    op_ft_percent_rank = int(row[99])
    pp_po_rank = int(row[100])
    pp_pd_rank = int(row[101])
    elite_sos_rank = int(row[102])
    yearMap[team] = [k_tempo, k_tempo_rank, k_adj_t, k_adj_t_rank, k_off, ko_rank, k_adj_o, k_adj_o_rank, k_def, kd_rank, k_adj_d, kadj_d_rank, kadj_em, kadj_em_rank, badj_em, badj_o, badj_d, barthag, games, w, l, win_percent, efg_percent, efgd_percent, ftr, ftrd, tov_percent, tovd_percent, oreb_percent, dreb_percent, op_oreb_percent, op_dreb_percent, raw_t, twopt_percent, twoptd_percent, threept_percent, threeptd_percent, blk_percent, blked_percent, ast_percent, op_ast_percent, twoptr, threeptr, twoptrd, threeptd, badjt, avg_hgt, eff_hgt, exp, talent, ft_percent, op_ft_percent, pp_po, pp_pd, elite_sos, wab, badj_em_rank, badj_o_rank, badj_d_rank, barthag_rank, efg_percent_rank, efgd_percent_rank, ftr_rank, ftrd_rank, tov_percent_rank, tovd_percent_rank, oreb_percent_rank, dreb_percent_rank, op_oreb_percent_rank, op_dreb_percent_rank, raw_t_rank, twopt_percent_rank, twoptd_percent_rank, threept_percent_rank, threeptd_percent_rank, blk_percent_rank, blked_percent_rank, ast_percent_rank, op_ast_percent_rank, twoptr_rank, threeptr_rank, twoptrd_rank, threeptd_rank, badjt_rank, avg_hgt_rank, eff_hgt_rank, exp_rank, talent_rank, ft_percent_rank, op_ft_percent_rank, pp_po_rank, pp_pd_rank, elite_sos_rank]

statMap_3[currYear] = yearMap

# Combine all the stats
realStatsMap = {}

# Fresh 2024 Team Map
data_2024 = pd.read_csv('2024/data/Tournament Simulation.csv').to_numpy()
teamList = []
for i in range(len(data_2024)):
    teamList.append(data_2024[i][4])

for team in teamList:
    # create array for each team from each statmap
    for year in range(2024, 2025, 1):
        teamArray = []
        if year in statMap_2:
            if team in statMap_2[year]:
                teamArray += statMap_2[year][team]
            else:
                teamArray += [0 for i in range(9)]
        else:
            teamArray += [0 for i in range(9)]
        if year in statMap_3:
            if team in statMap_3[year]:
                teamArray += statMap_3[year][team]
            else:
                teamArray += [0 for i in range(93)]
        else:
            teamArray += [0 for i in range(93)]
        realStatsMap[(year, team)] = teamArray


"""
ROUND 1
"""

# Gather 2024 Matchups
matchupData_2024 = []
matchupTeams = []

for i in range(0, len(data_2024), 2):
    matchupA = []
    year = 2024
    teamA = data_2024[i][4]
    teamB = data_2024[i + 1][4]
    round = 1
    teamAStats = realStatsMap[(year, teamA)]
    teamBStats = realStatsMap[(year, teamB)]
    # add each element from statMap to matchupA
    matchupA.append(year)
    matchupA.append(round)
    for j in range(len(teamAStats)):
        matchupA.append(teamAStats[j])
    for j in range(len(teamBStats)):
        matchupA.append(teamBStats[j])
    matchupData_2024.append(matchupA)
    matchupTeams.append((teamA, teamB))

# Make Predictions
matchupData_2024 = np.array(matchupData_2024)
x_test = matchupData_2024[:, 2:]

# Cast to tensor
x_test = torch.tensor(x_test)

# Change to float
x_test = x_test.float()


# Make Predictions
predictions = []
with torch.no_grad():
    for data in x_test:
        inputs = data
        outputs = model(inputs)
        # turn outputs to probabilities
        outputs = torch.softmax(outputs, 0)
        predicted = torch.argmax(outputs)
        predictions.append(predicted)

# Get Winners
winners = []
for i in range(len(matchupTeams)):
    if predictions[i] == 0:
        winners.append(matchupTeams[i][0])
    else:
        winners.append(matchupTeams[i][1])

print(winners)

"""
ROUND 2
"""

# Gather 2024 Matchups
matchupData_2024 = []
matchupTeams = []

for i in range(0, len(winners), 2):
    matchupA = []
    year = 2024
    teamA = winners[i]
    teamB = winners[i + 1]
    round = 2
    teamAStats = realStatsMap[(year, teamA)]
    teamBStats = realStatsMap[(year, teamB)]
    # add each element from statMap to matchupA
    matchupA.append(year)
    matchupA.append(round)
    for j in range(len(teamAStats)):
        matchupA.append(teamAStats[j])
    for j in range(len(teamBStats)):
        matchupA.append(teamBStats[j])
    matchupData_2024.append(matchupA)
    matchupTeams.append((teamA, teamB))

# Make Predictions
matchupData_2024 = np.array(matchupData_2024)
x_test = matchupData_2024[:, 2:]

# Cast to tensor
x_test = torch.tensor(x_test)

# Change to float
x_test = x_test.float()


# Make Predictions
predictions = []
with torch.no_grad():
    for data in x_test:
        inputs = data
        outputs = model(inputs)
        # turn outputs to probabilities
        outputs = torch.softmax(outputs, 0)
        predicted = torch.argmax(outputs)
        predictions.append(predicted)

# Get Winners
winners = []
for i in range(len(matchupTeams)):
    if predictions[i] == 0:
        winners.append(matchupTeams[i][0])
    else:
        winners.append(matchupTeams[i][1])

print(winners)

"""
SWEET SIXTEEN
"""

# Gather 2024 Matchups
matchupData_2024 = []
matchupTeams = []

for i in range(0, len(winners), 2):
    matchupA = []
    year = 2024
    teamA = winners[i]
    teamB = winners[i + 1]
    round = 3
    teamAStats = realStatsMap[(year, teamA)]
    teamBStats = realStatsMap[(year, teamB)]
    # add each element from statMap to matchupA
    matchupA.append(year)
    matchupA.append(round)
    for j in range(len(teamAStats)):
        matchupA.append(teamAStats[j])
    for j in range(len(teamBStats)):
        matchupA.append(teamBStats[j])
    matchupData_2024.append(matchupA)
    matchupTeams.append((teamA, teamB))

# Make Predictions
matchupData_2024 = np.array(matchupData_2024)
x_test = matchupData_2024[:, 2:]

# Cast to tensor
x_test = torch.tensor(x_test)

# Change to float
x_test = x_test.float()


# Make Predictions
predictions = []
with torch.no_grad():
    for data in x_test:
        inputs = data
        outputs = model(inputs)
        # turn outputs to probabilities
        outputs = torch.softmax(outputs, 0)
        predicted = torch.argmax(outputs)
        predictions.append(predicted)

# Get Winners
winners = []
for i in range(len(matchupTeams)):
    if predictions[i] == 0:
        winners.append(matchupTeams[i][0])
    else:
        winners.append(matchupTeams[i][1])

print(winners)

"""
ELITE EIGHT
"""

# Gather 2024 Matchups
matchupData_2024 = []
matchupTeams = []

for i in range(0, len(winners), 2):
    matchupA = []
    year = 2024
    teamA = winners[i]
    teamB = winners[i + 1]
    round = 4
    teamAStats = realStatsMap[(year, teamA)]
    teamBStats = realStatsMap[(year, teamB)]
    # add each element from statMap to matchupA
    matchupA.append(year)
    matchupA.append(round)
    for j in range(len(teamAStats)):
        matchupA.append(teamAStats[j])
    for j in range(len(teamBStats)):
        matchupA.append(teamBStats[j])
    matchupData_2024.append(matchupA)
    matchupTeams.append((teamA, teamB))

# Make Predictions
matchupData_2024 = np.array(matchupData_2024)
x_test = matchupData_2024[:, 2:]

# Cast to tensor
x_test = torch.tensor(x_test)

# Change to float
x_test = x_test.float()


# Make Predictions
predictions = []
with torch.no_grad():
    for data in x_test:
        inputs = data
        outputs = model(inputs)
        # turn outputs to probabilities
        outputs = torch.softmax(outputs, 0)
        predicted = torch.argmax(outputs)
        predictions.append(predicted)

# Get Winners
winners = []
for i in range(len(matchupTeams)):
    if predictions[i] == 0:
        winners.append(matchupTeams[i][0])
    else:
        winners.append(matchupTeams[i][1])

print(winners)

"""
FINAL FOUR
"""

# Gather 2024 Matchups
matchupData_2024 = []
matchupTeams = []

for i in range(0, len(winners), 2):
    matchupA = []
    year = 2024
    teamA = winners[i]
    teamB = winners[i + 1]
    round = 5
    teamAStats = realStatsMap[(year, teamA)]
    teamBStats = realStatsMap[(year, teamB)]
    # add each element from statMap to matchupA
    matchupA.append(year)
    matchupA.append(round)
    for j in range(len(teamAStats)):
        matchupA.append(teamAStats[j])
    for j in range(len(teamBStats)):
        matchupA.append(teamBStats[j])
    matchupData_2024.append(matchupA)
    matchupTeams.append((teamA, teamB))

# Make Predictions
matchupData_2024 = np.array(matchupData_2024)
x_test = matchupData_2024[:, 2:]

# Cast to tensor
x_test = torch.tensor(x_test)

# Change to float
x_test = x_test.float()


# Make Predictions
predictions = []
with torch.no_grad():
    for data in x_test:
        inputs = data
        outputs = model(inputs)
        # turn outputs to probabilities
        outputs = torch.softmax(outputs, 0)
        predicted = torch.argmax(outputs)
        predictions.append(predicted)

# Get Winners
winners = []
for i in range(len(matchupTeams)):
    if predictions[i] == 0:
        winners.append(matchupTeams[i][0])
    else:
        winners.append(matchupTeams[i][1])

print(winners)
    
"""
CHAMPIONSHIP
"""

# Gather 2024 Matchups
matchupA = []
year = 2024
teamA = winners[0]
teamB = winners[1]
round = 6
teamAStats = realStatsMap[(year, teamA)]
teamBStats = realStatsMap[(year, teamB)]
# add each element from statMap to matchupA
matchupA.append(year)
matchupA.append(round)
for j in range(len(teamAStats)):
    matchupA.append(teamAStats[j])
for j in range(len(teamBStats)):
    matchupA.append(teamBStats[j])

# Make Predictions
matchupData_2024 = np.array([matchupA])
x_test = matchupData_2024[:, 2:]

# Cast to tensor
x_test = torch.tensor(x_test)

# Change to float
x_test = x_test.float()


# Make Predictions
predictions = []
with torch.no_grad():
    for data in x_test:
        inputs = data
        outputs = model(inputs)
        # turn outputs to probabilities
        outputs = torch.softmax(outputs, 0)
        predicted = torch.argmax(outputs)
        predictions.append(predicted)

# Get Winners
winner = ""
if predictions[0] == 0:
    winner = teamA
else:
    winner = teamB

print(winner)

