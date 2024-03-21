# The Annual March Madness Neural Network Challenge
Every year I give myself 6 hours to hack together a neural network (including data processing) to predict the outcome of 
the March Madness bracket each year. This challenge has evolved from something rather simple when I first started it into 2022, to an investigation on the power of machine learning models, and how different models have their strengths and limitations in predicting an event with both deterministic and stochastic events. 

March Madness as a whole presents a really intriguting challenge for probabilistic and statistical sports modeling, due to the sport's basis on cold numbers and the fact that one team typically is better than the other, while also enabling randomness due to its one-and-done nature.

Over the years, I've evolved from simple neural networks evaluating seed, to more complex neural networks taking in statistical information. The current model uses over 100 advanced statistics over both regular season and tournament matches to capture the most data and trends amongst March Madness over the years.

## Current Methodology

### 2024 Model
The network is trained on the last 16 years of regular season and postseason games. Unlike the last two years, along with seeding, 
the model also takes in various advanced stats calculated by KenPom and HeatCheck. This data was sourced from Kaggle at the following [link](https://www.kaggle.com/datasets/nishaanamin/march-madness-data). Overall, each team now has 102 datapoints 
associated with it. In addition, when the datapoints was being constructed for the model, each datapoint was given twice, once in the form of (Team A, Team B, Team B Win Status) and (Team B, Team A, Team A Win Status), to capture and dissuade any trend of picking the first entry in each set (as each match was always structured in high seed, followed by low seed).

Thank you to [KenPom](https://kenpom.com/) and [HeatCheck](https://heatcheckcbb.com/) for the data!

The architecture of the neural network is a 4-layer vanilla neural network with 204 parameters in the input, 500 in the first hidden layer, 75 in the second hidden layer, 25 in the third hidden layer, and 2 in the output. The network was trained with a 
batch size of 32, using the [Adamax](https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html) optimizer and CrossEntropyLoss.

### 2023 Models
The network is trained on the last 35 years of regular season and postseason games. The network takes in seeding, offensive, defensive, and net ratings per season, along with the win ratio, and uses it to predict performance in head 
to head matchups each season.

The architecture of the neural network is a 3-layer standard neural network with 10 parameters in the input, 36 in the first hidden layer, 12 in the second hidden layer, and 2 in the output layer. The network was trained with a batch size of 64, using the Stochastic Gradient Descent Optimizer, and Mean-Squared Error Loss.

### 2022 Model
The network is trained on over 2000 past games and makes probabilistic predictions based on direct seed matchups. It then simulates roughly 80 games and takes a moving average, before finally applying a second probabilistic prediction based on 
the results of the 80 games to make its final bracket. 

## Stats
2022 Model: 2 Layers, Vanilla Architecture, 880 parameters

2023 Model: 3 Layers, Vanilla Architecture, 1272 parameters

2024 Model: 4 Layers, Vanilla Architecture, 345425 parameters

## Future Improvements
Due to lack of time in 2024, many of the ensemble techniques that I wanted to explore this year were unused. In 2025, I plan to revisit using ensembles and non-deep models, perhaps some combination of the two involving simulations to take into account the probabilities outputed by the model. In general, the actual model architecture seems rather well optimized, and aside from hyperparameter tuning, it seems like the way to progress is through pipelining, expert advice, and other multi-model approaches.

In addition, reinforcement learning and recurrent neural networks are strategies worth exploring as well.

## Technologies Used
The project is developed in Python 3
- Pandas: Data Importing
- Numpy: Data Manipulation
- PyTorch: Neural Network Training for 2023/2024 Model
- Tensorflow: Neural Network Training for 2022 Model

Data Source for 2024: https://www.kaggle.com/datasets/nishaanamin/march-madness-data

Data Source 2022/2023: https://www.kaggle.com/competitions/march-machine-learning-mania-2023

## 2024's Predictions
Due to time, only the 2024 version of the model was ran.
![2024 March Madness Predictions](/2024.png)

For even more fun, I also ran this year's bracket through both [ChatGPT](https://chat.openai.com/) and [Claude](https://claude.ai/).
Their brackets are shown [here](https://fantasy.espn.com/tc/sharer?challengeId=240&from=espn&context=CHAMPION_PICK&entryId=498b8d70-e74b-11ee-a97a-352ad3702aa6&outcomeId=a497b9a1-c12d-11ee-b568-d9cd047f74cf&propositionId=a497b980-c12d-11ee-b568-d9cd047f74cf) and [here](https://fantasy.espn.com/tc/sharer?challengeId=240&from=espn&context=CHAMPION_PICK&entryId=e43874e0-e74c-11ee-a97a-352ad3702aa6&outcomeId=a497b9b7-c12d-11ee-b568-d9cd047f74cf&propositionId=a497b980-c12d-11ee-b568-d9cd047f74cf) respectively. With the explosion of LLMs over the last year, I wanted to see how well general purpose models would perform when compared to a specially statistically trained model.

## 2023's Predictions
First, I ran it using the 2022 version of the model.
![2023 March Madness Predictions](/2023-22.png)

Then, I ran it using the 2023 version of the model.
![2023 March Madness Predictions](/2023-23.png)

The actual results were pretty mediocre. Due to the high nature of upsets in this year's tournament, the 2022 model scored
45/192 points (placing in the 55th percentile), and the 2023 model scored 56/192 points (placing in the 88th percentile).
Both models heavily struggled in picking upsets in the Sweet 16, causing a cascading effect in the rest of the bracket.
The 2023 model was also a lot more conservative in picking upsets, causing the 2022 model to do a little better in the second round due to its nature to pick more wild upsets in the first round.

As a bonus, the 2023 model was also given a chance to participate in ESPN's Second Chance Challenge, allowing the model to re-create a bracket before the Sweet 16 began. The 2023 model scored 72/128 points, placing in the 98th percentile (top 70000 brackets)!!

A large proportion of this was actually the model recalibrating and picking UConn to go all the way, which it did. It turns out that the 2023 model is actually very good at picking upsets in the later rounds, but struggles in the first round. Hence, once given a correct Sweet 16 bracket, the model was able to do very well in the rest of the tournament, 
picking some rather interesting and unexpected upsets. However, the model still loved to favor 1 seeds, and so it was heavily penalized in the Sweet 16 for going book. That being said, the model performed exceptionally well at this stage.

In general, due to the nature of these models differing in the way they pick upsets, some form of ensemble model could work as one of the 2024 models.

For fun, the 2024 model was ran on the 2023 bracket, as an evaluatory metric. This model scored a ridiculous 119/192 points, which is a whopping 102% increase in performance over last year! The biggest improvement came in the fact that due to the 
nature of the advanced statistics being used, the increase in datasets, and the larger size of the model, it was able to more reliablty pick upsets in earlier rounds, which strongly benefitted it in later rounds. 
In fact, it had Florida Atlantic making the Final Four, which, as a 9 seed, is HIGHLY unlikely. Furthermore, it correctly identified the 5-seed San Diego State making the Championship, which again, is rather rare. The fact that the model picked these two upsets shows that the new datasets and the deeper model help improve performance dramatically.

## 2022's Prediction Recap!
So, this is what the model predicted.
![2022 March Madness Predictions](/2022.png)
The actual results weren't that great. Out of a possible 192 points, the model scored a low 56.

For fun, the 2023 model was also ran on the 2022 bracket, as an evaluatory metric. This model scored 76/192 points, 
which is a 35% improvement. The biggest improvement came in the fact that it correctly picked Kansas making the final which gave it a huge boost in score. On the contrary, the 2023 model was a lot more conservative in picking upsets, and so it was heavily penalized in the second round and Sweet Sixteen.

