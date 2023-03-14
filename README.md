# The March Madness Neural Network
This is a small project I hacked together for fun in about 2 hours. The goal of it is to create a neural network
that will predict the outcome of the March Madness Tournament.

## Current Methodology

### 2023 Models
The network is trained on the last 35 years of regular season and postseason games. The network takes in seeding, offensive, defensive, and net ratings per season, along with the win ratio, and uses it to predict performance in head 
to head matchups each season.

The architecture of the neural network is a 3-layer standard neural network with 10 parameters in the input, 36 in the first hidden layer, 12 in the second hidden layer, and 2 in the output layer. The network was trained with a batch size of 64, using the Stochastic Gradient Descent Optimizer, and Mean-Squared Error Loss.

### 2022 Model
The network is trained on over 2000 past games and makes probabilistic predictions based on direct seed matchups. It then simulates roughly 80 games and takes a moving average, before finally applying a second probabilistic prediction based on 
the results of the 80 games to make its final bracket. 

## Future Improvements
For the 2024 model, the primary area of focus is to consider strength of schedule, as well as strength of opponent play. Another area that is also being considered is to consider more stats, such as rebounding, assisting, efficiency, turnovers, steals, and blocks. This could make a more enriching dataset for the model.

## Technologies Used
The project is developed in Python 3
- Pandas: Data Importing
- Numpy: Data Manipulation
- PyTorch: Neural Network Training for 2023 Model
- Tensorflow: Neural Network Training for 2022 Model

Data Source: https://www.kaggle.com/competitions/march-machine-learning-mania-2023

## 2023's Predictions
First, I ran it using the 2022 version of the model.
![2023 March Madness Predictions](/2023-22.png)

Then, I ran it using the 2023 version of the model.
![2023 March Madness Predictions](/2023-23.png)

## 2022's Prediction Recap!
So, this is what the model predicted.
![2022 March Madness Predictions](/2022.png)
The actual results weren't that great. Out of a possible 192 points, the model scored a low 56.

