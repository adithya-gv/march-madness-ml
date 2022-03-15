# The March Madness Neural Network
This is a small project I hacked together for fun in about 2 hours. The goal of it is to create a neural network
that will predict the outcome of the March Madness Tournament.

## Current Methodology
The network is trained on over 2000 past games and makes probabilistic predictions based on direct seed matchups. It then simulates roughly 80 games and takes a moving average, before finally applying a second probabilistic prediction based on 
the results of the 80 games to make its final bracket. 

## Future Improvements
One key area that can be added to improve this neural network is to consider a team's net rating, which takes into 
consideration both how many points a team scores as well as how efficient their defense is. Due to time restrictions, 
net rating was not added in the 2022 tournament's bracket, but is most likely planned to be used in some way for the 2023 bracket. 

## Technologies Used
The project is developed in Python 3
- Pandas: Data Importing
- Numpy: Data Manipulation
- Tensorflow: Neural Network Training

Data Source: https://www.kaggle.com/c/ncaam-march-mania-2021/data

## 2022's Predictions
Some of these results are... interesting. 
![2022 March Madness Predictions](/2022.png)