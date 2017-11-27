# Upper Confidence Bound
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


# Implement UCB
import math
d = 10
N = 10000
ad_selected = []
numbers_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if (numbers_of_selections[i] > 0):
            average_reward = sum_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/numbers_of_selections[i])
            upperbound = average_reward + delta_i
        else:
            upperbound=1e400            
        if upperbound > max_upper_bound:
            max_upper_bound = upperbound
            ad = i
    ad_selected.append(ad)
    numbers_of_selections[ad]=numbers_of_selections[ad]+1
    sum_of_rewards[ad]=sum_of_rewards[ad]+dataset.iloc[n,ad]
    total_reward=sum(sum_of_rewards)
        
# Plot the Histogram of distribution
plt.hist(ad_selected)
plt.title("Histogram of Ads Selected")
plt.xlabel("Ads")
plt.ylabel("Number of times shown")
plt.show