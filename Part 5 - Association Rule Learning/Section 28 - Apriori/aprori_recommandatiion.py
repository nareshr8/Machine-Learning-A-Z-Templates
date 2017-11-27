# Apriori Based Recommendation
# We need apyori.py in our Working directory to run APriory Algorithm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)

# Convert the dataset to List of Lists
transactions = []

for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
dataset.values

# Training Data Set
from apyori import apriori
# Considering items that is purchased atleast 3 times a day (3*7/7500), 
# 20% confident with the association
rules = apriori(transactions, min_support=0.003 ,
                min_confidence=0.2,min_lift= 3,min_length=2)

# Visualising the results
results =list(rules)

def inspect(results):
    rh          = [tuple(result[2][0][0])[0] for result in results]
    lh          = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))
# the line creates a date frame which is accessible from Variable explorer
resultDataFrame=pd.DataFrame(inspect(results))
