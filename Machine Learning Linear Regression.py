#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#reading the data 

df = pd.read_csv('/content/drive/MyDrive/Datasets/50_Startups.csv')
df.head()


