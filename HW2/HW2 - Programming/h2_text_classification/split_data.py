import pandas as pd
import numpy as np

path = 'data/'

df = pd.read_csv(path + 'full_data.csv')
train_indices = np.random.rand(len(df)) <= 0.8

train = df[train_indices]
test = df[~train_indices]

train.to_csv(path + 'train.csv', index=False)
test.to_csv(path + 'test.csv', index=False)
