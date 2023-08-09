import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


import string
import warnings
warnings.filterwarnings('ignore')


# set display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
# set printing options to show all elements
np.set_printoptions(threshold=np.inf)

dataset = pd.read_csv('dataset.csv')

#1. Get a brief information on the data.
print('\nThe columns are:  ')
print(dataset.columns)

print('\n\nNumber of X tuples = {}'.format(dataset.shape[0]))

print('\n\n')
print(dataset.info())
print('\n\n')
print(dataset.head())

#2.Handling Missing Values

dataset_columns = dataset.columns.tolist()
print('\n')

for col in dataset_columns:
    print('{} column missing values: {}'.format(col, dataset[col].isnull().sum()))
    
#find duplicated row

#get all possible genre value
print('number of genre : {}'.format(dataset.track_genre.nunique()))
print(dataset.track_genre.unique())

#TODO
#drop row where genre=sleep
#drop loudness column
#normilize popularity using min-max normalixzatioin
#binarize explicite

#normalize data betweeb [0-1]
def minMax_normalization(column):
    clean_dataset[column] = (clean_dataset[column] - clean_dataset[column].min()) / (clean_dataset[column].max() - clean_dataset[column].min())
    return clean_dataset[column]       

#backup data 
clean_dataset=dataset.copy()

#drop dup data
clean_dataset.drop_duplicates(subset=['track_id'],inplace=True)

#drop loudness column
clean_dataset.drop('loudness',axis=1,inplace=True)

#drop the song where genre is sleep since there is mostly asmr like song
clean_dataset.drop(clean_dataset.index[clean_dataset['track_genre'] == 'sleep'], inplace=True)

#remove row where tempo is 0
clean_dataset.drop(clean_dataset.index[clean_dataset['tempo'] == 0], inplace=True)

#normalize popularity
minMax_normalization('popularity')

#binarize explicit True False to 1 and 0
clean_dataset["explicit"] = clean_dataset["explicit"].astype(int)

#export
clean_dataset.to_csv('clean_data.csv',index=False)



