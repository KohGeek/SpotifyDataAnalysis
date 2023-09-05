#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 1. [Importing Libraries](#importing-libraries)
# 2. [Data Description](#data-description)
#     * [Multiple Genres](#multiple-genres)
# 3. [Data Preprocessing](#data-cleaning)
# 4. [Data Modelling](#data-modelling)

# ## Importing Libraries <a class="anchor" id="importing-libraries"></a>
# 
# Here, we import all the necessary libraries for our work.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# ## Data Description <a class="anchor" id="data-description"></a>
# 
# First, the data is loaded and basic information about the data is displayed.

# In[ ]:


tracks = pd.read_csv('csvs/dataset.csv', index_col=0)

tracks.head()


# Our goal is to identify and predict the genres of the song, so we try display and see how many genres are there in the dataset.

# In[ ]:


print('Number of genres: {}'.format(tracks.track_genre.nunique()))

# Get a count of all genre
tracks.track_genre.value_counts()


# ### Multiple Genres <a class="anchor" id="multiple-genres"></a>
#  
# We discover that some song may have multiple genres. To improve our modelling, we will be using tracks with one genre only. 

# In[ ]:


# Sort by popularity first, so when we drop duplicate we drop lower popularity
# Drop duplicate if track_name, duration_ms, artists and track_genre are all the same
tracks.sort_values(by=['popularity'],ascending=False,inplace=True)
tracks.drop_duplicates(subset=['track_name','duration_ms','artists','track_genre'],inplace=True)

# If track_name, duration_ms and artists are same, but genre is different, aggregate the genre
tracks = tracks.groupby(['track_name','duration_ms','artists'],as_index=False).agg({'track_genre':lambda x: ','.join(x),
                                                                                                  'album_name': 'first',
                                                                                                  'track_id': 'first',
                                                                                                  'popularity': 'max',
                                                                                                  'explicit': 'first',
                                                                                                  'danceability': 'first',
                                                                                                  'energy': 'first',
                                                                                                  'loudness': 'first',
                                                                                                  'speechiness': 'first',
                                                                                                  'acousticness': 'first',
                                                                                                  'instrumentalness': 'first',
                                                                                                  'liveness': 'first',
                                                                                                  'valence': 'first',
                                                                                                  'tempo': 'first',
                                                                                                  'key': 'first',
                                                                                                  'mode': 'first'})


# Remove all tracks with more than one genre
tracks = tracks[tracks['track_genre'].str.contains(',') == False]
tracks.track_genre.value_counts()


# Additionally, any genre with less than 500 tracks does not constitute enough training and test sample, and will be removed from the dataset.

# In[ ]:


# Remove all genres with less than 500 tracks, maintain all columns
tracks = tracks.groupby('track_genre').filter(lambda x: len(x) > 500)
tracks.track_genre.value_counts()


# ## Data Preprocessing <a class="anchor" id="data-cleaning"></a>
# 
# We start off with basic data cleaning, removing null data and removing unnecessary columns according to our EDA.

# In[ ]:


# Drop the row where track_name = null
tracks.drop(tracks.index[tracks['track_name'].isnull()], inplace=True)


# To make our modelling easier, we will limit our selection to a hand selected few genres. As much as the top 10 genre present an interesting opportunity, a cursory glance at the data shows that the top 10 genres are not very distinct from each other. Hence, we will select a few genres that are more significantly distinct from one another.

# In[ ]:


genre_popularity = tracks.groupby('track_genre')['popularity'].mean()
genre_popularity.sort_values(ascending=False)

# What is the difference between pop-film, k-pop, pop? 
# And what is the difference between sad and emo?


# We choose the following genre for our modelling, and remove the rest of the genres from the dataset.
# - Country
# - Chill
# - K-Pop
# - Club
# - Rock-n-Roll
# - Classical
# - Sleep
# - Electronic
# - Ambient
# - Opera

# In[ ]:


# Retain only the genres listed above
tracks = tracks[tracks['track_genre'].isin(['country', 'chill', 'k-pop', 'club', 'rock-n-roll', 'classical', 'sleep', 'electronic', 'ambient', 'opera'])]


# We will also remove Track ID from our dataset as the ID is randomly generated data. Additionally, track name, artist name and album name will be removed as well. These three category are too diverse and will be hard to generalize, even if they provide very useful information. 
# 
# We will also drop the track key, as it will present too many dimension for our model to handle.

# In[ ]:


# Drop the track_id column
tracks.drop('track_id', axis=1, inplace=True)

# Drop the track_name, artists, album_name columns
tracks.drop(['track_name', 'artists', 'album_name'], axis=1, inplace=True)

# Drop the key column
tracks.drop('key', axis=1, inplace=True)


# Next, we will discretize both loudness, tempo and duration_ms into 10 bins each. The exact value of these columns are not important, but their rough bins will help better inform the model.
# 
# We will also normalise the popularity columns, as they are on a different scale from the rest of the data.

# In[ ]:


# Discretize the loudness column into 10 bins, normalised within 0 and 1
tracks['loudness'] = pd.cut(tracks['loudness'], 10, labels=False)
tracks['loudness'] = MinMaxScaler().fit_transform(tracks[['loudness']])

# Discretize the tempo column into 10 bins, normalised within 0 and 1
tracks['tempo'] = pd.cut(tracks['tempo'], 10, labels=False)
tracks['tempo'] = MinMaxScaler().fit_transform(tracks[['tempo']])

# Normalise the duration_ms column through the use of log transformation, then normalise within 0 and 1
tracks['duration_ms'] = np.log(tracks['duration_ms'])
tracks['duration_ms'] = MinMaxScaler().fit_transform(tracks[['duration_ms']])

# Normalise the popularity column through MinMaxScaler
tracks['popularity'] = MinMaxScaler().fit_transform(tracks[['popularity']])

# Describe the dataset
tracks.describe()


# Next, we make sure each of the genres has 500 sample exactly.

# In[ ]:


# Drop individual rows until the number of tracks per genre is equal
tracks = tracks.groupby('track_genre').apply(lambda x: x.sample(tracks.track_genre.value_counts().min(), random_state=42).reset_index(drop=True))
tracks.track_genre.value_counts()

tracks["explicit"] = tracks["explicit"].astype(int)


tracks.to_csv('clean_data_dog.csv',index=False)
