import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import string
import warnings
warnings.filterwarnings('ignore')

# set display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
# set printing options to show all elements
np.set_printoptions(threshold=np.inf)

df = pd.read_csv('titanic.csv')

#1. Get a brief information on the data.
print('\nThe columns are:  ')
print(df.columns)

print('\n\nNumber of X tuples = {}'.format(df.shape[0]))

print('\n\n')
print(df.info())
print('\n\n')
print(df.head())

#2.Handling Missing Values

df_columns = df.columns.tolist()
print('\n')

for col in df_columns:
    print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    
#2.1 Handling Missing Values: Age

sns.heatmap(df.drop(['PassengerId'], axis=1).corr(), annot = True, cmap='coolwarm')
plt.show()

age_by_pclass_sex = df.groupby(['Sex', 'Pclass']).median()['Age']

print('\n')
for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(df['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

#2.2 Handling Missing Values: Embarked
print('\n')
print(df[df['Embarked'].isnull()])

df['Embarked'] = df['Embarked'].fillna('S')

#2.3 Handling Missing Values: Cabin

cabin = df['Cabin'].unique()

print(cabin)

# Creating Deck column from the first letter of the Cabin column (M stands for Missing)
df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df_all_decks = df.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 
                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()
print(df_all_decks)

df_all_decks_survived = df.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                                                                                   'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()

print('\n')
print(df_all_decks_survived)

print('\n')
df['Deck'] = df['Deck'].replace(['A', 'B', 'C', 'T'], 'ABC')
df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')
df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')

print(df['Deck'].value_counts())

#drop Cabin as we have replaced it just now
df.drop(['Cabin'], inplace=True, axis=1)

print(df.head())

#3.1 Data Distribution - Features with continuous values
sns.distplot(df[df['Survived'] == 0]['Age'], label='Not Survived', hist=True, color='green')
sns.distplot(df[df['Survived'] == 1]['Age'], label='Survived', hist=True, color='blue')
plt.legend()
plt.show()

sns.distplot(df[df['Survived'] == 0]['Fare'], label='Not Survived', hist=True, color='green')
sns.distplot(df[df['Survived'] == 1]['Fare'], label='Survived', hist=True, color='blue')
plt.legend()
plt.show()

#3.2 Data Distribution - Features with categorical values


plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.countplot(x = 'Embarked', hue = 'Survived', data = df)
plt.subplot(2,3,2)
sns.countplot(x = 'Parch', hue = 'Survived', data = df)
plt.subplot(2,3,3)
sns.countplot(x = 'Pclass', hue = 'Survived', data = df)
plt.subplot(2,3,4)
sns.countplot(x = 'Sex', hue = 'Survived', data = df)
plt.subplot(2,3,5)
sns.countplot(x = 'SibSp', hue = 'Survived', data = df)
plt.subplot(2,3,6)
sns.countplot(x = 'Deck', hue = 'Survived', data = df)

plt.show()

#4. Discretized continuous features
print(df['Fare'].describe())

df['Fare'] = pd.qcut(df['Fare'], 4)
plt.figure(figsize=(20, 12))
sns.countplot(x='Fare', hue='Survived', data=df)
plt.show()
      
print('\n')
print(df['Fare'])

print(df['Age'].describe())

df['Age'] = pd.qcut(df['Age'], 4)
plt.figure(figsize=(20, 12))
sns.countplot(x='Age', hue='Survived', data=df)
plt.show()
      
print('\n')
print(df['Age'])

#5. Feature Engineering - Family_size and Ticket Frequency

df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df['Family_Size_Grouped'] = df['Family_Size'].map(family_map)

print(df.head())

print('\nHow many unique number of tickets:')
print(df['Ticket'].nunique())

df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')

print('\nThe highest number of the same ticket being used:')
print(df['Ticket_Frequency'].nunique())

print(df.head())

#6. Feature Transformation - Label encoding non-numeric features

non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Family_Size_Grouped', 'Age', 'Fare']

for feature in non_numeric_features:        
    df[feature] = LabelEncoder().fit_transform(df[feature])

#6. Feature Transformation - One-Hot-Encode categorical features
cat_features = [ 'Sex', 'Deck', 'Embarked', 'Family_Size_Grouped']
encoded_features = []


for feature in cat_features:
    encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
    n = df[feature].nunique()
    cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
    encoded_df = pd.DataFrame(encoded_feat, columns=cols)
    encoded_df.index = df.index
    encoded_features.append(encoded_df)

df = pd.concat([df, *encoded_features[:6]], axis=1)

#7. Drop all unneeded features
print('\n')
print(df.columns)
df = df.drop(['PassengerId', 'Name','Ticket', 'Parch', 'Family_Size', 'Family_Size_Grouped', 'Sex', 'SibSp', 'Embarked', 'Deck' ], axis=1)

print('\nResulting dataframe:')
print(df.head())
