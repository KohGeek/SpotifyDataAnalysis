#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('clean_data_dog.csv')
data.head()


# In[3]:


data.drop(data.columns[0], axis=1, inplace=True)
data.shape


# In[4]:


data.info()


# In[5]:


features_with_nan=[feature for feature in data.columns if data[feature].isna().sum()>0]
features_with_nan


# In[6]:


for feature in features_with_nan:
    print('Number of missing value in {}: {}'.format(feature,np.round(data[feature].isna().sum())))


# In[7]:


data=data.dropna()
data.isna().sum()


# In[8]:


data=data.drop_duplicates()
data.shape


# In[9]:


feature_numerical=[feature for feature in data.columns if data[feature].dtype!='O']
print('Number of numerical columns=', len(feature_numerical))
data[feature_numerical].head()


# In[10]:


feature_discrete_numerical=[feature for feature in feature_numerical if data[feature].nunique()<50]
feature_discrete_numerical


# In[11]:


for feature in feature_discrete_numerical:
    dataset=data.copy()
    sns.barplot(x=feature, y=dataset['popularity'], data=dataset, estimator=np.median)
    plt.show()


# In[12]:


features_continuous_numerical=[features for features in feature_numerical if features not in feature_discrete_numerical]
features_continuous_numerical


# In[13]:


feature_categorical=[feature for feature in data.columns if data[feature].dtypes=='O']
print('Number of categorical features:', len(feature_categorical))
data[feature_categorical].head()


# In[14]:


feature_categorical


# In[15]:


for feature in feature_categorical:
    dataset=data.copy()
    print(feature, ': Number of unique entries:', dataset[feature].nunique())


# In[16]:


data['speechiness'].sort_values()


# In[17]:


speechiness_type=[]
for i in data.speechiness:
    if i<0.33:
        speechiness_type.append('Low')
    elif 0.33<=i<=0.66:
        speechiness_type.append('Medium')
    else:
        speechiness_type.append('High')


# In[18]:


data['speechiness_type']=speechiness_type
print(data.speechiness_type.value_counts())
data.head()


# In[19]:


data.drop(data.columns[0], axis=1, inplace=True)
#Selecting the numerical features:
feature_numerical=[feature for feature in data.columns if data[feature].dtypes!='O']
#Selecting the discrete numerical features
feature_discrete_numerical=[feature for feature in feature_numerical if data[feature].nunique()<50]
#Selecting the continuous features
feature_continuous_numerical=[feature for feature in feature_numerical if feature not in feature_discrete_numerical]
feature_continuous_numerical


# In[20]:


data.shape


# In[21]:


feature_categorical=[feature for feature in data.columns if feature not in feature_numerical]
dataset=data.copy()
for feature in feature_categorical:
    print(feature,': {}, missing values {}'.format(data[feature].nunique(), data[feature].isna().sum()))


# In[22]:




# In[23]:


import category_encoders as ce


# In[24]:





from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
features_scaling=[feature for feature in feature_numerical if feature not in ['popularity','mode']]
scaler.fit(data[features_scaling])


# In[28]:


scaler.transform(data[features_scaling])


# In[29]:


data_to_replace=pd.DataFrame(scaler.transform(data[features_scaling]), columns=features_scaling)
data_to_replace.head()


# In[30]:


for feature in features_scaling:
    data[feature]=data_to_replace[feature].values
data.isna().sum()


# In[31]:


X=data.drop(['track_genre_0, '], axis=1)
y=data['track_genre']


# In[ ]:


X.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=7)


# In[ ]:


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[ ]:


def correlation(dataset,threshold):
    correlated_columns=set()
    correlation_matrix=dataset.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i,j])>threshold:
                colname=correlation_matrix.columns[i]
                correlated_columns.add(colname)
    return correlated_columns
        


# In[ ]:


corr_features=correlation(X_train,0.7)
print(len(set(corr_features)))
print(corr_features)


# In[ ]:


X_train_corr=X_train.copy()
X_test_corr=X_test.copy()


# In[ ]:


X_train_corr.drop(corr_features, axis=1, inplace=True)
X_test_corr.drop(corr_features, axis=1, inplace=True)
print(X_train_corr.shape, X_test_corr.shape)


# In[ ]:


X_train_corr.isna().sum()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtree=DecisionTreeRegressor()
def model(name):
    name.fit(X_train_corr,y_train)
    prediction=name.predict(X_test_corr)
    residual=y_test-prediction
    
    plt.figure(figsize=(15,6))
    
    plt.subplot(1,2,1)
    plt.scatter(y_test,prediction)
    
    plt.subplot(1,2,2)
    sns.distplot(residual, hist=False, kde=True)
    plt.show()


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


model(dtree)


# In[32]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score


# In[33]:


algos=[dtree]
MSE=[]
ABMSE=[]
R2_score=[]
for feature in algos:
    prediction=feature.predict(X_test_corr)
    mse=mean_squared_error(y_test, prediction)
    abmse=mean_absolute_error(y_test, prediction)
    score=r2_score(y_test, prediction)
    MSE.append(mse)
    ABMSE.append(abmse)
    R2_score.append(score)


# In[34]:


algosname=['DecisionTree']
metrics=pd.DataFrame(list(zip(algosname,MSE,ABMSE,R2_score)), columns=['Model','MSE', 'ABMSE', 'R2_score'])


# In[ ]:


plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
sns.barplot(x='Model', y='MSE', data=metrics)
plt.xticks(rotation=90)

plt.subplot(1,2,2)
sns.barplot(x='Model', y='R2_score', data=metrics)
plt.xticks(rotation=90)
plt.show()


# In[ ]:




