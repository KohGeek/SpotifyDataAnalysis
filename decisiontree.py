from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# To import the Iris dataset
df = pd.read_csv('clean_data_dog.csv')

# Preview dataset
print("\n")
print(df.head())

# Get the summary of the dataset
print("\n")
print("Dataset summary:")
print(df.info())

# Check missing value
print("\n")
print("Check missing value:")
print(df.isnull().sum())

# Prepare data for training and test
# Define feature vecytor and target variable (class)
x = df.drop([ 'track_genre'], axis=1)
y = df['track_genre']

print(x.head())

# split data into separate training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# check shape of x_train and x_test
print("\n")
print("train & test sample size", x_train.shape, x_test.shape)

# Import and instantiate the DecisionTreeClassifier model with gini
dt = DecisionTreeClassifier(criterion='gini', max_depth=4)

# Fit the decision tree model
dt = dt.fit(x_train, y_train)

# Visualise the tree
features = ['duration_ms','popularity','explicit','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','mode']
class_names = ['country', 'chill', 'k-pop', 'club', 'rock-n-roll', 'classical', 'sleep', 'electronic', 'ambient', 'opera']

tree.plot_tree(dt, feature_names=features, class_names=class_names)

# Predict test set results
y_pred = dt.predict(x_test)

# Create a confusion matrix
popularity = confusion_matrix(y_test, y_pred)
print("----------\n")
print(popularity)

# Transform to df for plotting
popularity_df = pd.DataFrame(
    popularity, 
    index=['country', 'chill', 'k-pop', 'club', 'rock-n-roll', 'classical', 'sleep', 'electronic', 'ambient', 'opera'], 
    columns=['country', 'chill', 'k-pop', 'club', 'rock-n-roll', 'classical', 'sleep', 'electronic', 'ambient', 'opera'])

# Print the confusion matrix as heatmap
plt.figure(figsize=(5.5,4))
sns.heatmap(popularity_df, annot=True)
plt.title(f'Decision Tree with Gini \nAccuracy:{accuracy_score(y_test, y_pred)}')
plt.ylabel('True Label')
plt.xlabel('Predicted label')
plt.show()
