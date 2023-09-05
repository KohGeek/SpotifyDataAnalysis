
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv('clean_data_dog.csv')


X = df.drop('track_genre', axis=1)
y = df['track_genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print('Random Forest Classifier')
print('Accuracy: {}'.format(accuracy_score(y_test, rfc_pred)))
print('Confusion Matrix: \n{}'.format(confusion_matrix(y_test, rfc_pred)))
print('Classification Report: \n{}'.format(classification_report(y_test, rfc_pred)))