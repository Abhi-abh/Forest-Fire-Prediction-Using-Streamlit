import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

df = pd.read_csv('Algerian_forest_fires_dataset_Cleaned.csv')

## encoding of the categories in classes
df['Classes']=np.where(df['Classes'].str.contains('not fire'),0,1)

df.corr(numeric_only=True)

# Split the data into features (X) and target variable (y)
X = df[[ 'FFMC', 'ISI','FWI',]]
y = df['Classes']

# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# Create logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# check accuracy and cofusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)

loreg_cm = ConfusionMatrixDisplay.from_estimator( model,X_test, y_test)

precision=metrics.precision_score(y_test, y_pred)
sensitivity_recall=metrics.recall_score(y_test, y_pred)
f1_score=metrics.f1_score(y_test, y_pred)
Specificity = metrics.recall_score(y_test, y_pred, pos_label=0)


print({"Accuracy":accuracy,"Precision":precision,"Sensitivity_recall":sensitivity_recall,"Specificity":Specificity,"F1_score":f1_score})

import pickle
pickle.dump(model, open('./model.sav', 'wb'))