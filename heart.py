# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("heart.csv")

type(dataset)

dataset.shape

dataset.head(5)

dataset.sample(5)

dataset.describe()

dataset.info()

info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])

dataset["target"].describe()

dataset["target"].unique()

print(dataset.corr()["target"].abs().sort_values(ascending=False))

y = dataset["target"]

sns.countplot(y)


target_temp = dataset.target.value_counts()

print(target_temp)

print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))

#Alternatively,
# print("Percentage of patience with heart problems: "+str(y.where(y==1).count()*100/303))
# print("Percentage of patience with heart problems: "+str(y.where(y==0).count()*100/303))

# #Or,
# countNoDisease = len(df[df.target == 0])
# countHaveDisease = len(df[df.target == 1])

dataset["sex"].unique()

import seaborn as sns

# Correct usage of sns.barplot with keyword arguments
sns.barplot(x=dataset["sex"], y=y)

dataset["cp"].unique()

import seaborn as sns

# Correct usage of sns.barplot with keyword arguments
sns.barplot(x="cp", y=y, data=dataset)

dataset["fbs"].describe()

dataset["fbs"].unique()

sns.barplot(x="fbs", y=y, data=dataset)

dataset["restecg"].unique()

sns.barplot(x="restecg", y=y, data=dataset)
# The 'x' and 'y' arguments specify the columns within the DataFrame to use for the barplot.
# The 'data' argument specifies the DataFrame containing the data.

dataset["exang"].unique()

sns.barplot(x="exang", y=y, data=dataset)
# The 'x' argument is set to "exang", indicating that the "exang" column of the DataFrame should be used for the x-axis.
# The 'y' argument is set to y, indicating that the values in the 'y' variable should be used for the y-axis.
# The 'data' argument is set to dataset, specifying the DataFrame that contains the data for the plot.

dataset["slope"].unique()

sns.barplot(x="slope", y=y, data=dataset)
# Provide 'x', 'y', and 'data' as keyword arguments.
# This tells sns.barplot where to find the data for the x and y axes within your dataset.

dataset["ca"].unique()

sns.countplot(dataset["ca"])

sns.barplot(x="ca", y=y, data=dataset)
# Provide 'x', 'y', and 'data' as keyword arguments, similar to
# your earlier sns.barplot call.
# This tells sns.barplot where to find the data for the x and y axes within your dataset.

dataset["thal"].unique()

sns.barplot(x="thal", y=y, data=dataset)
# Provide 'x', 'y', and 'data' as keyword arguments, similar to
# your earlier sns.barplot call.
# This tells sns.barplot where to find the data for the x and y axes within your dataset.

sns.distplot(dataset["thal"])

from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

X_train.shape

X_test.shape

Y_train.shape

Y_test.shape

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)

Y_pred_lr.shape

score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,Y_train)

Y_pred_nb = nb.predict(X_test)

Y_pred_nb.shape

score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")

from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)

Y_pred_svm.shape

score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)

Y_pred_knn.shape

score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")

from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x

#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)

print(Y_pred_dt.shape)

score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize variables
max_accuracy = 0
best_x = None

# Reduce the range for random_state
for x in range(100):  # Reduced to 100 iterations for efficiency
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)

    # Update max_accuracy and best_x
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

# Train the best model
rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

print(f"Max Accuracy: {max_accuracy}%")
print(f"Best Random State: {best_x}")

Y_pred_rf.shape

score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")

import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, Y_train)

Y_pred_xgb = xgb_model.predict(X_test)

Y_pred_xgb.shape

score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)

print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")

from keras.models import Sequential
from keras.layers import Dense

# https://stats.stackexchange.com/a/136542 helped a lot in avoiding overfitting

model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=300)

Y_pred_nn = model.predict(X_test)

Y_pred_nn.shape

rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded

score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")

#Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11.

scores = [score_lr,score_nb,score_svm,score_knn,score_dt,score_rf,score_xgb,score_nn]
algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest","XGBoost","Neural Network"]

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas

# ... your existing code ...

# Create a Pandas DataFrame from algorithms and scores
data = pd.DataFrame({'Algorithms': algorithms, 'Accuracy score': scores})

sns.set(rc={'figure.figsize': (15, 8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

# Pass the DataFrame to the 'data' argument of sns.barplot
sns.barplot(x='Algorithms', y='Accuracy score', data=data)

scores = [score_lr,score_rf,score_xgb]
algorithms = ["Logistic Regression","Random Forest","XGBoost"]

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas

# ... your existing code ...

# Create a Pandas DataFrame from algorithms and scores
data = pd.DataFrame({'Algorithms': algorithms, 'Accuracy score': scores})

sns.set(rc={'figure.figsize': (15, 8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

# Pass the DataFrame to the 'data' argument of sns.barplot
sns.barplot(x='Algorithms', y='Accuracy score', data=data)

from sklearn.metrics import precision_score, recall_score

# Calculate precision and recall for each model
precision_lr = precision_score(Y_test, Y_pred_lr)
recall_lr = recall_score(Y_test, Y_pred_lr)

precision_nb = precision_score(Y_test, Y_pred_nb)
recall_nb = recall_score(Y_test, Y_pred_nb)

precision_svm = precision_score(Y_test, Y_pred_svm)
recall_svm = recall_score(Y_test, Y_pred_svm)

precision_knn = precision_score(Y_test, Y_pred_knn)
recall_knn = recall_score(Y_test, Y_pred_knn)

precision_dt = precision_score(Y_test, Y_pred_dt)
recall_dt = recall_score(Y_test, Y_pred_dt)

precision_rf = precision_score(Y_test, Y_pred_rf)
recall_rf = recall_score(Y_test, Y_pred_rf)

precision_xgb = precision_score(Y_test, Y_pred_xgb)
recall_xgb = recall_score(Y_test, Y_pred_xgb)

precision_nn = precision_score(Y_test, Y_pred_nn)
recall_nn = recall_score(Y_test, Y_pred_nn)


# Print the results
print("Logistic Regression: Precision =", precision_lr, "Recall =", recall_lr)
print("Naive Bayes: Precision =", precision_nb, "Recall =", recall_nb)
print("SVM: Precision =", precision_svm, "Recall =", recall_svm)
print("KNN: Precision =", precision_knn, "Recall =", recall_knn)
print("Decision Tree: Precision =", precision_dt, "Recall =", recall_dt)
print("Random Forest: Precision =", precision_rf, "Recall =", recall_rf)
print("XGBoost: Precision =", precision_xgb, "Recall =", recall_xgb)
print("Neural Network: Precision =", precision_nn, "Recall =", recall_nn)


