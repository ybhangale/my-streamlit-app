import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv("heart.csv")  # Update the path

# Streamlit app title
st.title("Heart Disease Prediction")

# Display dataset
if st.checkbox("Show Dataset"):
    st.write(dataset)

# Display data analysis
if st.checkbox("Show Data Analysis"):
    st.write(dataset.describe())
    st.write(dataset.info())

# Add your data visualization code here
if st.checkbox("Show Count Plot of Target"):
    sns.countplot(x='target', data=dataset)
    st.pyplot()

# Add model training and predictions
def train_model():
    predictors = dataset.drop("target", axis=1)
    target = dataset["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
    
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(Y_test, Y_pred)
    return accuracy

if st.button("Train Model"):
    accuracy = train_model()
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")