import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Load dataset
st.title("Heart Disease Prediction App")
st.write("This app predicts the likelihood of heart disease based on input features.")

@st.cache
def load_data():
    return pd.read_csv("heart.csv")

data = load_data()

# Sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", int(data.age.min()), int(data.age.max()), int(data.age.mean()))
    sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", data.cp.unique())
    trestbps = st.sidebar.slider("Resting Blood Pressure", int(data.trestbps.min()), int(data.trestbps.max()), int(data.trestbps.mean()))
    chol = st.sidebar.slider("Serum Cholesterol", int(data.chol.min()), int(data.chol.max()), int(data.chol.mean()))
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", data.restecg.unique())
    thalach = st.sidebar.slider("Max Heart Rate Achieved", int(data.thalach.min()), int(data.thalach.max()), int(data.thalach.mean()))
    exang = st.sidebar.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", float(data.oldpeak.min()), float(data.oldpeak.max()), float(data.oldpeak.mean()))
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0-2)", data.slope.unique())
    ca = st.sidebar.slider("Number of Major Vessels (0-3)", int(data.ca.min()), int(data.ca.max()), int(data.ca.mean()))
    thal = st.sidebar.selectbox("Thalassemia (1-3)", data.thal.unique())

    return pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

input_df = user_input_features()

# Display dataset
if st.checkbox("Show raw data"):
    st.write(data)

# Preprocess and split data
X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(objective="binary:logistic", random_state=42)
}

st.subheader("Model Selection and Accuracy")
accuracy_scores = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    accuracy_scores[model_name] = accuracy

    st.write(f"Accuracy of {model_name}: {accuracy}%")

# Predict user input
selected_model = st.selectbox("Choose a model for prediction", list(models.keys()))
prediction_model = models[selected_model]
prediction = prediction_model.predict(input_df)

st.subheader("Prediction")
st.write("Prediction: ", "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease")

# Visualizations
st.subheader("Data Correlation Heatmap")
st.write("This heatmap shows the correlation between different features in the dataset.")

fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size for better clarity
sns.heatmap(
    data.corr(), 
    annot=True,           # Annotates the heatmap with the correlation values
    fmt=".2f",            # Format to show values with two decimal places
    cmap="coolwarm",      # Improved color scheme for better contrast
    cbar_kws={'shrink': 0.8}  # Shrink the color bar for better fit
)
st.pyplot(fig)
