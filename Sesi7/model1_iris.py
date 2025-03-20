import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    return df

df = load_data()

# Sidebar for user input
st.sidebar.header("User Input Features")
sepal_length = st.sidebar.slider("Sepal Length", float(df["sepal_length"].min()), float(df["sepal_length"].max()))
sepal_width = st.sidebar.slider("Sepal Width", float(df["sepal_width"].min()), float(df["sepal_width"].max()))
petal_length = st.sidebar.slider("Petal Length", float(df["petal_length"].min()), float(df["petal_length"].max()))
petal_width = st.sidebar.slider("Petal Width", float(df["petal_width"].min()), float(df["petal_width"].max()))

# Split data
X = df.drop(columns=['species'])
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on user input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Streamlit UI
st.title("🌸 Iris Flower Classification")
st.write("This app predicts the species of an Iris flower based on its features.")

# Display dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Display prediction
st.subheader("Prediction Result")
st.write(f"Predicted Species: **{prediction[0]}**")
st.write("Prediction Probability:")
st.write(pd.DataFrame(prediction_proba, columns=model.classes_))

# Confusion Matrix
st.subheader("Model Performance")
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(fig)

st.write(f"Model Accuracy: **{accuracy_score(y_test, y_pred) * 100:.2f}%**")