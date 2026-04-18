import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

# Load data
df = pd.read_csv("traffic_congestion_dataset.csv")

encoder= OneHotEncoder(sparse_output=False)

encoded= encoder.fit_transform(df[['day', 'rain']])
encoded_df=pd.DataFrame(
    encoded, 
    columns=encoder.get_feature_names_out(['day', 'rain'])
)

df = df.drop(['day', 'rain'], axis=1)

df=pd.concat([df, encoded_df], axis=1)

df = df.drop(columns=['rain_No'])

# Drop traffic_count
df = df.drop(columns=['traffic_count'])

# Prepare data
X = df.drop(columns=['congestion_level'])
y = df['congestion_level']


model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X, y)

st.title("Traffic Congestion Predictor")

st.write("Predict traffic congestion based on time, speed, day, and weather.")

# Inputs
hour = st.slider("Hour of Day", 0, 23)
avg_speed = st.slider("Average Speed", 10, 60)
day = st.selectbox(
    "Day of the Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

rain = st.selectbox("Rain Condition", ["No", "Yes"])

# Simple encoding (all zero except selected)
input_data = {col: 0 for col in X.columns}
input_data['hour'] = hour
input_data['avg_speed'] = avg_speed
day_col = f"day_{day}"
if day_col in input_data:
    input_data[day_col] = 1

if rain == "Yes":
    input_data['rain_Yes'] = 1

input_df = pd.DataFrame([input_data])

# Prediction
prediction = model.predict(input_df)
proba = model.predict_proba(input_df)

st.subheader(f"Predicted Congestion: {prediction[0]}")

st.write("### Prediction Confidence")
for i, cls in enumerate(model.classes_):
    st.write(f"{cls}: {proba[0][i]*100:.2f}%")

st.write("### Feature Importance")

importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

st.bar_chart(importance)