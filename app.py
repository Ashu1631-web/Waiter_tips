import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page Config
st.set_page_config(page_title="Waiter Tips Dashboard", layout="wide")

st.title("💰 Waiter Tips Prediction & Analysis Dashboard")
st.markdown("Interactive + Animated Charts using Plotly 🚀")

# Load Dataset
url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/tips.csv"
data = pd.read_csv(url)

st.subheader("📌 Dataset Preview")
st.dataframe(data.head())

# ==============================
# 🎯 Animated Scatter Graph
# ==============================
st.subheader("📍 Total Bill vs Tip (Animated Scatter)")

fig1 = px.scatter(
    data_frame=data,
    x="total_bill",
    y="tip",
    size="size",
    color="day",
    animation_frame="time",   # Animation Added
    title="Tips Based on Bill Amount"
)
st.plotly_chart(fig1, use_container_width=True)

# ==============================
# 📊 Bar Graph (Day-wise Tips)
# ==============================
st.subheader("📊 Total Tips by Day (Bar Graph)")

day_tips = data.groupby("day")["tip"].sum().reset_index()

fig2 = px.bar(
    day_tips,
    x="day",
    y="tip",
    color="day",
    title="Total Tips Collected Each Day",
    text_auto=True
)
st.plotly_chart(fig2, use_container_width=True)

# ==============================
# 🍩 Pie Chart (Gender Tips)
# ==============================
st.subheader("🍩 Tips Distribution by Gender")

fig3 = px.pie(
    data,
    values="tip",
    names="sex",
    hole=0.5,
    title="Who Tips More?"
)
st.plotly_chart(fig3, use_container_width=True)

# ==============================
# 🤖 Tips Prediction Model
# ==============================
st.subheader("🤖 Waiter Tip Prediction Model")

# Convert categorical to numeric
data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})

x = np.array(data[["total_bill", "sex", "smoker", "day", "time", "size"]])
y = np.array(data["tip"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(xtrain, ytrain)

# ==============================
# 🎛 User Inputs
# ==============================
st.markdown("### Enter Details to Predict Tip 👇")

bill = st.number_input("Total Bill Amount", 1.0, 100.0, 20.0)
gender = st.selectbox("Gender", ["Male", "Female"])
smoker = st.selectbox("Smoker?", ["Yes", "No"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])
size = st.slider("Group Size", 1, 6, 2)

# Convert inputs
gender_val = 1 if gender == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0
day_val = {"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3}[day]
time_val = 0 if time == "Lunch" else 1

features = np.array([[bill, gender_val, smoker_val, day_val, time_val, size]])

if st.button("Predict Tip 💸"):
    prediction = model.predict(features)
    st.success(f"✅ Predicted Tip Amount: ₹{prediction[0]:.2f}")
