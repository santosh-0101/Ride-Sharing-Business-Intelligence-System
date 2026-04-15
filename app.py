import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("🚕 Ride-Sharing BI Dashboard")

# Load Data
df = pd.read_csv("ride_sharing_data.csv")
df['pickup_time'] = pd.to_datetime(df['pickup_time'])

# Feature Engineering
df['hour'] = df['pickup_time'].dt.hour
df['day'] = df['pickup_time'].dt.day_name()

# Sidebar Filters
st.sidebar.header("Filters")

location = st.sidebar.selectbox("Select Pickup Location", df['pickup_location'].unique())

filtered_df = df[df['pickup_location'] == location]

# KPIs
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"₹{filtered_df['fare_amount'].sum():,.0f}")
col2.metric("Total Rides", len(filtered_df))
col3.metric("Avg Fare", f"₹{filtered_df['fare_amount'].mean():.2f}")

# Rides by Hour
st.subheader("⏰ Rides by Hour")

fig1, ax1 = plt.subplots()
filtered_df.groupby('hour').size().plot(ax=ax1)
st.pyplot(fig1)

# Revenue by Location
st.subheader("💰 Revenue by Drop Location")

fig2, ax2 = plt.subplots()
filtered_df.groupby('drop_location')['fare_amount'].sum().plot(kind='bar', ax=ax2)
st.pyplot(fig2)

# Distance Distribution
st.subheader("📍 Distance Distribution")

fig3, ax3 = plt.subplots()
sns.histplot(filtered_df['distance_km'], bins=20, ax=ax3)
st.pyplot(fig3)
# ---------------- ML MODEL ----------------
from sklearn.linear_model import LinearRegression

X = df[['distance_km', 'trip_duration_min']]
y = df['fare_amount']

model = LinearRegression()
model.fit(X, y)

# ---------------- USER INPUT ----------------
st.subheader("🔮 Fare Prediction")

distance = st.number_input("Enter Distance (km)", min_value=1.0, max_value=50.0, value=5.0)
duration = st.number_input("Enter Duration (minutes)", min_value=5, max_value=120, value=15)

if st.button("Predict Fare"):
    prediction = model.predict([[distance, duration]])
    st.success(f"Estimated Fare: ₹{prediction[0]:.2f}")
