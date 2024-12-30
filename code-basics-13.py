import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

base_path = 'datasets/csv_files/'

# Load datasets
city_target_passenger_rating = pd.read_csv(base_path + 'city_target_passenger_rating.csv')
dim_city = pd.read_csv(base_path + 'dim_city.csv')
dim_date = pd.read_csv(base_path + 'dim_date.csv')
dim_repeat_trip_distribution = pd.read_csv(base_path + 'dim_repeat_trip_distribution.csv')
fact_passenger_summary = pd.read_csv(base_path + 'fact_passenger_summary.csv')
fact_trips = pd.read_csv(base_path + 'fact_trips.csv')
monthly_target_new_passengers = pd.read_csv(base_path + 'monthly_target_new_passengers.csv')
monthly_target_trips = pd.read_csv(base_path + 'monthly_target_trips.csv')

# Utility Function
def calculate_city_metrics(df):
    city_metrics = df.groupby('city_name').agg(
        total_trips=('trip_id', 'nunique'),
        total_fare=('fare', 'sum'),
        total_distance=('distance', 'sum')
    ).reset_index()
    
    city_metrics['avg_fare_per_km'] = city_metrics['total_fare'] / city_metrics['total_distance']
    city_metrics['avg_fare_per_trip'] = city_metrics['total_fare'] / city_metrics['total_trips']
    
    total_trips = city_metrics['total_trips'].sum()
    city_metrics['%_contribution_to_total_trips'] = (city_metrics['total_trips'] / total_trips) * 100
    
    return city_metrics

# Streamlit pages
st.title("City-Level Trip and Fare Summary")

dim_date['date'] = pd.to_datetime(dim_date['date'])

start_date = st.date_input("Start Date", value=dim_date['date'].min(), min_value=dim_date['date'].min(), max_value=dim_date['date'].max())
end_date = st.date_input("End Date", value=dim_date['date'].max(), min_value=dim_date['date'].min(), max_value=dim_date['date'].max())

cities = st.multiselect("Select Cities", options=dim_city['city_name'].unique(), default=dim_city['city_name'].unique())

fact_trips['date'] = pd.to_datetime(fact_trips['date'])
d = pd.merge(fact_trips, dim_city, on='city_id')
d = d.rename({'fare_amount': 'fare', 'distance_travelled(km)': 'distance'}, axis='columns')
d['month_name'] = d['date'].dt.month_name()

month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
d['month_name'] = pd.Categorical(d['month_name'], categories=month_order, ordered=True)

x = dim_repeat_trip_distribution.copy()
x['month'] = pd.to_datetime(x['month'])
x['month_name'] = x['month'].dt.month_name()

d = pd.merge(d, x, on=['month_name', 'city_id'])
x = fact_passenger_summary[['month', 'city_id', 'total_passengers']].copy()
x['month'] = pd.to_datetime(x['month'])
x['month_name'] = x['month'].dt.month_name()

d = pd.merge(d, x, on=['city_id', 'month_name'])
filtered_df = d[(d['date'] >= pd.to_datetime(start_date)) & (d['date'] <= pd.to_datetime(end_date))]

if cities:
    filtered_df = filtered_df[filtered_df['city_name'].isin(cities)]

# City metrics
city_report = calculate_city_metrics(filtered_df)
st.subheader("Report Summary")
st.write("This table shows the total trips, average fare per km, average fare per trip, and percentage contribution to total trips for each city.")
st.dataframe(city_report[['city_name', 'total_trips', 'avg_fare_per_km', 'avg_fare_per_trip', '%_contribution_to_total_trips']])

# Page 1: Total Trips per City
st.subheader("Total Trips per City")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=city_report, x='city_name', y='total_trips', ax=ax, palette='viridis')
ax.set_title('Total Trips per City')
ax.set_xlabel('City')
ax.set_ylabel('Total Trips')
st.pyplot(fig)

# Page 2: Percentage Contribution to Total Trips
st.subheader("Percentage Contribution to Total Trips")
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(city_report['%_contribution_to_total_trips'], labels=city_report['city_name'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2', len(city_report)))
ax.set_title('Percentage Contribution of Each City to Total Trips')
st.pyplot(fig)

# Page 3: Average Fare per Kilometer Over Time
st.subheader("Average Fare per Kilometer Over Time")
fare_per_km_time = filtered_df.groupby(['date'])['fare'].sum() / filtered_df.groupby(['date'])['distance'].sum()
fig, ax = plt.subplots(figsize=(10, 6))
fare_per_km_time.plot(ax=ax, color='blue', marker='o')
ax.set_title('Average Fare per Kilometer Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Average Fare per Kilometer')
st.pyplot(fig)

# Page 4: Average Fare per Trip Over Time
st.subheader("Average Fare per Trip Over Time")
fare_per_trip_time = filtered_df.groupby(['date'])['fare'].sum() / filtered_df.groupby(['date'])['trip_id'].nunique()
fig, ax = plt.subplots(figsize=(10, 6))
fare_per_trip_time.plot(ax=ax, color='red', marker='o')
ax.set_title('Average Fare per Trip Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Average Fare per Trip')
st.pyplot(fig)

# Page 5: Performance Report (Actual vs Target Trips)
target_df = monthly_target_trips.copy()
target_df['month'] = pd.to_datetime(target_df['month'])
target_df['month_name'] = target_df['month'].dt.month_name()
target_df = pd.merge(target_df, dim_city, on='city_id')
target_df = target_df.rename({'total_target_trips': 'target_trips'}, axis='columns')

actual_trips = filtered_df.groupby(['city_name', 'month_name']).agg(actual_trips=('trip_id', 'nunique')).reset_index()
report = pd.merge(actual_trips, target_df, on=['city_name', 'month_name'], how='left')

report['month_name'] = pd.Categorical(report['month_name'], categories=month_order, ordered=True)
report = report.sort_values(by=['city_name', 'month_name'])
report['performance_status'] = report.apply(lambda row: "Above Target" if row['actual_trips'] > row['target_trips'] else "Below Target", axis=1)
report['%_difference'] = ((report['actual_trips'] - report['target_trips']) / report['target_trips']) * 100

st.subheader("Performance Report")
st.write("This table evaluates the target performance for trips at the monthly and city level.")
st.dataframe(report[['city_name', 'month_name', 'actual_trips', 'target_trips', 'performance_status', '%_difference']])

# Page 6: Trip Frequency Distribution (Repeat Trip Analysis)
st.subheader("Trip Frequency Distribution (Repeat Trips)")
trip_order = ["2-Trips", "3-Trips", "4-Trips", "5-Trips", "6-Trips", "7-Trips", "8-Trips", "9-Trips", "10-Trips"]
new_cities = ["All Cities"] + filtered_df["city_name"].unique().tolist()
selected_city = st.selectbox("Select City", new_cities)

if selected_city == "All Cities":
    city_filtered_df = filtered_df
else:
    city_filtered_df = filtered_df[filtered_df["city_name"] == selected_city]

grouped_df = city_filtered_df.groupby(["city_name", "trip_count"])["repeat_passenger_count"].sum().reset_index()
total_passengers = grouped_df.groupby("city_name")["repeat_passenger_count"].sum().reset_index()
grouped_df = pd.merge(grouped_df, total_passengers, on="city_name", suffixes=("", "_total"))
grouped_df["percentage"] = (grouped_df["repeat_passenger_count"] / grouped_df["repeat_passenger_count_total"]) * 100

grouped_df["trip_count"] = pd.Categorical(grouped_df["trip_count"], categories=trip_order, ordered=True)
grouped_df = grouped_df.sort_values("trip_count")

if selected_city == "All Cities":
    consolidated_fig = px.bar(grouped_df, x="trip_count", y="percentage", color="city_name", title="Repeat Trip Distribution by City")
else:
    consolidated_fig = px.bar(grouped_df, x="trip_count", y="percentage", title=f"Repeat Trip Distribution for {selected_city}")

st.plotly_chart(consolidated_fig)
