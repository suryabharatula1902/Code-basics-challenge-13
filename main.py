# & "C:\Users\dell\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\streamlit.exe" run main.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

base_path='datasets/csv_files/'

city_target_passenger_rating=pd.read_csv(base_path+'city_target_passenger_rating.csv')

dim_city=pd.read_csv(base_path+'dim_city.csv')

dim_date=pd.read_csv(base_path+'dim_date.csv')

dim_repeat_trip_distribution=pd.read_csv(base_path+'dim_repeat_trip_distribution.csv')

fact_passenger_summary=pd.read_csv(base_path+'fact_passenger_summary.csv')

fact_trips=pd.read_csv(base_path+'fact_trips.csv')

monthly_target_new_passengers=pd.read_csv(base_path+'monthly_target_new_passengers.csv')

monthly_target_new_passengers=pd.read_csv(base_path+'monthly_target_new_passengers.csv')

monthly_target_trips=pd.read_csv(base_path+'monthly_target_trips.csv')

def calculate_city_metrics(df):
    # Grouping by city_name to get required metrics
    city_metrics = df.groupby('city_name').agg(
        total_trips=('trip_id', 'nunique'),  # Assuming 'trip_id' is unique for each trip
        total_fare=('fare', 'sum'),
        total_distance=('distance', 'sum')
    ).reset_index()
    
    # Calculate additional metrics
    city_metrics['avg_fare_per_km'] = city_metrics['total_fare'] / city_metrics['total_distance']
    city_metrics['avg_fare_per_trip'] = city_metrics['total_fare'] / city_metrics['total_trips']
    
    # Calculate percentage contribution of each city to total trips
    total_trips = city_metrics['total_trips'].sum()
    city_metrics['%_contribution_to_total_trips'] = (city_metrics['total_trips'] / total_trips) * 100
    
    return city_metrics

st.title("City-Level Trip and Fare Summary")

dim_date['date']=pd.to_datetime(dim_date['date'])

start_date = st.date_input("Start Date", value=dim_date['date'].min(), min_value=dim_date['date'].min(), max_value=dim_date['date'].max())
end_date = st.date_input("End Date", value=dim_date['date'].max(), min_value=dim_date['date'].min(), max_value=dim_date['date'].max())

cities = st.multiselect("Select Cities", options=dim_city['city_name'].unique(), default=dim_city['city_name'].unique())


fact_trips['date'] = pd.to_datetime(fact_trips['date'])

d=pd.merge(fact_trips,dim_city,on='city_id')
d=d.rename({'fare_amount':'fare','distance_travelled(km)':'distance'},axis='columns')
d['month_name']=d['date'].dt.month_name()

month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

d['month_name'] = pd.Categorical(d['month_name'], categories=month_order, ordered=True)

x=dim_repeat_trip_distribution.copy()

x['month']=pd.to_datetime(x['month'])

x['month_name']=x['month'].dt.month_name()

d=pd.merge(d,x,on=['month_name','city_id'])

x=fact_passenger_summary[['month','city_id','total_passengers']].copy()

x['month']=pd.to_datetime(x['month'])

x['month_name']=x['month'].dt.month_name()

d=pd.merge(d,x,on=['city_id','month_name'])

filtered_df = d[(d['date'] >= pd.to_datetime(start_date)) & (d['date'] <= pd.to_datetime(end_date))]

if cities:
    filtered_df = filtered_df[filtered_df['city_name'].isin(cities)]


city_report = calculate_city_metrics(filtered_df)


st.subheader("Report Summary")
st.write("This table shows the total trips, average fare per km, average fare per trip, and percentage contribution to total trips for each city.")

st.dataframe(city_report[['city_name', 'total_trips', 'avg_fare_per_km', 'avg_fare_per_trip', '%_contribution_to_total_trips']])


st.subheader("Total Trips per City")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=city_report, x='city_name', y='total_trips', ax=ax, palette='viridis')
ax.set_title('Total Trips per City')
ax.set_xlabel('City')
ax.set_ylabel('Total Trips')
st.pyplot(fig)


st.subheader("Percentage Contribution to Total Trips")
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(city_report['%_contribution_to_total_trips'], labels=city_report['city_name'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2', len(city_report)))
ax.set_title('Percentage Contribution of Each City to Total Trips')
st.pyplot(fig)


st.subheader("Average Fare per Kilometer Over Time")
fare_per_km_time = filtered_df.groupby(['date'])['fare'].sum() / filtered_df.groupby(['date'])['distance'].sum()
fig, ax = plt.subplots(figsize=(10, 6))
fare_per_km_time.plot(ax=ax, color='blue', marker='o')
ax.set_title('Average Fare per Kilometer Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Average Fare per Kilometer')
st.pyplot(fig)

st.subheader("Average Fare per Trip Over Time")
fare_per_trip_time = filtered_df.groupby(['date'])['fare'].sum() / filtered_df.groupby(['date'])['trip_id'].nunique()
fig, ax = plt.subplots(figsize=(10, 6))
fare_per_trip_time.plot(ax=ax, color='red', marker='o')
ax.set_title('Average Fare per Trip Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Average Fare per Trip')
st.pyplot(fig)


target_df=monthly_target_trips.copy()

target_df['month']=pd.to_datetime(target_df['month'])

target_df['month_name']=target_df['month'].dt.month_name()

target_df=pd.merge(target_df,dim_city,on='city_id')
target_df=target_df.rename({'total_target_trips':'target_trips'},axis='columns')


actual_trips = (
    filtered_df.groupby(['city_name', 'month_name'])
    .agg(actual_trips=('trip_id', 'nunique'))  # Assuming 'trip_id' is unique for each trip
    .reset_index()
)

# Merge actual trips with target trips
report = pd.merge(actual_trips, target_df, on=['city_name', 'month_name'], how='left')

# Sort by month order
report['month_name'] = pd.Categorical(report['month_name'], categories=month_order, ordered=True)
report = report.sort_values(by=['city_name', 'month_name'])

# Calculate performance status and % difference
report['performance_status'] = report.apply(
    lambda row: "Above Target" if row['actual_trips'] > row['target_trips'] else "Below Target",
    axis=1
)
report['%_difference'] = ((report['actual_trips'] - report['target_trips']) / report['target_trips']) * 100

report["month_name"] = pd.Categorical(report["month_name"], categories=month_order, ordered=True)


# Display the report table
st.subheader("Performance Report")
st.write("This table evaluates the target performance for trips at the monthly and city level.")
st.dataframe(report[['city_name', 'month_name', 'actual_trips', 'target_trips', 'performance_status', '%_difference']])

# Dropdown for city selection
selected_city = st.selectbox("Select a City to View Plots", options=report['city_name'].unique())


# Filter report for the selected city
city_report = report[report['city_name'] == selected_city]

# Extract the months available for the selected city
available_months = city_report['month_name'].unique()

# Plot 1: Actual vs Target Trips for the selected city
st.subheader(f"Actual vs Target Trips for {selected_city}")
fig_actual_vs_target = px.bar(
    city_report,
    x="month_name",
    y="actual_trips",
    title=f"Actual vs Target Trips ({selected_city})",
    labels={"actual_trips": "Trips", "month_name": "Month"},
    color_discrete_sequence=["#636EFA"],
    category_orders={"month_name": available_months}  # Dynamically order based on available months
)
fig_actual_vs_target.add_scatter(
    x=city_report['month_name'],
    y=city_report['target_trips'],
    mode='lines+markers',
    name="Target Trips",
    line=dict(dash='dash', color='black')
)
st.plotly_chart(fig_actual_vs_target)

# Plot 2: % Difference in Trips for the selected city
st.subheader(f"% Difference in Trips for {selected_city}")
fig_percent_difference = px.bar(
    city_report,
    x="month_name",
    y="%_difference",
    title=f"% Difference Between Actual and Target Trips ({selected_city})",
    labels={"%_difference": "% Difference", "month_name": "Month"},
    text="%_difference",
    color_discrete_sequence=["#EF553B"],
    category_orders={"month_name": available_months}  # Dynamically order based on available months
)
fig_percent_difference.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig_percent_difference.update_layout(yaxis_title="% Difference", xaxis_title="Month", showlegend=False)
st.plotly_chart(fig_percent_difference)

trip_order = ["2-Trips", "3-Trips", "4-Trips", "5-Trips", "6-Trips", "7-Trips", "8-Trips", "9-Trips", "10-Trips"]

st.subheader("Filter Cities for Plots")
new_cities = ["All Cities"] + filtered_df["city_name"].unique().tolist()
selected_city = st.selectbox("Select City", new_cities)

# Filter data by selected city
if selected_city == "All Cities":
    city_filtered_df = filtered_df
else:
    city_filtered_df = filtered_df[filtered_df["city_name"] == selected_city]


grouped_df = (
    city_filtered_df.groupby(["city_name", "trip_count"])["repeat_passenger_count"]
    .sum()
    .reset_index()
)

# Calculate percentage distribution
total_passengers = grouped_df.groupby("city_name")["repeat_passenger_count"].sum().reset_index()
grouped_df = pd.merge(grouped_df, total_passengers, on="city_name", suffixes=("", "_total"))
grouped_df["percentage"] = (grouped_df["repeat_passenger_count"] / grouped_df["repeat_passenger_count_total"]) * 100

# Order trip categories
grouped_df["trip_count"] = pd.Categorical(grouped_df["trip_count"], categories=trip_order, ordered=True)
grouped_df = grouped_df.sort_values("trip_count")

# Consolidated Plot (stacked bar chart)
st.subheader("Consolidated Plot: Trip Frequency Distribution")
if selected_city == "All Cities":
    consolidated_fig = px.bar(
        grouped_df,
        x="trip_count",
        y="percentage",
        color="city_name",
        title="Consolidated Repeat Trip Frequency Distribution",
        labels={"percentage": "% of Repeat Passengers", "trip_count": "Trip Category"},
        barmode="stack",
    )
    st.plotly_chart(consolidated_fig)

# City-Specific Plot (bar chart for selected city)
if selected_city != "All Cities":
    st.subheader(f"Trip Frequency Distribution for {selected_city}")
    city_specific_fig = px.bar(
        grouped_df[grouped_df["city_name"] == selected_city],
        x="trip_count",
        y="percentage",
        title=f"Repeat Trip Frequency for {selected_city}",
        labels={"percentage": "% of Repeat Passengers", "trip_count": "Trip Category"},
        text="percentage",
    )
    city_specific_fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(city_specific_fig)


st.title("City-Level New Passenger Analysis")

selected_cities = cities


filtered_city_df = filtered_df[filtered_df["city_name"].isin(selected_cities)]


city_new_passengers = (
    filtered_city_df[filtered_city_df["passenger_type"] == "new"]
    .groupby("city_name")
    .size()
    .reset_index(name="total_new_passengers")
)

# Handle cases where some cities have no new passengers
if selected_cities:
    all_cities_df = pd.DataFrame({"city_name": selected_cities})
    city_new_passengers = pd.merge(
        all_cities_df, city_new_passengers, on="city_name", how="left"
    ).fillna(0)

# Rank cities
city_new_passengers["rank"] = city_new_passengers["total_new_passengers"].rank(ascending=False, method="dense")

# Identify Top 3 and Bottom 3 cities
top_3_cities = city_new_passengers.nlargest(3, "total_new_passengers")
bottom_3_cities = city_new_passengers.nsmallest(3, "total_new_passengers")

# Add category column
city_new_passengers["city_category"] = "Other"
city_new_passengers.loc[city_new_passengers["city_name"].isin(top_3_cities["city_name"]), "city_category"] = "Top 3"
city_new_passengers.loc[city_new_passengers["city_name"].isin(bottom_3_cities["city_name"]), "city_category"] = "Bottom 3"

# Display the data
st.subheader("City-Level New Passenger Report")
st.write(city_new_passengers)

# Plot Top and Bottom 3 cities
st.subheader("Top and Bottom 3 Cities by Total New Passengers")
fig = px.bar(
    city_new_passengers[city_new_passengers["city_category"].isin(["Top 3", "Bottom 3"])],
    x="city_name",
    y="total_new_passengers",
    color="city_category",
    title="Top and Bottom 3 Cities",
    labels={"total_new_passengers": "Total New Passengers", "city_name": "City"},
    text="total_new_passengers",
)
fig.update_traces(texttemplate="%{text}", textposition="outside")
st.plotly_chart(fig)


city_month_revenue = (
    filtered_city_df.groupby(["city_name", "month_name"])["fare"]
    .sum()
    .reset_index(name="revenue")
)

# Identify the month with the highest revenue for each city
city_max_revenue = (
    city_month_revenue.sort_values(["city_name", "revenue"], ascending=[True, False])
    .drop_duplicates(subset=["city_name"], keep="first")
)

# Calculate total revenue per city
city_total_revenue = (
    city_month_revenue.groupby("city_name")["revenue"]
    .sum()
    .reset_index(name="total_revenue")
)

# Merge total revenue into the max revenue DataFrame
city_max_revenue = city_max_revenue.merge(city_total_revenue, on="city_name")
city_max_revenue["percentage_contribution"] = (
    city_max_revenue["revenue"] / city_max_revenue["total_revenue"] * 100
).round(2)

# Rename columns for clarity
city_max_revenue = city_max_revenue.rename(
    columns={
        "month_name": "highest_revenue_month",
        "revenue": "highest_revenue",
        "percentage_contribution": "percentage_contribution (%)",
    }
)[["city_name", "highest_revenue_month", "highest_revenue", "percentage_contribution (%)"]]

# Display the data
st.subheader("City-Level Revenue Report")
st.write(city_max_revenue)

# Plot highest revenue by city
st.subheader("Highest Revenue by City")
fig = px.bar(
    city_max_revenue,
    x="city_name",
    y="highest_revenue",
    color="highest_revenue_month",
    title="Highest Revenue by City",
    labels={"highest_revenue": "Revenue", "city_name": "City"},
    text="highest_revenue",
)
fig.update_traces(texttemplate="%{text}", textposition="outside")
st.plotly_chart(fig)

filtered_city_df = filtered_df[filtered_df["city_name"].isin(selected_cities)]


st.title("Repeat Passenger Rate Analysis")

city_options = filtered_df['city_name'].unique()
city_options = list(city_options) + ['All Cities Combined']  # Adding option for all cities combined

# Create a dropdown for selecting city
selected_city = st.selectbox('Select City', options=city_options, index=0)

# Dropdown for months is not required as we will display all months in the plots
month_options = filtered_df['month_name'].unique()

# Filter the data based on the selected city
if selected_city == 'All Cities Combined':
    filtered_data = filtered_df  # Include all cities
else:
    filtered_data = filtered_df[filtered_df['city_name'] == selected_city]

# Group by city_name, month_name, and passenger_type
grouped_df = filtered_data.groupby(['city_name', 'month_name', 'passenger_type']).size().unstack(fill_value=0)

# Calculate total passengers (new + repeated) for each city and month
grouped_df['total_passengers'] = grouped_df['new'] + grouped_df['repeated']

# Calculate the repeat passenger percentage
grouped_df['repeat_passenger_percentage'] = (grouped_df['repeated'] / grouped_df['total_passengers']) * 100

# Reset the index for easier display
grouped_df = grouped_df.reset_index()

# Display the report in Streamlit
st.write("### Repeat Passenger Rate Analysis by City and Month")
st.dataframe(grouped_df[['city_name', 'month_name', 'repeat_passenger_percentage']])

# Plotting the repeat passenger percentage for each city and month
st.write("### Repeat Passenger Percentage Over Time")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data for each city or combined data
if selected_city == 'All Cities Combined':
    combined_df = grouped_df.groupby(['month_name'])['repeat_passenger_percentage'].mean().reset_index()
    ax.plot(combined_df['month_name'], combined_df['repeat_passenger_percentage'], label='All Cities Combined', marker='o')
else:
    city_data = grouped_df[grouped_df['city_name'] == selected_city]
    ax.plot(city_data['month_name'], city_data['repeat_passenger_percentage'], label=selected_city, marker='o')

ax.set_title('Repeat Passenger Percentage by City and Month')
ax.set_xlabel('Month')
ax.set_ylabel('Repeat Passenger Percentage')
ax.legend(title='City')

# Show the plot
st.pyplot(fig)

# For bar plot of repeat passenger percentages
st.write("### Repeat Passenger Percentage as Bar Plot")
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))

# Create a bar plot for each city or combined data
if selected_city == 'All Cities Combined':
    combined_df = grouped_df.groupby(['month_name'])['repeat_passenger_percentage'].mean().reset_index()
    sns.barplot(x='month_name', y='repeat_passenger_percentage', data=combined_df, ax=ax_bar)
else:
    sns.barplot(x='month_name', y='repeat_passenger_percentage', hue='city_name', data=grouped_df[grouped_df['city_name'] == selected_city], ax=ax_bar)

ax_bar.set_title('Repeat Passenger Percentage by City and Month')
ax_bar.set_xlabel('Month')
ax_bar.set_ylabel('Repeat Passenger Percentage')

# Show the bar plot
st.pyplot(fig_bar)
