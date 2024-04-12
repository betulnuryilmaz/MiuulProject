import streamlit as st
import pandas as pd
import folium
from geopy.distance import geodesic
from geopy.distance import great_circle
import pickle
from datetime import datetime, timedelta
import numpy as np
from joblib import load
import joblib
@st.cache_data
def get_data():
    file_path = "flight_delay_df01.csv"
    df = pd.read_csv(file_path)
    file_path2 = "delay1_sample_df.csv"
    df1 = pd.read_csv(file_path2)
    ap = pd.read_csv("airport.csv")
    aiport_enc = pd.read_csv("airport_encoded.csv", sep=";")
    time_enc = pd.read_csv("time_encoded.csv", sep=";")
    airline_enc = pd.read_csv("air_encoded.csv", sep=";")
    return df, df1, ap, aiport_enc, time_enc, airline_enc

st.set_page_config(layout="wide")

st.header("✈ Flight Delay Prediction Model ✈")
tab_home, tab_pred, tab_vis = st.tabs(["Main Page", "Model Examination & Graphics", "Prediction"])


column_dataset, column_analysis = tab_home.columns([1,1.5])

column_dataset.subheader("Scope of the Dataset")

column_dataset.markdown("""This includes data from 2018 to 2023-04 and contains information about the scheduled arrival
                            time, actual arrival time, scheduled departure time, and actual departure time focusing on 
                            delays, not cancellations or diverted flights. """)
# Add a clickable link
column_dataset.markdown("[Click here to visit Kaggle's website](https://www.kaggle.com/datasets/arvindnagaonkar/flight-delay?select=Flight_Delay.parquet)")

# List of items
item_list = ["Organizing Dataset - Sampling, Distribution", "Feature Engineering - Adding Features, Encoding, Normalization",
             "Modelling - Hyperparameter Optimization, XGBoost", "Visualization", "Model Deployment"]

# Displaying the list using Streamlit
column_dataset.markdown("**Model 1 Preparation Process:**")
for item in item_list:
    column_dataset.write(f"- {item}")


# List of items
item_list = ["Organizing Dataset - Sampling, Distribution", "Feature Engineering - Adding Features, Outlier, Encoding, Normalization",
             "Modelling - Hyperparameter Optimization, CatBoost", "Visualization", "Model Deployment"]

# Displaying the list using Streamlit
column_dataset.markdown("**Model 2 Preparation Process:**")
for item in item_list:
    column_dataset.write(f"- {item}")


df, df1, ap, aiport_enc, time_enc, airline_enc = get_data()
column_analysis.write("### Data from CSV file for Flight Delay:")
column_analysis.dataframe(df, width = 1500)
column_analysis.write("### Data from CSV file for Delay Minutes:")
column_analysis.dataframe(df1, width = 1500)

with tab_pred:
    column1, column2 = tab_pred.columns([1, 1])

    column1.markdown(
        "<h4 style='text-align: center;'>Performance Metrics Comparison for Different Data Distribution</h4>",
        unsafe_allow_html=True)
    column1.image("graph/even_uneven_data_dist.png", use_column_width=True)

    column1.markdown(
        "<h4 style='text-align: center;'>Feature Importance of XGBoost</h4>",
        unsafe_allow_html=True)
    column1.image("graph/feature_importance_delayEVEN_xgboost.png", use_column_width=True)

    column1.markdown(
        "<h4 style='text-align: center;'>Feature Importance of CatBoost</h4>",
        unsafe_allow_html=True)
    column1.image("graph/feature_importance_minuteEVEN_catboost.png", use_column_width=True)

    column1.markdown(
        "<h4 style='text-align: center;'>ROC AUC Values for Different Models</h4>",
        unsafe_allow_html=True)
    column1.image("graph/model_performance_roc_auc_EVEN.png", use_column_width=True)
##----------------------------

    column2.markdown(
        "<h4 style='text-align: center;'>Airline vs Flight Count Delay Time Analysis</h4>",
        unsafe_allow_html=True)
    column2.image("graph/airline_delaytime.png", use_column_width=True)

    column2.markdown(
        "<h4 style='text-align: center;'>Delay Time Analysis According to Weekday</h4>",
        unsafe_allow_html=True)
    column2.image("graph/delaytime_vs_week.png", use_column_width=True)

    column2.markdown(
        "<h4 style='text-align: center;'>Delay Time Analysis According to Month</h4>",
        unsafe_allow_html=True)
    column2.image("graph/delaytime_vs_month.png", use_column_width=True)

    column2.markdown(
        "<h4 style='text-align: center;'>RMSE Values for Different Models</h4>",
        unsafe_allow_html=True)
    column2.image("graph/RMSE_comparison_EVEN.png", use_column_width=True)

#TAB VİS

with tab_vis:

    today = datetime.today().date()
    tomorrow = today + timedelta(days=1)

    day_selection = st.sidebar.date_input(
        'Flight Day',today)

    def extract_day_month(full_date):
        # Convert the full date string to a datetime object
        date_obj = datetime.strptime(full_date, '%Y-%m-%d')  # Adjust the format based on your date's actual format

        # Extract day and month from the datetime object
        day = date_obj.day
        month = date_obj.month
        year =date_obj.year
        # Get the day of the week as a number (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)

        day_of_week = date_obj.weekday()

        return day, month,day_of_week, year


    # Extract day and month from the full date
    day, month,day_of_week, year = extract_day_month(str(day_selection))


    option1 = st.sidebar.selectbox(
        'Departure Airport',
        (ap["City"].tolist()))

    #st.sidebar.write('You selected:', option1)

    option2 = st.sidebar.selectbox(
        'Arrival Airport',
        (ap["City"].tolist()))

    departure_encode = (aiport_enc.loc[aiport_enc["City"] == option1, "airport_encoded"]).values[0]
    arrival_encode = (aiport_enc.loc[aiport_enc["City"] == option2, "airport_encoded"]).values[0]
    def create_csv_from_dict(dictionary):
        dff = pd.DataFrame(list(dictionary.items()), columns=['Abbreviation', 'Full Name'])
        return dff


    airline_abb = {
        "UA": "United Airlines",
        "DL": "Delta Airlines",
        "F9": "Frontier Airlines",
        "NK": "Spirit Airlines",
        "AA": "American Airlines",
        "WN": "Southwest Airlines",
        "AS": "Alaska Airlines",
        "HA": "Hawaiian Airlines",
        "VX": "Virgin America",
        "B6": "JetBlue Airways",
        "G4": "Allegiant Air",
    }

    dff = create_csv_from_dict(airline_abb)

    airline = st.sidebar.selectbox(
        'Airline Selection',
        (dff["Full Name"].tolist()))
    # airline_name_abbr = (dff.loc[dff["Full Name"] == airline, "Abbreviation"]).values[0]
    airline_name_encoded = (airline_enc.loc[airline_enc["airlinename"] == airline, "airline_encoded"]).values[0]
    #st.sidebar.write('You selected:', option2)

    latitude1 = ap.loc[ap["City"] == option1, "Latitude"].values[0]
    longitude1 = ap.loc[ap["City"] == option1, "Longitude"].values[0]

    latitude2 = ap.loc[ap["City"] == option2, "Latitude"].values[0]
    longitude2 = ap.loc[ap["City"] == option2, "Longitude"].values[0]

    # default_time = datetime.now().time()  # Set default time to current time
    selected_time = st.sidebar.time_input('Departure Time')

    hour = str(selected_time.hour)
    minute = str(selected_time.minute).zfill(2)

    departure_time = hour+minute

    if int(hour) > 3 and int(hour) <= 5:
        CRSDepTimeHourDis = "EarlyMorning"
    elif int(hour) >= 6 and int(hour) <= 11:
        CRSDepTimeHourDis = "Morning"
    elif int(hour) >= 12 and int(hour) <= 16:
        CRSDepTimeHourDis = "Afternoon"
    elif int(hour) >= 17 and int(hour) <= 21:
        CRSDepTimeHourDis = "Evening"
    else:
        CRSDepTimeHourDis = "Night"

    crs_time_encode = (time_enc.loc[time_enc["Time"] == CRSDepTimeHourDis, "Time_encoded"]).values[0]
    # Convert input to
    # float (handle invalid input gracefully)
    try:
        lat1 = float(latitude1)
        lon1 = float(longitude1)

        lat2 = float(latitude2)
        lon2 = float(longitude2)

    except ValueError:
        st.warning("Please enter valid numerical values for latitude and longitude.")

    # Create a Folium map centered around a specific location


    m = folium.Map(location=[lat1, lon1], zoom_start=5)  # Set the zoom level as needed

    folium.Marker([latitude1, longitude1], popup=option1, icon=folium.Icon(icon="cloud")).add_to(m)
    folium.Marker([latitude2, longitude2], popup=option2, icon=folium.Icon(icon="cloud")).add_to(m)

    # Create a PolyLine connecting the two airports
    line = folium.PolyLine(
        locations=[[lat1, lon1], [lat2, lon2]],
        color='blue',
        weight=2,
        opacity=1
    )
    line.add_to(m)

    folium_map = m._repr_html_()
    st.components.v1.html(folium_map, width=1350, height=600)


    #######################################################################################################

    # Example function to use the loaded model for prediction
#     def predict_with_model(model, input_data):
    #         # Your prediction logic here
    #         prediction = model.predict(input_data)
    #         return prediction

    # Path to your pickled file
    file_path = 'flight_delay01_xgboost.pkl'
    file_path2 = 'finalmodel_EVEN_catboost.pkl'

    input_data = pd.DataFrame({
        "Year": [year],
        "DayofMonth": [day],
        "DayofWeek": [day_of_week],
        "Month": [month],
        "DepTime": [int(departure_time)],
        "Marketing_Airline_Network": [airline_name_encoded],
        "OriginCityName": [departure_encode],
        "DestCityName": [arrival_encode],
        "CRSDepTimeHourDis": [crs_time_encode],
        "WheelsOffHourDis": [crs_time_encode]
    })

#    st.write(input_data)

    with open(file_path, 'rb') as file:
        loaded_model = joblib.load(file)
        # Use the loaded model (perform predictions or any other operations)
        button_clicked = st.sidebar.button('Predict')
        if button_clicked:
            prediction = loaded_model.predict(input_data)  # input_data'nın modele uygun hale getirilmesi gerekebilir
#            st.write('Tahmin Sonucu:', prediction)




    with open(file_path2, 'rb') as file2:
        minute_model = joblib.load(file2)
        # Use the loaded model (perform predictions or any other operations)

        if button_clicked:
            if prediction == 1:
                prediction2 = minute_model.predict(input_data)  # input_data'nın modele uygun hale getirilmesi gerekebilir
#                st.write('Tahmin Sonucu Dakika:', np.round(prediction2.astype(int), 0))
            else:
                prediction2 = 0
#                st.write('Tahmin Sonucu Dakika:', prediction2)
    try:
        if prediction == 0:
            # Display the PNG image for prediction result 0
            st.title('Delay is not expected.')
            st.image('graph/result_0.png')
        else:
            # Display the PNG image for prediction result 1
            st.title(f'Delay is expected, and the expected delay time is {np.round(prediction2.astype(int), 0)} minutes.')
            st.image('graph/result_1.png')
    except NameError:
            # Varsayılan olarak prediction 0 varsayalım
            st.write('Please, make prediction')

