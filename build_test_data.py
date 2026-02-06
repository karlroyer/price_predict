#!python3

#
# Data manipulation
# ==============================================================================
import pandas as pd

import os
os.environ["KERAS_BACKEND"] = 'torch'

from sklearn.pipeline import make_pipeline

from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures

data_dir = 'data/'
wind_direction_cols = []

def read_price_data(filename):
    # Read data
    priceData = pd.read_csv(data_dir + filename, usecols=['datetime', 'price'])

    # Convert datetime to datetime objects and Drop timezone data
    priceData['datetime'] = pd.to_datetime(priceData['datetime'])
    priceData['price'] = priceData['price'].astype(float)
    priceData['datetime'] = priceData['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
    priceData.drop_duplicates(inplace=True)
    priceData.set_index('datetime', inplace=True)
    priceData.sort_index(inplace=True)
    return priceData

def readSolarData(name):
    filename = data_dir + name + ".csv";
    solar_data = pd.read_csv(
        filename,
        skiprows = 3,
        usecols = ['time','global_tilted_irradiance (W/m²)'],
    )
    solar_data.columns = ['datetime', name]
    solar_data['datetime'] = pd.to_datetime(solar_data['datetime'])
    solar_data[name] = solar_data[name].astype(float)
    solar_data.drop_duplicates(inplace=True)
    solar_data.set_index('datetime', inplace=True)
    solar_data.sort_index(inplace=True)
    return resample_data_to_csv(solar_data);     


def readWindData(name):
    filename = data_dir + name + ".csv";
    wind = pd.read_csv(
        filename,
        skiprows = 3,
        usecols = ['time','wind_speed_100m (km/h)','wind_direction_100m (°)'],
    )
    wind.columns = ['datetime', name + '_speed', name + '_direction']
    wind['datetime'] = pd.to_datetime(wind['datetime'])
    wind[name + '_speed'] = wind[name + '_speed'].astype(float)
    wind[name + '_direction'] = wind[name + '_direction'].astype(float)
    wind_direction_cols.append(name + '_direction')
    wind.drop_duplicates(inplace=True)
    wind.set_index('datetime', inplace=True)
    wind.sort_index(inplace=True)
    return resample_data_to_csv(wind)

def readTempHumidityRain(name):
    filename = data_dir + name + ".csv";
    thr = pd.read_csv(
        filename,
        skiprows = 3,
        usecols = ['time','temperature_2m (°C)','relative_humidity_2m (%)','rain (mm)'],
    )
    thr.columns = ['datetime', 'temp', 'humidity', 'rain']
    
    thr['datetime'] = pd.to_datetime(thr['datetime'])
    
    thr['temp'] = thr['temp'].astype(float)
    thr['humidity'] = thr['humidity'].astype(float)
    thr['rain'] = thr['rain'].astype(float)
   
    thr.drop_duplicates(inplace=True)
    thr.set_index('datetime', inplace=True)
    thr.sort_index(inplace=True)
    return resample_data_to_csv(thr)

def read_holidays(name):
    filename = data_dir + name + ".csv";
    thr = pd.read_csv(
        filename,
        skiprows = 3,
        usecols = ['datetime','holidays'],
    )
    thr['datetime'] = pd.to_datetime(thr['datetime'])
    thr['holidays'] = thr['holidays'].astype(float) 
    thr.drop_duplicates(inplace=True)
    thr.set_index('datetime', inplace=True)
    thr.sort_index(inplace=True)
    return resample_data_to_csv(thr)

def resample_data_to_csv(data):

    # Now calculate per minute data
    return data.resample('15min').ffill()


# Read data
priceData = read_price_data('nl_day_ahead_prices_raw_utc.csv')
print('Price data columns ',priceData.columns);
print(priceData.head(100))

# Now calculate per minute data
priceData = resample_data_to_csv(priceData)
print('Resampled data to 15 minute intevals')


# Now we need to load the exog data
exog_features = [
    'week_day_sin',
    'week_day_cos',
    'year_day_sin',
    'year_day_cos',
    'solar_radiation_Dorhout_Mees_NL',
    'solar_radiation_Fledderbosch_NL',
    'solar_radiation_Midden_Groningen_NL',
    'solar_radiation_Vlagtwedde_NL',
    'solar_radiation_Vloeivelden_Hollandia_NL',
    'wind_speed_Borsselle_NL',
    'wind_speed_Hollanse_Kust_4_NL',
    'wind_speed_Windplan_Groen_NL',
    'wind_speed_Gemini_NL',
    'wind_speed_Hollanse_Kust_5_NL',
    'temp_humidity_rain',
    'holidays_NL'
]

for exog_col_name in exog_features:
    print("Exog column " + exog_col_name)
    if (exog_col_name.startswith('solar_radiation_')):
        print("Importing solar radiation "+exog_col_name)
        sr = readSolarData(exog_col_name)
        priceData = pd.concat([priceData, sr], axis=1)
    elif (exog_col_name.startswith('wind_speed_')):
        print("Importing wind info for "+exog_col_name)
        wind = readWindData(exog_col_name)
        priceData = pd.concat([priceData, wind], axis=1)
    elif (exog_col_name == 'temp_humidity_rain'):
        print("Importing temp humidity and rain average")
        thr = readTempHumidityRain(exog_col_name)
        priceData = pd.concat([priceData, thr], axis=1)
    elif (exog_col_name.startswith('holidays_')):
        print("Importing holidays")
        holidays = read_holidays(exog_col_name)
        priceData = pd.concat([priceData, holidays], axis=1)
    else:
        print("Skipping exog column " + exog_col_name)

# Add month sin and cos and weekday sin and cos
# Calendar features
# ==============================================================================
features_to_extract_and_encode = [
    'month',
    'week',
    'day_of_week',
    'hour'
]
max_values = {
    "month":       12,
    "week":        52,
    "day_of_week": 7,
    "hour":        24,
}
calendar_transformer = DatetimeFeatures(
    variables           = 'index',
    features_to_extract = features_to_extract_and_encode,
    drop_original       = False,
)
cyclical_encoder = CyclicalFeatures(
    variables     = features_to_extract_and_encode,
    max_values    = max_values,
    drop_original = True
)

exog_transformer = make_pipeline(
    calendar_transformer,
    cyclical_encoder
)
priceData = exog_transformer.fit_transform(priceData);


# Trim data window to 2024-01-01 to 2025-09-01
mask = (priceData.index > '2024-01-01') & (priceData.index <= '2025-09-01')
priceData = priceData.loc[mask]

# Lets estimate 7 days of data, and use 4 times us much to train model
estimation_steps = 4 * 24 * 7

priceDataTrain = priceData[:-estimation_steps].copy()
priceDataTest  = priceData[-estimation_steps:].copy()

print("Orig size ", priceData['price'].size)
print("Train size: ", priceDataTrain['price'].size)
print("Test size: ", priceDataTest['price'].size)

print(
    f"Dates train      : {priceDataTrain.index.min()} --- " 
    f"{priceDataTrain.index.max()}  (n={len(priceDataTrain)})"
)
print(
    f"Dates test       : {priceDataTest.index.min()} --- " 
    f"{priceDataTest.index.max()}  (n={len(priceDataTest)})"
)

exog_columns = priceDataTrain.columns.to_list();
exog_columns.pop(0)
print("Number of exog variables in all_exog_ files: ", len(exog_columns))

priceDataFuture = priceDataTest.drop(
    labels = 'price',
    axis = 1
)

priceDataTrain.to_csv(  data_dir + 'all_exog_train.csv');
priceDataTest.to_csv(   data_dir + 'all_exog_test.csv');
priceDataFuture.to_csv( data_dir + 'all_exog_future.csv');

priceDataTrain.drop(
    labels  = wind_direction_cols,
    axis    = 1,
    inplace = True
)
priceDataTest.drop(
    labels  = wind_direction_cols,
    axis    = 1, 
    inplace = True
)
priceDataFuture.drop(
    labels  = wind_direction_cols,
    axis    = 1,
    inplace = True
)

exog_columns = priceDataTrain.columns.to_list();
exog_columns.pop(0)
print("Number of exog variables in exog_no_wind_dir_ files: ", len(exog_columns))

priceDataTrain.to_csv(  data_dir + 'exog_no_wind_dir_train.csv');
priceDataTest.to_csv(   data_dir + 'exog_no_wind_dir_test.csv');
priceDataFuture.to_csv( data_dir + 'exog_no_wind_dir_future.csv');

