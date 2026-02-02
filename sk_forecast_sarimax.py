# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

import os
os.environ["KERAS_BACKEND"] = 'torch'

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
import skforecast
import skforecast.stats
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import FunctionTransformer
from skforecast.datasets import fetch_dataset
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterStats
from skforecast.model_selection import TimeSeriesFold, backtesting_sarimax, grid_search_sarimax
from skforecast.plot import set_dark_theme


data_dir = 'data/'
# Cyclical encoding with sine/cosine transformation
# ==============================================================================
def sin_transformer(period):
    """
    Returns a transformer that applies sine transformation to a variable using
    the specified period.
    """
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    """
    Returns a transformer that applies cosine transformation to a variable using
    the specified period.
    """
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

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
priceData['weekday_sin'] = sin_transformer(7).fit_transform(priceData.index.day_of_week)
priceData['weekday_cos'] = cos_transformer(7).fit_transform(priceData.index.day_of_week)
priceData['monthday_sin'] = sin_transformer(12).fit_transform(priceData.index.month)
priceData['monthday_cos'] = cos_transformer(12).fit_transform(priceData.index.month)

# Trim data window to 2024-01-01 to 2025-09-01
mask = (priceData.index > '2024-01-01') & (priceData.index <= '2025-09-01')
priceData = priceData.loc[mask]
        
print('Creating sklearn models')

# Lets estimate 7 days of data
estimation_steps = 4 * 24 * 7

trimmedPriceData = priceData[:-estimation_steps].copy()
priceDataTest = priceData[-estimation_steps:].copy()

print("Orig size ", priceData['price'].size)
print("Train size: ", trimmedPriceData['price'].size)
print("Test size: ", priceDataTest['price'].size)

print(
    f"Dates train      : {trimmedPriceData.index.min()} --- " 
    f"{trimmedPriceData.index.max()}  (n={len(trimmedPriceData)})"
)
print(
    f"Dates test       : {priceDataTest.index.min()} --- " 
    f"{priceDataTest.index.max()}  (n={len(priceDataTest)})"
)

set_dark_theme()

exog_columns = trimmedPriceData.columns.to_list();
exog_columns.pop(0)
print("Number of exog variables: ", len(exog_columns))

forecaster = ForecasterStats(
    estimator=Sarimax(order=(12, 1, 1), seasonal_order=(0, 0, 0, 0), maxiter=200),
)

forecaster.fit(
    y                 = trimmedPriceData['price'], 
    exog              = trimmedPriceData[exog_columns],
    suppress_warnings = True
)

forecaster.fit(
    series = trimmedPriceData[['price']],
    exog   = trimmedPriceData[exog_columns]
)

# Remove price from test data so we are sure we estimate the 7 days...
priceDataForGraph = priceDataTest.copy()
priceDataTest.drop(
    labels = 'price',
    inplace = True,
    axis = 1
)

# Predict with exogenous variables
predictions = forecaster.predict(
    steps = estimation_steps,
    exog = priceDataTest[exog_columns]
)

fig, ax = plt.subplots(figsize=(7, 3.5))
priceDataForGraph['price'].plot(ax=ax, label='test')
predictions['pred'].plot(ax=ax, label="predictions")
ax.legend()
plt.show()

