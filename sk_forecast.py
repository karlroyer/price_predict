# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

import os
os.environ["KERAS_BACKEND"] = 'torch'

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
from skforecast.plot import set_dark_theme

# Modeling and Forecasting
# ==============================================================================
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster
from skforecast.deep_learning import create_and_compile_model
from skforecast.deep_learning import ForecasterRnn
from skforecast.utils import save_forecaster
import sys
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model

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

priceData.to_csv('data/banana.csv')
mask = (priceData.index > '2024-01-01') & (priceData.index <= '2025-09-01')
priceData = priceData.loc[mask]
priceData.to_csv('data/banana2.csv')
        
# Now fit the prophet model to data
print('Creating sklearn models')
# Create and fit forecaster

# Lets estimate 7 days of data, and use 4 times us much to train model
estimation_steps = 4 * 24 * 7
tune_steps       = 4 * estimation_steps
total_steps      = estimation_steps + tune_steps

trimmedPriceData = priceData[:-estimation_steps].copy()
priceDataTrain = priceData[:-total_steps].copy()
priceDataTune  = priceData[-total_steps:-estimation_steps].copy()
priceDataTest = priceData[-estimation_steps:].copy()

print("Orig size ", priceData['price'].size)
print("Train size: ", priceDataTrain['price'].size)
print("Tune size: ", priceDataTune['price'].size)
print("Test size: ", priceDataTest['price'].size)

print(
    f"Dates train      : {priceDataTrain.index.min()} --- " 
    f"{priceDataTrain.index.max()}  (n={len(priceDataTrain)})"
)
print(
    f"Dates validation : {priceDataTune.index.min()} --- " 
    f"{priceDataTune.index.max()}  (n={len(priceDataTune)})"
)
print(
    f"Dates test       : {priceDataTest.index.min()} --- " 
    f"{priceDataTest.index.max()}  (n={len(priceDataTest)})"
)

set_dark_theme()

exog_columns = priceDataTrain.columns.to_list();
exog_columns.pop(0)
print("Number of exog variables: ", len(exog_columns))

series = ['price']
levels = ['price']
lags = 72

print("Exog columns ",exog_columns)
model = create_and_compile_model(
    series                  = trimmedPriceData[series],         # Single-series
    levels                  = levels,                    # One target series to predict
    lags                    = lags, 
    steps                   = estimation_steps, 
    exog                    = trimmedPriceData[exog_columns],  # Exogenous variables
    recurrent_layer         = "LSTM",
    recurrent_units         = [128, 64],
    recurrent_layers_kwargs = {"activation": "tanh"},
    dense_units             = [64, 32],
    compile_kwargs          = {'optimizer': Adam(learning_rate=0.01), 'loss': 'mse'},
    model_name              = "Single-Series-Multi-Step-Exog2"
)

print(model.summary())

# ==============================================================================
forecaster = ForecasterRnn(
    estimator=model,
    levels=levels,
    lags=lags, 
    transformer_series=MinMaxScaler(),
    transformer_exog=MinMaxScaler(),
    fit_kwargs={
        "epochs": 25, 
        "batch_size": 512, 
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
        ],  # Callback to stop training when it is no longer learning and to reduce learning rate.
        "series_val": priceDataTune[series],      # Validation data for model training.
        "exog_val": priceDataTune[exog_columns]  # Validation data for exogenous variables
    },
)


# Create and fit forecaster
# ==============================================================================
forecaster.fit(
    series = trimmedPriceData[series], 
    exog   = trimmedPriceData[exog_columns]
)

# Save the forecaster
# ==============================================================================
save_forecaster(
    forecaster, 
    file_name = 'forecaster_custom_features.joblib', 
    save_custom_functions = True, 
    verbose = False
)

# Training and overfitting tracking
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 3))
_ = forecaster.plot_history(ax=ax)
plt.show();
sys.exit()


# Prediction with exogenous variables
# ==============================================================================
predictions = forecaster.predict(
    exog = priceDataTest[exog_columns]
)

fig, ax = plt.subplots(figsize=(7, 3.5))
priceDataTest['price'].plot(ax=ax, label='test')
priceDataTune['price'].plot(ax=ax, label='tune')
predictions.loc[predictions["level"] == levels, "pred"].plot(ax=ax, label="predictions")
ax.legend()
plt.show()
sys.exit()
cv = TimeSeriesFold(
         steps                 = 10,
         initial_train_size    = priceDataTrain['price'].size,
         refit                 = True,
         fixed_train_size      = False,
         gap                   = 0,
         skip_folds            = None,
         allow_incomplete_fold = True
     )

metric, predictions_backtest = backtesting_forecaster(
    forecaster    = forecaster,
    y             = priceData['price'],
    exog          = priceData[exog_columns],
    cv            = cv,
    metric        = 'mean_squared_error',
)

# Plot predictions
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3))
priceDataTest['price'].plot(ax=ax)
priceDataTune['price'].plot(ax=ax)
predictions_backtest['pred'].plot(ax=ax)
ax.legend(loc="upper left")
plt.show();
print(f"Backtest error: {metric}")
sys.exit()
