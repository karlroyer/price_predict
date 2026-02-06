# Data manipulation
# ==============================================================================
import pandas as pd

from lightgbm import LGBMRegressor

from skforecast.recursive import ForecasterRecursiveMultiSeries

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
from skforecast.plot import set_dark_theme

# Modelling and Forecasting
# ==============================================================================
import sys

import argparse

data_dir = ''

def read_datetime_csv(filename):
    # Read data
    data = pd.read_csv(data_dir + filename)
    # Convert datetime to datetime objects and Drop timezone data
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.drop_duplicates(inplace=True)
    data.set_index('datetime', inplace=True)
    data.sort_index(inplace=True)
    return data.resample('15min').ffill()

parser = argparse.ArgumentParser(description = 'Use rms forecast on data')
parser.add_argument('-c', '-csv_prefix', required = True, help = 'The prefix for the files _test, _train and _test_no_price')
parser.add_argument('-s', '--show_accuracy', action='store_true')
parser.add_argument('-f','--filename', help = 'Output filename, if not supplied output to stdout')
parser.add_argument('-g', '--graphname', help = 'Name of the accuracy graph', default = 'Predicted Price vs Test')

args = parser.parse_args()
    
dataTrain       = read_datetime_csv(args.c + '_train.csv')

dataTestNoPrice = read_datetime_csv(args.c + '_future.csv')

forecaster = ForecasterRecursiveMultiSeries(
    estimator          = LGBMRegressor(random_state=123, verbose=-1, max_depth=10), 
    lags               = 400,
    encoding           = "ordinal", 
    dropna_from_series = False
)

exog_columns = dataTrain.columns.to_list();
exog_columns.pop(0)

# Create and fit forecaster
# ==============================================================================
forecaster.fit(
    series = dataTrain[['price']],
    exog   = dataTrain[exog_columns]
)


# Prediction with exogenous variables
# ==============================================================================
predictions = forecaster.predict(
    steps = dataTestNoPrice[exog_columns[0]].size,
    exog  = dataTestNoPrice[exog_columns]
)

if (args.show_accuracy):
    dataTest = read_datetime_csv(args.c + '_test.csv')
    fig, ax = plt.subplots(figsize=(7, 3.5))
    dataTest['price'].plot(ax=ax, xlabel="Actual")
    predictions['pred'].plot(ax=ax, ylabel="price", xlabel="Predicted")
    ax.legend()
    plt.show()

if (args.filename):
    predictions.to_csv(args.filename)
else:
    print(predictions.to_csv())
