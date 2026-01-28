
# Import CSV library
import pandas as pd
from pandas import to_datetime
import matplotlib.pyplot as plt

# Import prophet
from prophet import Prophet


def read_price_data(filename):
  # Read data
  priceData = pd.read_csv(filename)
  priceData = priceData.drop(['country','date'], axis=1)

  # Convert datetime to datetime objects and Drop timezone data
  priceData['datetime'] = to_datetime(priceData['datetime'])
  priceData['datetime'] = priceData['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
  priceData.set_index('datetime', inplace=True)
  priceData.drop_duplicates(inplace=True)
  return priceData

def resample_data_to_csv(data, filename):

  # Now calculate per minute data
  resampledData = data.resample('1min').ffill()

  resampledData.rename_axis('ds', inplace=True)
  resampledData.reset_index()
  resampledData.rename(columns = {"price": "y"}, inplace=True)

  resampledData.to_csv(filename)
  
# Read data
priceData = read_price_data('/home/karl/Amster/data/enso/nl_day_ahead_prices_raw_utc.csv')
print('Price data columns ',priceData.columns);

# Now calculate per minute data
resample_data_to_csv(priceData, '/home/karl/Amster/data/enso/smooth_data.csv')
print('Resampled data to 1 minute intevals')
priceData = pd.read_csv('/home/karl/Amster/data/enso/smooth_data.csv')


# Now fit the prophet model to data
print('Creating prophet models')
model = Prophet()
model.fit(priceData)

#futureData = model.make_future_dataframe(
#  periods = 60 * 24 * 7,    # Seven days of 1 min intervals
#  freq    = 60          # Every 5 mins
#)

#model.predict(futureData)


  
