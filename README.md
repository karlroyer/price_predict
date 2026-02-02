Prediction of Energy Prices

sk_forecast.py builds a full AI prediction based on estimation data - takes considerable time and CPU the model is saved in a file named forecaster_custom_features.joblib. The sk_forecast_load.py uses this to predict 7 days of data and display response.

sk_forecast_rnn.py builds and shows 7 days forcast using a Rnn forecaster.

sk_forecast_sarimax.py does the same using a sarimax statistical model.

