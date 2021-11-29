import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import datetime as dt
from pprint import pprint
import json
 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from os.path import exists


def predict_tomorrow_price(asset, prediction_days = 30, start = dt.datetime(2010,1,1).strftime('%s')):
    
    cache_file_path = f"cache/{asset}.json"
    
    if exists(cache_file_path):
        return json.load(open(cache_file_path))

    end = dt.datetime.now().strftime('%s') # current time

    scaler = MinMaxScaler(feature_range=(0,1))

    # Dataset with all the prices of the asset up until the most recent day
    historic_prices = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{asset}?period1={start}&period2={end}&interval=1d&events=history&crumb=ydacXMYhzrn').dropna()

    # Dataset normalized between 0 and 1, 0 being the lowest price and 1 being the highest price of all the dataset
    scaled_data = scaler.fit_transform(historic_prices['Close'].values.reshape(-1,1))

    x_train = [] # 30 day price before the date
    y_train = [] # actual price the day in question
    
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    # Convert it to numpy
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Convert array into a 3 dimensional array, from shape (2368, 30) to shape (2368, 30, 1) so it can be entered in the NN
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=1))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Prediction of the next closing price
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Latest 30 days closing prices
    model_inputs = historic_prices[-prediction_days : ].Close.values
    model_inputs = model_inputs.reshape(-1, 1) # N rows, 1 column
    model_inputs = scaler.transform(model_inputs) 

    x_test = np.array([model_inputs[ -prediction_days : , 0]])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    predicted_price = predicted_prices[0, 0] # Predicted price for next day
    last_actual_price = historic_prices.Close.iat[-1] # Last price ever recorded
    percentage_change = 100 * predicted_price / last_actual_price - 100

    final_object = {
        'last_actual_price': float(last_actual_price),
        'predicted_price': float(predicted_price),
        'percentage_change': float(percentage_change),
        'value_change': float(predicted_price - last_actual_price),
        'last_30_day_prices': historic_prices[-prediction_days : ].to_dict(orient="records")
    }

    with open(cache_file_path, 'w+') as f:
        json.dump(final_object, f)

    return final_object

if __name__ == '__main__':
    pprint(predict_tomorrow_price('FB'))