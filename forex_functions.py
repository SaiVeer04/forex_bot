import MetaTrader5 as mt5
import datetime
import pandas as pd
import talib
import pprint
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import load_model
import pytz
import os
from keras.models import Sequential

class functions:
    ticks = 60
    forecast = 9
    def get_training_data(symbol):
        #Start mt5
        mt5.initialize()
        #time frame is 5 minutes can make shorter if needed
        timeframe = mt5.TIMEFRAME_M5
        #get the data
        symbol_ticks = mt5.copy_rates_from(symbol, timeframe,datetime.datetime.now(),20025)
        #close mt5
        mt5.shutdown()
        
        df = pd.DataFrame(symbol_ticks)
        #convert the time units
      
        df['time'] = pd.to_datetime(df['time'],unit ='s')

        return df
    
    def create_custom_indicators(df):
        #values will be edited in here

        #get ema and rsi
        ema = talib.EMA(df['close'], timeperiod=25)
        sma = talib.SMA(df['close'], timeperiod=25)

        df['ema'] = ema
        df['sma'] = sma
        #drop first 25 values
        df.drop(index=df.index[:25],inplace=True)

        return df

    def get_train_arrays(train_data):
        
        x_train = []
        y_train = []
        for i in range(functions.ticks, len(train_data) - functions.forecast-1):
            x_train.append(train_data[i-functions.ticks:i, 0])
            y_train.append(train_data[i:i+functions.forecast, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        return x_train,y_train

    def process_train_data(df,get_train_arrays):
        #get data from the datasets
        close = df['close']
        ema = df['ema']
        rsi = df['sma']

        #change to values
        close_val = close.values
        ema_val = ema.values
        rsi_val = rsi.values

        #get length
        training_len = math.ceil(len(close_val)*0.8)
        
        scaler = MinMaxScaler(feature_range=(0,1))

        scaled_data = scaler.fit_transform(close_val.reshape(-1,1))
        train_data_close = scaled_data[0: training_len, :]

        scaled_data = scaler.fit_transform(ema_val.reshape(-1,1))
        train_data_ema = scaled_data[0: training_len, :]

        scaled_data = scaler.fit_transform(rsi_val.reshape(-1,1))
        train_data_rsi = scaled_data[0: training_len, :]

        x_close_train,y_close_train = get_train_arrays(train_data_close)
        x_ema_train,y_ema_train = get_train_arrays(train_data_ema)
        x_rsi_train,y_rsi_train = get_train_arrays(train_data_rsi)
        
        return x_close_train,x_ema_train,x_rsi_train,y_close_train

    def train_model(train1,train2,train3,y_train):
        input_1 = layers.Input(shape=(train1.shape[1], 1))
        input_2 = layers.Input(shape=(train2.shape[1], 1))
        input_3 = layers.Input(shape=(train3.shape[1],1))

        lstm1 = layers.LSTM(100, return_sequences=True)(input_1)
        lstm2 = layers.LSTM(100, return_sequences=True)(input_2)
        lstm3 = layers.LSTM(100, return_sequences=True)(input_3)

        merged = concatenate([lstm1, lstm2,lstm3])
        

        hidden = layers.LSTM(100, return_sequences=False)(merged)
        output = layers.Dense(20)(merged)

        model = keras.Model(inputs=[input_1, input_2,input_3], outputs=output)

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit([train1, train2, train3], y_train, batch_size=1, epochs=3)
        
        return model
    
    def new_model(train1,train2,train3,y_train):
        n_lookback = functions.ticks
        n_forecast = functions.forecast
        model = Sequential()
        model.add(layers.LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 3)))
        model.add(layers.LSTM(units=50))
        model.add(layers.Dense(n_forecast))

        x_input = np.concatenate((train1, train2, train3), axis=1)
        x_input = x_input.reshape(x_input.shape[0], n_lookback, 3)
        print("compiling model")
        model.compile(loss='mean_squared_error', optimizer='adam')
        print("compiling model1")
        model.fit(x_input, y_train, epochs=100, batch_size=32)
        print("compiling model2")
        return model

    def save_model(symbol,model):
        append_write = 'a'
        if os.path.exists('currencypairs.txt'):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
        
        f = open("currencypairs.txt",append_write)

        f.write("\n")
        f.write(symbol)
        

        f.close()

        model.save(symbol+'.h5')

    def check_model_exists(symbol):
        check = False
        f = open("currencypairs.txt", "r")
        for x in f:
            if(symbol in x):
                check = True
        f.close()
        return check


    def load(symbol):
        model = load_model(symbol+'.h5')
        return model

    def prepare_test_data(training_data_len,values):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(values.reshape(-1,1))
        test_data = scaled_data[training_data_len-functions.ticks: , : ]
        x_test = []
        y_test = values[training_data_len:]

        for i in range(functions.ticks, len(test_data)):
            x_test.append(test_data[i-functions.ticks:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test,y_test

    def test_model(model):
        close = df['close']
        ema = df['ema']
        rsi = df['sma']

        close_val = close.values
        ema_val = ema.values
        rsi_val = rsi.values

        #get length
        training_len = math.ceil(len(close_val)*0.8)

        x_close_test,y_test1 = functions.prepare_test_data(training_len,close_val)
        x_ema_test,y_test2 = functions.prepare_test_data(training_len,ema_val)
        x_rsi_test,y_test3 = functions.prepare_test_data(training_len,rsi_val)

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(close_val.reshape(-1,1))

        predictions = model.predict([x_close_test,x_ema_test,x_rsi_test])
        confidence_rate = predictions
        predictions = scaler.inverse_transform(predictions)

        # Calculate the root mean squared error
        rmse = np.sqrt(np.mean(predictions - y_test1)**2)
        print(rmse)
        print(confidence_rate)
    
    def prepare_predict(training_data_len,values):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(values.reshape(-1,1))
        test_data = scaled_data[training_data_len-functions.ticks: , : ]
        x_test = []
        #print(test_data)
        y_test = values[training_data_len:]

    
        x_test.append(test_data[0:training_data_len, 0])
            
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        return x_test

    def get_confidence(symbol,model):
        mt5.initialize()
        
        timeframe = mt5.TIMEFRAME_M5
        symbol_ticks = mt5.copy_rates_from_pos(symbol, timeframe,0,100)
        mt5.shutdown()

        df_test = pd.DataFrame(symbol_ticks)
        df_test['time'] = pd.to_datetime(df_test['time'],unit ='s')

        new_df = functions.create_custom_indicators(df_test)

         #get data from the datasets
        close = df_test['close']
        ema = df_test['ema']
        rsi = df_test['sma']

        #change to values
        close_val = close.values
        ema_val = ema.values
        rsi_val = rsi.values

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(close_val.reshape(-1,1))

        close_test = functions.prepare_predict(len(df_test),close_val)
        ema_test = functions.prepare_predict(len(df_test),ema_val)
        rsi_test = functions.prepare_predict(len(df_test),rsi_val)

        x_input = np.concatenate((close_test, ema_test, rsi_test), axis=1)
        x_input = x_input.reshape(x_input.shape[0], functions.ticks, 3)
        predictions = model.predict(x_input)
        predictions = scaler.inverse_transform(predictions)
        gmt2 = pytz.timezone('Etc/GMT-2')
        df_test['time'] = df_test['time'].dt.tz_localize(gmt2)

        # Convert the 'time' column to EST
        est = pytz.timezone('EST')
        df_test['time'] = df_test['time'].dt.tz_convert(est)
        #print(df_test.tail(1))
        #print(df_test['close'].tail(1))
        return predictions
    

    def get_confidence(symbol,model):
        mt5.initialize()
        
        timeframe = mt5.TIMEFRAME_M5
        symbol_ticks = mt5.copy_rates_from_pos(symbol, timeframe,0,100)
        mt5.shutdown()

        df_test = pd.DataFrame(symbol_ticks)
        df_test['time'] = pd.to_datetime(df_test['time'],unit ='s')

        new_df = functions.create_custom_indicators(df_test)

         #get data from the datasets
        close = df_test['close']
        ema = df_test['ema']
        rsi = df_test['sma']

        #change to values
        close_val = close.values
        ema_val = ema.values
        rsi_val = rsi.values

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(close_val.reshape(-1,1))

        close_test = functions.prepare_predict(len(df_test),close_val)
        ema_test = functions.prepare_predict(len(df_test),ema_val)
        rsi_test = functions.prepare_predict(len(df_test),rsi_val)

        x_input = np.concatenate((close_test, ema_test, rsi_test), axis=1)
        x_input = x_input.reshape(x_input.shape[0], functions.ticks, 3)
        predictions = model.predict(x_input)
        predictions = scaler.inverse_transform(predictions)
        gmt2 = pytz.timezone('Etc/GMT-2')
        df_test['time'] = df_test['time'].dt.tz_localize(gmt2)

        # Convert the 'time' column to EST
        est = pytz.timezone('EST')
        df_test['time'] = df_test['time'].dt.tz_convert(est)
        #print(df_test.tail(1))
        #print(df_test['close'].tail(1))
        return predictions,df_test['close'].tail(1).values
    
   
    





        
        

    
        


    
        

        


