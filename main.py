from forex_functions import functions
import MetaTrader5 as mt5
import datetime
from keras.models import Sequential
symbol = "EURUSD"

model = Sequential()

if(functions.check_model_exists(symbol) == True):
    model = functions.load(symbol)
else:
    df = functions.get_training_data(symbol)
    df = functions.create_custom_indicators(df)
    x_close_train,x_ema_train,x_rsi_train,y_close_train = functions.process_train_data(df,functions.get_train_arrays)
    model = functions.new_model(x_close_train,x_ema_train,x_rsi_train,y_close_train)
    functions.save_model(symbol,model)



print(functions.get_confidence(symbol,model))
arr = functions.get_confidence(symbol,model).ravel()
dimensions = arr

# Print the dimensions
print(dimensions)



