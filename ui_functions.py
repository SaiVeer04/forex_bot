from forex_functions import functions as f
import MetaTrader5 as mt5
import time
import csv
class functions:
    def create_dict():
        dict1 = {}
        keys = []
        values = []
        f = open("currencypairs.txt",'r')
        for x in f:
            x1 = x.rstrip()
            if len(x1) != 0:
                keys.append(x1)
                values.append(1)
        for i in range(len(keys)):
            dict1[keys[i]] = values[i]
        return dict1
    
    def get_options(dict_val):
        option = []
        for key, value in dict_val.items():
            if(value == 1):
                option.append(key)
        return option
    
    def best_choice(options):
        best_str = ""
        best_int = 0
        for symbol in options:
            temp_int = best_int
            model = f.load(symbol)
            predictions,close = f.get_confidence(symbol,model)
            predict = predictions.ravel()
            change = 0
            close_price = close[0]
            #print(close_price)
            for x in predict:
                if(change < x):
                    change = x
            #print(change)
            change = change - close_price
            if(change > 0):
                best_str = symbol
                best_int = change
        return best_str

    def trading(best_choice):
        mt5.initialize()
        
        #get the data
        price = mt5.symbol_info_tick(best_choice).ask
        #close mt5
        
        #remeber to add a buy function right here

        og_price = price

        trailing_loss = og_price * 0.95
        sold = False
        sold_price = 0
        highest_price = price
        while sold == False:
            price = mt5.symbol_info_tick(best_choice).ask
            if(price <= trailing_loss):
                #sell the forex
                #add to csv
                print(f"Sold: {price}" )
                sold_price = price
                sold = True
            elif(price >= highest_price ):
                trailing_loss = price * 0.95
                print("higher price found")
            else:
                print(f"holding at: {price}")
            if(og_price < trailing_loss):
                print("profits guranteed")
            time.sleep(10)
        
        profits = sold_price - og_price
        profit = False
        if(profits > 0):
            profit = True
        with open('profits.csv', 'a', newline='') as csvfile:
            # Create a CSV writer
            writer = csv.writer(csvfile)
            
            # Add some data to the CSV file
            writer.writerow([best_choice,og_price, sold_price, profits,profit])
            
        mt5.shutdown()


        