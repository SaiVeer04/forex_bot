from datetime import datetime
import threading
import time
import os
from ui_functions import functions as u
from forex_functions import functions as f

dict_val = u.create_dict()

while True:
    options = u.get_options(dict_val)
    best_choice = u.best_choice(options)
    if len(best_choice) > 0:
        u.trading(best_choice)
    else:
        print("waiting for next in hit in one min...")
        time.sleep(60)
    break
    
    