"""
from types import MethodType
 
def print_classname(a):
print a.__class__.__name__
 
class A(object):
pass

# this assigns the method to the instance a, but not to the class definition
a = A()
a.print_classname = MethodType(print_classname, a, A)
 
# this assigns the method to the class definition
A.print_classname = MethodType(print_classname, None, A)
"""
import pandas
import threading

class SharedObjects:
    pd = 'm1'
    size = 300
    pairs = dict()
    
    def __init__(self, con):
        self.connection = con
        #self.connection.connect()
        self.tradable_pairs = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 
                  'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP', 
                  'USD/ZAR', 'ZAR/JPY', 'Copper']
        self.discrepancies = {'EUR/USD': 0.57, 'USD/JPY': 55.5, 'GBP/USD': 0.63, 'USD/CHF': 0.49, 
                  'AUD/USD': 0.35, 'USD/CAD': 0.68, 'NZD/USD': 0.33, 'EUR/GBP': 0.45, 
                  'USD/ZAR': 7.3, 'ZAR/JPY': 4, 'Copper': 1.3}
        
    def get_status(self):
        status = ""
        if(self.connection.is_connected() == True):
            status = "You are logged in"
        else:
            status =  "You are logged out"
        return(status)
    
    def get_subscribed_symbols(self):
        return(self.connection.get_subscribed_symbols())
    
    def historical_data(self, sym, pd, sz):
        data = self.connection.get_candles(sym, period=pd, number=sz)
        dif = self.discrepancies.get(sym)
        prices = list( ( data['askclose'] + data['bidclose'] / 2 ) - dif ) 
        opens = list( (data['askopen'] + data['bidopen'] / 2) - dif )
        highs = list( (data['askhigh'] + data['bidhigh'] / 2) - dif )
        lows = list( (data['asklow'] + data['bidlow'] / 2) - dif )
        #ticks_no = data['tickqty']
        dates = data.axes[0].tolist()
        d = {'Open': opens, 'High': highs, 'Low': lows, 'Close': prices}
        df = pandas.DataFrame(data = d, index = dates)
        return(df)
    
    def stream_data(self, sym, handler):
        self.connection.subscribe_market_data(sym, add_callbacks = handler)
        return(None)
    
    def unstream_data(self):
        try:
            def unstream():
                for symbol in self.tradable_pairs:
                    self.connection.unsubscribe_market_data(symbol)
            
            threading.Thread(target=unstream).start()
        except:
            print("Error: unable to start thread")
        return(None)