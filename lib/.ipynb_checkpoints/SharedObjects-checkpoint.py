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
import numpy

class SharedObjects:
    pd = 'm1'
    size = 300
    pairs = dict()
    coint_mat = pandas.DataFrame()
    spreads = dict()
    prs = dict()
    
    def __init__(self, con):
        self.connection = con
        #self.connection.connect()
        self.tradable_pairs = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 
                  'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP', 
                  'USD/ZAR', 'ZAR/JPY']
        
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
        prices = list(  ( data['askclose'] + data['bidclose']  ) / 2 ) 
        opens = list( (data['askopen'] + data['bidopen']) / 2)
        highs = list( (data['askhigh'] + data['bidhigh']) / 2)
        lows = list( (data['asklow'] + data['bidlow']) / 2 )
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
    
    def set_var(self, dic, datf0, prs):
        self.spreads = dic
        self.coint_mat = datf0
        self.prs = prs
    
    def mergeall_byrow(self, dtfs):
        dtf = pandas.DataFrame()
        for key in dtfs:
            dtf = pandas.concat([dtf, dtfs.get(key)])
        return dtf
    
    def pairwise_spreadplot(self, a, b):
        dff = pandas.DataFrame()
        dff[a] = self.spreads.get(a)[b]
        dff['mean'] = dff[a].mean()
        dff['upper'] = dff['mean'] + 1.96*dff[a].std()
        dff['lower'] = dff['mean'] - 1.96*dff[a].std()
        dff['buy'] = dff[a][((dff[a] < dff['lower']) & (dff[a].shift(1) > dff['lower']) | 
                          (dff[a] <  dff['mean']) & (dff[a].shift(1) >  dff['mean']))]

        dff['sell'] = dff[a][((dff[a] > dff['upper']) & (dff[a].shift(1) < dff['upper']) | 
                           (dff[a] >  dff['mean']) & (dff[a].shift(1) <  dff['mean']))]
        return(dff.plot(figsize =(17,10), style=['g', '--r', '--b', '--b', 'm^','cv']))
    
    def pairwise_spread(self, a):
        dff = self.spreads.get(a)
        dff = dff.drop(columns=[a])
        return(dff)
    
    def pairwise_plot(self, a, b):
        dtf = pandas.DataFrame()
        yy = self.prs.get(a).prices['Close'].tolist()
        y_np = numpy.array(yy)
        standardized_y = ((y_np-y_np.mean())/y_np.std() ).tolist()
        dtf[a] = standardized_y
        xx = self.prs.get(b).prices['Close'].tolist()
        x_np = numpy.array(xx)
        standardized_x = ((x_np-x_np.mean())/x_np.std() ).tolist()
        dtf[b] = standardized_x
        return(dtf.plot(figsize =(17,10)))