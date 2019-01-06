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
from scipy import stats

class SharedObjects:
    pd = 'm1'
    size = 300
    ipairs = []
    coint_mat = pandas.DataFrame()
    spreads = dict()
    prs = dict()
    quantity = 2
    corr_mat = pandas.DataFrame()
    dataset0 = pandas.DataFrame()
    traded_currencies = []
    automate = False
    corr_bd = 0.3
    
    def __init__(self, con):
        self.connection = con
        #self.connection.connect()
        self.tradable_pairs = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 
                  'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP', 
                  'USD/ZAR', 'ZAR/JPY']
        self.coint_mat = pandas.DataFrame()
        self.spreads = dict()
        self.corr_bd = 0.3
        return
        
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
        self.automate = True
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
    
    def mergeall_byrow(self, dtfs):
        dtf = pandas.DataFrame()
        for key in dtfs:
            dtf = pandas.concat([dtf, dtfs.get(key)])
        return dtf
    
    def pairwise_spreadplot(self, a, b):
        dff = pandas.DataFrame()
        dff[a] = self.spreads.get(a)[b]
        dff['mean'] = dff[a].mean()
        dff['upper'] = dff['mean'] + (2.05*dff[a].std())
        dff['lower'] = dff['mean'] - (2.05*dff[a].std())
        dff['buy'] = dff[a][((dff[a] < dff['lower']) & (dff[a].shift(1) > dff['lower']) | 
                          (dff[a] <  dff['mean']) & (dff[a].shift(1) >  dff['mean']))]

        dff['sell'] = dff[a][((dff[a] > dff['upper']) & (dff[a].shift(1) < dff['upper']) | 
                           (dff[a] >  dff['mean']) & (dff[a].shift(1) <  dff['mean']))]
        return(dff.plot(figsize =(17,10), style=['g', '--r', '--b', '--b', 'm^','cv']))
    
    def update(self, prs, coint_mat, spreads):
        self.prs = prs
        self.coint_mat = coint_mat.fillna(0.999)
        self.spreads = spreads
        self.pair_selection()
        return
    
    def pairwise_spread(self, a):
        dff = self.spreads.get(a)
        dff = dff.drop(columns=[a])
        return(dff)
    
    def pairwise_plot(self, a, b):
        dtf = pandas.DataFrame()
        yy = self.prs.get(a).standardized_prices
        dtf[a] = yy
        xx = self.prs.get(b).standardized_prices
        dtf[b] = xx
        return(dtf.plot(figsize =(17,10)))
    
    def pair_selection(self):
        prs = self.prs
        self.ipairs = []
        dtf = pandas.DataFrame()
        for key in prs:
            yy = prs.get(key).standardized_prices
            dtf[key] = pandas.Series(yy)
        self.dataset0 = dtf
        self.corr_mat = dtf.corr(method='kendall').replace(1, 0)
        for key in prs.keys():
            for ky in prs.keys():
                #if(self.corr_mat.loc[key][ky] >= self.corr_bd and 
                #   self.coint_mat.loc[key][ky] < 0.05 and 
                #   self.coint_mat.loc[ky][key] < 0.05 ):
                #    if ([ky, key] not in self.ipairs):
                #        self.ipairs.append([key,ky])
                if(self.corr_mat.loc[key][ky] <= -self.corr_bd and 
                     self.coint_mat.loc[key][ky] < 0.05 and 
                     self.coint_mat.loc[ky][key] < 0.05):
                    if ([key, ky] not in self.ipairs):
                        self.ipairs.append([key,ky])
                else:
                    return
        if(self.automate == True):
            self.signal()
        return
    
    def signal(self):
        signals = []
        for pr in self.ipairs:
            dff = pandas.DataFrame()
            dff['spread'] = self.spreads.get(pr[0])[pr[1]]
            dff['mean'] = dff['spread'].mean()
            dff['upper'] = dff['mean'] + (2.05*dff['spread'].std())
            dff['lower'] = dff['mean'] - (2.05*dff['spread'].std())
            index = len(dff.index.values) - 1 
            
            y = self.prs.get(pr[0]).standardized_prices()
            x = self.prs.get(pr[1]).standardized_prices()
            b = stats.linregress(x, y).slope
            
            if(dff['spread'][index] < dff['lower'][index]):
                signals.append({pr[0]: ['Buy', 1], pr[1]: ['Sell', abs(b)]})
                self.Buy(pr[0], self.quantity * 1)
                self.Sell(pr[1], self.quantity * abs(b))
            elif(dff['spread'][index] > dff['upper'][index]):
                signals.append({pr[0]: ['Sell', 1], pr[1]: ['Buy', abs(b)]})
                self.Sell(pr[0], self.quantity * 1)
                self.Buy(pr[1], self.quantity * abs(b))
        return(signals)
            
    def Buy(self, symbol, amount, limit=None):
        self.check()
        if((symbol in self.traded_currencies) is False):
            self.connection.create_market_buy_order(symbol, amount)
        return
    
    def Sell(self, symbol, amount, limit=None):
        self.check()
        if((symbol in self.traded_currencies) is False):
            self.connection.create_market_sell_order(symbol, amount)
        return
    
    def check(self):
        df = self.connection.get_open_positions().T
        if(len(df.index) > 0):
            tradeIds = df.loc['tradeId'].tolist()
            limits = df.loc['limit'].tolist()
            self.traded_currencies = df.loc['currency'].tolist()
        
            for lim in limits:
                if(lim == 0):
                    tradeId = tradeIds[limits.index(lim)]
                    self.connection.change_trade_stop_limit(tradeId, is_in_pips=True, is_stop=False, rate=0.1)
        return