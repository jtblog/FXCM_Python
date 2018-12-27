import pandas
import sklearn

class Pair:
    
    def __init__(self, symbol, dataframe):
        self.sym = symbol
        self.prices = dataframe
        
    def update(self, df):
        df['Close'] = ( df['Bid'] + df['Ask'] ) / 2
        df = df.drop(columns=['Bid', 'Ask'])
        #self.df = df
        ts0 = self.prices.axes[0].tolist()[len(self.prices.axes[0].tolist())-1]
        ts1 = df.axes[0].tolist()[len(df.axes[0].tolist())-1]
        cl = df['Close'][len(df.axes[0].tolist())-1]
        high = df['High'][len(df.axes[0].tolist())-1]
        low = df['Low'][len(df.axes[0].tolist())-1]
        opn =  self.prices['Close'][len(self.prices.axes[0].tolist())-1]
        d = {'Open': [opn], 'High': [high], 'Low': [low], 'Close': [cl]}
        dtf = pandas.DataFrame(data = d, index = [ts1])
        if(ts0.minute == ts1.minute):
            ts2 = self.prices.axes[0].tolist()[len(self.prices.axes[0].tolist())-2]
            self.prices = self.prices[:ts2]
            self.prices = pandas.concat([self.prices, dtf])
        else:
            ts = self.prices.axes[0].tolist()[0]
            self.prices = self.prices.drop(ts)
            self.prices = pandas.concat([self.prices, dtf])
        df = pandas.DataFrame()
        return(None)
    
    def on_price_update(self, data, dataframe):
        if(data['Symbol'] == self.sym):
            self.update(dataframe)
        else:
            return(None)