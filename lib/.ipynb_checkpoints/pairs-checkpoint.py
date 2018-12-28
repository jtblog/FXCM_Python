import pandas
import numpy
from sklearn import *

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
        
    def multiple_linear_regression(self, dictn):
        lm = linear_model.LinearRegression()
        for key in dictn:
            keys = set(dictn.keys())
            excludes = set([self.sym])
            diff = keys.difference(excludes)
            x = pandas.DataFrame()
            y = self.prices['Close'].tolist()

            for ky in diff:
                x[ky] = dictn.get(ky).prices['Close'].tolist()
            model = lm.fit(x,y)

            #R squared
            rs = lm.score(x,y)
            #Coefficients
            bs = model.coef_.tolist()
            #Intercept
            intercept = model.intercept_
            #Prediction (first)
            pred = lm.predict(x)[0]

            dez = {}
            if(y[len(y)-1] > pred):
                dez = {True: 'SELL', False: 'BUY'}
            elif(y[len(y)-1] < pred):
                dez = {True: 'BUY', False: 'SELL'}
            dez[1.0] = dez[True]
            dez[-1.0] = dez[False] 

            df = pandas.DataFrame()
            df[0] = [self.sym, '-', dez[True]]
            df[1] = ['R_Squared', rs, '-']
            df[2] = ['Prediction', pred, '-' ]
            df[3] = ['Intercept', intercept, '-']
            hd = x.columns.tolist()
            for s in hd:
                c = bs[hd.index(s)]
                sgn = numpy.sign(c)
                df[hd.index(s)+4] = [s, c, dez[sgn] ]
            
            return(df)
        