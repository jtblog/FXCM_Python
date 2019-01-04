import pandas
import numpy
from sklearn import *
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import warnings

class Pair:
    
    standardized_prices = []
    sym = ''
    
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
        df = None
        return
    
    def standardize_prices(self):
        y_np = numpy.array(self.prices['Close'].tolist())
        self.standardized_prices = ( (y_np-y_np.mean())/y_np.std() ).tolist()
        return
        
    
    def on_price_update(self, data, dataframe):
        if(data['Symbol'] == self.sym):
            self.update(dataframe)
        else:
            return
        return
        
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
        
    def co_integration(self, prs, ct_mat, spreads):
        ct_mat = ct_mat
        spreads = spreads
        adfs = dict()
        spd = pandas.DataFrame()
        
        for ky in prs:
            x = prs.get(ky).standardized_prices
            y = prs.get(self.sym).standardized_prices
            if(len(x) == len(y) and len(x) > 0 and len(y) > 0):
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                s_x = list(map(lambda a: a*slope, x))
                spread = [a - b for a, b in zip(y, s_x)]
                spd[ky] = pandas.Series(spread)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adf = adfuller(spread, maxlag=1)
                adfs[ky] = adf[1]
                
        spreads[self.sym] = spd
            
        df0 = pandas.DataFrame(adfs, index = [self.sym])
        if( (ct_mat is not None and self.sym in ct_mat.index) is True):
            ct_mat.loc[self.sym] = df0.loc[self.sym]
        else:
            ct_mat = pandas.concat([ct_mat, df0], sort=False)
        
        adfs = None
        spd = None
        spread = None
        
        return([ct_mat, spreads])