import pandas
import numpy
from sklearn import *
from statsmodels.tsa.stattools import adfuller
import warnings

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
    
    def reg(self, x, y):
        regr = linear_model.LinearRegression()
        x_constant = pandas.concat([x, pandas.Series([1]*len(x),index = x.index)], axis=1)
        regr.fit(x_constant, y)    
        beta = regr.coef_[0]
        alpha = regr.intercept_
        spread = y - x*beta - alpha
        return spread
    
    def co_integration(self, prs):
        adfmat = dict()
        st_df = dict()
        spread_df = dict()
        for pr in prs:
            adfmat[pr] = []
        
        
        for key in prs:
            dtf = pandas.DataFrame()
            yy = prs.get(key).prices['Close'].tolist()
            y_np = numpy.array(yy)
            standardized_y = ((y_np-y_np.mean())/y_np.std() ).tolist()
            dtf[key] = standardized_y
            spd = dict()

            for ky in prs:
                xx = prs.get(ky).prices['Close'].tolist()
                x_np = numpy.array(xx)
                standardized_x = ((x_np-x_np.mean())/x_np.std() ).tolist()
                dtf[ky] = standardized_x
                spread = self.reg(dtf[ky], dtf[key])
                spd[ky] = spread
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adf = adfuller(spread, maxlag=1)
                adfmat.get(ky).append(adf[1])

            st_df[key] = dtf
            spread_df[key] = pandas.DataFrame(spd)
            
        adf_ps = pandas.DataFrame(index = list(adfmat.keys()))
        for ke in adfmat:
            adf_ps[ke] = adfmat.get(ke)
        for key in spread_df:
            spread_df[key] = pandas.DataFrame(spread_df.get(key))  
        adf_ps = adf_ps.fillna(0.999)

        adfmat = None
        dtf = None
        spd = None
        spread = None
        
        return([spread_df, adf_ps])