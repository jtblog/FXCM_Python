#
# fxcm_data_reader
#
# a class for fetching historical data provided by
# FXCM
#
# by the Python Quants GmbH
# September 2017
#

import datetime as dt
from io import BytesIO, StringIO
import gzip
import urllib.request
import pandas as pd


class fxcmpy_tick_data_reader(object):
    """ fxcm_tick_data_reader, a class to fetch historical tick data provided 
    by FXCM
    """

    symbols = ('AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'CADCHF', 'EURAUD',
               'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD', 'GBPCHF', 'GBPJPY',
               'GBPNZD', 'GBPUSD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'NZDCAD',
               'NZDCHF', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY')

    def __init__(self, symbol, start, end, verbosity=False):
        """ Constructor of the class.

        Arguments:
        ==========

        symbol: string, one of fxcm_data_reader.symbols,
            defines the instrument to deliver data for.

        start: datetime.datetime, datetime.date or string (defaut None),
            the first date to receive data for.

        end: datetime.datetime, datetime.date or string (defaut None),
            the first date to receive data for.

        verbosity: boolean (default: False), 
            whether to print output or not.

        """
        if isinstance(start, str):
            try:
                start = pd.Timestamp(start).to_pydatetime()
            except:
                msg = "Can not convert parameter start to datetime."
                raise ValueError(msg)

        elif isinstance(start, dt.datetime) or isinstance(start, dt.date):
            pass
        else:
            msg = "start must either be a datetime object or a string"
            raise ValueError(msg)

        self.start = start

        if isinstance(end, str):
            try:
                end = pd.Timestamp(end).to_pydatetime()
            except:
                msg = "Can not convert parameter end to datetime."
                raise ValueError(msg)

        elif isinstance(end, dt.datetime) or isinstance(end, dt.date):
            pass
        else:
            msg = "end must either be a datetime object or a string."
            raise ValueError(msg)

        self.stop = end

        if self.start > self.stop:
            raise ValueError('Invalid date range')

        if symbol not in self.symbols:
            msg = 'Symbol %s is not supported. For a list of supported'
            msg += ' symbols, see get_available_symbols()'
            raise ValueError(msg % symbol)
        else:
            self.symbol = symbol

        if verbosity != True and verbosity != False:
            raise TypeError('verbosity must be a boolean')
        else:
            self.verbosity = verbosity

        self.data = None
        self.url = 'https://tickdata.fxcorporate.com/%s/%s/%s.csv.gz'
        self.codec = 'utf-16'
        if not isinstance(self, fxcmpy_candles_data_reader):
            self.__fetch_data__()
            
    def get_raw_data(self):
        """ Returns the raw data set as pandas DataFrame """
        return self.data

    def get_data(self, start=None, end=None):
        """ Returns the requested data set as pandas DataFrame;
        DataFrame index is converted to DatetimeIndex object 
        
        Arguments:
        ==========

        start: datetime.datetime, datetime.date or string (defaut None),
            the first date to receive data for. If None, the value of start as 
            given in the constructor is used. 

        end: datetime.datetime, datetime.date or string (defaut None),
            the first date to receive data for. If None, the value of end as 
            given in the constructor is used. 

        """
        if start is not None:
            if isinstance(start, str):
                try:
                    start = pd.Timestamp(start).to_pydatetime()
                except:
                    msg = "Can not convert parameter start to datetime."
                    raise ValueError(msg)

            elif isinstance(start, dt.datetime) or isinstance(start, dt.date):
                pass
            else:
                msg = "start must either be a datetime object or a string."
                raise ValueError(msg)

        if end is not None:
            if isinstance(end, str):
                try:
                    end = pd.Timestamp(end).to_pydatetime() 
                except:
                    msg = "Can not convert parameter end to datetime."
                    raise ValueError(msg)

            elif isinstance(end, dt.datetime) or isinstance(end, dt.date):
                pass
            else:
                msg = "end must either be a datetime object or a string"
                raise ValueError(msg)

        try:
            self.data_adj
        except:
            data = self.data.copy()
            index = pd.to_datetime(data.index.values,
                                   format='%m/%d/%Y %H:%M:%S.%f')
            data.index = index
            self.data_adj = data
        data = self.data_adj
        if start is not None:
            data = data[data.index >= start]
        if end is not None:
            data = data[data.index <= end]
        return data
    
    @classmethod
    def get_available_symbols(cls):
        """ Return all symbols available"""
        return cls.symbols

    def __fetch_data__(self):
        """ Retrieve the data for the given symbol and the given time window """
        self.data = pd.DataFrame()
        running_date = self.start
        seven_days = dt.timedelta(days=7)
        while running_date <= self.stop:
            year, week, noop = running_date.isocalendar()
            url = self.url % (self.symbol, year, week)
            data = self.__fetch_dataset__(url)
            if len(self.data) == 0:
                self.data = data
            else:
                self.data = pd.concat((self.data, data))
            running_date = running_date + seven_days
        
    def __fetch_dataset__(self, url):
        """ Fetch the data for the given symbol and for one week."""
        if self.verbosity:
            print('Fetching data from: %s' % url)
        requests = urllib.request.urlopen(url)
        buf = BytesIO(requests.read())
        f = gzip.GzipFile(fileobj=buf)
        data = f.read()
        data_str = data.decode(self.codec)
        data_pandas = pd.read_csv(StringIO(data_str), index_col=0)
        return data_pandas

class fxcmpy_candles_data_reader(fxcmpy_tick_data_reader):
    """ fxcm_candles_data_reader, a class to fetch historical candles data 
    provided by FXCM
    """

    def __init__(self, symbol, start, end, period, verbosity=False):
        """ Constructor of the class.

        Arguments:
        ==========

        symbol: string, one of fxcm_data_reader.symbols,
            defines the instrument to deliver data for.

        start: datetime.datetime, datetime.date or string (defaut None),
            the first date to receive data for.

        end: datetime.datetime, datetime.date or string (defaut None),
            the first date to receive data for.

        period: string, one of ('m1', 'H1', 'D1'), 
            the granularity of the data.

        verbosity: boolean (default: False), 
            whether to print output or not.

        """


        fxcmpy_tick_data_reader.__init__(self, symbol, start, end, verbosity)
        if period not in ['m1', 'H1', 'D1']:
            raise ValueError("period must be one of 'm1', 'H1' or 'D1'")
        self.period = period
        self.codec = 'utf-8'
        self.url = 'https://candledata.fxcorporate.com/%s/%s/%s/%s.csv.gz'
        self.__fetch_data__()


    def __fetch_data__(self):
        """ Fetch the data for the given symbol, period 
        and the given time window. """
        self.data = pd.DataFrame()
        if self.period != 'D1':
            running_date = self.start
            seven_days = dt.timedelta(days=7)
            while running_date <= self.stop:
                year, week, noop = running_date.isocalendar()
                url = self.url % (self.period, self.symbol, year, week)
                data = self.__fetch_dataset__(url)
                if len(self.data) == 0:
                    self.data = data
                else:
                    self.data = pd.concat((self.data, data))
                running_date = running_date + seven_days
        else:
            start, noop, noop2 = self.start.isocalendar()
            stop, noop, noop2 = self.stop.isocalendar()
            if stop >= dt.datetime.now().year:
                msg = "Candles with period 'D1' are restricted to years before %s"
                raise ValueError(msg % dt.datetime.now().year ) 
            for year in range(start, stop+1):
                url = 'https://candledata.fxcorporate.com/%s/%s/%s.csv.gz'
                data = self.__fetch_dataset__(url % (self.period, self.symbol,
                                                     year) )

                if len(self.data) == 0:
                    self.data = data
                else:
                    self.data = pd.concat((self.data, data))

        
