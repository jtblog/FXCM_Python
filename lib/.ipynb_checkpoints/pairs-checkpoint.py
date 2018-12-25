

class Pair:
    
    def __init__(self, symbol, prices=None):
        self.sym = symbol
        self.prices = prices
        return ''
    
    @classmethod
    def f1(self, x, y):
        return min(x, x+y)