import os, re, sys

import numpy as np
import pandas as pd

class quantFeatures:

    def __init__(self, data):
        self.data = data
    
    def BollingerBand(self, win:int=10):
        cl = self.data['close']
        k = 2

        mbb = cl.rolling(window=win).mean()
        ubb = mbb + k * cl.rolling(window=win).std()
        lbb = mbb - k * cl.rolling(window=win).std()

        bb_width = ubb - lbb    #width of bollinger band
        d_to_ubb = ubb - cl     #distance from close price to upper bollinger band
        d_to_lbb = cl - lbb     #distance from close price to lower bollinger band

        ret = pd.concat([bb_width, d_to_ubb, d_to_lbb], axis=1)
        ret.columns = ['bb_width', 'd_to_ubb', 'd_to_lbb']

        return ret

    def MACD(self):
        cl = self.data['close']

        macd_short, macd_long, macd_signal = 12, 26, 9
        macd_short_term = cl.ewm(span=macd_short).mean()
        macd_long_term = cl.ewm(span=macd_long).mean()
        
        MACD = macd_short_term - macd_long_term             #MACD
        MACD_signal = MACD.ewm(span=macd_signal).mean()
        MACD_oscillator = MACD - MACD_signal                #MACD Oscillator

        ret = pd.concat([MACD, MACD_oscillator], axis=1)
        ret.columns = ['MACD', 'MACD_Oscillator']
        return ret

    def RSI(self, period:int=14):
        cl = self.data["close"]
        delta = cl.diff()

        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        _gain = up.ewm(com=(period - 1), min_periods=period).mean()
        _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

        RS = _gain / _loss
        RSI = pd.Series(100 - (100 / (1 + RS)), name="RSI")

        ret = pd.DataFrame(RSI)
        ret.columns = ['RSI']

        return ret
    
    def pivot(self):
        cl = self.data['close']
        hi = self.data['high']
        lo = self.data['low']

        pivot = (cl + hi + lo) / 3
        sup1 = pivot*2 - hi
        sup2 = pivot - hi + lo
        res1 = pivot*2 - lo
        res2 = pivot + hi - lo

        #d_to_sup1 = cl - sup1
        d_to_sup2 = cl - sup2
        #d_to_res1 = res1 - cl
        d_to_res2 = res2 - cl

        ret = pd.concat([d_to_sup2,d_to_res2], axis=1)
        ret.columns = ['d_to_sup2','d_to_res2']
        return ret
    
    def Stochastic(self, win:int=5):
        cl = self.data['close']
        lo = self.data['low']
        hi = self.data['high']

        fast_k = ((cl - lo.rolling(window=win).min()) / (hi.rolling(window=win).max() - lo.rolling(window=win).min())) * 100
        slow_k = fast_k.rolling(window=win).mean()

        ret = pd.concat([fast_k,slow_k], axis=1)
        ret.columns = ['fast_k','slow_k']
        return ret
    
    def Momentum(self):
        cl = self.data['close']

        tick3 = cl.shift(3)
        tick6 = cl.shift(6)
        tick12 = cl.shift(12)
        
        mtm = 0.5*(cl/tick3-1) + 0.3*(cl/tick6-1) + 0.2*(cl/tick12-1) 

        ret = pd.DataFrame(mtm)
        ret.columns = ['Momentum']

        return ret
    
    def Ichimoku(self):
        hi = self.data['high']
        cl = self.data['close']
        lo = self.data['low']
        
        nine_period_high =  hi.rolling(window=9).max()
        nine_period_low = lo.rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) /2
        
        period26_high = hi.rolling(window=26).max()
        period26_low = lo.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        period52_high = hi.rolling(window=52).max()
        period52_low = lo.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Metrics
        ten_to_sen = tenkan_sen - kijun_sen
        cloud_thickness = senkou_span_a - senkou_span_b
        d_to_cloud = cl - senkou_span_b + cloud_thickness

        ret = pd.concat([ten_to_sen,cloud_thickness,d_to_cloud], axis=1)
        ret.columns = ['ten_to_sen','cloud_thickness','d_to_cloud']
        return ret
    
    def Volume(self):
        return self.data['volume']
    
    def Close(self):
        return self.data['close']


class targetVariable:

    def __init__(self, data):
        self.data = data
    
    def lagReturn_pred(self):
        lag = self.data['close'].shift(1) 
        dr = (self.data['close']/lag)-1
        dr = dr*100     #Transform to percentage

        return_next_tick = pd.DataFrame(dr.shift(-1))
        return_next_tick.columns = ['return_next_tick']
        
        return return_next_tick