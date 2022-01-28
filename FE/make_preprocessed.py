import os, re, sys

import numpy as np
import pandas as pd

from datetime import datetime

from make_features import quantFeatures

#Enable importing files in upper dir
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import utils

from make_features import quantFeatures as qf
from make_features import targetVariable as tv

date = datetime.today().strftime("%Y%m%d%H%M")

tFreq = sys.argv[1]     #trading frequency: ['minute5', 'minute15', 'minute30', 'minute60']
uFreq = sys.argv[2]     #updating frequency: ['24h', '3d', '7d', '15d', '30d']
option = sys.argv[3]    #features option: ['quant', 'all'] --> can be added more

#------------------------------------------------------------
# 몇분봉, update freq를 인자로 받는다.
# 인자에 해당하는 데이터만 트레인 데이터 생성
# 트레인 데이터 옵션도 인자로 받는다
# 옵션에 따른 피쳐 조합으로 구성된 데이터셋을 제공
# 모든 티커 트레인 데이터 생성
# (옵션) 추후에 어떤 데이터 셋을 썼는지도 인클루드 할 수 있게끔 한다
#------------------------------------------------------------

#------------------------------------------------------------
# Ticker Names
ticker_names = []
for ticker in utils.Tickers:
    ticker_names.append(utils.Tickers[ticker][1])
#------------------------------------------------------------

class makePreprocessedQuant:

    def __init__(self, data):
        self.data = data

    def generate(self):
        # Making Preprocessed Data (Quant Option)

        # Independent Variables
        ret = qf(self.data).BollingerBand()
        ret = pd.concat([ret, qf(self.data).MACD()], axis=1)
        ret = pd.concat([ret, qf(self.data).RSI()], axis=1)
        ret = pd.concat([ret, qf(self.data).pivot()], axis=1)
        ret = pd.concat([ret, qf(self.data).Stochastic()], axis=1)
        ret = pd.concat([ret, qf(self.data).Momentum()], axis=1)
        ret = pd.concat([ret, qf(self.data).Ichimoku()], axis=1)
        ret = pd.concat([ret, qf(self.data).Volume()], axis=1)

        # Target Variable
        ret = pd.concat([ret, tv(self.data).lagReturn_pred()], axis=1)

        # Close Price
        ret = pd.concat([ret, qf(self.data).Close()], axis=1)

        ret = ret.dropna()  # Drop rows containing NaN 
        ret = ret.reset_index(drop=True) # Reset Index

        return ret

if __name__ == '__main__':
    dirNow = os.getcwd()
    dirParent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    # Iterate All Tickers
    for ticker in ticker_names:
        # directory that contains csv files
        dDir = dirParent + '/data/raw_data/{}/{}/{}/'.format(ticker, tFreq, uFreq)

        #Get the first file
        recent_file_name = utils.firstCSVFile(dDir)

        # Raw Data
        data = pd.read_csv(dDir + recent_file_name, encoding='euc-kr')

        # Saving Dir
        sDir = dirParent + '/data/preprocessed_data/{}/{}/{}/{}/'.format(option, ticker, tFreq, uFreq)
        utils.createDirectory(sDir) #Create Dir if there does not exist
        sName ='{}_{}_{}_{}_pre_{}.csv'.format(ticker, tFreq, uFreq, option, date)

        if option == 'quant':
            output = makePreprocessedQuant(data).generate()
            output.to_csv(sDir + sName)
        
        # Later we add more options


