import os, re, sys

import numpy as np
import pandas as pd

from datetime import datetime

#Enable importing files in upper dir
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import utils

date = datetime.today().strftime("%Y%m%d%H%M")

#Command Example: python3 make_traintest.py minute5 3d quant 0.8 1 d tv0.10

tFreq = sys.argv[1]             #trading frequency: ['minute5', 'minute15', 'minute30', 'minute60']
uFreq = sys.argv[2]             #updating frequency: ['24h', '3d', '7d', '15d', '30d']
option = sys.argv[3]            #features option: ['quant', 'all'] --> can be added more
ratio = float(sys.argv[4])             #ratio of trainset: [0~1] --> recommended to be 0.7, 0.75, or 0.8
mock_test_len = int(sys.argv[5])     #timeframe of mock trading: Int
mock_test_unit = sys.argv[6]    #timeframe unit of mock trading ['h', 'd'] 
version = sys.argv[7]           #version i.e. 'v0.2' / 'tv0.1' --> tv stands for test version

#------------------------------------------------------------
# Ticker Names
ticker_names = []
for ticker in utils.Tickers:
    ticker_names.append(utils.Tickers[ticker][1])
#------------------------------------------------------------

class makeMock:

    def __init__(self, data):
        self.data = data
    
        self.multiplier_h = {'minute5' : 12, 'minute15' : 4, 'minute30' : 2, 'minute60' : 1}
        self.multiplier_d = {'minute5' : 288, 'minute15' : 96, 'minute30' : 48, 'minute60' : 24}
    
    def generate(self):
        len_p_data = len(self.data)
        if mock_test_unit == 'h':
            len_mock_data = mock_test_len * self.multiplier_h[tFreq]
            p_data = self.data.iloc[:-len_mock_data,:]
            mock_data = self.data.iloc[-len_mock_data:,:]
            mock_data = mock_data.reset_index(drop=True)
            return p_data, mock_data
        elif mock_test_unit == 'd':
            idx_mock_data = mock_test_len * self.multiplier_d[tFreq]
            p_data = self.data.iloc[:-idx_mock_data,:]
            mock_data = self.data.iloc[-idx_mock_data:,:]
            mock_data = mock_data.reset_index(drop=True)
            return p_data, mock_data
        else:
            raise NameError

class makeTrainTest:

    def __init__(self, data):
        self.data = data 
        self.idx_trainData = int(ratio * len(self.data))

    def generateTrain(self):
        out = self.data.iloc[-self.idx_trainData:,:]
        out = out.reset_index(drop=True)
        return out
    
    def generateTest(self):
        out = self.data.iloc[:-self.idx_trainData,:]
        out = out.reset_index(drop=True)
        return out
        
if __name__ == '__main__':
    dirNow = os.getcwd()
    dirParent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    for ticker in ticker_names:
        # directory that contains csv files
        dDir = dirParent + '/data/preprocessed_data/{}/{}/{}/{}/'.format(option, ticker, tFreq, uFreq)
        #Get the first file
        recent_file_name = utils.firstCSVFile(dDir)
        # Raw Data
        data = pd.read_csv(dDir + recent_file_name, encoding='euc-kr')

        sDir_mock = dirParent + '/data/model_input/{}/mock_data/{}/'.format(version, ticker)
        sDir_train = dirParent + '/data/model_input/{}/train_data/{}/'.format(version, ticker)
        sDir_test = dirParent + '/data/model_input/{}/test_data/{}/'.format(version, ticker)
        utils.createDirectory(sDir_train)
        utils.createDirectory(sDir_test)
        utils.createDirectory(sDir_mock)
        sName_mock = '{}_mock_{}.csv'.format(ticker, version)
        sName_train ='{}_train_{}.csv'.format(ticker, version)
        sName_test ='{}_test_{}.csv'.format(ticker, version)

        if mock_test_len == 0:
            train_data = makeTrainTest(data).generateTrain()
            test_data = makeTrainTest(data).generateTest()
            train_data.to_csv(sDir_train + sName_train)
            test_data.to_csv(sDir + sName_test)
        else:
            updated_data, mock_data = makeMock(data).generate()
            train_data = makeTrainTest(updated_data).generateTrain()
            test_data = makeTrainTest(updated_data).generateTest()
            mock_data.to_csv(sDir_mock + sName_mock)
            train_data.to_csv(sDir_train + sName_train)
            test_data.to_csv(sDir_test + sName_test)
    
    print("Ver Info ({})\ntFreq: {}\nuFreq: {}\noption: {}\ntrain set ratio: {}\nmock data length: {}{}".format(version, tFreq, uFreq, option, ratio, mock_test_len, mock_test_unit))
