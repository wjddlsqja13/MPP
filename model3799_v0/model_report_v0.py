import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error 

#Enable importing files in upper dir
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils

# input parameters
version = 'tv0.10'
lossFn = 'MSE'
OptMock = False
target_ret = 0.01

ticker_names = []
for ticker in utils.Tickers:
    ticker_names.append(utils.Tickers[ticker][1])

class testAccuracy:

    def __init__(self, data, pred):
        self.data = data
        self.pred = pred

    def MSE(self):
        ret = mean_squared_error(self.data['return_next_tick'], self.pred['Prediction'])
        return round(ret, 5)
    
    def mvPredAcc(self):
        self.data['move'] = 0
        self.pred['move'] = 0
        for i in range(len(self.data)):
            self.pred['move'][i] = 0 if self.pred['Prediction'][i] < 0 else 1
            self.data['move'][i] = 0 if self.data['return_next_tick'][i] < 0 else 1
        con = pd.concat( [self.pred['move'], self.data['move']], axis=1)
        con.columns = ['pred move', 'true move']

        corrected = 0
        for i in range(len(con)):
            if con['pred move'][i] == con['true move'][i]:
                corrected += 1
        accuracy = round((corrected/len(con)),5)
        return accuracy
    
    def testMockTrading(self):
        self.pred['move'] = 0
        for i in range(len(self.pred)):
            self.pred['move'][i] = 0 if self.pred['Prediction'][i] < target_ret else 1
        con = pd.concat([self.pred['move'], self.data['return_next_tick']], axis=1)
        con.columns = ['pred move', 'return']
        
        ret = 100
        index_ls = []
        for index, row in con.iterrows():
            if con['pred move'][index] == 1:
                index_ls.append(index)
                ret += ret * (con['return'][index]/100)
        return round(ret,5)

if __name__ == '__main__':
    dirNow = os.getcwd()
    dirParent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    for ticker in ticker_names:
        true_data = pd.read_csv(dirParent+'/data/model_input/{}/test_data/{}/{}_test_{}.csv'.format(version, ticker, ticker, version))
        pred_data = pd.read_csv(dirNow+'/{}/{}/prediction/model3799_{}_{}_y.csv'.format(version, ticker, version, ticker))

        mse = testAccuracy(true_data, pred_data).MSE()
        mvAcc = testAccuracy(true_data, pred_data).mvPredAcc()
        mock_ret = testAccuracy(true_data, pred_data).testMockTrading()

        out_file = open('./{}/{}/model_report_{}_{}.txt'.format(version, ticker, version, ticker), 'w')
        out_file.write('MSE: {} \nCorrect Prediction in Moves: {} \nTest Mock Return: {}'.format(mse, mvAcc, mock_ret))
