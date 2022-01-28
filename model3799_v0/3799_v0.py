import os, sys, re
import csv
import time

import numpy as np
import pandas as pd

#Enable importing files in upper dir
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils 

import torch
from torch.nn.modules.activation import ReLU
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

#-----------------------------Here Should Be Modified-------------------------------------
# Parameters
# a. Tickers
ticker_names = []
for ticker in utils.Tickers:
    ticker_names.append(utils.Tickers[ticker][1])
# b. Version
version = 'tv0.10'
# c. nn parameters
input_len = 15
dropout_rate = 0.1
hidden_size = 40
# d. training parameters
lr = 0.08
criterion = nn.MSELoss()
optimizer_option = 'Adadelta'
batch_size = 20
epochs = 30
log_batch = 50
#------------------------------------------------------------------------------------------
dirNow = os.getcwd()
dirParent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

#-----------------------------------------DataSet-------------------------------------------------
class dataset(Dataset):
    def __init__(self, version, ticker, option):
        # super(dataset).__init__()
        self.version = version
        self.ticker = ticker
        self.option = option

        self.samples, self.gt = utils.parse_csv_quant(self.version, self.ticker, self.option)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        input = self.samples[index]
        input_tensor = torch.tensor(input, dtype=torch.float32)
        true_val = self.gt[index]
        gt_tensor = torch.tensor(true_val, dtype=torch.float32)
        return input_tensor, gt_tensor
#-------------------------------------------------------------------------------------------------

#-----------------------------------3799v0 structure----------------------------------------------
#mainModel
class model3799v0(nn.Module):

    def __init__(self):
        super(model3799v0, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(input_len)
        self.dropout0 = nn.Dropout(0.2)

        self.dense1 = nn.Linear(input_len, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+input_len, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size+hidden_size, 1)

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)

        x = torch.cat([x3, x4], 1)

        x = self.dense5(x)

        return x
#-------------------------------------------------------------------------------------------------

#---------------------------------------Attach Model----------------------------------------------
def attach_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model3799v0()

    if optimizer_option == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    return model, device, optimizer
#-------------------------------------------------------------------------------------------------

#----------------------------------------Train Model----------------------------------------------
def trainModel(model, device, trainloader, optimizer, criterion, ticker):
    model.to(device)
    model.train()
    running_loss = 0.0
    total = 0
    for i, data in enumerate(trainloader):
        input, gt = data
        inputs, true_val = input.to(device), gt.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, true_val)
        loss.backward()
        optimizer.step()
        total += true_val.size(0)
        running_loss += loss.item()
        # if i % log_batch == log_batch - 1:
        #     avg_loss = running_loss / log_batch
        #     print('Epoch: %d/%d Batch: %5d running loss: %.3f' % (epoch + 1, epochs, i + 1, avg_loss))
        #     running_loss = 0.0
        sDir = dirNow + "/{}/{}/model/".format(version, ticker)
        utils.createDirectory(sDir)
        torch.save(model.state_dict(), sDir+"model3799_{}_{}.pth".format(version, ticker))
#-------------------------------------------------------------------------------------------------

#-----------------------------------------Test Model----------------------------------------------
def testModel(version, ticker, option):
    test_data = dataset(version, ticker, option)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    PATH = dirNow + "/{}/{}/model/model3799_{}_{}.pth".format(version, ticker, version, ticker)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model3799v0()
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    model.eval()

    pred_list = []

    with torch.no_grad():
        for data, gt in test_loader:
            data, gt = data.to(device), gt.to(device)
            prediction = float(model(data.detach()))
            pred_list.append(prediction)

    ret = pd.DataFrame(pred_list)
    ret.columns = ['Prediction']

    sDir = dirNow + "/{}/{}/prediction/".format(version, ticker)
    utils.createDirectory(sDir)
    ret.to_csv(sDir+"model3799_{}_{}_y.csv".format(version, ticker))
#-------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    for ticker in ticker_names:
        train_data = dataset(version, ticker, 'train')
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        model, device, optimizer = attach_model()

        for epoch in range(epochs):
            trainModel(model, device, trainloader, optimizer, criterion, ticker)

        testModel(version, ticker, 'test')