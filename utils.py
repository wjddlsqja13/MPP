import os, re, sys
import pandas as pd
import numpy as np
import csv

#------------------------------------1. Tickers------------------------------------------
Tickers = {
    #[Ticker, Saved_Name]
    0:['KRW-BTC', 'BTC'], #Bitcoin
    # 1:['KRW-ETH', 'ETH'], #Etherium
    # 2:['KRW-XRP', 'XRP'], #Ripple
    # 3:['KRW-SAND', 'SAN'], #The Sandbox
    # 4:['KRW-NEAR', 'NEA'], #Near Protocol
    # 5:['KRW-LINK', 'LIN'], #Chainlink
    # 6:['KRW-BTT', 'BTT'], #BitTorrent
    # 7:['KRW-POWR', 'POW'], #Power Ledger
    # 8:['KRW-BORA', 'BOR'], #BORA
    # 9:['KRW-STX', 'STX'], #Stacks
    # 10:['KRW-ATOM', 'COS'], #Cosmos
    # 11:['KRW-MATIC', 'POL'], #Polygon
    # 12:['KRW-DOGE', 'DOG'], #Dogecoin
    # 13:['KRW-VET', 'VET'], #VeChain
    # 14:['KRW-SXP', 'SXP'], #Swipe
    # 15:['KRW-SOL', 'SOL'], #Solana
    # 16:['KRW-DOT', 'DOT'], #Polkadot
    # 17:['KRW-HUNT', 'HUN'], #HUNT
    # 18:['KRW-MANA', 'MAN'], #Decentraland
    # 19:['KRW-PLA', 'PLA'], #PlayDapp
    # 20:['KRW-ADA', 'ADA'], #ADA
    # 21:['KRW-EOS', 'EOS'], #EOS
    # 22:['KRW-HIVE', 'HIV'], #Hive
    # 23:['KRW-NU', 'NUC'], #Nuchypher
    # 24:['KRW-XTZ', 'XTZ'], #Tezos
    # 25:['KRW-ELF', 'ELF'], #Aelf
    # 26:['KRW-AXS', 'AXS'], #Axie Infinity
    # 27:['KRW-LSK', 'LSK'], #Lisk
    # 28:['KRW-ETC', 'ETC'], #Ethereum Classic
    # 29:['KRW-AERGO', 'AER'], #Aergo
    # 30:['KRW-CRO', 'CRO'] #Crypto.com Chain
}

#------------------------------------2. createDirectory------------------------------------------
def createDirectory(directory): #Create directory if there is no such path
    try: 
        if not os.path.exists(directory): 
            os.makedirs(directory) 
    except OSError: 
        print("Error: Failed to create the directory.")

#------------------------------------3. return recent csv filename in the directory------------------------------------------
def firstCSVFile(directory):
    # getting the recent file only
    file_name_and_time_lst = []
    for f_name in os.listdir(f"{directory}"):
        if f_name.endswith('.csv'):
            written_time = os.path.getctime(f"{directory}{f_name}")
            file_name_and_time_lst.append((f_name, written_time))
    # sort by time created --> recent one first, 
    sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)
    # get the first one
    recent_file = sorted_file_lst[0]
    recent_file_name = recent_file[0]

    return recent_file_name

#------------------------------------4. Parsing CSV for quant------------------------------------------
def parse_csv_quant(version, ticker, option):
    
    dirNow = os.getcwd()
    dirParent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if option == 'train':
        root = dirParent + '/data/model_input/{}/train_data/{}/{}_train_{}.csv'.format(version, ticker, ticker, version)
    elif option == 'test':
        root = dirParent + '/data/model_input/{}/test_data/{}/{}_test_{}.csv'.format(version, ticker, ticker, version)
    elif option == 'mock':  
        root = dirParent + '/data/model_input/{}/mock_data/{}/{}_mock_{}.csv'.format(version, ticker, ticker, version)        
    input = []
    gt = []
    with open(root, 'r') as f:
        rdr = csv.reader(f)
        for idx, line in enumerate(rdr):
            if idx == 0:
                continue
            single_data = [
                float(line[2]),
                float(line[3]),
                float(line[4]),
                float(line[5]),
                float(line[6]),
                float(line[7]),
                float(line[8]),
                float(line[9]),
                float(line[10]),
                float(line[11]),
                float(line[12]),
                float(line[13]),
                float(line[14]),
                float(line[15]),
                float(line[16]),
            ]
            input.append(single_data)
            gt.append([float(line[17])])
    # print(input[0])
    # print(gt[0])
    return input, gt

#test_driver
# version = 'tv0.10'
# ticker = 'ADA'
# option = 'train'
# parse_csv_quant(version, ticker, option)