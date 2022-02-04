__tv0.10__ <br>
- minute5 data for all tickers --> tFreq: minute5, uFreq: 3d, option: quant, train ratio: 0.2, mock: 1d 
- features: quant v0
- nn structure = same as 3799
  
- input_len = 15
- dropout_rate = 0.1
- hidden_size = 40
- lr = 0.08
- criterion = nn.MSELoss()
- optimizer_option = 'Adadelta'
- batch_size = 100
- epochs = 30
- log_batch = 50
  
__tv0.11__