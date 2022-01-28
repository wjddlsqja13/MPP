1. Raw Data Collection (Source file at local) <br>
  
Will integrate to this later, once it needs MLOps <br>
  
2. make_preprocessed.py tFreq uFreq option <br>
  
option = [quant] : will add more options later <br>
  
3. make_traintest.py tFreq uFreq option trainset_ratio mock_data_len mock_data_unit version <br>
  
- option = [quant]
- trainset_ratio = float between 0~1 --> testset_ratio = 1 - trainset_ratio
- mock_data_len = integer
- mock_data_unit = [h, d] h: hour, d: day
- __So 24 h indicates 24 hours and 3 d indicates 3 days of mock data__
- version = tv (test ver) v (final ver) <br>
  
__Common:__ <br>
- tFreq = trading frequency: ['minute5', 'minute15', 'minute30', 'minute60']
- uFreq = updating frequency: ['24h', '3d', '7d', '15d', '30d']
  
  
__Features Version Log__ <br>
- quant v0: [BollingerBand, MACD, RSI, pivot, Stochastic, Momentum, Ichimoku, Volume]