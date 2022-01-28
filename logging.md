## TODOS (MD)

- __tv0.11__
- __validation tool__
- v0: Original 3799
- v1: Improve interpolation model network
- Add volatility to FE/make_features.py - quantFeatures
- Find more quantFeatures
- Model Testing Code
  
## TODOS (AnV)
  
1. EDA
2. Time Series Analysis
3. Regression Analysis -- Feature and Target
4. IRF
5. Tools Devel
6. etc.

## Logging
  
- 19 Jan 2022: (MD) FE/make_features.py - class quantFeatures done
- 19 Jan 2022: (MD) Raw Data Imported --> Saved to data/raw_data
- 21 Jan 2022: (MD) FE/make_preprocess.py - done (will add more options later) + make some part object and mv to utils
- 21 Jan 2022: (Data) preprocess data for [minute5, 7d, quant] and [minute5, 3d, quant] done --> check at data/preprocessed_data/quant
- 22 Jan 2022: (utils) done w/ createDir.py
- 22 Jan 2022: (MD) Started working on FE/make_traintest.py --> it divides preprocessed data into three (train, test, mock)
- 22 Jan 2022: (Data) preprocess data for [minute5, 24h, quant]
- 23 Jan 2022: (MD) utils.py updated 
- 24 Jan 2022: (MD) FE/make_traintest.py done
- 24 Jan 2022: (Data) Mock, train, test set downloaded for tv0.10 --> check at data/model_input/
- 25 Jan 2022: (MD) utils.py:parse_csv_quant done
- 25 Jan 2022: (MD) model3799_v0/3799_v0.py - torch.Dataset and 3799v0 structure done
- 25 Jan 2022: (MD) model3799_v0/3799_v0.py - attach_model, testModel, and trainModel done - not tested yet
- 26 Jan 2022: Pytorch is not work for python 3.10 --> downgraded to 3.9.5
- 27 Jan 2022: (MD) model3799_v0/3799_v0.py done
- 27 Jan 2022: (MD) tv0.10 done

 
## Work Progress
__Ongoing:__ <br>
- TODO: tv0.11, validation_tool.py, 
  
__Completed (For Now):__ <br>
- FE/make_features.py
- FE/make_preprocessed.py
- data/raw_Data
- data/preprocessed_data {[minute5, 7d, quant], [minute5, 3d, quant], [minute5, 24h, quant]}
- utils: tickers, createDirectory, firstCSVFile, parse_csv_quant
- FE/make_traintest.py
- data/model_input
- model3799_v0/3799_v0.py
