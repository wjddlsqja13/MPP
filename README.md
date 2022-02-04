# MPP_Phase1
  
## Develing Aim
1. Model Development (MD)
2. Analytics and Validation (AnV)
3. Feature Development (FD)

### 1) Model Development
  
__1.1 3799 Original (v0)__ <br>
  
Implementation of the model that I developed during the undergraduate capstone project (STAT3799): _Cryptocurrency Market Prediction Using Deep Neural Network - Mimicking Quant Traders_. The major purpose is to restructure the programs.<br>
  
The model is based on interpolation method which learns the relationship between return_next_tick and quant features. Details can be checked through the report (/others/). In order for practical usage, first, the model should able to predict more future (so it can reduce turnover). Second (Phase 2), the continuous self updating environment (CI, CD, CML) must be established, i.e. using MLOps frameworks. <br>
  
__1.2 3799 Nested Learning + Integration of Long Term Factors (v1)__ <br>
  
As mentioned above, there is a need for the model to predict more future returns with short term trading frequency data. I am proposing two ideas 1) Nested Learning 2) Transformer Model. The idea of nested learning is to repeatedly generate the return_next_tick using the interactive property between the price/return and quant features. It follows the logic of markov model to some extent. More details will be updated as v0 is done. <br>
  
As think of traders, they do not make investment decisions by just looking at technical indicators. This project is also aiming at integrating all quantitative and qualitative variables into the model. However, it is currently aiming at finishing quantitative variables as soon as possible. Thus, through the version 1, I am looking into incorporating macroeconomic factors into the model learning. Details will be suggested when v1 starts to develop. <br>
  
__1.3 3799 Transformer + Discovery of More Features (v2)__ <br>
  
Second idea is to implement the transformer model to make inferences of features sequences for consequent n ticks from the present, and use the interpolation model to regress target variable. --> Therefore, transformer + Interpolation. More details will be updated as v1 is done. <br>
  
More features must be discovered in order for v3. <br>
  
__1.4 Selection of set of features depends on macroeconomic situation__ <br>
  
One of my hypotheses is that the model will perform better if it ables to learn which combination of features will perform better in a specific economic situation. More details will be updated as v2 is done. <br>
  
__1.5 TF Transformation (Optional)__ <br>
    
___
  
### 2) Analytics and Validation
  
With Frank, the aim of analytics and validation (AnV) is to conduct financial data analysis (time-series / statistical ML) to validate the usage of features, and develop automation analytics tools to assess model performances and to perform essential data analyses. <br>
  
  
### 3) Directory Info + Current Progress
  
- /analytics/: AnV section, just started. Frank working on EDA.
- /FE/: Feature Engineering; it generates the preprocessed data (make features), train, test, mock data. Please refer to dataset_prep_procedures.md
- /data/: collecting program of raw_data is in local. contains preprocessed, train, test, mock data for testing usage.
- /model3799v0/: v0 is done. Model Analytics Tools is under development. Some updates still remaining - adjusting some parameters and network structure. Deadline: Feb 10
- utils.py: contains tickers that we are focusing on. also some frequently used functions.
- /others/: each version will have reports, and it is saved in the corresponding dir.
