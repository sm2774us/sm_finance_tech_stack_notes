<h1 align="center">Description</h1>
This hobby project compares different methods for time series prediction on stock market data. The methods used are 

1. Autoregressive integrated moving average (ARIMA) model built from scratch
2. Implemented SARIMAX model
3. LSTM model
4. Attention-based Transformer model.

The aim of the project is to demonstrate how build and use these different models on time series and show their efficiency on it.

Let's first begin with some analysis of the data!:)

<hr>

## 1. Data analysis
Some questions of interest in time series analysis are

1. What are the overall behavioural trend?
2. Is there any recurrent seasonality involved?
3. How much noise is there?

In [Analysis.py](https://github.com/olof98johansson/StockPrediction/blob/main/Analysis.py) we use the [statsmodels API](https://www.statsmodels.org/stable/index.html) to analyse these questions. The results of this for the Amazon stock data (AMZN) are the following

![Analysis](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/analysis_ex.png?raw=true)

As seen, there are some clear seasonal behaviour in the data. This is unwanted, and therefore, the data is made stationary. This is done by logarithmic differencing as
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=y'_t&space;=&space;\log{\left(\frac{y_t}{y_{t-1}}&space;\right&space;)}-y_{t-i}" target="_blank"><img src="https://latex.codecogs.com/png.latex?y'_t&space;=&space;\log{\left(\frac{y_t}{y_{t-1}}&space;\right&space;)}-y_{t-i}" title="y'_t = \log{\left(\frac{y_t}{y_{t-1}} \right )}-y_{t-i}" /></a> </p>
where <i>i</i> is the differencing range, here set to 12. To check that the data is stationary after the transformation, the Augmented Dickey-Fuller test is performed. The result of the transformation is seen in the following

![Stationary](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/stationary_data_demo.png?raw=true)

where it is seen that the data has both stationary mean and stationary variance.

The inverse transformation to obtain the original values from the stationary data is
<p align="center"> <a href="https://www.codecogs.com/eqnedit.php?latex=y_t&space;=&space;y_{t-1}e^{y'_{t}&plus;y_{t-i}}" target="_blank"><img src="https://latex.codecogs.com/png.latex?y_t&space;=&space;y_{t-1}e^{y'_{t}&plus;y_{t-i}}" title="y_t = y_{t-1}e^{y'_{t}+y_{t-i}}" /></a></p>

## 2. ARIMA model
The first model to be used for the predictions is the Autoregressive integrated moving average (ARIMA) model in [ARIMA.py](https://github.com/olof98johansson/StockPrediction/blob/main/ARIMA.py). The ARIMA model consists of first performing regression on the variable of interest prior (lagged) values where the regression error are linear combination of error terms of recurrent occurrence in the past. The formula for the ARIMA model is
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex={\displaystyle&space;Y_{t}-r&space;_{1}Y_{t-1}-\dots&space;-r&space;_{p'}Y_{t-p'}=e&space;_{t}&plus;M&space;_{1}e&space;_{t-1}&plus;\cdots&space;&plus;M&space;_{q}e&space;_{t-q}}" target="_blank"><img src="https://latex.codecogs.com/png.latex?{\displaystyle&space;Y_{t}-r&space;_{1}Y_{t-1}-\dots&space;-r&space;_{p'}Y_{t-p'}=e&space;_{t}&plus;M&space;_{1}e&space;_{t-1}&plus;\cdots&space;&plus;M&space;_{q}e&space;_{t-q}}" title="{\displaystyle Y_{t}-r _{1}Y_{t-1}-\dots -r _{p'}Y_{t-p'}=e _{t}+M _{1}e _{t-1}+\cdots +M _{q}e _{t-q}}" /></a></p>

where <i>Y</i> are the data values, <i>r</i> are the autoregressive parameters, <i>M</i> are the parameter of the moving average and <i>e</i> are the error terms. Furthermore, <i>p'</i> is the number of time lags of the autoregressive model and <i>q</i> is the order of the moving average model.

<br>
The test results of the ARIMA model is 

![arimares](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/arima_predictions.png?raw=true)


and the density plot of the residuals are

![density](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/arima_residuals.png?raw=true)


## 3. SARIMAX
The second model is the Seasonal Autoregressive Integrated Moving Average Exogenous (SARIMAX) model which is implemented from the [statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) library. From the [pmdarima](https://pypi.org/project/pmdarima/) library, the most optimal values of the orders <i>p</i>, <i>q</i> and <i>d</i> for the SARIMAX model can be conducted approximately. The test result of the SARIMAX model with orders <i>(p, q, d) = (1, 1, 1)</i> on the Amazon stock is

![sarimax](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/sarimax_predictions.png?raw=true)


## 4. LSTM
Now to the fun bits of machine learning! :D 

### 4.1 Preprocessing
For the machine learning methods, some additional information about the data is added in form of technical indicators. The first indicator that is added is just the daily return which is the current closing price subtracted by the previous closing price. The second indicator is rate of change (ROC) which is the percentage of the daily change. The third indicator is the Williams %R indicator which measures overbought and oversold levels of the stock. Williams %R is calculated as
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=WR&space;=&space;\frac{HH_{t-n}&space;-&space;C_t}{HH_{t-n}-LL_{t-n}}" target="_blank"><img src="https://latex.codecogs.com/png.latex?WR&space;=&space;\frac{HH_{t-n}&space;-&space;C_t}{HH_{t-n}-LL_{t-n}}" title="WR = \frac{HH_{t-n} - C_t}{HH_{t-n}-LL_{t-n}}" /></a></p>

where <i>HH</i> is the highest high within the <i>n</i> previous period, <i>C</i> the closing price and <i>LL</i> the lowest low within the <i>n</i> previous period.

Next, the money flow index (MFI) indicator was computed which measures the buying and selling pressure based on the volume of the stock. It is computed as

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;MFI&space;&=&space;100&space;-&space;\frac{100}{1&plus;MFR},\\&space;MFR&space;&=&space;\frac{\sum_{i=t-n}^{t}(RMF_i)_&plus;}{\sum_{i=t-n}^t(RMF_{i})_-},\\&space;RMF&space;&=&space;TP&space;\cdot&space;V,\\&space;TP&space;&=&space;\frac{H&space;&plus;&space;L&space;&plus;&space;C}{3}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\begin{align*}&space;MFI&space;&=&space;100&space;-&space;\frac{100}{1&plus;MFR},\\&space;MFR&space;&=&space;\frac{\sum_{i=t-n}^{t}(RMF_i)_&plus;}{\sum_{i=t-n}^t(RMF_{i})_-},\\&space;RMF&space;&=&space;TP&space;\cdot&space;V,\\&space;TP&space;&=&space;\frac{H&space;&plus;&space;L&space;&plus;&space;C}{3}&space;\end{align*}" title="\begin{align*} MFI &= 100 - \frac{100}{1+MFR},\\ MFR &= \frac{\sum_{i=t-n}^{t}(RMF_i)_+}{\sum_{i=t-n}^t(RMF_{i})_-},\\ RMF &= TP \cdot V,\\ TP &= \frac{H + L + C}{3} \end{align*}" /></a></p>

where <i>MFI</i> is the money flow index, <i>MFR</i> is the raw money flow, <i>TP</i> is the typical price, <i>V</i> is the volume and <i>H, L, C</i> is the high, low and closing price.

Thereafter was the Ulcer index also used which is a volatility indicator that measures the downside risk in terms of depth and duration of price declines. The Ulcer index is computed as 
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;UI_t&space;&=&space;\sqrt{SA_t},\\&space;SA_t&space;&=&space;\frac{1}{n}\sum_{i=t-n}^t&space;(PD_i)^2,\\&space;PD_t&space;&=&space;100&space;\hspace{1mm}\frac{C_t-HH_{t-n}}{HH_{t-n}}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\begin{align*}&space;UI_t&space;&=&space;\sqrt{SA_t},\\&space;SA_t&space;&=&space;\frac{1}{n}\sum_{i=t-n}^t&space;(PD_i)^2,\\&space;PD_t&space;&=&space;100&space;\hspace{1mm}\frac{C_t-HH_{t-n}}{HH_{t-n}}&space;\end{align*}" title="\begin{align*} UI_t &= \sqrt{SA_t},\\ SA_t &= \frac{1}{n}\sum_{i=t-n}^t (PD_i)^2,\\ PD_t &= 100 \hspace{1mm}\frac{C_t-HH_{t-n}}{HH_{t-n}} \end{align*}" /></a></p>

where <i>UI</i> is the Ulcer index, <i>SA</i> is the squared average and <i>PD</i> is the percentage drawdown.

The consecutive indicator added was the average true range index which measures the market volatility. It measures this by decomposing the asset's price for a period <i>n</i> as

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;ATR&space;&=&space;\frac{1}{n}\sum_{i=t-n}^tTR_i,\\&space;TR&space;&=&space;\max{\left(H-L,&space;|H-C|,&space;|L-C|&space;\right&space;)}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\begin{align*}&space;ATR&space;&=&space;\frac{1}{n}\sum_{i=t-n}^tTR_i,\\&space;TR&space;&=&space;\max{\left(H-L,&space;|H-C|,&space;|L-C|&space;\right&space;)}&space;\end{align*}" title="\begin{align*} ATR &= \frac{1}{n}\sum_{i=t-n}^tTR_i,\\ TR &= \max{\left(H-L, |H-C|, |L-C| \right )} \end{align*}" /></a></p>

where <i>H</i> is the highest asset price of time t, <i>L</i> is the lowest asset price of time t, <i>C</i> is the closing price at time t and <i>TR</i> is the true range.

The penultimate and last indicators are just the simple moving average (SMA) and the exponential moving average (EMA).


### 4.2 Results
The result for using the LSTM model with the above indicators as additional input features is

![lstm](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/lstm_predictions.png?raw=True)


![lstmz](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/lstm_predictions_zoom.png?raw=True)


### 5 Transformer
The [original paper](https://arxiv.org/abs/1706.03762) of the attention-based transformer model was published in 2017 and since then its popularity has explode. The model in [transformer.py](https://github.com/olof98johansson/StockPrediction/blob/main/transformer.py) is the implementation from the original paper, but here, only the encoder of the transformer model is used. This attention-based model uses positional encoding whose values represent the importance of the different input features in each time step, i.e which features that the model should pay more attention to. This along with multihead scaled dot product layers with residual connections and feed-forward linear bottleneck creates the transformer encoder. The results, with the additional indicators as for the lstm, for the transformer model is

![transformer](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/transformer_predictions.png?raw=True)

![transformerz](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/transformer_predictions_zoom.png?raw=True)

