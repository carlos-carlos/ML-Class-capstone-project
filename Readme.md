1. Introduction
--------------------

This is my first machine learning project ever. It is a capstone project meant 
to showcase basic machine learning concepts and skills learned during a 
machine learning introductory course focusing primarily on the sklearn library, 
pandas, and numpy. It also demonstrates  techniques learned to aquire, clean, 
structure, and transform data from scratch using a free API as the source. 
The project is also a means through which to learn the basics of using
machine learning in the realm of finance, and the specific techniques and 
best pratices in that context.

The project goal is to attempt to use sklearn models with cryptocoin OHLCV
data to see if future returns can be predicted which can outperform 
a benchmark, which  will be Bitcoin. 
This is a very basic experiment meant to showcase basic skills learned
as part of a course and is neither intended nor expected to generate a strategy
that can actually lead to financial gain.  Such a strategy would require more 
data and complexity than is demonstrated here.

The strategy used here assumes a momentum based approach to a trading strategy.
Please see the file "capstone proposal" for a more detailed description of the
assumptions made in this project.

Below is a description of the purpose of each file in the porject directory.


1.1 Project Structure

Step1_gatherdata.py
Step2_preparedata.py
Step4_PCA_Risk_Factors.ipnyb
Step3_Initial_insights.ipnyb
Step5_Statistical_Inference.ipnyb (optional)
Step6_Predictions.ipnyb
Step7_Alphalens_Evaluation.ipnyb
Step8_Backtest.ipnyb


2. Project Overview
------------------------


2.1 Step 1 - Gather the data

This file pulls complete hourly OHLCV data for the top 100 cryptocoins by market
capitalization listed on Coingecko. The data ranges from roughly 3 years back
to the present day. Due to API limitations the script grabs data in 90 day 
increments (the maximum range for which hourly data is provided) and then
puts it all toghether in a single .csv file per coin. The script filters out
known stablecoins as well as wrapper coins such as WBTC. The script can be
adjusted to filter coins based on current market capitalization and/or USD
trade volume. It can also be adjusted to grab more than just the top 100 coins.

Running it may take time depending on how burdened the API is. The script pauses
intermittently when requesting data so as not to abuse the API with repeated
requests.

2.2 Step 2 Prepare the data

This file imports all the .csv files as Pandas DataFrames which it then 
assembles into one big MultiIndex data frame with indexes structures 
by datetime and coin name. It then saves this mdf as a .csv file.

The purpose of this is to limit damage to the base data in case something goes
wrong. It also makes it easier to limit or add coins to the base data as desired
without having to re-request the data from coingecko each time. 

The script can also update the data and filter the data on demand. Coins can be
filtered by market capitalization or by USD trade volume. This differs from the
filter in step one because here you can specify a rolling mean as the condition
to filter coins. For example I can set it to keep coins that have consistently
demonstrated a market capitalization above XX USD over X time period. This can 
be preferable because market cap and trade volume can vary wildly in the
cryptocurrency world. For example, todays 7 out of the top 10 might not be 
tomorrows, if you only consider the present day values. By using a rolling mean
of market cap and volume instead, its a better way to separate "sustained"
investor interest from potentially anomalous interest in a given coin.

2.3 Step 3 Initial insights 

NOTE: Despite the name of the file and its position here, the following step 2.4
should be run before running this step, if it is desired to check exposure
to "known" risk factors.

This script takes the OHLCV data aquired in the previous steps and
uses it to generate the features that will actually be used as the data for the
models in the later steps.

Please see the file "detailed feature description" for more detailed info on the
features. This file also "cuts" the data for the later steps based on the 
global settings specified. The script can also drop coins that do not have
data for the full time range. This is done to avoid errors due to missing data 
later in the program. 

The script will generate several scatter plots, histograms, and heatmaps using
seaborn and matplotlib to render the data. This provides useful info about 
distribution and concentration of values within the features. 

2.4 Step 4 PCA Risk Factors

NOTE: Please Run this step before the aforementioned step. 

The purpose of this step is to compare the coins in our initital data pool to
"known" risk factors, that is to say, known drivers of returns and which factors
are the largest drivers of those returns. 

If this experiment was being run on the traditional stock market it would be 
preferable to use a method such as Fama-French to measure exposure to
known risk factors. However this is not possible for
cryptocoins without access to more data for all of the coins being used.

Instead, PCA() is used to find the risk factors using a purely data driven
approach. This has the advantage that we do not need any prior knowelege of risk
factors, since in theory, PCA can find which factors in the data are the largest
drivers of returns by way of clustering. This is far from perfect but it is 
none the less applicable as an attempt to find return drivers in volatile 
cryptocurrency data, which can be quite volatile. 

A more detailed description of this step and the rationale behind it can be 
found in the file "proposal".

2.5 Statistical Inference

This step isn't strictly essential in terms of the goals of this project. 
I undertook this step in order to explore the statsmodels library and experiment
with OLS to see how it works. The goal here was to estimate a linear regression
model using the same data later used to make predictions. 

I also wanted to become more familiar with statistics used in data science such
as Jarque-Bara, Durbin-Watson, p-value, skew, and kertosis, what they mean in 
this context. 
For example, a negative skew and high kurtosis implies that a model could be 
making a large number of errors. The plot is meant to visualize this by
plotting residual vs. normal distribution. A low Durbin-Watson statistic 
suggests the residuals may be positivly correlated. 
The residual autocorrection plot visualizes correlations 
by returns lags periods. 

2.6 Predictions

This step predicts "future" returns.
The target "y" labels are shifted returns used as "future" returns to measure
the performance of the model, and give it something to 
measure performance against.
Linear Regression, Ridge Regression, and Lasso Regression are used to make 
predictions. Cross Validation is done with TimeSeriesCV from SciKitLearn which
is invoked in the class MultipleTimeSeriesCV. The time frame to test/train is 
set at the top of the script. Metrics used to evaluate the model performance
in this step include, the Information Coefficient and Root Mean Squared Error. 

The class MultipleTimeSeriesCV was written by 
Marcos Lopez de Prado and adapted by Stefan Jansen. I have also adapted it 
slightly for use with this specific dataset. TimeSeriesCV is invoked in the 
class MultipleTimeSeriesCV in this way because, with financial data, labels are 
usually derived from overlapping data points. Since returns are calculated 
from prices over many time periods, this can lead to leakage of information 
from test to training data. The class addresses this by invoking TimeSeriesCV 
in a way that ensures the data is used point-in-time, or rather, 
"known at the time" that it is being used. Techniques like embargoing,
purging, and  combinatorial-cross-validation are leveraged to achieve this. 

2.7 Alphalens Analysis

This step further evaluates the performance of the models, in addition to the 
evaluation metrics used in Step 6. Here, the Alphalens library is used to 
evaluate the model predictions with Finance specific metrics
such as Alpha, Beta, and the mean returns performance by quantile for each lag.
Alphalens also shows useful Information Coefficient statistics such as p-value
and t-statistic to further evaluate the performance of each model.
A positive Alpha value above zero and maxing out at 4 
means that the model(s) was able to find some predictive signal within the data
from the features. Alpha signals decay over time but can also indicate returns
based on different holding periods. For example, the model might have found a 
Alpha signal that is negative over 30 days but then trends positive after 90.
Beta is a measure of volatillity, or in finance terminology, "systemic risk".
The higher the Beta, the more volatillity and the higher the risk. 


2.8 Vectorized Backtest

This script transforms the predicted returns from our model(s) into trading
signals. To do this, it applies a very simple trading strategy for the sake
of demonstration. The script "goes long" (buy and hold) and "goes short" 
(sell and recover at a lower price) on the X highest and lowest coins on a 
daily basis, over a given timeframe, based on the predicted returns 
of the model(s).The time frame and number of long/short positions 
can be adjusted in the script. This is the final step in this project and the 
third and final evalutation of the model(s) predictions from step 6. 
Here the predictions of the model(s) are compared to a benchmark, in this case, 
the returns of Bitcoin over the same time period. 


3. Results and Findings
--------------------------------------------------

3.1 Results

Experiments were run using an initial pool of 52 cryptocoins as well as a 
smaller pool of 6 cryptocoins. The latter pool was made up of the coins that
had the highest trade volume and market capitalization while also having data 
for the full time frame in question.

Experiments were run with Linear Regression, Ridge Regression, 
and Lasso Regression being trained on several train/test splits 
(30 train, 7 test) to  predict daily returns (1 day returns) for 3, 2, 
and 1 year time frames. Each time frame ends on the present day (today).
Mean Squared Error averaged between 3 - 8% while Information Coefficient 
averaged between 4 - 6%. The models performed better when trianed over 3 and 2
years as compared to 1 year. This was expected.

Alpha signals found by the models were weak, ranging from negative values to 
the 0.11xx range. The Lasso Regression predictions were slightly better
than Ridge and Linear Regression. Alpha signals appeared to be strongest for 30
and 90 day lag periods. This suggests the signals are weak overall but stronger
for predicting returns over those time periods. This was also expected, since
this is a very basic strategy and no weights are being applied to any coins in 
the pool. 

Beta signals were mixed, ranging from negative values to the 0.3xx range. They
trended more positive over periods of 30 days and longer. I expected high
Betas given that we are working with cryptocoins. However, the Betas do seem to
reflect the stronger returns over longer time periods, and also higher losses
during downturns. 

The backtest for all time frames seemed to backup what was suggested by the 
Alpha and Beta metrics mentioned before, with the "strategy" breifly performing 
better than the benchmark for the first 30 - 50 days before performing very
poorly for the rest of the time period. I had expected stronger performance,
in the nearterm, but the backtest results do seem consistent with the alpha
signals found in step 7. 

Taken collectivly, the results seem to indicate better performance over short
time periods but very poor performance longterm, compared to the benchmark.

3.2 Limitations

Ideally there would have been at least 5 years of data for each coin to 
engineer features and train/test models on. This was limited by both API 
limitations as well as the fact that many of the cryptocoins that are in the 
data pool did not exist going back that long. Indeed many of them are less than
2 years old. This leaves less data to train the model on, but also limits the
coins that can be used in a sample portfolio to test against Bitcoin. 

Furthermore, due to limited data, heavy use of imputation was made to fill in
NaN values created from shifting the returns to create the "y" labels and also
the the shifted returns for various lag periods. Imputation was used in order
to preserve a full 3,2, and 1 years worth of data during the 
feature engineering phase.

Not imputing these NaNs and dropping them instead, would have 
resulted in much less data to train and test the models on. However, 
imputation likely had some kind of biasing effect on the model results.


3.3 Future work

The results of this model could be improved in many ways, using the same base
data. For example, a strategy for wieghting the coins in the coin pool to create
a more logical portfolio could be applied. There are many approaches to take to
weighting the portfolio, such running further experiments with PCA() to produce
Eigenportfolios, or using the "Kelly Criterion".

Other features could be added or removed to the data based on the 
results of the initital insights of step 3. Furthermore, the Alpha signals could
be made stronger by adding sentiment indicators as features, in combination with
the momentum indicators already used. This would require code for 
Natural Language Processing (NLP), in order to be implemented properly. 

Using the existing features, the Alpha signals could be further pre-processed
to remove noise, using a technique such as the Kalman Filter. 

Finally, a classicfication approach could be taken 
using logistic regression, as a linear classification approach 
to model qualitative output variables. This would require obtaining binary
output variables, so a condition would have to be created that results in 
either a 0 or 1 outcome, a moving average for example, or for the returns,
1 when the return is positive, and 0 if it is negative. 

Ultimately more data would also be needed, or perhaps adjusting the models to 
make predictions for shorter timeframes. Since cryptocoin market change often
and significantly, perhaps it makes more since to train the models over a 
shorter period of prior trends, to make predictions for the next week or the
next month, before rebalancing the portfolio and training the models with newer
data. This also takes into account that unlike with the stock market, market
trends from 3 or 4 years ago in the crypto space might have very little to do
with market dynamics within the past year or the present day. 


Concluding Remarks
------------------------

This was was my first machine learning project that I have ever attempted and I 
undertook it hoping to learn while exercising the skills that I learned during
my course. I've come away from it with more questions than answers but with a
better idea of what to try and learn next. I hope it will serve a baseline for
further development and many more projects. 








 






