# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 19:08:23 2021

@author: nived
"""

""" 1.3 - Import LIBRARIES"""
                             #MAIN Libraries
# import 'pandas' 
import pandas as pd 
# import 'numpy' 
import numpy as np
# import subpackage of matplotlib
import matplotlib.pyplot as plt
# import 'seaborn'
import seaborn as sns
# import 'random' to generate random sample
import random
#Altair
import altair as alt
#Datapane
import datapane as dp
# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
# import subpackage of Matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# import 'Seaborn' 
import seaborn as sns


# scaling
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler


#EDA Reports
import pandas_profiling as pp
import sweetviz as sv
import dtale as de
import autoviz as au
from Dora import Dora



# import various functions from statsmodels
import statsmodels
import statsmodels.api as sm
import scipy.stats as stats
import statistics
from scipy import stats
from statsmodels.stats import weightstats as stests
from scipy.stats import shapiro
from statsmodels.stats import power
import statsmodels.formula.api as smf

# train-test split 
from sklearn.model_selection import train_test_split


#  feature selection
from sklearn.feature_selection import RFE
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


#AutoML

import pycaret

# machine learning 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm  import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from IPython.display import Image  
from sklearn.ensemble import RandomForestClassifier


# performance metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import fbeta_score


# to suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')
# to suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')
# display all columns of the dataframe
pd.options.display.max_columns = None
# display all rows of the dataframe
pd.options.display.max_rows = None
# to display the float values upto 6 decimal places     
pd.options.display.float_format = '{:.6f}'.format
# to suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')
# display all columns of the dataframe
pd.options.display.max_columns = None



'1 - Data Import'

ts = pd.read_excel('D:/shopster/Model Building/Modified Data/pos-ts.xlsx')



'2 - Datatype'
ts.info()


'3- Datatype conversion'
ts['Orderdate'] =  pd.to_datetime(ts['Orderdate'])


'4 - Adfuller test to determine stationarity'
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(ts['Ordercount'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


## Time series is stationary



import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})



# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(ts['Ordercount'].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(ts['Ordercount'].diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

## Time series is stationary

from statsmodels.tsa.arima_model import ARIMA

# 1,1,2 ARIMA Model
model = ARIMA(ts['Ordercount'], order=(1,1,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show() 

ypred = model_fit.plot_predict(dynamic=False)

ypred = model_fit.predict()


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(s,ypred)


from sklearn.metrics import mean_absolute_error
s = ts['Ordercount'].head(302)
mean_absolute_error(s,ypred)

axes[2, 0].plot(ts['Ordercount'].diff()); axes[2, 0].set_title('1st Order Differencing')
plot_acf(ts['Ordercount'].diff().dropna(), ax=axes[2, 1])
