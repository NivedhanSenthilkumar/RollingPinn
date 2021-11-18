# Commit  : Skeleton Commit
# Changes : NA

"1-IMPORT LIBRARIES"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import altair as alt
import datapane as dp
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import lazypredict

# scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler

#EDA Reports
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

# feature selection
from sklearn.feature_selection import RFE
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

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
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm  import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from IPython.display import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR

# Import required libraries for machine learning classifiers
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

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
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate

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
from lazypredict.Supervised import LazyClassifier, LazyRegressor


'DATA IMPORT'
Admin = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Admin/MODEL1/DATA/Admin.csv')
Google = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Admin/MODEL1/DATA/GOOGLE.csv')
Fb = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Admin/MODEL1/Data/FB.csv')
Fbads = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Admin/MODEL1/Data/FB-ADS.csv')
Ig = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Admin/MODEL1/Data/IG.csv')
Broadcast = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Admin/MODEL1/Data/Broadcast.csv')

#Setting Datatype of Orderdate to date
Admin['Orderdate'] = pd.to_datetime(Admin['Orderdate']).dt.date
Google['Orderdate'] = pd.to_datetime(Google['Orderdate']).dt.date
Fb['Orderdate'] = pd.to_datetime(Fb['Orderdate']).dt.date
Ig['Orderdate'] = pd.to_datetime(Ig['Orderdate']).dt.date
Fb['Orderdate'] = pd.to_datetime(Fb['Orderdate']).dt.date
Fbads['Orderdate'] = pd.to_datetime(Fbads['Orderdate']).dt.date
Broadcast['Orderdate'] = pd.to_datetime(Broadcast['Orderdate']).dt.date

#Sorting by Date
Admin =  Admin.sort_values(by = 'Orderdate')
Google =  Google.sort_values(by = 'Orderdate')
Fb =  Fb.sort_values(by = 'Orderdate')
Fbads =  Fbads.sort_values(by = 'Orderdate')
Ig =  Ig.sort_values(by = 'Orderdate')
Broadcast =  Broadcast.sort_values(by = 'Orderdate')

#DATAFRAMES JOINING
concat = Admin.merge(Google,on='Orderdate',how = 'left').merge(Fb,on='Orderdate',how = 'left').merge(Ig,on='Orderdate',how = 'left').merge(Fbads,on='Orderdate',how = 'left').merge(Broadcast,on='Orderdate',how = 'left')

#Sort orderdates for matching
concat['Orderdate'] = pd.to_datetime(concat['Orderdate'])

#Altering Broadcast columns
concat['Line-Deliveredcount'] =  concat['Line-Deliveredcount'].fillna(0)
concat['Line-Broadcastopened'] =  concat['Line-Broadcastopened'].fillna(0)
concat['Line-Broadcastclick'] =  concat['Line-Broadcastclick'].fillna(0)

