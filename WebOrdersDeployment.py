
'LIBRARIES'
"1-IMPORT LIBRARIES"
# import 'pandas'
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
# import 'Seaborn'
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
#  feature selection
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
from lazypredict.Supervised import LazyClassifier, LazyRegressor
import pycaret

'DATA PREPARATION'
Web = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Web/Model1/Data/Web.csv')
Google = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Web/Model1/Data/GOOGLE.csv')
Fb = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Web/Model1/Data/FB.csv')
Fbads = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Web/Model1/Data/FB-ADS.csv')
Ig = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Web/Model1/Data/IG.csv')
Broadcast = pd.read_csv('D:/Rollingpinn/MODEL BUILDING/Web/Model1/Data/Broadcast.csv')

#Converting to date
Web['Orderdate'] = pd.to_datetime(Web['Orderdate']).dt.date
Google['Orderdate'] = pd.to_datetime(Google['Orderdate']).dt.date
Fb['Orderdate'] = pd.to_datetime(Fb['Orderdate']).dt.date
Ig['Orderdate'] = pd.to_datetime(Ig['Orderdate']).dt.date
Fb['Orderdate'] = pd.to_datetime(Fb['Orderdate']).dt.date
Fbads['Orderdate'] = pd.to_datetime(Fbads['Orderdate']).dt.date
Broadcast['Orderdate'] = pd.to_datetime(Broadcast['Orderdate']).dt.date

#Sorting by Date
Web =  Web.sort_values(by = 'Orderdate')
Google =  Google.sort_values(by = 'Orderdate')
Fb =  Fb.sort_values(by = 'Orderdate')
Fbads =  Fbads.sort_values(by = 'Orderdate')
Ig =  Ig.sort_values(by = 'Orderdate')
Broadcast =  Broadcast.sort_values(by = 'Orderdate')

#DATAFRAMES JOINING
concat = Web.merge(Google,on='Orderdate',how = 'left').merge(Fb,on='Orderdate',how = 'left').merge(Ig,on='Orderdate',how = 'left').merge(Fbads,on='Orderdate',how = 'left').merge(Broadcast,on='Orderdate',how = 'left')

#Sort orderdates for matching
concat['Orderdate'] = pd.to_datetime(concat['Orderdate'])

#Altering Broadcast columns - Filling 0 for days with no broadcast
concat['Line-Deliveredcount'] =  concat['Line-Deliveredcount'].fillna(0)
concat['Line-Broadcastopened'] =  concat['Line-Broadcastopened'].fillna(0)
concat['Line-Broadcastclick'] =  concat['Line-Broadcastclick'].fillna(0)

#Brodcast - for broadcast return the broadcastdata or else 0
def replace(x):
    if x == 0:
        return 0
    else:
        return x

#Brodcast - for broadcast days return 1 and nonbroadcast days 0
    if x == 0:
        return 0
    else:
        return 1

#Splitting Day into 4 Weeks
def Week(x):
    if x >= 1 and x <= 8:
        return 1
    elif x >= 9 and x <= 16:
        return 2
    elif x >= 17 and x <= 24:
        return 3
    else:
        return 4

#Weekday or Weekend
def Weekdayend(x):
    if x >= 0 and x <= 4:
        return 1
    else:
        return 2

##Splitting a Month into 4 bins
def Monthsplitup(x):
    if x >= 1 and x <= 10:
        return 1
    elif x >= 11 and x <= 20:
        return 2
    elif x >= 21 and x <= 25:
        return 3
    else:
        return 4

#Splitting Year into 4 Quarters
def Quarter(x):
    if x >= 1 and x <= 3:
        return 1
    elif x >= 4 and x <= 6:
        return 2
    elif x >= 7 and x <= 9:
        return 3
    else:
        return 4

#Splitting date into day,month,year
concat["day"] = concat['Orderdate'].map(lambda x: x.day)
concat["month"] = concat['Orderdate'].map(lambda x: x.month)
concat["year"] = concat['Orderdate'].map(lambda x: x.year)
concat['dayofweek'] = pd.to_datetime(concat['Orderdate']).dt.dayofweek


#NULL VALUES
Total = concat.isnull().sum().sort_values(ascending=False)
Percent = (concat.isnull().sum()*100/len(concat)).sort_values(ascending=False)
Missingdata = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])
print(Missingdata)

#HIGH NULL VALUE COLUMN DROP
concat = concat.drop(['Line-Videostart','Line-VideoComplete','Orderdate'],axis=1)

#IMPUTATION
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
concat = pd.DataFrame(imputer.fit_transform(concat),columns = concat.columns)

concat = concat[['GA-Organicsearches','GA-Users','Weborders']]
from pycaret.regression import *
s = setup(concat, target = 'Weborders', transform_target = True, log_experiment = True, experiment_name = 'WEB')
#Enable models
models(internal=True)[['Name', 'GPU Enabled']]

# compare baseline models
best = compare_models()
print(best)

et_model = create_model('et')
predict_model(et_model)
save_model(et_model, model_name = 'extra_tree_model')

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, df):
    predictions_data = predict_model(estimator=model, data=concat)
    return predictions_data['Label'][0]


model = load_model('extra_tree_model')

st.title('Wine Quality Classifier Web App')
st.write('This is a web app to classify the quality of your wine based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction of the classifier.')

Organicsearch = st.sidebar.slider(label='OS', min_value=4.0,
                                  max_value=16.0,
                                  value=10.0,
                                  step=0.1)
Users = st.sidebar.slider(label='Users', min_value=0.00,
                          max_value=2.00,
                          value=1.00,
                          step=0.01)

features = {'GA-Organicsearches': Organicsearch, 'GA-Users': Users}

features_df = pd.DataFrame([features])

st.table(features_df)

if st.button('Predict'):
    prediction = predict_quality(model, features_df)

    st.write(' Based on feature values, your wine quality is ' + str(prediction))



