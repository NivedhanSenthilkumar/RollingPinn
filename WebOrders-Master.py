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

                       'CUSTOM FUNCTIONS'
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


                           'Exploratory Data Analysis'
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

#Extended Numerical Summary
def numericalattributes(X):
    Output = pd.DataFrame()
    Output['Variables'] = X.columns
    Output['Skewness'] = X.skew().values
    Output ['Kurtosis'] = X.kurt().values
    Output ['Standarddeviation'] = X.std().values
    Output ['Variance'] = X.var().values
    Output ['Mean'] = X.mean().values
    Output ['Median'] = X.median().values
    Output ['Minimum'] = X.min().values
    Output ['Maximum'] = X.max().values
    Output ['Sum'] = X.sum().values
    Output ['Count'] = X.count().values
    return Output

numericalattributes(concat).to_excel('D:/Rollingpinn/MODEL BUILDING/Web/Model1/EDA/Data Summary/summary.xlsx')

#VIF
def variableinflation(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)

variableinflation(concat).to_excel('D:/Rollingpinn/MODEL BUILDING/Web/Model1/EDA/VIF/VIF-web.xlsx')

#CORRELATION
correlation = concat.corr()
correlation.to_excel('D:/Rollingpinn/MODEL BUILDING/Web/Model1/EDA/Correlation/corr-web.xlsx')

#Visualization



#SCALING
sc = StandardScaler()
datasc = sc.fit_transform(concat)
datasc = pd.DataFrame(concat,columns = concat.columns)

#TRANSFORMATION
logt = np.log(datasc+0.0000000000000000000000001)
numericalattributes(logt).to_excel('D:/Rollingpinn/MODEL BUILDING/Web/Model1/EDA/Data Summary/summary-logt.xlsx')

#OUTLIER ANALYSIS


                         'FEATURE STORE'
#TIME FUNCTIONS
concat['Week'] = concat['day'].apply(Week)
concat['Weekday/Weekend'] = concat['dayofweek'].apply(Weekdayend)
concat['Monthsplitup'] = concat['day'].apply(Monthsplitup)
concat['Quarter'] = concat['month'].apply(Quarter)

#BROADCAST
concat['broadcast-yes/no'] = concat['Line-Deliveredcount'].apply(replacedate)
concat['Broadcastopenrate'] =  concat['Broadcastopenrate'].fillna(0)
concat['Broadcastclickrate'] =  concat['Broadcastclickrate'].fillna(0)
concat['Broadcastopenrate'] = (concat['Line-Broadcastopened']/concat['Line-Deliveredcount'])*100
concat['Broadcastclickrate'] = (concat['Line-Broadcastclick']/concat['Line-Deliveredcount'])*100
concat['Broadcastopenrate'] = concat['Broadcastopenrate'].apply(replace)
concat['Broadcastclickrate'] = concat['Broadcastclickrate'].apply(replace)

#MARKETING ATTRIBUTES
concat['AverageOrderValue'] = concat['Webprice'] / concat['Weborders']
concat['NewUsersRate'] = concat['GA-Newusers']/concat['GA-Users']
concat['TotalLikes'] = concat['IG - Likecount'] + concat['FB-Totallikes']
concat['Totalreach'] = concat['IG-Profilereach'] + concat['FB-Totalreach']
concat['Totalclicks'] = concat['IG - Clickswebsite'] + concat['FA-Clicks'] + concat['FB-Contentclicks'] + concat['GA-Productlistclicks']
concat['TotalImpressions'] = concat['IG-Profileimpressions'] + concat['FB-Totalimpressions'] + concat['FA-Impressions']
concat['AdusageRate'] = concat['FA-Amountspent']/concat['GA-Newusers']


                          'TRAIN TEST SPLIT'
X = datasc.drop(['Weborders','Webprice'],axis = 1)
Y = datasc['Weborders']
xtrain, xtest, ytrain, ytest = train_test_split( X, Y, test_size=0.33, random_state=42)


                         'MODEL BUILDING'
'1-AUTOML FUNCTION'
scoring = {
    'R2-Square': make_scorer(r2_score),
    'MSE': make_scorer(mean_squared_error),
    'MAE': make_scorer(mean_absolute_error)}

# Instantiate the machine learning classifiers
lin_model = LinearRegression()
svr_model = SVR()
rf_model = RandomForestRegressor()
gb_model = GradientBoostingRegressor()
dt_model = DecisionTreeRegressor()
lgbm_model = LGBMRegressor()
ridge_model = Ridge()
lasso_model = Lasso()
knn_model = KNeighborsRegressor()

# Define the models evaluation function
def Regression(X, y, folds):
    # Perform cross-validation to each machine learning classifier
    lin = cross_validate(lin_model, X, y, cv=folds, scoring=scoring)
    svr = cross_validate(svr_model, X, y, cv=folds, scoring=scoring)
    rf = cross_validate(rf_model, X, y, cv=folds, scoring=scoring)
    gb = cross_validate(gb_model, X, y, cv=folds, scoring=scoring)
    dt = cross_validate(dt_model, X, y, cv=folds, scoring=scoring)
    lgbm = cross_validate(lgbm_model, X, y, cv=folds, scoring=scoring)
    ridge = cross_validate(ridge_model, X, y, cv=folds, scoring=scoring)
    lasso = cross_validate(lasso_model, X, y, cv=folds, scoring=scoring)
    knn = cross_validate(knn_model, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'SVR Regression': [
        svr['test_R2-Square'].mean(),
        svr['test_MSE'].mean(),
        svr['test_MAE'].mean()],
        'LGBM Regression': [
            lgbm['test_R2-Square'].mean(),
            lgbm['test_MSE'].mean(),
            lgbm['test_MAE'].mean()],
        'Linear Regression': [
            lin['test_R2-Square'].mean(),
            lin['test_MSE'].mean(),
            lin['test_MAE'].mean()],
        'Ridge Regression': [
            ridge['test_R2-Square'].mean(),
            ridge['test_MSE'].mean(),
            ridge['test_MAE'].mean()],

        'Lasso Regression': [
            lasso['test_R2-Square'].mean(),
            lasso['test_MSE'].mean(),
            lasso['test_MAE'].mean()],
        'KNN Regression': [
            knn['test_R2-Square'].mean(),
            knn['test_MSE'].mean(),
            knn['test_MAE'].mean()],

        'XGB Regression': [
            gb['test_R2-Square'].mean(),
            gb['test_MSE'].mean(),
            gb['test_MAE'].mean()],

        'DecisionTree Regression': [
            dt['test_R2-Square'].mean(),
            dt['test_MSE'].mean(),
            dt['test_MAE'].mean()],

        'Randomforest Regression': [
            rf['test_R2-Square'].mean(),
            rf['test_MSE'].mean(),
            rf['test_MAE'].mean()]},

        index=['R2', 'MSE', 'MAE'])

    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmin(axis=1)
    # Return models performance metrics scores data frame
    return (models_scores_table)

#Applying Function
Regression(X,Y,5).to_excel('D:/Rollingpinn/MODEL BUILDING/Web/Model1/Validation/Basemodel/Basemodel-InbuiltFunction.xlsx')


                            'FEATURE SELECTION'
#-BACKWARD ELIMINATION
rf = RandomForestRegressor()
rfb = sfs(estimator = rf, k_features = 'best', forward = False,
                     verbose = 2, scoring = 'r2')
sfs_backward = rfb.fit(xtrain, ytrain)
print('Features selelected using backward elimination are: ')
print(sfs_backward.k_feature_names_)
print('\nR-Squared: ', sfs_backward.k_score_)

#FORWARD SELECTION
rf = RandomForestRegressor()
rfb = sfs(estimator = rf, k_features = 'best', forward = True,
                     verbose = 2, scoring = 'r2')
sfs_backward = rfb.fit(xtrain, ytrain)
print('Features selelected using backward elimination are: ')
print(sfs_backward.k_feature_names_)
print('\nR-Squared: ', sfs_backward.k_score_)

#RECURSIVE FEATURE ELIMINATION
rf = RandomForestRegressor()
rfe_model = RFE(estimator=rf, n_features_to_select = 12)
rfe_model = rfe_model.fit(xtrain, ytrain)
feat_index = pd.Series(data = rfe_model.ranking_, index = xtrain.columns)
print(feat_index)

                       'ENSEMBLE MODEL BUILDING'
#1-bagging model
#2-boosting model





#MODEL EVALUATION
def Regressionerrormetric(model):
    ypred = model.predict(xtest)
    scorecard = pd.DataFrame({
        'Mean Absolute Error': metrics.mean_absolute_error(ytest, ypred),
        'Mean Squared Error': metrics.mean_squared_error(ytest, ypred),
        'Root Mean Squared Error': np.sqrt(((ypred - ytest) ** 2).mean()),
        'Mean Absolute Percentage error' : np.mean(np.abs((ytest - ypred)/ytest))*100},
        'Mean Squared Log Error':mean_squared_log_error( ytest, ypred),
        'Root Mean Square Log error' : np.sqrt(mean_squared_log_error( ytest, ypred )),
        index=['ERROR', 'MSE', 'RMSE','MAPE','MSLE','RMSLE'])
    return scorecard.head(1)

a = Regressionerrormetric(model1)
b = Regressionerrormetric(model2)
c = Regressionerrormetric(model3)

Experiments = pd.DataFrame()
Experiments['ModelName'] = ['Bagging Model','Boosting Model','Bagging&Boosting']
Experiments = pd.concat([a,b,c],axis=0)
print(Experiments)

#FINAL MODEL





#FEATURE IMPORTANCE
model = RandomForestRegressor()
model.fit(X, y)
importance = model.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
important_features = pd.DataFrame({'Features': X_train_xfs.columns,'Importance': xgb_model.feature_importances_})
fe_imp=important_features.sort_values(by='Importance',ascending=False)






'SINGLE MESSAGE'
import pywhatkit
pywhatkit.sendwhatmsg('+919790766998','HI',12,19)

'BULK MESSAGES'
# Program to send bulk customized message through WhatsApp web application
# Author @inforkgodara

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
import pandas
import time

# Load the chrome driver
driver = webdriver.Chrome('/path/to/chromedriver')
count = 0

# Open WhatsApp URL in chrome browser
driver.get("https://web.whatsapp.com/")
wait = WebDriverWait(driver, 20)

# Read data from excel
excel_data = pandas.read_excel('D:/Customer bulk email data.xlsx', sheet_name='Customers')

# Iterate excel rows till to finish
for column in excel_data['Name'].tolist():
    # Assign customized message
    message = excel_data['Message'][0]

    # Locate search box through x_path
    search_box = '//*[@id="side"]/div[1]/div/label/div/div[2]'
    person_title = wait.until(lambda driver:driver.find_element_by_xpath(search_box))

    # Clear search box if any contact number is written in it
    person_title.clear()

    # Send contact number in search box
    person_title.send_keys(str(excel_data['Contact'][count]))
    count = count + 1

    # Wait for 2 seconds to search contact number
    time.sleep(2)

    try:
        # Load error message in case unavailability of contact number
        element = driver.find_element_by_xpath('//*[@id="pane-side"]/div[1]/div/span')
    except NoSuchElementException:
        # Format the message from excel sheet
        message = message.replace('{customer_name}', column)
        person_title.send_keys(Keys.ENTER)
        actions = ActionChains(driver)
        actions.send_keys(message)
        actions.send_keys(Keys.ENTER)
        actions.perform()

# Close chrome browser
driver.quit()

