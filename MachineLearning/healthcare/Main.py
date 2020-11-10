import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pandas.api.types import is_numeric_dtype
import seaborn as sns
###################################################################
# Data Load
###################################################################

data_path = 'healthcare'
train_f_name = 'train_data_2.csv'
test_fname = 'test_data.csv'

train_data = pd.read_csv(os.path.join(data_path, train_f_name))
test_data = pd.read_csv(os.path.join(data_path, test_fname))
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

r_scaler = RobustScaler()
s_scaler = StandardScaler()
m_scaler = MinMaxScaler()

###################################################################
# Inspection
# foundWrongValue...Wrong value may actually be subjective.
# Outlier and Wrong Value may not be well discerned, but since there are not many, we searched for found and removed them manually.
##################################################

# Check if there are outliers through the plot.
# There were no outliers in this project.
##################################################
def plotSeries(data,key):
    plt.title(key)
    plt.hist(data[key])
    plt.show()

# Preprocessing and Inspection
# print K_best value of data.
# Data divided into categorical data can also be checked
#################################################
def select_k_best(X, Y, n, visualization=False):
    # Num of Top feature select...
    bestFeatures = SelectKBest(score_func=chi2, k=n)
    fit = bestFeatures.fit(X, Y)
    dfColumns = pd.DataFrame(X.columns)
    dfscores = pd.DataFrame(fit.scores_)
    featureScores = pd.concat([dfColumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    print(featureScores.nlargest(n, 'Score'))
    if visualization:
        df = pd.concat([X, Y], axis=1)
        plt.figure(figsize=(20, 20))
        g = sns.heatmap(df.corr(), annot=True, cmap="RdYlGn")
        plt.show()
    return featureScores.nlargest(n,'Score')

def found_wrong_value(data):
    for idx in data:
        print('---------------------------------------------------')
        print(idx)
        print(data[idx].unique())
        #print(data[idx].min())
        #print(data[idx].max())
        #print(data[idx].mean())
        print('---------------------------------------------------')
def foundNan(df):
    print(df.isna().any())
    print(df.isna().sum())

foundNan(train_data)
found_wrong_value(train_data)


###################################################################
# Preprocessing
###################################################################
# Perform scaling according to the entered scaler.
def ScalingData(data, scaler):
    df_scaled = scaler.fit_transform(data)
    return pd.DataFrame(df_scaled, columns=data.columns)

# Encoding String value to Numeric Value
def columnEncoding(col,df):
    for c in col:
        # get All distinguish Value of column
        print(df[c].dtypes)
        print(df[c].isna().sum())
        if is_numeric_dtype(df[c]):
            continue
        c_unique = df[c].unique()
        encoder = LabelEncoder()
        encoder.fit(df[c])
        df[c] = encoder.transform(df[c])
        result_unique = df[c].unique()
        print("%s is replaced %s"%(c_unique,result_unique)) # print replaced label compared origin label
    return df

# Part that determines how to handle na value of dataframe
def handleNa(col, df,work="mean"):
    # fill na value for 'mean'
    if work=="mean":
        for c in col:
            mean = df[c].mean()
            df[c]=df[c].fillna(mean)
    # fill na value for 'median'
    elif work=="median":
        for c in col:
            median = df[c].median()
            df[c]=df[c].fillna(median)
    # fill na value for 'mode'
    elif work=="mode":
        for c in col:
            mode = df[c].mode()[0]
            df[c]=df[c].fillna(mode)
    # drop row which contains na value
    elif work=="drop":
        df = df.dropna(subset=col)
    return df

# City Code Patient, Bed grade 2개에 Nan값이 존재, handle Na가 필요하다.
# 둘 다 Categorical data, mode값을 이용해 채우는 것이 좋다고 판단하였다.

train_data = handleNa(['City_Code_Patient', 'Bed Grade', 'Stay','Age'], train_data, work="drop")
train_data= columnEncoding(train_data.columns, train_data)

#m_scaler만 되는 이유 분석이 필요.
#train data의 축소가 필요.

X = train_data.iloc[:, :-1]
X=ScalingData(X, m_scaler)
Y = train_data.iloc[:, -1]
X, _, Y, _ = train_test_split(X, Y, test_size=0.8, random_state=42)

select_k_best(X, Y, len(X.columns), visualization=True)
#nan이 존재하는 column만 따로 분류
categorical =['City_Code_Patient', 'Bed Grade']
numeric =[]
###################################################################
# Modeling
###################################################################
#Emsemble
#SVM
#KNN --> 각각 parameter화
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
param_rf = {
    'n_estimators':np.arange(50, 210, 40),
    'criterion':['gini','entropy'],
    'max_depth':[None,1,3,5,10, 15, 20],#
    'bootstrap':[True, False]
}
param_bagging={
    'n_estimators':np.arange(4,20,2),
    'bootstrap':[True,False],
}
param_svc={
    'C':[0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'rbf'],
    'degree':[3, 4, 5, 6],
    'gamma':['scale', 'auto']
}
param_knn={
    'weight':['uniform','distance'],
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size':[5, 10, 30, 50, 100],
    'p':[1,2,3,4,5]
}

rf = RandomForestClassifier()
bagging = BaggingClassifier()
svc=SVC()
knn = KNeighborsClassifier()

def process(X, Y, model, param, cv=10):
    grid_search =GridSearchCV(model, param_grid=param, cv=cv)
    grid_search.fit(X, Y)
    return grid_search
print('process start')
rf_result=process(X, Y, rf, param_rf)
rf_result = pd.DataFrame(rf_result.cv_results_)
rf_result.to_csv('rf_preprocessed.csv')
print('process done')
#프로세스가 굉장히 오래걸린다. 이를 해결할 수 있는 방법은?
###################################################################
# Evaluation
###################################################################
