# -*- coding: utf-8 -*- 
# @Time : 2022/12/6 09:38 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler  # LabelEncoder #OneHotEncoder
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import mutual_info_regression
from sklearn import set_config
import lightgbm as lgb
import xgboost as xgb
set_config(display='diagram')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)


# --------------------------- data processing ------------------------
Xy = pd.read_csv('train.csv', index_col='Id')
X_test = pd.read_csv('test.csv', index_col='Id')
# Remove rows with missing target
Xy = Xy.dropna(axis=0, subset=['SalePrice'])
# Separate target from predictors
X = Xy.drop(['SalePrice'], axis=1)
y = Xy.SalePrice

# Columns with missing values in more than half number of rows
null_cols = [col for col in X.columns if X[col].isnull().sum() > len(X)/2]

X.drop(null_cols,axis=1,inplace=True)
X_test.drop(null_cols,axis=1,inplace=True)

X.drop('Utilities',axis=1,inplace=True)
X_test.drop('Utilities',axis=1,inplace=True)

# Merge the datasets so we can process them together
df = pd.concat([X, X_test])
# df.to_csv("df.csv")


# ------------------- feature engineering --------------
df1 = pd.DataFrame()  # dataframe to hold new features
# Age of House when sold
df1['Age'] = df['YrSold'] - df['YearBuilt']
# Years between Remodeling and sales
df1['AgeRemodel'] = df['YrSold'] - df['YearRemodAdd']
year_cols = ['YrSold', 'YearBuilt', 'AgeRemodel', 'Age']
df_1 = pd.concat([df, df1], axis=1).loc[:, year_cols]

df2 = pd.DataFrame()  # dataframe to hold new features
df2['Remodel'] = df['YearRemodAdd'] != df['YearBuilt']
df2['Garage'] = df['GarageQual'].notnull()
df2['Fireplace'] = df['FireplaceQu'].notnull()
df2['Bsmt'] = df['BsmtQual'].notnull()
df2['Masonry'] = df['MasVnrType'].notnull()

# Converting boolean columns [False,True] into numerical columns [0,1]
df2 = df2.replace([False, True], [0, 1])
object_cols = df.select_dtypes(include=['object']).columns
# Categorical Columns with number of unuque categoies in them
# df[object_cols].nunique().sort_values()

ordinal_cols = [i for i in object_cols if
                ('QC' in i) or ('Qu' in i) or ('Fin' in i) or ('Cond' in i) and ('Condition' not in i)]
df.loc[:, ordinal_cols] = df.loc[:, ordinal_cols].fillna('NA')
# print("Column Names: [Unique Categories in each column]")
# {col: [*df[col].unique()] for col in ordinal_cols}

# 1] Columns with similar ordered categories [Poor<Fair<Typical/Average<Good<Excellent]
ordinal_cols1 = [i for i in object_cols if ('QC' in i) or ('Qu' in i) or ('Cond' in i) and ('Condition' not in i)]
df.loc[:, ordinal_cols1] = df.loc[:, ordinal_cols1].replace(['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])

# 2] Columns with similar ordered categories [No Garage/Basement<Unfinished<Rough Finished<Finished,etc]
ordinal_cols2 = ['BsmtFinType1', 'BsmtFinType2']
df.loc[:, ordinal_cols2] = df.loc[:, ordinal_cols2].replace(['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
                                                            [0, 1, 2, 3, 4, 5, 6])

# 3] Column with ordered categories [No Basement<No Exposure<Mimimum Exposure<Average Exposure<Good Exposure]
ordinal_cols3 = ['BsmtExposure']
df.loc[:, ordinal_cols3] = df.loc[:, ordinal_cols3].fillna('NA')
df.loc[:, ordinal_cols3] = df.loc[:, ordinal_cols3].replace(['NA', 'No', 'Mn', 'Av', 'Gd'], [0, 1, 2, 3, 4])

# 4] Column with ordered categories [Regular<Slightly irregular<Moderately Irregular<Irregular]
ordinal_cols4 = ['LotShape']
df.loc[:, ordinal_cols4] = df.loc[:, ordinal_cols4].replace(['Reg', 'IR1', 'IR2', 'IR3'], [0, 1, 2, 3])

# 5] Column with ordered categories [No Garage<Unfinished<Rough Finished<Finished]
ordinal_cols5 = ['GarageFinish']
df.loc[:, ordinal_cols5] = df.loc[:, ordinal_cols5].replace(['NA', 'Unf', 'RFn', 'Fin'], [0, 1, 2, 3])

# 6] Home functionality Column
ordinal_cols6 = ['Functional']
df.loc[:, ordinal_cols3] = df.loc[:, ordinal_cols3].fillna('Mod')
df.loc[:, ordinal_cols6] = df.loc[:, ordinal_cols6].replace(
    ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"], list(range(8)))

o_columns = ordinal_cols1 + ordinal_cols2 + ordinal_cols3 + ordinal_cols4 + ordinal_cols5 + ordinal_cols6
df.loc[:, o_columns].dtypes.value_counts()
Bath_cols = [i for i in df.columns if 'Bath' in i]
SF_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
df[SF_cols + Bath_cols] = df[SF_cols + Bath_cols].fillna(0)


df3 = pd.DataFrame()  # dataframe to hold new features
df3["Liv_Qual"] = (df.OverallQual + df.OverallCond / 3) * df.GrLivArea
df3["GarageArea_Qual"] = (df.GarageQual + df.GarageCond / 3) * df.GarageArea * df.GarageCars
df3['BsmtArea_Qual'] = (df.BsmtQual * df.BsmtCond / 3) * df.TotalBsmtSF
df3["LivLotRatio"] = df.GrLivArea / df.LotArea
df3["Spaciousness"] = (df['1stFlrSF'] + df['2ndFlrSF']) / df.TotRmsAbvGrd
df3['TotalSF'] = df[SF_cols].sum(axis=1)
df3['TotalBath'] = df.FullBath + df.BsmtFullBath + df.HalfBath / 2 + df.BsmtHalfBath / 2
# df3["Garage_Spaciousness"] = df.GarageArea / (df.GarageCars+1)
# df3["BsmtQual_SF"] = ((df.BsmtQual + df.BsmtCond/2 + df.BsmtExposure/3) * df.TotalBsmtSF) + (df.BsmtFinType1 * df.BsmtFinSF1) + (df.BsmtFinType2 * df.BsmtFinSF2)


df4 = pd.DataFrame()
Porches = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
df4["PorchTypes"] = df[Porches].gt(0.0).sum(axis=1)

df5 = pd.DataFrame()
df5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")

df6 = pd.DataFrame()  # dataframe to hold new features
df6 = pd.get_dummies(df.BldgType, prefix="Bldg")
df6 = df6.mul(df.GrLivArea, axis=0)

df7 = pd.DataFrame()
# df7['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] # df3
df7['OverallQual_TotalSF'] = df['OverallQual'] * df[SF_cols].sum(axis=1)
df7['OverallQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
df7['OverallQual_TotRmsAbvGrd'] = df['OverallQual'] * df['TotRmsAbvGrd']
df7['GarageArea_YearBuilt'] = df['GarageArea'] + df['YearBuilt']

# Note: `MSSubClass` feature is read as an `int` type, but is actually a (nominative) categorical.

features_nom = ["MSSubClass"] + list(df.select_dtypes('object').columns)

# Cast each of the above 23 columns into 'category' DataType
for name in features_nom:
    df[name] = df[name].astype("category")
    # Add a None category for missing values
    if "None" not in df[name].cat.categories:
        df[name] = df[name].cat.add_categories("None")

# Label encoding for categoricals
for colname in df.select_dtypes(["category"]):
    df[colname] = df[colname].cat.codes

df = pd.concat([df, df1, df2, df3, df4, df5, df6, df7], axis=1)
for i in ['GarageArea','GarageCars','GarageYrBlt','Functional','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1','MasVnrArea','LotFrontage','GarageArea_YearBuilt','GarageArea_Qual']:
    df[i] = df[i].fillna(0)
# df=df.dropna(axis=0)


# Reform splits
X = df.loc[X.index, :]
X_test = df.loc[X_test.index, :]
X_y = X.copy()
X_y['SalesPrice'] = y

train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2) # it would cause index mxture
train_y = np.log(train_y)
val_y = np.log(val_y)
## ----------------------------- model 1 rf ----------------------
model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = LGBMRegressor()
# model = XGBRegressor()


pipe = Pipeline([('scaler', StandardScaler()),
                 ('reduce_dim', PCA()),
                 ('regression', model)])
pipe.fit(train_x, train_y)
train_pred = pipe.predict(train_x)
val_pred = pipe.predict(val_x)



# # ----------------------- 1 opt -------------------
n_estimators = [int(x) for x in np.linspace(start=10, stop=500, num=10)]
max_features = [x+1 for x in range(11)]
max_depth = [int(x) for x in np.linspace(start=1, stop=11, num=5)]
min_samples_split = [int(x) for x in np.linspace(start=2, stop=50, num=5)]
params = {'n_estimators': n_estimators,
          'max_depth': max_depth,
          'max_features': max_features,
          'min_samples_split': min_samples_split
          }


# ------------------ model opt-----------------
model_cv = RandomizedSearchCV(model, params, cv=5, n_iter=10,n_jobs = -1)
# model_cv = GridSearchCV(model, params, cv=5, n_jobs = -1)
model_cv.fit(train_x, train_y)
val_pred = model_cv.predict(val_x)
train_pred = model_cv.predict(train_x)

best_estimator = model_cv.best_estimator_
print(best_estimator)
print(model_cv.best_score_)


# ---------------------- model performance ----------------
print('Training RMSE:{}'.format(np.sqrt(mean_squared_error(train_y, train_pred))))
print('Test RMSE:{}'.format(np.sqrt(mean_squared_error(val_y, val_pred))))
print('Training R-squared:{}'.format(r2_score(train_y, train_pred)))
print('Test R-squared:{}'.format(r2_score(val_y, val_pred)))


# ---- output profile -------
# X_test = pd.DataFrame(X_test)
submission = model_cv.predict(X_test)
submission = pd.DataFrame(pow(np.e, submission))
submission.to_csv("submission")