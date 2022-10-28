import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import Lasso

data_train = pd.read_csv('../data/train_data.csv')
data_test_a = pd.read_csv('../data/test_data.csv')
features = [f for f in data_train.columns if f not in ['id','issueDate','isDefault'] and '_outliers' not in f]
x_train = data_train[features]
x_test = data_test_a[features]
y_train = data_train['isDefault']

train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size=0.2)

model_list = [XGBClassifier(),
              GradientBoostingClassifier(),
              RandomForestClassifier(),
              DecisionTreeClassifier(),
              MultinomialNB(),
              LinearSVC(),
              Lasso(alpha=0.005)]

for model in model_list:
    model.fit(train_x, train_y)
    val_pred = model.predict(val_x)
    fpr, tpr, threshold = metrics.roc_curve(val_y, val_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print('model name is {} and roc_auc is {}'.format(model, roc_auc))

