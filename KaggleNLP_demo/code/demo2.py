# -*- coding: utf-8 -*- 
# @Time : 2022/11/4 23:15 
# @Author : YeMeng 
# @File : demo2.py 
# @contact: 876720687@qq.com
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
pd.set_option('display.max_columns',None)

# --------------------- load data -------------------
data=pd.read_csv("./input/train.csv",nrows=1000)
testdata = pd.read_csv("./input/test.csv")

# data[["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]].corr()

# split data for offline testing randomly
mask = np.random.rand(len(data))<0.8
traindf = data[mask]
validdf = data[~mask]

vectorizer = TfidfVectorizer()
train_transformed = vectorizer.fit_transform(traindf.full_text)
valid_transformed = vectorizer.transform(validdf.full_text)


# -------------------- model training & validation  ---------------------
model = lgb.LGBMRegressor()
for col in tqdm(["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]):
    model.fit(train_transformed, traindf[col])
    validdf[f"{col}_preds"] = model.predict(valid_transformed)


errors=[]
for col in tqdm(["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]):
    error=np.sqrt(mean_squared_error(validdf[col], validdf[f"{col}_preds"]))
    print(f"{col}:", error)
    errors.append(error)
print(f"MCRMSE{col}:", np.mean(error))
# ------------- Whole process & full data training and submission ---------------
# 全量训练
data_transformed = vectorizer.fit_transform(data.full_text)
test_transformed = vectorizer.transform(testdata.full_text)

model = lgb.LGBMRegressor()
for col in tqdm(["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]):
    model.fit(data_transformed,data[col])
    testdata[f"{col}"] = model.predict(test_transformed)

testdata.drop(columns=["full_text"],inplace=True)
testdata.to_csv("submission.csv",index=False)