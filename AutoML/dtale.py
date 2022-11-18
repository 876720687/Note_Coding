# -*- coding: utf-8 -*- 
# @Time : 2022/11/18 19:05 
# @Author : YeMeng 
# @File : dtale.py 
# @contact: 876720687@qq.com
# autovisualizaion
##＃ Importing Seaborn Library For Some Datasets
import seaborn as sns

### Printing Inbuilt Datasets of Seaborn Library
print(sns.get_dataset_names())

### Loading Titanic Dataset
df = sns.load_dataset("titanic")

### Importing The Library
import dtale

### Generating Quick Summary
dtale.show(df)