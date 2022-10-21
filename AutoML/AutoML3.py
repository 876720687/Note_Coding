# -*- coding: utf-8 -*- 
# @Time : 2022/9/24 20:42 
# @Author : YeMeng 
# @File : AutoML3.py 
# @contact: 876720687@qq.com
# TODO 不是很理解



from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import pandas as pd
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion, Pipeline




#
# model = Pipeline(
#     [
#         # ('vectorizer', TfidfVectorizer()),
#         ("union", FeatureUnion(transformer_list=
#             [
#                 ("handpicked", TfidfVectorizer(vocabulary=vocab)),
#                 ("bigrams", TfidfVectorizer(ngram_range=(2, 2)))
#             ])
#         ),
#         ("classifier", LinearSVC(class_weight='balanced')),
#     ]
# )


def extract_feature_names(model, name):
    """从任意 sklearn 模型中提取特征名称

    Args:
      model: The Sklearn model, transformer, clustering algorithm, etc. 我们想要获得命名特征
      name: 我们所在管道中当前步骤的名称.

    Returns:
      特征名称列表。如果模型没有命名特征，它会构造特征名称 通过将索引附加到提供的名称.
    """
    if hasattr(model, "get_feature_names"):
        return model.get_feature_names()
    elif hasattr(model, "n_clusters"):
        return [f"{name}_{x}" for x in range(model.n_clusters)]
    elif hasattr(model, "n_components"):
        return [f"{name}_{x}" for x in range(model.n_components)]
    elif hasattr(model, "components_"):
        n_components = model.components_.shape[0]
        return [f"{name}_{x}" for x in range(n_components)]
    elif hasattr(model, "classes_"):
        return classes_
    else:
        return [name]


def get_feature_names(model, names, name):
    """这个方法从 Sklearn 管道中按顺序提取特征名称 此方法仅适用于组合管道和 FeatureUnions。
它会 使用 DFS 从模型中提取所有名称

    Args:
    model: model
    names: 最终特征化步骤的名称列表
    name: 我们要评估的步骤的当前名称


    Returns:
    feature_names: 从管道中提取的特征名称列表。
    """
    # 检查名称是否是我们的特征步骤之一。这是基本情况。
    if name in names:
        # 如果它具有 named_steps 属性，则它是一个管道，我们需要访问这些功能
        if hasattr(model, "named_steps"):
            return extract_feature_names(model.named_steps[name], name)
        # 否则直接获取特征
        else:
            return extract_feature_names(model, name)
    elif type(model) is Pipeline:
        feature_names = []
        for name in model.named_steps.keys():
            feature_names += get_feature_names(model.named_steps[name], names, name)
        return feature_names
    elif type(model) is FeatureUnion:
        feature_names= []
        for name, new_model in model.transformer_list:
            feature_names += get_feature_names(new_model, names, name)
        return feature_names
    # 如果以上都不是，返回空
    else:
        return []



vocab = {"worst": 0, "awful": 1, "waste": 2,
         "boring": 3, "excellent": 4}


model = Pipeline(
    [
        ("union", FeatureUnion(transformer_list=
            [
                ("h1", TfidfVectorizer(vocabulary={"worst": 0})),
                ("h2", TfidfVectorizer(vocabulary={"best": 0})),
                ("h3", TfidfVectorizer(vocabulary={"awful": 0})),
                ("tfidf_cls", Pipeline(
                    [
                        ("vectorizer", CountVectorizer()),
                        ("transformer", TfidfTransformer()),
                        ("tsvd", TruncatedSVD(n_components=2))
                    ]
                ))
            ])
        ),
        ("classifier", LinearSVC(C=1.0, class_weight="balanced")),
    ])



imdb_data = load_dataset('imdb')
x_train = [x["text"]for x in imdb_data["train"]]
y_train = [x["label"]for x in imdb_data["train"]]
model.fit(x_train, y_train)

# # G获取特征的名字
# feature_names = model.named_steps["vectorizer"].get_feature_names()
# print(feature_names)
# # 获取每个特征的系数，由于我们使用的是svm，所以获取的是特征系数，其代表了每个特征的重要度
# # 返回的数组是一个二维数组，我们平铺方便后面画图
# coefs = model.named_steps["classifier"].coef_.flatten()
# print(coefs)

get_feature_names(model,)