from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    构造函数，传递列参数用于列抽取
    可以加入一些打印信息，看看执行的流程
    """

    def __init__(self, feature_names):
        self.feature_names = feature_names
        print('FeatureSelector init exce...')

    # 返回对象本身
    def fit(self, X, y=None):
        print('FeatureSelector fit exce...')
        return self

    # 我们需要重写transform方法
    def transform(self, X, y=None):
        print('FeatureSelector transform exce...')
        return X[self.feature_names]


# 构建自定义的分类列Transformer
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, use_dates=['year', 'month', 'day']):
        self._use_dates = use_dates
        print('CategoricalTransformer init exce...')

    def fit(self, X, y=None):
        return self

    def get_year(self, obj):
        return str(obj)[:4]

    def get_month(self, obj):
        return str(obj)[4:6]

    def get_day(self, obj):
        return str(obj)[6:8]

    def create_binary(self, obj):
        if obj == 0:
            return 'No'
        else:
            return 'Yes'

    def transform(self, X, y=None):
        print('CategoricalTransformer transform exce...')
        for spec in self._use_dates:
            exec("X.loc[:,'{}'] = X['date'].apply(self.get_{})".format(spec, spec))
        X = X.drop(columns=['date'], axis=1)

        X.loc[:, 'view'] = X['view'].apply(self.create_binary)
        X.loc[:, 'waterfront'] = X['waterfront'].apply(self.create_binary)
        X.loc[:, 'yr_renovated'] = X['yr_renovated'].apply(self.create_binary)

        return X.values


# 自定义数值列的转换处理器
class NumericalTransformer(BaseEstimator, TransformerMixin):
    # 构造函数，bath_per_bed ,years_old控制是否计算卧室和时间处理
    def __init__(self, bath_per_bed=True, years_old=True):
        self._bath_per_bed = bath_per_bed
        self._years_old = years_old

    # 直接返回转换器本身
    def fit(self, X, y=None):
        return self

        # 我们编写的自定义变换方法创建了上述特征并删除了冗余特征

    def transform(self, X, y=None):
        if self._bath_per_bed:
            # 创建新列
            X.loc[:, 'bath_per_bed'] = X['bathrooms'] / X['bedrooms']
            # 删除冗余列
            X.drop('bathrooms', axis=1)
        if self._years_old:
            # 创建新列
            X.loc[:, 'years_old'] = 2019 - X['yr_built']
            # 删除冗余列
            X.drop('yr_built', axis=1)

        # 将数据集中的任何无穷大值转换为 Nan
        X = X.replace([np.inf, -np.inf], np.nan)
        # 返回一个 numpy 数组
        return X.values


# 将其保留为数据帧，因为我们的管道被调用
data = pd.read_csv("kc_house_data.csv")
# pandas 数据框提取适当的列
X = data.drop('price', axis=1)
# 您可以将目标变量转换为 numpy
y = np.array(data['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 传递分类管道的分类特征
categorical_features = ['date', 'waterfront', 'view', 'yr_renovated']

# 传递数值管道的数值特征
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                      'condition', 'grade', 'sqft_basement', 'yr_built']

# 定义管道
# 定义分类管道中的步骤
categorical_pipeline = Pipeline(
    [
        ('cat_selector', FeatureSelector(categorical_features)),

        ('cat_transformer', CategoricalTransformer()),

        ('one_hot_encoder', OneHotEncoder(sparse=False))
    ])

# 定义数值管道中的步骤
numerical_pipeline = Pipeline(
    [
        ('num_selector', FeatureSelector(numerical_features)),

        ('num_transformer', NumericalTransformer()),

        ('imputer', SimpleImputer(strategy='median')),

        ('std_scaler', StandardScaler())
    ])

# 将数值和分类管道水平组合成一个完整的大管道
# 使用 FeatureUnion
full_pipeline = FeatureUnion(transformer_list=
    [
        ('categorical_pipeline', categorical_pipeline),

        ('numerical_pipeline', numerical_pipeline)
    ])

# 完整管道作为另一个管道中的一个步骤，将估算器作为最后一步
full_pipeline_m = Pipeline(
    [
        ('full_pipeline', full_pipeline),

        ('model', LinearRegression())
    ])

# 可以像任何其他管道一样调用它
full_pipeline_m.fit(X_train, y_train)

# 可以像任何其他管道一样使用它进行预测
y_pred = full_pipeline_m.predict(X_test)

pd.DataFrame(y_pred).to_csv("ans.csv")


# 最终使用模块进行查看
# 不显示
from sklearn import set_config

set_config(display='diagram')