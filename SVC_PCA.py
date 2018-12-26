# coding:utf-8

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
#Pipelines are initialized with a list of (name, estimator) tuples.
estimators = [('reduce_dim', PCA()), ('svm', SVC())]
clf = Pipeline(estimators)
#这里初始化了一个sklearn的管道类
#初始化的方式是在数组内放置包含估计器以及名称的元祖

#查看初始化的结果
#print(clf)
#print([x for x in dir(clf) if not x.startswith('_')])

#这里写入一个优化器
from sklearn.model_selection import GridSearchCV
params ={'reduce_dim__n_components':[1,5,10,12,15,20],'svm__kernel':['linear','rbf']}
gs = GridSearchCV(clf,param_grid = params)

from sklearn.datasets import make_classification
gs.fit(*make_classification())
print(gs.best_params_)